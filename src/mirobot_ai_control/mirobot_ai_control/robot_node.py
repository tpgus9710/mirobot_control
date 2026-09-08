import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Float32MultiArray, Bool
import serial
import time
import threading

# 미로봇 하드웨어 시리얼 인터페이스 및 ROS 2 통신 처리 드라이버 클래스
class MirobotDriverNode(Node):

    # WLKATA Mirobot 하드웨어 매뉴얼(p.55~56) 기준 실제 가동범위.
    #
    # [주의] 매뉴얼 값이 실제보다 느슨한 축이 있다. J2 하한은 매뉴얼상 -60 이지만
    # 실기는 -35 에서 멈춘다(2026-09-03 슬라이더로 확인). 이 표가 하드웨어보다
    # 느슨하면 범위 밖 명령이 경고 없이 시리얼로 나가고, 컨트롤러는 명령을 받고도
    # 자세를 못 만든다 — 로그만 보면 정상으로 보여 원인을 찾기 매우 어렵다.
    # 실측으로 확인한 축은 실측값을 쓴다. 상한은 아직 확인하지 못했다.
    # AI/GUI 등 상위 노드가 어떤 값을 보내든, 이 노드가 시리얼로 내보내기 직전
    # 마지막으로 한 번 더 검증하는 최종 방어선 역할만 함.
    JOINT_HARD_LIMITS = [
        (-100.0, 100.0),  # J1
        (-35.0,   90.0),  # J2  ← 하한은 매뉴얼(-60)이 아니라 실측값이다
        (-180.0,  50.0),  # J3
        (-180.0, 180.0),  # J4
        (-180.0,  40.0),  # J5
        (-180.0, 180.0),  # J6
    ]

    # ── [재설계 v2] "Idle 텔레메트리 기반 판정" → "ok 응답 기반 판정" ──────
    #
    # v1에서는 <Idle,...> 상태 텔레메트리를 보고 로봇이 멈췄는지 판단하려 했는데,
    # 효과가 거의 없었음. 이 컨트롤러는 GRBL처럼 Run/Idle을 실시간으로 구분해서
    # 스트리밍하는 게 아니라, 명령 하나를 물리적으로 다 처리할 때까지 블로킹하고
    # 있다가 끝나야 'ok'를 돌려주는 단순(동기식) 방식일 가능성이 높음 — 실제로
    # "초기 버전"이 '명령 보내고 ok 받으면 다음 명령' 방식으로 잘 동작했다는
    # 사실도 이 가설과 일치함.
    #
    # 그래서 이번엔 'ok' 응답 자체를 "직전 명령이 물리적으로 완료됐다"는 신호로
    # 사용함. 그리퍼(M3)도 같은 시리얼 채널에서 'ok'를 주고받으므로, 팔 이동과
    # 그리퍼 명령이 서로의 'ok'를 가로채 오판하지 않도록 하나의 대기열로 통합해서
    # "항상 시리얼 상에 명령이 하나만 떠 있는" 상태를 보장함.
    BUSY_TIMEOUT = 3.0  # 이 시간 안에 'ok'가 안 오면 안전을 위해 강제로 busy 해제
                        # (노이즈 등으로 'ok' 파싱을 놓쳐서 영구 정지되는 것 방지)

    # 드라이버 노드 초기화 및 구독자/퍼블리셔 설정
    def __init__(self):
        super().__init__('mirobot_driver_node')
        
        # ROS 2 파라미터 선언 및 연결 포트 설정
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        
        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baudrate').get_parameter_value().integer_value
        
        # 동시성 제어를 위한 시리얼 쓰기 락 및 안전 호밍 플래그 선언
        self.serial_lock = threading.Lock()
        self.is_homing = False  # 드라이버 구동 시작 시 False로 초기화

        self.last_limit_warn_time = 0.0
        self.limit_warn_interval = 1.0  # 범위 초과 경고 로그 스팸 방지 (1초당 최대 1회)

        # 컨트롤러가 스스로 밀어 주는 "<Idle,...>" 텔레메트리에는 실제 관절각이
        # 들어 있는데, 예전에는 맨 앞 상태 토큰만 쓰고 나머지를 버렸다. 그래서
        # "지금 팔이 몇 도인가"를 볼 방법이 시스템에 아예 없었다 — GUI 슬라이더도
        # ai_node 로그도 전부 '보낸 값'이지 '도달한 값'이 아니다.
        # 보낸 값과 도달한 값이 어긋나는 상황(가동범위 밖 명령 등)을 잡으려면
        # 실제 값이 반드시 필요하므로, 원문 그대로 주기 로그로 남긴다.
        # 0 으로 두면 끈다. 별도의 '?' 질의는 보내지 않는다 — 이 드라이버는
        # 시리얼에 명령이 하나만 떠 있게 유지하는 설계라 질의를 끼워 넣으면
        # ok 카운트가 어긋난다.
        self.telemetry_log_interval = 1.0
        self.last_telemetry_log_time = 0.0

        # ── 시리얼 링크 상태 ────────────────────────────────────────────
        # USB 시리얼은 끊긴다. 파이 전원이 순간 처지면 CH341 이 통째로 재열거되고
        # (dmesg 에 'ch341-uart converter now attached to ttyUSB0' 가 다시 찍힌다)
        # 그 순간 진행 중이던 write 가 [Errno 5] 로 실패한다.
        #
        # 예전에는 그 예외가 콜백 밖으로 튀어나가 rclpy.spin 을 무너뜨려서
        # 노드가 통째로 죽었다. USB 는 곧 다시 붙는데 드라이버만 사라지는
        # 상황이라, "로봇이 멈췄다"로 보였다. 이제는 링크만 끊긴 것으로 보고
        # 노드는 살려 둔 채 재연결한다.
        self.port = port
        self.baud = baud
        self.link_ok = False          # 시리얼이 살아 있고 명령을 보내도 되는가
        self.link_lost_logged = False
        # 재연결 직후에는 컨트롤러가 리셋됐는지 알 수 없다. 위치를 모르는 채
        # G0 를 보내면 엉뚱한 자세로 급이동하므로, 사용자가 호밍하기 전까지는
        # 관절 명령을 내보내지 않는다. 자동 호밍은 일부러 하지 않는다 —
        # 저전압이 반복되면 그때마다 팔이 움직여 오히려 위험하다.
        self.needs_homing_after_reconnect = False
        self.RECONNECT_DELAY_SEC = 3.0

        # ── 단일 대기열 기반 큐잉 상태 ──────────────────────────────────────
        # 팔(joint)과 그리퍼(gripper) 명령을 각각 "가장 최신 값"으로만 저장해두고,
        # 시리얼 상에 명령이 하나도 떠 있지 않을 때(busy==False)만 순서대로 하나씩 꺼내 보냄.
        self.pending_joint_target   = None  # 아직 전송 안 된 최신 팔 목표값 (list[6])
        self.pending_gripper_target = None  # 아직 전송 안 된 최신 그리퍼 목표값 (float)
        self.busy = False          # 시리얼로 명령을 보내고 아직 'ok'를 못 받은 상태
        self.busy_since = 0.0      # busy로 전환된 시각 (타임아웃 안전장치용)
        self._expected_ok_count = 0  # busy 진입 시점의 ok_seen_count 기준값

        # [측정] 'ok' 응답 지연 계측용. 제어 경로에서는 읽지 않는다 —
        # 순수하게 진단 목적이라 값이 낡아도 동작에 영향이 없다.
        self._ok_lat = []
        self._ok_mv = []
        self._ok_csv = None
        self._ok_t0 = time.time()
        self._ok_last_summary = 0.0
        self._last_sent_joints = None
        self._pending_move_deg = 0.0

        # 상시 시리얼 리더 스레드가 채워주는 공유 상태
        self.robot_status = 'Unknown'  # 마지막으로 파싱된 상태 토큰 (호밍 재확인용으로만 사용)
        self.idle_streak = 0           # 연속으로 확인된 Idle 상태 횟수 (호밍용)
        self.ok_seen_count = 0         # 지금까지 수신한 'ok' 라인 누적 개수
        
        # 실시간 홈 복귀 가동 현황 공유용 퍼블리셔 추가
        self.homing_status_pub = self.create_publisher(Bool, '/mirobot/homing_status', 10)
        
        # 시리얼 통신 초기화 및 미로봇 연결 시도
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self.get_logger().info(f"Connected to Mirobot on {port}")
            time.sleep(2)  # 아두이노/컨트롤러 리셋 대기 시간

            # 상시 시리얼 리더 스레드 시작 (노드 생명주기 내내 동작)
            self.reader_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
            self.reader_thread.start()

            self.link_ok = True
            self.init_robot()
        except Exception as e:
            self.get_logger().error(f"Failed to connect to serial port: {e}")
            self.ser = None
            self.is_homing = False
            self.link_ok = False

        # 링크가 끊기면 되살리는 감시 스레드. 최초 연결 실패에도 계속 재시도한다.
        self.reconnect_thread = threading.Thread(
            target=self._reconnect_loop, daemon=True)
        self.reconnect_thread.start()

        # 6개 관절 각도 명령 수신 구독자 설정
        self.joint_sub = self.create_subscription(
            Float32MultiArray,
            '/mirobot/joint_commands',
            self.joint_callback,
            10
        )
        
        # 실측 데이터 기반 그리퍼 듀티 수치 수신 구독자 설정
        self.gripper_sub = self.create_subscription(
            Float32,
            '/mirobot/gripper',
            self.gripper_callback,
            10
        )

        # GUI 등 외부 수동 명령 및 동적 Homing($H)을 캐치할 raw command 구독자 추가
        self.raw_sub = self.create_subscription(
            String,
            '/mirobot/raw_commands',
            self.raw_callback,
            10
        )

        # 대기열에서 다음 명령을 꺼내 보낼 수 있는지 계속 확인하는 타이머 (33Hz)
        self.send_timer = self.create_timer(0.03, self._try_send_pending)

    # ── 시리얼 쓰기 (실패해도 노드를 죽이지 않는다) ──────────────────────────
    def _write(self, data, what=""):
        """시리얼에 쓴다. 실패하면 링크를 끊긴 것으로 표시하고 False 를 돌려준다.

        이 함수를 거치지 않는 ser.write 를 새로 만들지 말 것. 콜백이나 타이머
        안에서 예외가 나면 rclpy 가 그대로 노드를 내린다.
        """
        if not self.ser or not self.ser.is_open or not self.link_ok:
            return False
        try:
            with self.serial_lock:
                self.ser.write(data)
            return True
        except Exception as e:
            self._mark_link_lost(f"{what} 전송 실패: {e}")
            return False

    def _mark_link_lost(self, reason):
        if self.link_ok:
            self.get_logger().error(
                f"시리얼 링크가 끊겼습니다 — {reason}. "
                f"{self.RECONNECT_DELAY_SEC:.0f}초마다 재연결을 시도합니다.")
        self.link_ok = False
        # 끊긴 동안 쌓인 목표값은 버린다. 되살아난 뒤 낡은 값이 그대로 나가면
        # 로봇이 갑자기 그 자세로 급이동한다(호밍 직후와 같은 사고 경로).
        self.pending_joint_target = None
        self.pending_gripper_target = None
        self.busy = False
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass

    def _reconnect_loop(self):
        """링크가 끊겨 있으면 주기적으로 다시 연다."""
        while rclpy.ok():
            if self.link_ok:
                time.sleep(0.5)
                continue
            time.sleep(self.RECONNECT_DELAY_SEC)
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=1)
            except Exception as e:
                if not self.link_lost_logged:
                    self.get_logger().warn(f"재연결 대기 중 — {e}")
                    self.link_lost_logged = True
                continue
            self.link_lost_logged = False
            self.link_ok = True
            self.needs_homing_after_reconnect = True
            self.is_homing = False
            self.busy = False
            self.get_logger().warn(
                f"시리얼 재연결됨 ({self.port}). 다만 컨트롤러가 리셋됐는지 알 수 "
                "없어 현재 관절 위치를 신뢰할 수 없습니다. "
                "GUI 의 호밍 버튼을 누르기 전까지 관절 명령을 보내지 않습니다.")

    # ── 상시 시리얼 리더 스레드 ──────────────────────────────────────────────
    def _serial_reader_loop(self):
        while rclpy.ok():
            if not self.ser or not self.ser.is_open:
                time.sleep(0.1)
                continue
            try:
                raw = self.ser.readline()  # timeout=1초라서 데이터 없으면 최대 1초 후 빈 바이트 반환
            except Exception as e:
                # 읽기 실패도 대개 USB 가 빠진 것이다. 쓰기 쪽과 같은 경로로 처리해
                # 재연결 스레드가 되살리게 한다.
                self._mark_link_lost(f"읽기 실패: {e}")
                time.sleep(0.05)
                continue

            if not raw:
                continue  # 타임아웃, 새 데이터 없음

            line = raw.decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            self._handle_serial_line(line)

    def _handle_serial_line(self, line):
        low = line.strip().lower()
        if low == "ok":
            self.ok_seen_count += 1
            return

        if line.startswith("<"):
            # 예: "<Idle,MPos:...>" → 첫 콤마 앞의 토큰만 추출. 호밍 재확인용으로만 사용.
            token = line[1:].split(',')[0].split(':')[0].strip()
            self.robot_status = token
            if token == "Idle":
                self.idle_streak += 1
            elif token in ("Alarm", "Home", "Run", "Jog"):
                self.idle_streak = 0

            # 실제 관절각 관찰용. 파싱하지 않고 원문을 그대로 남긴다 —
            # 펌웨어마다 필드 이름과 순서가 달라, 섣불리 해석하면 틀린 값을
            # 확신에 차서 찍게 된다. 눈으로 보고 필요하면 그때 파서를 붙일 것.
            if self.telemetry_log_interval > 0:
                now = time.time()
                if now - self.last_telemetry_log_time >= self.telemetry_log_interval:
                    self.last_telemetry_log_time = now
                    self.get_logger().info(f"[실제상태] {line}")

    # 로봇 부팅 시 비동기 스레드로 홈 복귀 시퀀스 가동 (실행 병목 제거)
    def init_robot(self):
        if self.ser and self.ser.is_open:
            # 스레드가 실제로 뜨기 전의 짧은 틈에도 joint_callback이 새 목표값을
            # 큐에 쌓을 수 있으므로, 스레드 기동 전에 is_homing을 먼저 걸고
            # 혹시 이미 쌓여있던 목표값(대기 큐)도 여기서 미리 비워둠.
            self.is_homing = True
            self.pending_joint_target = None
            self.pending_gripper_target = None
            threading.Thread(target=self.run_homing_sequence, daemon=True).start()

    # 홈 복귀 완수를 모니터링하는 코어 가동 루프 수행.
    def run_homing_sequence(self):
        if not self.ser or not self.ser.is_open:
            return
            
        self.get_logger().info("하드웨어 호밍(원점 정렬)을 시작합니다. 완료될 때까지 다른 명령은 무시됩니다...")
        # 호출부(raw_callback/init_robot)에서 이미 걸어뒀겠지만, 이 함수가 다른 경로로
        # 호출될 가능성까지 대비해 여기서도 한 번 더 확실히 걸고 큐를 비움(idempotent).
        self.is_homing = True

        # 콜백의 is_homing 가드만으로는 '검사 후 대입' 사이에 호밍이
        # 시작되는 경합을 막지 못한다. 그래서 여기서 한 번 비운다.
        self.pending_joint_target = None
        self.pending_gripper_target = None
        
        status_msg = Bool()
        status_msg.data = True
        self.homing_status_pub.publish(status_msg)
        
        with self.serial_lock:
            self.ser.reset_input_buffer()
            self.ok_seen_count = 0
            self.idle_streak = 0
            ok_before = self.ok_seen_count
        if not self._write(b"$H\r\n", "호밍($H)"):
            self.get_logger().error("호밍 명령을 보내지 못했습니다 — 링크 끊김")
            self.is_homing = False
            return

        start_time = time.time()
        homing_acked = False
        while time.time() - start_time < 35.0:
            if self.ok_seen_count > ok_before:
                homing_acked = True
                break
            time.sleep(0.05)

        if not homing_acked:
            self.get_logger().warn("호밍 완료 ACK('ok')를 받지 못했습니다. 안전을 위해 추가 확인 후 진행합니다.")

        confirm_deadline = time.time() + 5.0
        while time.time() < confirm_deadline and self.idle_streak < 3:
            time.sleep(0.05)

        if self.idle_streak < 3:
            self.get_logger().warn("Idle 상태를 충분히 확인하지 못했습니다. 그래도 진행하지만 결과를 주의 깊게 확인하세요.")

        self._write(b"O105\r\n", "O105")
        time.sleep(0.5)

        # 호밍 중 어떤 경로로든 큐에 쌓였을 가능성까지 마지막으로 차단한다.
        # is_homing / busy 를 푸는 순간 _try_send_pending 이 남아있던 낡은
        # 목표값을 그대로 전송해 급이동하므로, 푸는 것보다 먼저 비워야 한다.
        self.pending_joint_target = None
        self.pending_gripper_target = None
        self.is_homing = False
        # 호밍이 끝났으면 자세를 다시 신뢰할 수 있다. 재연결 차단을 푼다.
        self.needs_homing_after_reconnect = False
        # 호밍 직후 로봇은 실제로 정지·대기 상태이므로 busy를 확실히 풀어줌
        self.busy = False
        self.get_logger().info("Mirobot 호밍 및 가동 초기화가 완전히 완료되었습니다!")
        
        status_msg.data = False
        self.homing_status_pub.publish(status_msg)

    # 실시간 동적 수동 G-Code 및 외부 $H 원격 입력에 대한 G-Code 시리얼 쓰기 핸들러 기능 수행
    def raw_callback(self, msg):
        if self.ser and self.ser.is_open:
            cmd = msg.data.strip()
            if cmd == "$H":
                if not self.is_homing:
                    # [호밍 전 목표값 잔류 방지] 스레드가 실제로 뜨기 전 짧은 틈에도
                    # joint_callback이 새 목표값을 큐에 쌓을 수 있고, 무엇보다
                    # "호밍이 시작되기 직전 이미 큐에 대기 중이던 목표값"이 그대로
                    # 남아있다가 호밍이 끝나는 순간 그대로 재생되면 방금 원점으로
                    # 돌아온 로봇이 갑자기 그 낡은 각도로 급이동하는 사고로 이어짐
                    # (실제로 발생했던 "호밍 후 이상한 자세로 갈리는" 증상의 원인).
                    # 그래서 스레드 기동 전에 is_homing부터 걸고 큐를 반드시 비움.
                    self.is_homing = True
                    self.pending_joint_target = None
                    self.pending_gripper_target = None
                    threading.Thread(target=self.run_homing_sequence, daemon=True).start()
            else:
                if not self.is_homing:
                    if self._write(f"{cmd}\r\n".encode(), f"raw({cmd})"):
                        self.get_logger().info(f"Sent Raw G-Code: {cmd}")

    # 수신된 J1~J6 관절 값의 유효성만 검사하고, "가장 최신 목표값"으로 저장만 해둠.
    # 실제 전송은 _try_send_pending()이 이전 명령의 'ok'를 확인한 뒤 담당함.
    def joint_callback(self, msg):
        if self.is_homing:
            return
        if len(msg.data) < 6:
            return

        out_of_range_idx = None
        for i, (lo, hi) in enumerate(self.JOINT_HARD_LIMITS):
            val = msg.data[i]
            if val < lo or val > hi:
                out_of_range_idx = i
                break

        if out_of_range_idx is not None:
            now_warn = time.time()
            if now_warn - self.last_limit_warn_time >= self.limit_warn_interval:
                lo, hi = self.JOINT_HARD_LIMITS[out_of_range_idx]
                self.get_logger().error(
                    f"J{out_of_range_idx+1} 명령값 {msg.data[out_of_range_idx]:.2f}°가 "
                    f"하드웨어 가동범위({lo}~{hi}°)를 벗어나 이번 6축 명령 전체를 무시했습니다. "
                    f"상위 노드(AI/GUI) 쪽 계산값을 확인하세요."
                )
                self.last_limit_warn_time = now_warn
            return

        # 팔을 빠르게 크게 움직이는 동안 도착하는 여러 메시지는 계속 이 변수를
        # 덮어쓰기만 하다가, 시리얼이 비는 순간 _try_send_pending()이 그 시점의
        # "가장 최신" 값 하나만 골라 전송함.
        self.pending_joint_target = list(msg.data[:6])

    # 그리퍼도 이제 즉시 전송하지 않고 최신값만 저장. 팔 이동과 'ok' 응답을
    # 공유하는 같은 시리얼 채널이라, 별도로 즉시 쏘면 서로의 'ok'를 가로채서
    # 완료 판정이 꼬일 수 있기 때문에 같은 대기열로 통합함.
    def gripper_callback(self, msg):
        if self.is_homing:
            return
        self.pending_gripper_target = float(msg.data)

    # 시리얼 상에 명령이 하나도 떠 있지 않을 때(busy==False)만 대기 중인 값을
    # 하나 골라 전송함. 팔 목표값을 그리퍼보다 우선 처리 — 그리퍼는 짧고 자주
    # 안 바뀌는 반면 팔은 연속적인 움직임이 더 중요하기 때문.
    # ══════════════════════════════════════════════════════════════════════
    #  [측정] 'ok' 응답 지연 계측
    #
    #  목적: 'ok'가 "명령을 파싱했다"인지 "동작이 끝났다"인지 판별한다.
    #  이 노드의 busy 가드는 후자를 전제로 만들어졌는데, 전자라면 가드가
    #  사실상 없는 것과 같고 명령이 펌웨어 버퍼에 쌓인다.
    #
    #  CSV 로도 남긴다 — 터미널에서 눈으로 세는 것보다 정확하고, 나중에
    #  다시 볼 수 있다. 측정이 끝나면 ENABLE_OK_LATENCY_LOG 를 False 로.
    # ══════════════════════════════════════════════════════════════════════
    ENABLE_OK_LATENCY_LOG = True
    OK_LATENCY_CSV = '/tmp/ok_latency.csv'
    OK_LATENCY_SUMMARY_SEC = 5.0     # 이 주기로 요약을 한 줄 찍는다

    def _log_ok_latency(self, ms):
        if not self.ENABLE_OK_LATENCY_LOG:
            return
        mv = getattr(self, '_pending_move_deg', 0.0)
        self._ok_lat.append(ms)
        self._ok_mv.append(mv)

        if self._ok_csv is None:
            try:
                self._ok_csv = open(self.OK_LATENCY_CSV, 'w')
                self._ok_csv.write('idx,latency_ms,move_deg\n')
            except Exception as e:
                self.get_logger().warn(f"[ok측정] CSV 열기 실패: {e}")
                self._ok_csv = False
        if self._ok_csv:
            try:
                self._ok_csv.write(f"{len(self._ok_lat)},{ms:.2f},{mv:.3f}\n")
                self._ok_csv.flush()
            except Exception:
                pass

        now = time.time()
        if now - self._ok_last_summary < self.OK_LATENCY_SUMMARY_SEC:
            return
        self._ok_last_summary = now
        if len(self._ok_lat) < 5:
            return

        lat = sorted(self._ok_lat[-200:])
        mv_recent = self._ok_mv[-200:]
        lat_recent = self._ok_lat[-200:]
        n = len(lat)
        med = lat[n // 2]
        p10, p90 = lat[int(n * 0.1)], lat[int(n * 0.9)]

        # 지연과 이동거리의 상관 — 완료 응답이라면 양의 상관이 나와야 한다
        corr = float('nan')
        if n >= 10:
            mx = sum(mv_recent) / n
            my = sum(lat_recent) / n
            num = sum((mv_recent[i] - mx) * (lat_recent[i] - my) for i in range(n))
            dx = sum((v - mx) ** 2 for v in mv_recent)
            dy = sum((v - my) ** 2 for v in lat_recent)
            if dx > 1e-9 and dy > 1e-9:
                corr = num / (dx ** 0.5 * dy ** 0.5)

        # 판정
        if med < 15.0:
            verdict = "★ 파싱 응답 — 명령이 펌웨어 버퍼에 쌓이는 중"
        elif med > 60.0 and (corr != corr or corr > 0.3):
            verdict = "동작 완료 응답 — busy 가드가 설계대로 작동 중"
        else:
            verdict = "판정 애매 — 아래 수치를 그대로 공유할 것"

        rate = len(self._ok_lat) / max(now - self._ok_t0, 1e-6)
        self.get_logger().info(
            f"[ok측정] 지연 중앙 {med:.1f}ms (10~90%: {p10:.1f}~{p90:.1f}) "
            f"이동거리 상관 {corr:+.2f}  전송률 {rate:.1f}회/초  "
            f"표본 {len(self._ok_lat)}  →  {verdict}")

    def _try_send_pending(self):
        if self.is_homing:
            return
        if not self.ser or not self.ser.is_open:
            return

        now = time.time()

        if self.busy:
            if self.ok_seen_count > self._expected_ok_count:
                # ── [측정] 'ok' 응답 지연 ─────────────────────────────────
                # 이 노드는 "WLKATA 펌웨어가 동작을 끝낸 뒤에 'ok'를 준다"고
                # 가정하고 설계됐지만, 그 가정은 검증된 적이 없다.
                # 표준 GRBL은 명령을 '파싱'했을 때 바로 'ok'를 돌려준다.
                #
                #   1~5ms 로 일정      → 파싱 응답. busy 가드가 무력화되고
                #                        명령이 펌웨어 플래너 버퍼에 쌓인다.
                #                        (손을 멈춰도 로봇이 옛 명령을 재생)
                #   80~250ms, 변동     → 동작 완료 응답. 설계대로 동작 중.
                #
                # 이동 거리도 같이 남긴다. 완료 응답이라면 거리가 클수록
                # 지연도 길어야 한다 — 상관이 없으면 파싱 응답이라는 뜻.
                self._log_ok_latency((now - self.busy_since) * 1000.0)
                self.busy = False
            elif now - self.busy_since > self.BUSY_TIMEOUT:
                # 안전장치: 'ok'를 못 받았어도 너무 오래 막혀있지 않도록 강제 해제
                self.get_logger().warn(
                    f"{self.BUSY_TIMEOUT}초 동안 'ok' 응답이 없어 강제로 대기 상태를 해제합니다.")
                self.busy = False
            else:
                return  # 아직 직전 명령 완료 안 됨 → 대기

        if self.needs_homing_after_reconnect:
            # 재연결 직후. 실제 자세를 모르는 상태라 목표값을 계속 버린다.
            self.pending_joint_target = None
            self.pending_gripper_target = None
            return

        if self.pending_joint_target is not None:
            target = self.pending_joint_target
            self.pending_joint_target = None
            # 이번 명령의 이동량(직전 전송값 대비 관절 변화 합). 'ok' 지연이
            # 동작 완료를 뜻한다면 이 값과 지연이 비례해야 한다.
            if self._last_sent_joints is not None:
                self._pending_move_deg = sum(
                    abs(target[i] - self._last_sent_joints[i]) for i in range(6))
            else:
                self._pending_move_deg = 0.0
            self._last_sent_joints = list(target)
            gcode = f"G0 X{target[0]:.2f} Y{target[1]:.2f} Z{target[2]:.2f} A{target[3]:.2f} B{target[4]:.2f} C{target[5]:.2f} F600\r\n"
            self._dispatch(gcode, f"Sent Joint G-Code: {gcode.strip()}")

        elif self.pending_gripper_target is not None:
            val = self.pending_gripper_target
            self.pending_gripper_target = None
            gcode = f"M3 S{val:.2f}\r\n"
            self._dispatch(gcode, f"Sent Gripper G-Code: {gcode.strip()}")

    def _dispatch(self, gcode, log_msg):
        """실제 시리얼 전송 + busy 상태 진입 (공통 로직 묶음)"""
        self._expected_ok_count = self.ok_seen_count
        if not self._write(gcode.encode(), "관절/그리퍼 명령"):
            return          # 링크가 끊겼다. busy 로 만들지 않는다.
        self.get_logger().info(log_msg)
        self.busy = True
        self.busy_since = time.time()

    # 노드 종료 시 로봇 하드웨어의 모터를 안전 대기 모드로 전환하고 세션 폐쇄함
    def destroy_node(self):
        # 종료 경로에서 예외가 나면 정리가 중단된다. USB 가 이미 빠진 상태로
        # 종료되는 경우가 실제로 있으므로 조용히 넘어간다.
        try:
            if self.ser and self.ser.is_open:
                with self.serial_lock:
                    self.ser.write(b"M18\r\n")
                    self.ser.close()
        except Exception as e:
            self.get_logger().warn(f"종료 중 시리얼 정리 실패(무시): {e}")
        super().destroy_node()

# ROS 2 시스템 초기화 및 드라이버 노드 스핀 실행 처리함
def main(args=None):
    rclpy.init(args=args)
    node = MirobotDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()