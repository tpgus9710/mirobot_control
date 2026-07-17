import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Float32MultiArray, Bool
import serial
import time
import threading

# 미로봇 하드웨어 시리얼 인터페이스 및 ROS 2 통신 처리 드라이버 클래스
class MirobotDriverNode(Node):

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
        
        # 실시간 홈 복귀 가동 현황 공유용 퍼블리셔 추가
        self.homing_status_pub = self.create_publisher(Bool, '/mirobot/homing_status', 10)
        
        # 시리얼 통신 초기화 및 미로봇 연결 시도
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self.get_logger().info(f"Connected to Mirobot on {port}")
            time.sleep(2)  # 아두이노/컨트롤러 리셋 대기 시간
            self.init_robot()
        except Exception as e:
            self.get_logger().error(f"Failed to connect to serial port: {e}")
            self.ser = None
            self.is_homing = False

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

    # 로봇 부팅 시 비동기 스레드로 홈 복귀 시퀀스 가동 (실행 병목 제거)
    def init_robot(self):
        if self.ser and self.ser.is_open:
            threading.Thread(target=self.run_homing_sequence, daemon=True).start()

    # 안전 동시성 락을 보장하고 홈 복귀 완수를 모니터링하는 코어 가동 루프 수행
    def run_homing_sequence(self):
        if not self.ser or not self.ser.is_open:
            return
            
        self.get_logger().info("하드웨어 호밍(원점 정렬)을 시작합니다. 완료될 때까지 다른 명령은 무시됩니다...")
        self.is_homing = True
        
        # 홈 복귀가 활성화되었음을 토픽으로 선언
        status_msg = Bool()
        status_msg.data = True
        self.homing_status_pub.publish(status_msg)
        
        with self.serial_lock:
            # WLKATA 표준 홈 복귀 명령 전송
            self.ser.write(b"$H\r\n")
            
            # 시리얼 반환값 모니터링 루프: 'ok'가 들어올 때까지 대기 (최대 35초 타임아웃)
            start_time = time.time()
            while time.time() - start_time < 35.0:
                if self.ser.in_waiting > 0:
                    try:
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            self.get_logger().info(f"[Mirobot Homing] {line}")
                            # 호밍 완료 시그널 수신 확인
                            if "ok" in line.lower():
                                break
                    except Exception:
                        pass
                time.sleep(0.05)
            
            # 절대 좌표 모드 G-Code(O105) 송출 및 대기
            self.ser.write(b"O105\r\n")
            time.sleep(0.5)
            
        self.is_homing = False
        self.get_logger().info("Mirobot 호밍 및 가동 초기화가 완전히 완료되었습니다!")
        
        # 홈 복귀 완료 상태 전송
        status_msg.data = False
        self.homing_status_pub.publish(status_msg)

    # 실시간 동적 수동 G-Code 및 외부 $H 원격 입력에 대한 G-Code 시리얼 쓰기 핸들러 기능 수행
    def raw_callback(self, msg):
        if self.ser and self.ser.is_open:
            cmd = msg.data.strip()
            if cmd == "$H":
                if not self.is_homing:
                    threading.Thread(target=self.run_homing_sequence, daemon=True).start()
            else:
                if not self.is_homing:
                    with self.serial_lock:
                        self.ser.write(f"{cmd}\r\n".encode())
                    self.get_logger().info(f"Sent Raw G-Code: {cmd}")

    # 수신된 J1~J6 관절 값을 받아 WLKATA Platinum G-Code 18번 가이드 표준형으로 가공 송신함
    def joint_callback(self, msg):
        if self.is_homing:
            # 호밍 수행 중에는 들어오는 모션 명령 유실 및 꼬임 차단
            return
            
        if len(msg.data) < 6:
            return
            
        if self.ser and self.ser.is_open:
            gcode = f"G0 X{msg.data[0]:.2f} Y{msg.data[1]:.2f} Z{msg.data[2]:.2f} A{msg.data[3]:.2f} B{msg.data[4]:.2f} C{msg.data[5]:.2f} F2000\r\n"
            with self.serial_lock:
                self.ser.write(gcode.encode())
            self.get_logger().info(f"Sent Joint G-Code: {gcode.strip()}")

    # 그리퍼 제어 수치(S35~S58)를 왈카타 M3 아날로그 서보 G-Code 규격으로 가공 송신함
    def gripper_callback(self, msg):
        if self.is_homing:
            # 호밍 중 그리퍼 신호 난입 차단
            return
            
        if self.ser and self.ser.is_open:
            gcode = f"M3 S{msg.data:.2f}\r\n"
            with self.serial_lock:
                self.ser.write(gcode.encode())
            self.get_logger().info(f"Sent Gripper G-Code: {gcode.strip()}")

    # 노드 종료 시 로봇 하드웨어의 모터를 안전 대기 모드로 전환하고 세션 폐쇄함
    def destroy_node(self):
        # 종료 시 하드웨어 안전 대기 상태 전환
        if self.ser and self.ser.is_open:
            with self.serial_lock:
                self.ser.write(b"M18\r\n")
                self.ser.close()
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