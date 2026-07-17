import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Float32, Float32MultiArray, Bool, String
import cv2
import mediapipe as mp
import math
import time
import threading
import os
import asyncio
import ssl as ssl_module
import numpy as np
from cv_bridge import CvBridge
from flask import Flask, Response, request
import websockets


# ══════════════════════════════════════════════════════════════════════════
#  One Euro Filter
#
#  기존 고정 alpha EMA는 "천천히 정교하게 움직일 때"와 "빠르게 확 움직일 때"를
#  구분하지 못해서, 떨림 제거와 반응성 중 하나를 항상 희생해야 했음.
#
#  One Euro Filter는 매 순간 추정 속도(dx/dt)를 보고 cutoff를 실시간으로 바꿈:
#    - 느리게 움직일 때(속도 작음) → cutoff가 min_cutoff에 가까워짐 → 강하게 스무딩(떨림 제거)
#    - 빠르게 움직일 때(속도 큼)   → cutoff가 커짐 → 스무딩이 약해짐(지연 최소화, 반응성 확보)
#
#  즉 "가만히 있을 땐 안정적, 빠르게 움직이면 바로 따라오는" 필터.
# ══════════════════════════════════════════════════════════════════════════
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        # min_cutoff: 정지/저속 구간에서의 저역통과 강도. 작을수록 떨림을 더 강하게 억제.
        # beta: 속도가 커질수록 cutoff를 얼마나 더 풀어줄지 결정. 클수록 빠른 동작 지연이 줄어듦.
        # d_cutoff: 속도 자체를 추정할 때 쓰는 저역통과 강도 (속도 추정값의 노이즈 억제용).
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self.t_prev is None:
            # 첫 샘플은 스무딩 없이 그대로 채택 (초기화)
            self.x_prev = x
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-3)  # dt=0 방지 (동일 타임스탬프 중복 호출 대비)

        # 1) 속도 추정 (그 자체도 노이즈가 있으므로 d_cutoff로 한 번 저역통과)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # 2) 추정 속도가 클수록 cutoff를 키워서 스무딩을 약하게 적용
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

    def reset(self):
        # AI ON/OFF 전환, 제어 팔 변경 등 "새로 시작"해야 하는 시점에 호출.
        # 이전 프레임과의 시간차/이전값이 남아있으면 재개 직후 순간적으로
        # 이상한 속도로 추정되어 튀는 현상이 생길 수 있어 완전히 리셋함.
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class MirobotAiNode(Node):

    # ══════════════════════════════════════════════════════
    #  튜닝 상수: 실제 로봇 동작을 보면서 이 값들만 조정하세요
    # ══════════════════════════════════════════════════════

    # ── J1 (베이스 회전): 반대쪽 팔 팔꿈치→손목의 수평 편향 ─────────────────
    # world landmark(3D, 미터 단위) 기준. 카메라와의 거리에 영향받지 않음.
    J1_LATERAL_MIN = -0.65  # 이 비율 이하 → J1_LEFT 로 클램프
    J1_LATERAL_MAX =  0.65  # 이 비율 이상 → J1_RIGHT 로 클램프
    J1_LEFT  = -90.0
    J1_RIGHT =  90.0

    # ── J2 (어깨 관절): 상완 벡터(어깨→팔꿈치)가 수직 기준선과 이루는 각도 ────
    # 0°=팔을 수직으로 늘어뜨림, 180°=팔을 수직으로 들어올림 방향.
    # [기본값] 캘리브레이션을 한 번도 안 했을 때 쓰이는 초기 추정치.
    # 실제로는 GUI의 "캘리브레이션" 버튼으로 본인 팔/카메라 위치 기준 실측값으로 덮어써짐.
    SHOULDER_ANGLE_DOWN = 25.0
    SHOULDER_ANGLE_UP   = 150.0
    # [방향 수정] 실제 하드웨어에서 "팔을 내리면 로봇이 올라가고, 올리면 내려가는"
    # 반전 현상이 확인되어 출력값(J2_DOWN/J2_UP) 두 값의 순서만 맞바꿈.
    # 입력 쪽(SHOULDER_ANGLE_DOWN/UP, 사람 팔의 실측 각도)은 그대로 두고
    # "그 각도를 로봇 관절값 어디로 매핑할지"만 반대로 바꾼 것이므로,
    # 캘리브레이션으로 SHOULDER_ANGLE_DOWN/UP이 갱신되어도 이 반전은 그대로 유지됨.
    J2_DOWN =  70.0
    J2_UP   = -30.0

    # ── J3 (팔꿈치 관절): 팔꿈치를 원점으로 한 (팔꿈치→어깨) vs (팔꿈치→손목) 각도 ─
    # 180°=곧게 폄, 각도가 작을수록 굽힘. J2와 완전히 독립적으로 움직임.
    ELBOW_ANGLE_BENT     = 40.0
    ELBOW_ANGLE_STRAIGHT = 165.0
    J3_BENT     = -60.0
    J3_STRAIGHT =   0.0

    # ── J5 (손목 굴곡) ────────────────────────────────────────────────────
    # [설계 노트] Pose의 world landmark(몸통 기준 좌표계)와 Hands의 world landmark
    # (손 자체 기준 로컬 좌표계)는 서로 다른 좌표계라서 3D로 직접 각도를 비교할 수
    # 없음. 그래서 J5는 지금처럼 같은 2D 이미지 투영 평면 안에서 전완 벡터와 손 벡터를
    # 비교하는 방식을 그대로 유지함 (이 부분은 카메라 거리 왜곡 문제가 J2/J3만큼
    # 크지 않기도 함 — 손목 굴곡은 전완/손이 항상 카메라와 비슷한 거리에 있음).
    # [조정] 60°로는 실측 최대 꺾임각(약 20°대)을 못 채워서 -90까지 안 갔음.
    # 화면의 "Wrist signed: XXdeg" 값을 손목을 최대로 꺾은 상태에서 확인하고,
    # 그 값으로 이 상수를 맞추면 최대로 꺾었을 때 정확히 -90까지 도달함.
    # 지금은 우선 25.0으로 낮춰둠 — 실측값이 이와 다르면 그 숫자로 다시 조정하세요.
    WRIST_BEND_MAX = 25.0
    J5_STRAIGHT    =   0.0
    J5_BENT        = -90.0
    # signed_bend_deg(부호 있는 손목 꺾임각)에 곱해지는 방향 계수.
    # 기본값(1.0)에서 실제로 "뒤로 꺾을 때 -90으로 가야 하는데 반대로 반응"하면
    # 이 값만 -1.0으로 바꾸면 됨. 화면의 "Wrist signed: XXdeg" 텍스트로 확인하면서 조정.
    J5_BACK_BEND_SIGN = 1.0

    # ── 안정성 가드 ──────────────────────────────────────────────────────────
    # world landmark는 카메라 거리 왜곡을 어느 정도 자체 보정하지만, 관절이
    # 완전히 겹쳐 보이거나 추정이 깨지는 극단적인 경우를 걸러내는 최소 안전장치.
    MIN_UPPERARM_LEN_M = 0.05   # 어깨→팔꿈치 3D 벡터 최소 길이 (미터)
    MIN_FOREARM_LEN_J3_M = 0.05  # 팔꿈치→손목 3D 벡터 최소 길이 (미터, J3용)
    MIN_FOREARM_VEC_J5 = 0.06   # J5용 2D 전완 벡터 최소 길이 (정규화 좌표, 기존과 동일)

    # [제거됨] MAX_DELTA_J1~J5 (발행 직전 최대 변화량 제한)
    # → 팔을 크게 움직였을 때 로봇이 계단식으로 나눠서 이동하는 원인이었어서 제거함.
    # 자세한 이유는 발행부(_try_send_pending 근처) 주석 참고.

    # ── One Euro Filter 파라미터 (관절별) ───────────────────────────────────
    # min_cutoff: 작을수록 정지 상태 떨림 억제가 강함
    # beta: 클수록 빠른 동작을 따라갈 때 지연이 줄어듦(대신 노이즈도 더 통과)
    D_CUTOFF = 1.0
    J1_MIN_CUTOFF, J1_BETA = 1.0, 0.3   # 베이스 회전: 조금 더 안정적으로
    J2_MIN_CUTOFF, J2_BETA = 1.0, 0.4
    J3_MIN_CUTOFF, J3_BETA = 1.0, 0.5   # 팔꿈치: 빠른 동작이 많아 반응성 조금 더 확보
    J5_MIN_CUTOFF, J5_BETA = 1.2, 0.6   # 손목: 노이즈가 큰 대신 반응성 우선

    # 데드밴드: 이 각도(°) 미만 변화는 떨림으로 간주하고 무시
    DEADBAND_J1      = 3.0
    DEADBAND_J2      = 2.0
    DEADBAND_J3      = 2.0
    DEADBAND_J5      = 3.0
    DEADBAND_GRIPPER = 1.0

    # 핀치 판정 비율 (엄지끝↔검지끝 / 손목↔중지MCP)
    PINCH_CLOSE_RATIO = 0.25
    PINCH_OPEN_RATIO  = 0.70

    # 랜드마크 visibility 최소 임계값
    VIS_THRESHOLD = 0.5

    # 토픽 발행 주기 (초)
    PUB_INTERVAL = 0.5  # 500ms = 2Hz

    # ── 캘리브레이션 ─────────────────────────────────────────────────────────
    # GUI의 "캘리브레이션 시작" 버튼을 누르면 3단계를 순서대로 거치며
    # SHOULDER_ANGLE_DOWN/UP, ELBOW_ANGLE_BENT/STRAIGHT 를 실측값으로 덮어씀.
    # 시간이 아니라 "유효 샘플 개수"로 단계 완료를 판정 → 팔이 잠깐 안 보여도
    # 안전하게 대기했다가 다시 보이면 이어서 채워짐 (타이밍 꼬임 없음).
    CALIB_REQUIRED_SAMPLES = 40   # 20Hz 처리 기준 약 2초 분량
    CALIB_MIN_RANGE_DEG = 10.0    # 이보다 좁은 범위로 캘리브레이션되면 실패 처리(자세 오류 방지)
    CALIB_STATUS_PUB_INTERVAL = 0.2  # 캘리브레이션 상태 토픽 발행 주기 (너무 자주 안 보내도록)

    CALIB_STAGES = ['shoulder_down', 'shoulder_up', 'elbow_bent']
    CALIB_INSTRUCTIONS = {
        'shoulder_down': "1/3 단계: 제어팔을 몸 옆으로 곧게 펴서 완전히 내린 자세를 유지하세요.",
        'shoulder_up':   "2/3 단계: 제어팔을 곧게 편 채로 최대한 높이 들어올린 자세를 유지하세요.",
        'elbow_bent':    "3/3 단계: 제어팔의 팔꿈치를 최대한 접은 자세를 유지하세요.",
    }

    # ══════════════════════════════════════════════════════

    def __init__(self):
        super().__init__('mirobot_ai_node')
        self.bridge = CvBridge()

        self.image_pub     = self.create_publisher(ROSImage,          '/mirobot/camera_image',       10)
        self.ai_joints_pub = self.create_publisher(Float32MultiArray, '/mirobot/ai_joint_commands',  10)
        self.gripper_pub   = self.create_publisher(Float32,           '/mirobot/ai_gripper_command', 10)

        self.ai_enable_sub = self.create_subscription(
            Bool, '/mirobot/ai_enable', self.ai_enable_callback, 10
        )
        self.ai_enabled = False

        # GUI에서 선택한 제어 팔('left' or 'right'). 기본값: 왼팔.
        self.control_arm = 'left'
        self.create_subscription(
            String, '/mirobot/control_arm', self.control_arm_callback, 10
        )

        # ── 호밍 감지 → 캘리브레이션 자동 초기화 ──────────────────────────────
        # 여러 사람이 번갈아 테스트하는 상황을 대비해, robot_node가 호밍을
        # 시작할 때마다(= 새로운 세션/새로운 사람이 시작한다는 신호로 간주)
        # 이전 사람의 캘리브레이션 값이 남아있지 않도록 기본값으로 되돌림.
        self.create_subscription(
            Bool, '/mirobot/homing_status', self.homing_status_callback, 10
        )

        # ── 캘리브레이션 토픽 ────────────────────────────────────────────────
        self.calibration_status_pub = self.create_publisher(
            String, '/mirobot/calibration_status', 10)
        self.create_subscription(
            String, '/mirobot/calibrate_cmd', self.calibrate_cmd_callback, 10)

        # 캘리브레이션 상태값을 인스턴스 속성으로 복사 (클래스 기본값은 건드리지 않고
        # 이 노드 인스턴스에서만 실측값으로 갱신되도록)
        self.SHOULDER_ANGLE_DOWN   = MirobotAiNode.SHOULDER_ANGLE_DOWN
        self.SHOULDER_ANGLE_UP     = MirobotAiNode.SHOULDER_ANGLE_UP
        self.ELBOW_ANGLE_BENT      = MirobotAiNode.ELBOW_ANGLE_BENT
        self.ELBOW_ANGLE_STRAIGHT  = MirobotAiNode.ELBOW_ANGLE_STRAIGHT

        self.calib_mode = None          # None 또는 CALIB_STAGES 중 하나 (현재 "수집 중"인 단계)
        self.calib_session_active = False  # 캘리브레이션 세션이 진행 중인지(단계 사이 대기 포함)
        self.calib_next_stage_idx = 0    # "시작" 버튼을 다시 누르면 진입할 다음 단계 인덱스
        self.calib_samples_shoulder = []  # shoulder_angle 샘플 버퍼
        self.calib_samples_elbow    = []  # elbow_angle 샘플 버퍼
        self.last_calib_status_pub_time = 0.0

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠에 접근하지 못했습니다.")

        # ── 웹 GUI용 MJPEG 스트리밍 서버 ──────────────────────────────────────
        # /mirobot/camera_image(ROS Image)를 rosbridge로 그대로 브라우저에 보내면
        # 프레임당 수백KB를 JSON+base64로 감싸서 20Hz로 쏘게 되어 네트워크가 못 버팀.
        # 대신 이 노드가 이미 만들고 있는 landmark 그려진 frame을 JPEG로 압축해서
        # 별도 HTTP 스레드로 직접 스트리밍함 (ROS 토픽 발행과는 완전히 독립적).
        self.latest_jpeg = None
        self.jpeg_lock = threading.Lock()

        # ── 원격 카메라(GUI 접속 기기) 프레임 수신용 상태 ─────────────────────
        # 폰/아이패드에서 getUserMedia()로 캡처한 프레임이 /upload_frame으로
        # 올라오면 여기 저장해두고, process_frame()에서 "최근 N초 안에 들어온
        # 원격 프레임이 있으면 그걸 우선 사용, 없으면 로컬 웹캠으로 자동 폴백"함.
        # → GUI를 여는 기기가 바뀌면(노트북→폰→아이패드) 자동으로 그 기기의
        #    카메라가 쓰이게 되고, 별도 선택 버튼이 필요 없음.
        self.remote_frame = None
        self.remote_frame_time = 0.0
        self.remote_frame_lock = threading.Lock()
        self.REMOTE_FRAME_TIMEOUT = 2.0  # 이 시간 안에 새 프레임이 없으면 로컬 웹캠으로 폴백

        self._start_mjpeg_server()
        self._start_upload_ws_server()

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose    = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # MediaPipe Hands: max_num_hands=2 → 어느 쪽 손인지 판별 가능
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            max_num_hands=2
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 상반신 뼈대 그리기용 연결선
        self.ARM_CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16)]

        # One Euro Filter 인스턴스 (관절별 독립)
        self.f_j1 = OneEuroFilter(self.J1_MIN_CUTOFF, self.J1_BETA, self.D_CUTOFF)
        self.f_j2 = OneEuroFilter(self.J2_MIN_CUTOFF, self.J2_BETA, self.D_CUTOFF)
        self.f_j3 = OneEuroFilter(self.J3_MIN_CUTOFF, self.J3_BETA, self.D_CUTOFF)
        self.f_j5 = OneEuroFilter(self.J5_MIN_CUTOFF, self.J5_BETA, self.D_CUTOFF)

        # 필터링된(스무딩된) 현재 관절값. 변수명은 기존 코드와의 일관성을 위해 ema_* 유지.
        self.ema_j1 = 0.0
        self.ema_j2 = 0.0
        self.ema_j3 = 0.0
        self.ema_j5 = 0.0
        self.ema_initialized = False  # AI ON 직후 첫 프레임에서 raw값으로 강제 초기화

        # 마지막으로 실제 발행된 값 (데드밴드 비교 기준)
        self.last_published_joints  = [0.0] * 6
        self.last_published_gripper = 35.0

        # 발행 시각 추적
        self.last_joint_pub_time   = 0.0
        self.last_gripper_pub_time = 0.0

        self.timer = self.create_timer(0.05, self.process_frame)  # 20Hz 처리
        self.get_logger().info(
            "AI 노드 준비 완료 — 3D world landmark 기반 J1/J2/J3 각도 계산 + One Euro Filter 적용")

    # ── 웹 GUI용 MJPEG 서버 ────────────────────────────────────────────────────
    def _start_mjpeg_server(self):
        flask_app = Flask(__name__)

        # ── CORS 허용 ──────────────────────────────────────────────────────────
        # index.html은 8443 포트, 이 서버는 5000 포트라 브라우저 입장에서
        # "다른 출처(origin)"로 취급됨. 특히 fetch()로 image/jpeg Blob을
        # POST하면 CORS "preflight"(OPTIONS) 요청이 먼저 나가는데, 이걸
        # 허용해주지 않으면 실제 POST 자체가 브라우저에서 막혀버림
        # (Safari에서는 이게 "Load failed"로 뜸).
        @flask_app.after_request
        def add_cors_headers(response):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response

        @flask_app.route('/video_feed')
        def video_feed():
            def generate():
                while rclpy.ok():
                    with self.jpeg_lock:
                        jpeg_bytes = self.latest_jpeg
                    if jpeg_bytes is not None:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                    time.sleep(0.05)  # 최대 약 20fps로 제한 (웹 대역폭 보호)
            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        # GUI를 여는 기기(폰/아이패드/노트북)의 카메라 프레임을 받는 엔드포인트.
        # 브라우저의 getUserMedia()가 캡처해서 JPEG bytes로 POST함.
        # 이 프레임이 들어오는 동안은 process_frame()이 로컬 웹캠 대신 이걸 사용함.
        @flask_app.route('/upload_frame', methods=['POST', 'OPTIONS'])
        def upload_frame():
            if request.method == 'OPTIONS':
                # CORS preflight 요청 — 실제 처리 없이 허용 응답만 돌려줌
                return ('', 204)
            try:
                jpeg_bytes = request.get_data()
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    with self.remote_frame_lock:
                        self.remote_frame = decoded
                        self.remote_frame_time = time.time()
                    return ('', 204)
                return ('decode failed', 400)
            except Exception as e:
                return (str(e), 500)

        def run_flask():
            # use_reloader=False 필수: reloader가 프로세스를 통째로 재실행시켜서
            # rclpy 노드가 두 번 초기화되는 사고가 남
            #
            # 폰/아이패드 카메라(getUserMedia)를 쓰려면 index.html이 https로
            # 열려야 하는데, 그 페이지가 이 서버로 fetch/img 요청을 보낼 때도
            # "mixed content" 정책 때문에 이 서버도 https여야 함.
            # generate_cert.sh로 만든 인증서가 있으면 자동으로 https로 뜨고,
            # 없으면 기존처럼 http로 뜸(노트북 localhost 테스트 등에는 http로 충분).
            cert_dir = os.path.expanduser("~/webgui_certs")
            certfile = os.path.join(cert_dir, "cert.pem")
            keyfile = os.path.join(cert_dir, "key.pem")

            if os.path.exists(certfile) and os.path.exists(keyfile):
                self.get_logger().info("인증서 발견 — MJPEG/업로드 서버를 HTTPS로 기동합니다.")
                flask_app.run(host='0.0.0.0', port=5000, threaded=True,
                               use_reloader=False, ssl_context=(certfile, keyfile))
            else:
                self.get_logger().warn(
                    "인증서를 찾지 못해 HTTP로 기동합니다 (폰/아이패드 카메라 업로드는 "
                    "이 상태에서 동작하지 않습니다. generate_cert.sh를 먼저 실행하세요).")
                flask_app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

        threading.Thread(target=run_flask, daemon=True).start()
        self.get_logger().info("MJPEG 스트리밍 서버 시작 (포트 5000, /video_feed)")

    # ── 웹소켓 기반 프레임 업로드 서버 (포트 5001) ────────────────────────────────
    # /upload_frame(HTTP POST)는 프레임마다 새로 HTTPS 연결을 맺어야 해서
    # TLS 핸드셔크 비용이 매번 붙고, 핫스팟처럼 지연이 큰 네트워크에서 특히
    # 끊김이 심해짐. 이 웹소켓 서버는 연결을 한 번만 맺어두고 그 위로 프레임을
    # 계속 binary 메시지로 흘려보내는 방식이라 그 반복 비용이 없음.
    async def _handle_upload_ws(self, websocket):
        self.get_logger().info("프레임 업로드 웹소켓 클라이언트 연결됨")
        try:
            async for message in websocket:
                if not isinstance(message, (bytes, bytearray)):
                    continue  # 텍스트 메시지는 무시(핑/디버그용 등)
                arr = np.frombuffer(message, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    with self.remote_frame_lock:
                        self.remote_frame = decoded
                        self.remote_frame_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"업로드 웹소켓 처리 중 오류: {e}")
        finally:
            self.get_logger().info("프레임 업로드 웹소켓 클라이언트 연결 종료")

    def _start_upload_ws_server(self):
        cert_dir = os.path.expanduser("~/webgui_certs")
        certfile = os.path.join(cert_dir, "cert.pem")
        keyfile = os.path.join(cert_dir, "key.pem")

        def run_ws_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            tls_ctx = None
            if os.path.exists(certfile) and os.path.exists(keyfile):
                tls_ctx = ssl_module.SSLContext(ssl_module.PROTOCOL_TLS_SERVER)
                tls_ctx.load_cert_chain(certfile, keyfile)
                self.get_logger().info("프레임 업로드 웹소켓 서버를 WSS(TLS)로 기동합니다 (포트 5001).")
            else:
                self.get_logger().warn(
                    "인증서를 찾지 못해 업로드 웹소켓을 평문 WS로 기동합니다 "
                    "(폰/아이패드에서는 동작하지 않음. generate_cert.sh 먼저 실행하세요).")

            # websockets 라이브러리 버전에 따라 핸들러가 (websocket) 또는
            # (websocket, path) 두 가지 방식으로 호출될 수 있어 *args로 흡수함.
            async def handler(*args):
                websocket = args[0]
                await self._handle_upload_ws(websocket)

            async def main():
                async with websockets.serve(
                        handler, "0.0.0.0", 5001, ssl=tls_ctx, max_size=5_000_000):
                    await asyncio.Future()  # 계속 대기 (서버 유지)

            loop.run_until_complete(main())

        threading.Thread(target=run_ws_server, daemon=True).start()
        self.get_logger().info("프레임 업로드 웹소켓 서버 시작 (포트 5001)")

    # ── 콜백 ─────────────────────────────────────────────────────────────────────

    def ai_enable_callback(self, msg):
        self.ai_enabled = msg.data
        if self.ai_enabled:
            self.ema_initialized = False
            # 재활성화 시 필터 내부 상태(이전 시각/이전값)를 완전히 리셋해서
            # 꺼져있던 동안의 시간차가 속도 추정에 섞여 들어가는 것을 방지
            self.f_j1.reset()
            self.f_j2.reset()
            self.f_j3.reset()
            self.f_j5.reset()
            self.get_logger().info(f"AI 제어 활성화 — 제어 팔: {self.control_arm}")
        else:
            self.get_logger().info("AI 제어 비활성화")

    def control_arm_callback(self, msg):
        """GUI에서 팔 선택이 바뀔 때 호출. 필터도 리셋해서 전환 직후 급이동 방지."""
        arm = msg.data.strip().lower()
        if arm in ('left', 'right') and arm != self.control_arm:
            self.control_arm = arm
            self.ema_initialized = False
            self.f_j1.reset()
            self.f_j2.reset()
            self.f_j3.reset()
            self.f_j5.reset()
            self.get_logger().info(f"제어 팔 변경: {arm}")

    def homing_status_callback(self, msg):
        """robot_node가 호밍을 시작하면(True) 캘리브레이션을 기본값으로 초기화.
        여러 사람이 번갈아 테스트할 때, 호밍 한 번으로 이전 사람 값이 안 남게 함."""
        if not msg.data:
            return  # 호밍 '완료'(False) 신호는 무시, '시작'(True) 시점에만 초기화

        was_calibrating = (self.calib_mode is not None) or self.calib_session_active

        self.SHOULDER_ANGLE_DOWN  = MirobotAiNode.SHOULDER_ANGLE_DOWN
        self.SHOULDER_ANGLE_UP    = MirobotAiNode.SHOULDER_ANGLE_UP
        self.ELBOW_ANGLE_BENT     = MirobotAiNode.ELBOW_ANGLE_BENT
        self.ELBOW_ANGLE_STRAIGHT = MirobotAiNode.ELBOW_ANGLE_STRAIGHT

        self.calib_mode = None
        self.calib_session_active = False
        self.calib_next_stage_idx = 0
        self.calib_samples_shoulder = []
        self.calib_samples_elbow = []

        self.get_logger().info("호밍 시작 감지 — 캘리브레이션을 기본값으로 초기화했습니다.")
        if was_calibrating:
            self._publish_calib_status(
                force=True,
                override_text="호밍 감지 — 진행 중이던 캘리브레이션이 취소되고 기본값으로 초기화되었습니다."
            )
        else:
            self._publish_calib_status(
                force=True,
                override_text="호밍 감지 — 캘리브레이션이 기본값으로 초기화되었습니다."
            )

    def calibrate_cmd_callback(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == 'start':
            if not self.calib_session_active:
                # 완전히 새 세션 시작
                self.calib_session_active = True
                self.calib_next_stage_idx = 0

            if self.calib_next_stage_idx < len(self.CALIB_STAGES):
                self.calib_mode = self.CALIB_STAGES[self.calib_next_stage_idx]
                self.calib_samples_shoulder = []
                self.calib_samples_elbow = []
                self.get_logger().info(f"캘리브레이션 단계 진행: {self.calib_mode}")
                self._publish_calib_status(force=True)
            else:
                # 이미 전 단계 완료된 상태에서 또 눌렸으면 세션 종료 처리
                self.calib_session_active = False

        elif cmd == 'cancel':
            if self.calib_mode is not None or self.calib_session_active:
                self.get_logger().info("캘리브레이션 취소됨")
            self.calib_mode = None
            self.calib_session_active = False
            self.calib_next_stage_idx = 0
            self.calib_samples_shoulder = []
            self.calib_samples_elbow = []
            self._publish_calib_status(force=True, override_text="캘리브레이션 취소됨")

    # ── 유틸리티 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _dist2d(a, b):
        """두 랜드마크 간 2D 유클리드 거리 (정규화 좌표)"""
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    @staticmethod
    def _lerp_clamp(val, in_min, in_max, out_min, out_max):
        """선형 보간 후 out 범위로 클램프"""
        t = (val - in_min) / (in_max - in_min + 1e-6)
        t = max(0.0, min(1.0, t))
        return out_min + t * (out_max - out_min)

    @staticmethod
    def _angle_between_2d(ax, ay, bx, by):
        """두 2D 벡터 사이의 각도(°) 반환 (0~180°)"""
        dot  = ax * bx + ay * by
        mag  = (math.sqrt(ax**2 + ay**2) + 1e-6) * (math.sqrt(bx**2 + by**2) + 1e-6)
        cos  = max(-1.0, min(1.0, dot / mag))
        return math.acos(cos) * 180.0 / math.pi

    @staticmethod
    def _angle_between_3d(ax, ay, az, bx, by, bz):
        """두 3D 벡터 사이의 각도(°) 반환 (0~180°). world landmark(미터 단위) 전용."""
        dot  = ax * bx + ay * by + az * bz
        mag  = (math.sqrt(ax**2 + ay**2 + az**2) + 1e-6) * (math.sqrt(bx**2 + by**2 + bz**2) + 1e-6)
        cos  = max(-1.0, min(1.0, dot / mag))
        return math.acos(cos) * 180.0 / math.pi

    def _publish_calib_status(self, force=False, override_text=None):
        """캘리브레이션 진행 상태를 GUI에 표시하기 위한 String 토픽 발행 (과도한 스팸 방지)"""
        now = time.time()
        if not force and (now - self.last_calib_status_pub_time < self.CALIB_STATUS_PUB_INTERVAL):
            return
        self.last_calib_status_pub_time = now

        if override_text is not None:
            text = override_text
        elif self.calib_mode is None:
            text = ""
        else:
            instr = self.CALIB_INSTRUCTIONS[self.calib_mode]
            if self.calib_mode in ('shoulder_down', 'shoulder_up'):
                count = len(self.calib_samples_shoulder)
            else:
                count = len(self.calib_samples_elbow)
            text = f"{instr}  ({min(count, self.CALIB_REQUIRED_SAMPLES)}/{self.CALIB_REQUIRED_SAMPLES})"

        msg = String()
        msg.data = text
        self.calibration_status_pub.publish(msg)

    def _finish_calib_stage(self):
        """현재 캘리브레이션 단계의 샘플을 평균 내어 실제 각도 상수에 반영하고, 다음 단계는
        자동으로 넘어가지 않고 사용자가 '캘리브레이션 시작' 버튼을 다시 누를 때까지 대기함
        (자세를 바꿀 시간을 충분히 주기 위함)."""
        stage = self.calib_mode
        current_idx = self.CALIB_STAGES.index(stage)

        if stage == 'shoulder_down':
            avg_shoulder = sum(self.calib_samples_shoulder) / len(self.calib_samples_shoulder)
            avg_elbow    = sum(self.calib_samples_elbow) / len(self.calib_samples_elbow)
            self.SHOULDER_ANGLE_DOWN  = avg_shoulder
            self.ELBOW_ANGLE_STRAIGHT = avg_elbow
            self.get_logger().info(
                f"[캘리브레이션] SHOULDER_ANGLE_DOWN={avg_shoulder:.1f}° "
                f"ELBOW_ANGLE_STRAIGHT={avg_elbow:.1f}°")

        elif stage == 'shoulder_up':
            avg_shoulder = sum(self.calib_samples_shoulder) / len(self.calib_samples_shoulder)
            self.SHOULDER_ANGLE_UP = avg_shoulder
            self.get_logger().info(f"[캘리브레이션] SHOULDER_ANGLE_UP={avg_shoulder:.1f}°")

            # shoulder_down/up 사이 범위가 너무 좁으면 자세를 잘못 취했을 가능성이 높음
            # → 값은 되돌리고, 같은 단계를 재시도할 수 있게 next_stage_idx를 그대로 둠
            if abs(self.SHOULDER_ANGLE_UP - self.SHOULDER_ANGLE_DOWN) < self.CALIB_MIN_RANGE_DEG:
                self.get_logger().warn("캘리브레이션 실패: 어깨 각도 범위가 너무 좁습니다. 이전 값을 유지합니다.")
                self.SHOULDER_ANGLE_DOWN = MirobotAiNode.SHOULDER_ANGLE_DOWN
                self.SHOULDER_ANGLE_UP   = MirobotAiNode.SHOULDER_ANGLE_UP
                self.calib_mode = None
                self.calib_next_stage_idx = current_idx  # 같은 단계 재시도
                self._publish_calib_status(
                    force=True,
                    override_text="캘리브레이션 실패: 어깨 각도 범위가 너무 좁습니다. "
                                  "준비되면 '캘리브레이션 시작' 버튼을 눌러 이 단계를 다시 시도하세요.")
                self.calib_samples_shoulder = []
                self.calib_samples_elbow = []
                return

        elif stage == 'elbow_bent':
            avg_elbow = sum(self.calib_samples_elbow) / len(self.calib_samples_elbow)
            self.ELBOW_ANGLE_BENT = avg_elbow
            self.get_logger().info(f"[캘리브레이션] ELBOW_ANGLE_BENT={avg_elbow:.1f}°")

            if abs(self.ELBOW_ANGLE_STRAIGHT - self.ELBOW_ANGLE_BENT) < self.CALIB_MIN_RANGE_DEG:
                self.get_logger().warn("캘리브레이션 실패: 팔꿈치 각도 범위가 너무 좁습니다. 이전 값을 유지합니다.")
                self.ELBOW_ANGLE_BENT     = MirobotAiNode.ELBOW_ANGLE_BENT
                self.ELBOW_ANGLE_STRAIGHT = MirobotAiNode.ELBOW_ANGLE_STRAIGHT
                self.calib_mode = None
                self.calib_next_stage_idx = current_idx  # 같은 단계 재시도
                self._publish_calib_status(
                    force=True,
                    override_text="캘리브레이션 실패: 팔꿈치 각도 범위가 너무 좁습니다. "
                                  "준비되면 '캘리브레이션 시작' 버튼을 눌러 이 단계를 다시 시도하세요.")
                self.calib_samples_shoulder = []
                self.calib_samples_elbow = []
                return

        # 이번 단계 수집 종료(자동으로 다음 단계 시작하지 않고 대기)
        self.calib_mode = None
        self.calib_samples_shoulder = []
        self.calib_samples_elbow = []

        if current_idx + 1 < len(self.CALIB_STAGES):
            self.calib_next_stage_idx = current_idx + 1
            next_instr = self.CALIB_INSTRUCTIONS[self.CALIB_STAGES[self.calib_next_stage_idx]]
            self._publish_calib_status(
                force=True,
                override_text=(
                    f"{current_idx + 1}단계 완료! 자세를 바꿀 시간을 가지신 뒤, "
                    f"준비되면 '캘리브레이션 시작' 버튼을 눌러 다음 단계로 진행하세요. "
                    f"(다음: {next_instr})"
                )
            )
        else:
            self.calib_session_active = False
            self.calib_next_stage_idx = 0
            self._publish_calib_status(
                force=True,
                override_text=(
                    f"캘리브레이션 완료! "
                    f"어깨 {self.SHOULDER_ANGLE_DOWN:.0f}°~{self.SHOULDER_ANGLE_UP:.0f}°, "
                    f"팔꿈치 {self.ELBOW_ANGLE_BENT:.0f}°~{self.ELBOW_ANGLE_STRAIGHT:.0f}° 로 반영되었습니다."
                )
            )

    # ── 메인 처리 루프 ────────────────────────────────────────────────────────────

    def process_frame(self):
        # ── 프레임 소스 결정: 원격(GUI 접속 기기 카메라) 우선, 없으면 로컬 웹캠 ──────
        # 최근 REMOTE_FRAME_TIMEOUT초 안에 /upload_frame으로 프레임이 들어왔으면
        # 그 기기(폰/아이패드/노트북 등 GUI를 열고 있는 기기)의 카메라를 그대로 씀.
        # 끊기거나 아무도 업로드를 안 하면 로컬 웹캠으로 자동 전환됨.
        frame = None
        with self.remote_frame_lock:
            if (self.remote_frame is not None and
                    time.time() - self.remote_frame_time < self.REMOTE_FRAME_TIMEOUT):
                frame = self.remote_frame.copy()

        if frame is None:
            if not self.cap.isOpened():
                return
            ret, frame = self.cap.read()
            if not ret:
                return

        # 원격 프레임은 기기마다 해상도가 제각각이라, 이후 계산에 쓰이는 픽셀 기준
        # 좌표(cv2.putText 위치 등)가 안정적으로 나오도록 표준 해상도로 맞춤.
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480))

        # 좌우 반전(거울 모드): 이 때문에 MediaPipe 핸드니스가 실제와 반대
        # → 실제 오른손 = MediaPipe "Left", 실제 왼손 = MediaPipe "Right"
        frame     = cv2.flip(frame, 1)
        h, w, _   = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)

        # [1번 개선] 3D world landmark: 엉덩이 중심 기준 실측(미터 단위) 3D 좌표.
        # 일반 pose_landmarks(2D 정규화 이미지 좌표)와 달리 카메라와의 거리/각도에
        # 따른 투영 왜곡이 없어서, "팔을 내림"과 "팔을 뻗음"처럼 2D에서는 헷갈리던
        # 동작이 3D 각도에서는 명확히 구분됨.
        world_lm = None
        if pose_results.pose_world_landmarks:
            world_lm = pose_results.pose_world_landmarks.landmark

        current_time = time.time()
        active_arm   = None

        # ── STEP 1: 사용할 팔 결정 (GUI 선택값 고정 사용, 자동 감지 없음) ──────────
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            sh_idx = 11 if self.control_arm == 'right' else 12
            wr_idx = 15 if self.control_arm == 'right' else 16

            if (lm[sh_idx].visibility > self.VIS_THRESHOLD and
                    lm[wr_idx].visibility > self.VIS_THRESHOLD):
                active_arm = self.control_arm

        # ── STEP 2: J1/J2/J3 계산 (3D world landmark 기반) ──────────────────────
        raw_j1 = None
        raw_j2 = None
        raw_j3 = None
        shoulder_angle = None
        elbow_angle = None
        geom_valid = False

        if active_arm is not None and world_lm is not None:
            lm = pose_results.pose_landmarks.landmark  # visibility 체크는 계속 2D 쪽 값 사용

            sh_idx = 11 if active_arm == 'right' else 12
            el_idx = 13 if active_arm == 'right' else 14
            wr_idx = 15 if active_arm == 'right' else 16

            # ── J1: 반대쪽 팔 팔꿈치→손목 수평 편향 (3D world 기준) ────────────
            # 실제 미터 단위 좌표라서 전완 길이로 정규화하면 사람 체형과 무관하게
            # 항상 같은 각도값이 나옴 (기존 2D 방식은 카메라 거리에 따라 흔들렸음).
            j1_el_idx = 13 if active_arm == 'left' else 14
            j1_wr_idx = 15 if active_arm == 'left' else 16
            j1_el_vis = lm[j1_el_idx]
            j1_wr_vis = lm[j1_wr_idx]

            if (j1_el_vis.visibility > self.VIS_THRESHOLD and
                    j1_wr_vis.visibility > self.VIS_THRESHOLD):
                j1_el = world_lm[j1_el_idx]
                j1_wr = world_lm[j1_wr_idx]
                forearm_len_3d = math.sqrt(
                    (j1_wr.x - j1_el.x) ** 2 + (j1_wr.y - j1_el.y) ** 2 + (j1_wr.z - j1_el.z) ** 2
                ) + 1e-6
                # 좌우 편향은 x축(수평) 성분만 사용, 3D 전완 길이로 정규화
                j1_lateral = (j1_wr.x - j1_el.x) / forearm_len_3d
                raw_j1 = self._lerp_clamp(j1_lateral,
                                           self.J1_LATERAL_MIN, self.J1_LATERAL_MAX,
                                           self.J1_LEFT,         self.J1_RIGHT)

                cx_j1 = int(lm[j1_wr_idx].x * w)
                cy_j1 = int(lm[j1_wr_idx].y * h)
                cv2.circle(frame, (cx_j1, cy_j1), 10, (255, 0, 255), -1)
                cv2.putText(frame, f"J1ctrl: {raw_j1:.1f}",
                            (cx_j1 + 12, cy_j1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # ── J2/J3: 3D 어깨 각도 / 팔꿈치 굽힘각 (완전 독립 계산) ────────────
            sh_w = world_lm[sh_idx]
            el_w = world_lm[el_idx]
            wr_w = world_lm[wr_idx]

            upperarm_dx = el_w.x - sh_w.x
            upperarm_dy = el_w.y - sh_w.y
            upperarm_dz = el_w.z - sh_w.z
            upperarm_len = math.sqrt(upperarm_dx**2 + upperarm_dy**2 + upperarm_dz**2)

            forearm_dx = wr_w.x - el_w.x
            forearm_dy = wr_w.y - el_w.y
            forearm_dz = wr_w.z - el_w.z
            forearm_len = math.sqrt(forearm_dx**2 + forearm_dy**2 + forearm_dz**2)

            geom_valid = (
                lm[el_idx].visibility > self.VIS_THRESHOLD and
                upperarm_len >= self.MIN_UPPERARM_LEN_M and
                forearm_len  >= self.MIN_FOREARM_LEN_J3_M
            )

            cv2.putText(frame,
                        f"geom: {'OK' if geom_valid else 'SKIP'}",
                        (w - 140, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if geom_valid else (0, 0, 255), 2)

            if geom_valid:
                # world 좌표계에서 y는 아래로 증가(MediaPipe 관례) → (0,1,0)이 "수직 아래" 기준벡터
                shoulder_angle = self._angle_between_3d(
                    0.0, 1.0, 0.0, upperarm_dx, upperarm_dy, upperarm_dz)
                raw_j2 = self._lerp_clamp(shoulder_angle,
                                           self.SHOULDER_ANGLE_DOWN, self.SHOULDER_ANGLE_UP,
                                           self.J2_DOWN,             self.J2_UP)

                # 팔꿈치를 원점으로 한 (팔꿈치→어깨) vs (팔꿈치→손목) 각도
                elbow_angle = self._angle_between_3d(
                    -upperarm_dx, -upperarm_dy, -upperarm_dz,
                    forearm_dx,   forearm_dy,   forearm_dz)
                raw_j3 = self._lerp_clamp(elbow_angle,
                                           self.ELBOW_ANGLE_BENT, self.ELBOW_ANGLE_STRAIGHT,
                                           self.J3_BENT,          self.J3_STRAIGHT)

                cv2.putText(frame,
                            f"shoulder:{shoulder_angle:.0f}deg elbow:{elbow_angle:.0f}deg",
                            (w - 280, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            # ── 캘리브레이션 샘플 수집 ──────────────────────────────────────
            # AI ON/OFF 여부와 무관하게, 캘리브레이션 모드 중이면 유효 각도를 계속 모음.
            if self.calib_mode is not None and geom_valid:
                if self.calib_mode in ('shoulder_down', 'shoulder_up'):
                    self.calib_samples_shoulder.append(shoulder_angle)
                    # shoulder_down 단계는 "곧게 펴고 내린" 자세이므로 elbow_straight도 같이 수집
                    if self.calib_mode == 'shoulder_down':
                        self.calib_samples_elbow.append(elbow_angle)
                    required_len = len(self.calib_samples_shoulder)
                elif self.calib_mode == 'elbow_bent':
                    self.calib_samples_elbow.append(elbow_angle)
                    required_len = len(self.calib_samples_elbow)
                else:
                    required_len = 0

                if required_len >= self.CALIB_REQUIRED_SAMPLES:
                    self._finish_calib_stage()

            if self.calib_mode is not None:
                self._publish_calib_status()
                cv2.putText(frame, "CALIBRATING", (10, h - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # 뼈대 시각화 (AI ON/OFF 무관하게 항상 표시)
            for idx in [11, 12, 13, 14, 15, 16]:
                pt = lm[idx]
                if pt.visibility > self.VIS_THRESHOLD:
                    cx, cy = int(pt.x * w), int(pt.y * h)
                    is_active = (active_arm == 'right' and idx in [11, 13, 15]) or \
                                (active_arm == 'left'  and idx in [12, 14, 16])
                    color = (0, 255, 0) if is_active else (120, 120, 120)
                    cv2.circle(frame, (cx, cy), 8, color, -1)
                    cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
            for s_idx, e_idx in self.ARM_CONNECTIONS:
                ps, pe = lm[s_idx], lm[e_idx]
                if ps.visibility > self.VIS_THRESHOLD and pe.visibility > self.VIS_THRESHOLD:
                    x1, y1 = int(ps.x * w), int(ps.y * h)
                    x2, y2 = int(pe.x * w), int(pe.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 150, 0), 4)

        # ── STEP 3: 활성 팔 쪽 손 매칭 → J5(손목 굴곡) + 그리퍼 계산 ───────────
        #
        # [핵심 설계] 핸드니스 라벨("Left"/"Right")을 사용하지 않음.
        # flip된 이미지에서 MediaPipe 핸드니스 분류는 불안정해서 틀리는 경우가 있고,
        # 이 경우 반대쪽 손이 제어권을 가로채는 문제가 생김.
        #
        # 대신 Pose의 손목 랜드마크 위치와 Hand의 손목(landmark[0]) 위치를 직접 비교해서
        # 선택한 팔의 Pose 손목과 가장 가까운 Hand만 사용함.
        # 반대쪽 손은 구조적으로 훨씬 멀리 있으므로 라벨 오류가 있어도 절대 끼어들지 못함.
        # MAX_HAND_MATCH_DIST 이상이면 "손이 없는 것"으로 처리해서 오감지도 차단.
        #
        # [2번 노트] J5는 Pose(몸통 world 기준)와 Hands(손 자체 로컬 world 기준)의
        # 좌표계가 서로 다르기 때문에 3D로 직접 비교할 수 없어, 여기서는 기존처럼
        # 같은 2D 이미지 투영 평면 위에서 계산함 (설계 노트는 클래스 상단 참고).
        MAX_HAND_MATCH_DIST = 0.15

        raw_j5           = None
        proposed_gripper = None
        matched_hand     = None

        if hand_results.multi_hand_landmarks and active_arm is not None:
            lm     = pose_results.pose_landmarks.landmark
            wr_idx = 15 if active_arm == 'right' else 16
            wr_pose = lm[wr_idx]

            best_dist = float('inf')
            for hlm_item in hand_results.multi_hand_landmarks:
                hw = hlm_item.landmark[0]
                dist = math.sqrt((hw.x - wr_pose.x) ** 2 + (hw.y - wr_pose.y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    matched_hand = hlm_item

            if best_dist > MAX_HAND_MATCH_DIST:
                matched_hand = None

            if matched_hand is not None:
                hlm = matched_hand.landmark
                lm  = pose_results.pose_landmarks.landmark

                # ── J5: 손목 굴곡 계산 (2D, 설계 이유는 상단 주석 참고) ──────────
                el_idx = 13 if active_arm == 'right' else 14
                el_lm  = lm[el_idx]
                wr_lm  = lm[15 if active_arm == 'right' else 16]

                forearm_dx = wr_lm.x - el_lm.x
                forearm_dy = wr_lm.y - el_lm.y
                forearm_mag = math.sqrt(forearm_dx ** 2 + forearm_dy ** 2)

                if forearm_mag >= self.MIN_FOREARM_VEC_J5:
                    # [원래 방식 복원] 손목(0)→중지 MCP(9) 대신 손목(0)→중지 손끝(12)을 사용.
                    # MCP는 손바닥 안쪽이라 손목을 굽혀도 벡터가 별로 안 바뀌어 둔감했음.
                    # 손끝(12)은 손목에서 가장 먼 점이라 손목을 꺾고 펴는 움직임이
                    # 벡터 각도 변화로 훨씬 크고 직관적으로 반영됨.
                    h_wrist    = hlm[0]
                    h_mid_tip  = hlm[12]
                    hand_dx = h_mid_tip.x - h_wrist.x
                    hand_dy = h_mid_tip.y - h_wrist.y

                    # [방향 구분] _angle_between_2d는 방향 상관없이 항상 0~180°
                    # 양수만 반환하기 때문에, 앞으로 꺾든 뒤로 꺾든 똑같이 "많이
                    # 꺾인 것"으로 처리되어 둘 다 -90 쪽으로 움직이는 문제가 있었음.
                    # 대신 cross-product 부호로 회전 방향을 구분하는 signed 각도를 써서
                    # 뒤로 꺾을 때만 -90 방향으로 반응하고, 앞으로 꺾을 땐 0에 고정시킴.
                    cross = forearm_dx * hand_dy - forearm_dy * hand_dx
                    dot   = forearm_dx * hand_dx + forearm_dy * hand_dy
                    signed_bend_deg = math.degrees(math.atan2(cross, dot))  # -180~180, 0=일직선

                    # 실측해보고 반대로 반응하면 이 값만 -1.0으로 뒤집으면 됨
                    effective_bend = signed_bend_deg * self.J5_BACK_BEND_SIGN

                    if effective_bend > 0.0:
                        # 뒤로 꺾은 방향 → 0°(폄) ~ -90°(최대 꺾음)로 매핑
                        raw_j5 = self._lerp_clamp(effective_bend,
                                                   0.0, self.WRIST_BEND_MAX,
                                                   self.J5_STRAIGHT, self.J5_BENT)
                    else:
                        # 앞으로 꺾은 방향 → 반응 없이 0°(폄) 고정
                        raw_j5 = self.J5_STRAIGHT

                    cv2.putText(frame,
                                f"Wrist signed: {signed_bend_deg:.1f}deg  J5: {self.ema_j5:.1f}deg",
                                (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 120, 255), 2)

                # ── 그리퍼: 엄지끝↔검지끝 핀치 비율 ─────────────────────────
                h_wrist   = hlm[0]
                h_mid_mcp = hlm[9]
                thumb_tip = hlm[4]
                index_tip = hlm[8]
                hand_size   = self._dist2d(h_wrist, h_mid_mcp) + 1e-6
                pinch_ratio = self._dist2d(thumb_tip, index_tip) / hand_size

                proposed_gripper = self._lerp_clamp(
                    pinch_ratio,
                    self.PINCH_CLOSE_RATIO, self.PINCH_OPEN_RATIO,
                    58.0, 35.0
                )

                self.mp_drawing.draw_landmarks(
                    frame, matched_hand, self.mp_hands.HAND_CONNECTIONS
                )
                cv2.putText(frame,
                            f"Pinch: {pinch_ratio:.2f}  Gripper: S{self.last_published_gripper:.1f}",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ── STEP 4: One Euro Filter 적용 후 발행 ─────────────────────────────
        if raw_j2 is not None:
            # AI ON 직후 첫 유효 프레임: raw값으로 강제 초기화 (켤 때 급이동 방지)
            if not self.ema_initialized:
                self.ema_j1 = raw_j1 if raw_j1 is not None else 0.0
                self.ema_j2 = raw_j2
                self.ema_j3 = raw_j3
                self.ema_j5 = raw_j5 if raw_j5 is not None else 0.0
                self.f_j1.reset()
                self.f_j2.reset()
                self.f_j3.reset()
                self.f_j5.reset()
                self.ema_initialized = True

            # J1: 반대쪽 팔이 안 보이면 마지막 필터값 유지
            if raw_j1 is not None:
                self.ema_j1 = self.f_j1.filter(raw_j1, current_time)
            self.ema_j2 = self.f_j2.filter(raw_j2, current_time)
            self.ema_j3 = self.f_j3.filter(raw_j3, current_time)
            if raw_j5 is not None:
                self.ema_j5 = self.f_j5.filter(raw_j5, current_time)
            # 손/반대팔이 안 잡히면 J1/J5는 마지막 값 유지 (급격한 복귀 방지)

            cv2.putText(frame,
                        f"Arm: {active_arm}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(frame,
                        f"J1: {self.ema_j1:.1f}  J2: {self.ema_j2:.1f}  J3: {self.ema_j3:.1f}  J5: {self.ema_j5:.1f}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)
            ai_state = "AI ON" if self.ai_enabled else "AI OFF"
            cv2.putText(frame, ai_state, (w - 120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if self.ai_enabled else (0, 0, 200), 2)

            # AI ON + 발행 주기 충족 시 토픽 발행 (데드밴드 필터). 캘리브레이션 중에는 발행 안 함.
            if (self.ai_enabled and self.calib_mode is None and
                    (current_time - self.last_joint_pub_time >= self.PUB_INTERVAL)):
                j1_changed = abs(self.ema_j1 - self.last_published_joints[0]) >= self.DEADBAND_J1
                j2_changed = abs(self.ema_j2 - self.last_published_joints[1]) >= self.DEADBAND_J2
                j3_changed = abs(self.ema_j3 - self.last_published_joints[2]) >= self.DEADBAND_J3
                j5_changed = abs(self.ema_j5 - self.last_published_joints[4]) >= self.DEADBAND_J5

                if j1_changed or j2_changed or j3_changed or j5_changed:
                    # [제거] 기존엔 여기서 _rate_limit()으로 "주기당 최대 변화량"을
                    # 강제로 깎아서 발행했는데, 이게 팔을 크게 움직였을 때도
                    # 로봇에는 조금씩 나눠서 목표값이 전달되게 만드는 원인이었음
                    # (One Euro Filter가 이미 노이즈를 적응적으로 걸러주고,
                    #  robot_node의 JOINT_HARD_LIMITS가 최종 안전망 역할을 하므로
                    #  이 추가 클램프는 불필요하게 움직임을 계단식으로 쪼개기만 했음).
                    # 이제 필터링된 값을 그대로 발행해서, 팔이 크게 움직이면
                    # 로봇도 한 번에 그 목표까지 이동하도록 함.
                    self.last_published_joints[0] = self.ema_j1
                    self.last_published_joints[1] = self.ema_j2
                    self.last_published_joints[2] = self.ema_j3
                    self.last_published_joints[4] = self.ema_j5

                    joint_msg      = Float32MultiArray()
                    joint_msg.data = [float(x) for x in self.last_published_joints]
                    self.ai_joints_pub.publish(joint_msg)
                    self.last_joint_pub_time = current_time

            # 그리퍼 발행 (캘리브레이션 중에는 발행 안 함)
            if (proposed_gripper is not None and self.ai_enabled and self.calib_mode is None and
                    current_time - self.last_gripper_pub_time >= self.PUB_INTERVAL):
                if abs(proposed_gripper - self.last_published_gripper) >= self.DEADBAND_GRIPPER:
                    self.last_published_gripper = proposed_gripper
                    grip_msg      = Float32()
                    grip_msg.data = float(proposed_gripper)
                    self.gripper_pub.publish(grip_msg)
                    self.last_gripper_pub_time = current_time

        # ── STEP 5: 카메라 이미지 ROS 토픽 발행 (PyQt GUI용, 기존 그대로 유지) ──────
        try:
            ros_img = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_pub.publish(ros_img)
        except Exception:
            pass

        # ── STEP 6: 웹 GUI용 JPEG 인코딩 (MJPEG 스트림 소스) ──────────────────────
        try:
            ok_enc, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok_enc:
                with self.jpeg_lock:
                    self.latest_jpeg = jpeg_buf.tobytes()
        except Exception:
            pass

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MirobotAiNode()
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