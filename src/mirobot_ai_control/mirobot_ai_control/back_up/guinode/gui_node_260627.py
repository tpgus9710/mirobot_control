import sys
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, String, Bool
from sensor_msgs.msg import Image as ROSImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
from cv_bridge import CvBridge

# ROS 2 토픽 스레드와 PyQt GUI 스레드 간 비동기 신호 중계를 위한 시그널 객체
class RosBridgeSignals(QObject):
    image_received = pyqtSignal(object)
    ai_joints_received = pyqtSignal(object)
    ai_gripper_received = pyqtSignal(float)
    homing_status_received = pyqtSignal(bool) # 드라이버 노드의 실시간 호밍 감지를 위한 PyQt 시그널 추가

# ROS 2 서브스크립션 데이터 수신 및 PyQt 시그널 브릿지 송출 노드 클래스
class MirobotGuiNode(Node):
    # GUI 노드 통신 구독자, 퍼블리셔 및 신호 전송 기구 정의함
    def __init__(self, signals):
        super().__init__('mirobot_gui_node')
        self.signals = signals
        self.bridge = CvBridge()
        
        self.joint_pub = self.create_publisher(Float32MultiArray, '/mirobot/joint_commands', 10)
        self.gripper_pub = self.create_publisher(Float32, '/mirobot/gripper', 10)
        self.raw_pub = self.create_publisher(String, '/mirobot/raw_commands', 10)
        self.ai_enable_pub = self.create_publisher(Bool, '/mirobot/ai_enable', 10) # AI 토글 전송 상태 퍼블리셔 추가
        
        self.image_sub = self.create_subscription(
            ROSImage, 
            '/mirobot/camera_image', 
            self.image_callback, 
            10
        )
        self.ai_joints_sub = self.create_subscription(
            Float32MultiArray,
            '/mirobot/ai_joint_commands',
            self.ai_joints_callback,
            10
        )
        self.ai_gripper_sub = self.create_subscription(
            Float32, 
            '/mirobot/ai_gripper_command', 
            self.ai_gripper_callback, 
            10
        )
        self.homing_status_sub = self.create_subscription(
            Bool,
            '/mirobot/homing_status',
            self.homing_status_callback,
            10
        ) # 드라이버 호밍 상태 트래커 추가

    # 수신된 OpenCV 비디오 프레임을 PyQt 이미지 처리 시그널로 중계함
    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.signals.image_received.emit(cv_img)
        except Exception:
            pass

    # 수신된 AI 예측 6축 관절각 데이터를 PyQt 슬라이더 매핑 시그널로 송출함
    def ai_joints_callback(self, msg):
        self.signals.ai_joints_received.emit(msg.data)

    # 수신된 AI 예측 손목-핑거 사잇각 캘리브레이션 듀티 데이터를 PyQt 슬라이더 매핑 시그널로 송출함
    def ai_gripper_callback(self, msg):
        self.signals.ai_gripper_received.emit(msg.data)

    # 드라이버로부터 홈 복귀 진행 상태 피드백을 수신하여 GUI 잠금 연동 시그널로 송출함
    def homing_status_callback(self, msg):
        self.signals.homing_status_received.emit(msg.data)


# Mirobot 제어 UI 렌더링 및 모션 오프셋 연동 기능 담당 PyQt5 윈도우 클래스
class MirobotGuiWindow(QMainWindow):
    # PyQt 창 상태 변수 초기화 및 비동기 ROS 시그널 핸들러 연결 수행함
    def __init__(self, node, signals):
        super().__init__()
        self.node = node
        self.signals = signals
        self.ai_mode_active = False
        self.is_homing = False # 수동 각도 스팸 전송 억제를 위한 호밍 감지 락 플래그 추가
        
        self.first_ai_frame = False
        self.ai_init = None
        self.robot_init = None
        
        self.signals.image_received.connect(self.update_image_display)
        self.signals.ai_joints_received.connect(self.update_ai_joints_display)
        self.signals.ai_gripper_received.connect(self.update_ai_gripper_display)
        self.signals.homing_status_received.connect(self.update_homing_status) # 시그널 동적 연결
        
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.publish_joint_commands)
        self.timer.start(250) # 각도 전송 빈도를 요구사항에 맞춰 250ms(4Hz) 완급조정형으로 완화 변경

    # 넓어진 1180px 창 면적 기반으로 카메라 뷰파인더, 원클릭 홈 단축키 및 슬라이더 UI 콤포넌트 렌더링 수행함
    def init_ui(self):
        self.setWindowTitle("Mirobot Advanced AI Control Center")
        self.setMinimumSize(1180, 600)
        self.resize(1180, 600)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QLabel { color: #e0e0e0; font-weight: bold; font-size: 12px; }
            QPushButton { 
                background-color: #2a2a2a; color: #ffffff; border: 1px solid #3d3d3d; 
                border-radius: 6px; padding: 10px; font-weight: bold; font-size: 13px; 
            }
            QPushButton:hover { background-color: #3a3a3a; }
            QPushButton:disabled { background-color: #1a1a1a; color: #555555; border: 1px solid #222222; }
            QSlider::groove:horizontal { height: 6px; background: #2d2d2d; border-radius: 3px; }
            QSlider::handle:horizontal { 
                background: #ff6b1a; width: 16px; margin-top: -5px; margin-bottom: -5px; border-radius: 8px; 
            }
            QSlider::handle:horizontal:disabled { background: #555555; }
        """)
        
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel("Camera Stream Waiting...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #292929; background-color: #050505; border-radius: 8px;")
        left_layout.addWidget(self.video_label)
        
        btn_layout = QHBoxLayout()
        
        self.btn_home = QPushButton("집 초기화 (Homing)")
        self.btn_home.setStyleSheet("background-color: #3b2314; border: 1px solid #d35400; color: #e67e22;")
        self.btn_home.clicked.connect(self.trigger_homing)
        btn_layout.addWidget(self.btn_home)
        
        self.btn_ai_toggle = QPushButton("MediaPipe AI 제어: OFF")
        self.btn_ai_toggle.setCheckable(True)
        self.btn_ai_toggle.setStyleSheet("background-color: #1f2a22; border: 1px solid #27ae60; color: #2ecc71;")
        self.btn_ai_toggle.clicked.connect(self.toggle_ai_mode)
        btn_layout.addWidget(self.btn_ai_toggle)
        
        left_layout.addLayout(btn_layout)
        
        right_layout = QVBoxLayout()
        
        self.sliders = []
        for i in range(6):
            row = QHBoxLayout()
            lbl = QLabel(f"Joint {i+1}: 0.0°")
            lbl.setMinimumWidth(110)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)
            
            row.addWidget(lbl)
            row.addWidget(slider)
            right_layout.addLayout(row)
            self.sliders.append((slider, lbl))
            
        grip_main_layout = QVBoxLayout()
        grip_title_lbl = QLabel("그리퍼 실시간 제어")
        grip_title_lbl.setStyleSheet("color: #ff6b1a; font-size: 13px; margin-top: 15px;")
        grip_main_layout.addWidget(grip_title_lbl)
        
        grip_row = QHBoxLayout()
        lbl_close = QLabel("닫힘")
        lbl_close.setStyleSheet("color: #ff4d4d; font-size: 12px;")
        
        self.grip_slider = QSlider(Qt.Horizontal)
        self.grip_slider.setRange(0, 100)
        self.grip_slider.setValue(100)
        self.grip_slider.valueChanged.connect(self.on_gripper_slider_changed)
        
        lbl_open = QLabel("열림")
        lbl_open.setStyleSheet("color: #4da6ff; font-size: 12px;")
        
        grip_row.addWidget(lbl_close)
        grip_row.addWidget(self.grip_slider)
        grip_row.addWidget(lbl_open)
        grip_main_layout.addLayout(grip_row)
        
        self.grip_status_lbl = QLabel("현재 상태: 열림 (35.0)")
        self.grip_status_lbl.setAlignment(Qt.AlignCenter)
        grip_main_layout.addWidget(self.grip_status_lbl)
        
        right_layout.addLayout(grip_main_layout)
        
        main_layout.addLayout(left_layout, stretch=6)
        main_layout.addLayout(right_layout, stretch=4)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.show()

    # 원터치 물리 홈 원점 정렬 명령어($H)를 발행하고 제어 UI를 1차 잠금 처리함
    def trigger_homing(self):
        self.btn_home.setEnabled(False)
        self.btn_home.setText("홈 복귀 진행 중...")
        self.btn_home.setStyleSheet("background-color: #1a1a1a; color: #555555; border: 1px solid #222222;")
        self.is_homing = True # 수동 각도 발행 루프 즉시 잠금
        
        msg = String()
        msg.data = "$H"
        self.node.raw_pub.publish(msg)

    # 드라이버로부터 반환된 실시간 호밍 감지 피드백 상태에 따라 GUI 수동 제어 잠금 및 버튼 상태 해제 처리함
    def update_homing_status(self, is_homing):
        self.is_homing = is_homing
        if not is_homing:
            self.btn_home.setEnabled(True)
            self.btn_home.setText("집 초기화 (Homing)")
            self.btn_home.setStyleSheet("background-color: #3b2314; border: 1px solid #d35400; color: #e67e22;")
        else:
            self.btn_home.setEnabled(False)
            self.btn_home.setText("홈 복귀 진행 중...")
            self.btn_home.setStyleSheet("background-color: #1a1a1a; color: #555555; border: 1px solid #222222;")

    # MediaPipe 토글 상태 변화에 따라 AI 노드 전송 트리거를 송출하고 수동 컨트롤러 위젯 가동 잠금(Read-Only)을 제어함
    def toggle_ai_mode(self, checked):
        self.ai_mode_active = checked
        
        # AI 노드 측에 미디어파이프 데이터 발행 정지/개시 동적 신호 전송
        enable_msg = Bool()
        enable_msg.data = checked
        self.node.ai_enable_pub.publish(enable_msg)
        
        if checked:
            self.btn_ai_toggle.setText("MediaPipe AI 제어: ON")
            self.btn_ai_toggle.setStyleSheet("background-color: #27ae60; border: 1px solid #2ecc71; color: #ffffff;")
            for slider, _ in self.sliders:
                slider.setEnabled(False)
            self.grip_slider.setEnabled(False)
            self.first_ai_frame = True
            self.ai_init = None
            self.robot_init = None
        else:
            self.btn_ai_toggle.setText("MediaPipe AI 제어: OFF")
            self.btn_ai_toggle.setStyleSheet("background-color: #1f2a22; border: 1px solid #27ae60; color: #2ecc71;")
            for slider, _ in self.sliders:
                slider.setEnabled(True)
            self.grip_slider.setEnabled(True)

    # 슬라이더 조작 수치를 역률 캘리브레이션 듀티 범위(S35~S58)로 가공 역매핑하여 그리퍼 토픽으로 송출함
    def on_gripper_slider_changed(self, value):
        if self.ai_mode_active or self.is_homing: # 호밍 중 수동 제어 신호 난입 원천 격리
            return
            
        g_val = 35.0 + ((100 - value) / 100.0) * (58.0 - 35.0)
        status_text = "열림" if value > 80 else "닫힘" if value < 20 else "동작 중"
        self.grip_status_lbl.setText(f"현재 상태: {status_text} ({g_val:.1f})")
        
        msg = Float32()
        msg.data = float(g_val)
        self.node.gripper_pub.publish(msg)

    # 수동 모드일 때 슬라이더 각도 상태를 패킷 어레이로 포장하여 주기(250ms)별로 송출함
    def publish_joint_commands(self):
        if self.ai_mode_active or self.is_homing: # 호밍 중이거나 AI 가동 중일 경우, 수동 슬라이더 G-Code 스팸 전송 락 작동
            return
            
        msg = Float32MultiArray()
        msg.data = [float(s[0].value()) for s in self.sliders]
        self.node.joint_pub.publish(msg)
        
        for i, (slider, lbl) in enumerate(self.sliders):
            lbl.setText(f"Joint {i+1}: {float(slider.value()):.1f}°")

    # 수신된 OpenCV Mat 객체를 QPixmap 포맷으로 보정 변환하여 실시간 비디오 프레임 라벨에 인카네이션 렌더함
    def update_image_display(self, cv_img):
        try:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            q_img = QImage(rgb_img.data, w, h, w * ch, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
        except Exception:
            pass

    # 최초 토글 1프레임 캡쳐본 대비 사용자의 상대 관절 회전 변위를 산정하여 슬라이더 UI 갱신 및 토픽으로 중계함
    def update_ai_joints_display(self, data):
        if not self.ai_mode_active:
            return
            
        if self.first_ai_frame or self.ai_init is None:
            self.ai_init = list(data)
            self.robot_init = [float(s[0].value()) for s in self.sliders]
            self.first_ai_frame = False
            return

        target_angles = []
        for i in range(6):
            if i < len(data) and i < len(self.ai_init):
                delta = data[i] - self.ai_init[i]
                target = self.robot_init[i] + delta
                clamped_val = max(-100.0, min(100.0, target))
                target_angles.append(clamped_val)
            else:
                target_angles.append(0.0)

        for i, val in enumerate(target_angles):
            if i < len(self.sliders):
                self.sliders[i][0].setValue(int(val))
                self.sliders[i][1].setText(f"Joint {i+1} (AI): {float(val):.1f}°")
                
        msg = Float32MultiArray()
        msg.data = [float(x) for x in target_angles]
        self.node.joint_pub.publish(msg)

    # 수신된 AI 손 끝 사잇각 데이터(35.0~58.0)를 역연산 퍼센테이지 슬라이더 가치로 동적 역투영 렌더링함
    def update_ai_gripper_display(self, g_val):
        if not self.ai_mode_active:
            return
            
        clamped_val = max(35.0, min(58.0, g_val))
        pct = 100 - int(((clamped_val - 35.0) / (58.0 - 35.0)) * 100)
        
        self.grip_slider.setValue(max(0, min(100, pct)))
        self.grip_status_lbl.setText(f"현재 상태 (AI): {pct}% ({clamped_val:.1f})")
        
        out_msg = Float32()
        out_msg.data = float(clamped_val)
        self.node.gripper_pub.publish(out_msg)


# rclpy 인프라 바인딩 및 PyQt 어플리케이션 백그라운드 ROS 동시 스핀 스레드 실행 핸들러 수행함
def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    signals = RosBridgeSignals()
    node = MirobotGuiNode(signals)
    window = MirobotGuiWindow(node, signals)
    
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()
    
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()