import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QCheckBox, QGroupBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

class RosGuiBridge(Node):
    """PyQt GUI 내부에서 백그라운드로 작동하는 ROS 2 통신 서브 노드입니다."""
    def __init__(self, gui_window):
        super().__init__('mirobot_gui_bridge_node')
        self.gui = gui_window
        self.bridge = CvBridge()

        # GUI 조작 값을 로봇 드라이버에게 전달할 퍼블리셔
        self.command_pub = self.create_publisher(Float32MultiArray, '/mirobot/joint_commands', 10)
        self.gripper_pub = self.create_publisher(Float32, '/mirobot/gripper', 10)

        # AI 연산 노드로부터 계산된 각도 수신
        self.ai_joint_sub = self.create_subscription(Float32MultiArray, '/mirobot/ai_joints', self.ai_joint_callback, 10)
        self.ai_grip_sub = self.create_subscription(Float32, '/mirobot/ai_gripper', self.ai_gripper_callback, 10)
        
        # 분석 영상 수신 및 GUI 캔버스 연동
        self.img_sub = self.create_subscription(Image, '/mirobot/camera_image', self.image_callback, 10)

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.gui.update_video_canvas(cv_img)
        except Exception as e:
            pass

    def ai_joint_callback(self, msg):
        # AI 주도 모드가 켜져 있을 때만 수신값을 슬라이더 및 로봇 제어에 실시간 동기화
        if self.gui.ai_mode_enabled:
            self.gui.sync_ai_joints(msg.data)

    def ai_gripper_callback(self, msg):
        if self.gui.ai_mode_enabled:
            self.gui.sync_ai_gripper(msg.data)

    def publish_commands(self, joints, gripper):
        # 최종 제어 신호 퍼블리싱
        j_msg = Float32MultiArray()
        j_msg.data = [float(val) for val in joints]
        self.command_pub.publish(j_msg)

        g_msg = Float32()
        g_msg.data = float(gripper)
        self.gripper_pub.publish(g_msg)


class MirobotGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mirobot AI & Manual Controller (ROS 2 Humble)")
        self.setGeometry(100, 100, 1100, 700)
        
        # 로직 관리용 전역 상태
        self.ai_mode_enabled = False
        self.joint_limits = [
            (-150, 150), # J1
            (-60, 60),   # J2
            (-50, 50),   # J3
            (-120, 120), # J4
            (-90, 90),   # J5
            (-120, 120)  # J6
        ]
        
        self.init_ui()
        
        # PyQt 환경에 ROS 2 노드 이식 및 스핀 타이머 구동
        self.ros_node = RosGuiBridge(self)
        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.spin_ros_once)
        self.ros_timer.start(15) # 15ms 주기로 ROS 이벤트 확인

    def init_ui(self):
        # 기본 위젯 및 다크 테마 시트 구현
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #ffffff; }
            QGroupBox { border: 2px solid #3d3d3d; border-radius: 8px; margin-top: 10px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; color: #00adb5; }
            QSlider::groove:horizontal { border: 1px solid #333; height: 8px; background: #3d3d3d; border-radius: 4px; }
            QSlider::handle:horizontal { background: #00adb5; border: 1px solid #00adb5; width: 18px; margin: -5px 0; border-radius: 9px; }
            QPushButton { background-color: #393e46; border: 1px solid #00adb5; border-radius: 5px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #00adb5; color: #1e1e1e; }
            QCheckBox { spacing: 8px; font-size: 14px; font-weight: bold; }
            QCheckBox::indicator { width: 18px; height: 18px; }
        """)

        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # [좌측 영역] 비디오 스트리밍 출력
        left_layout = QVBoxLayout()
        self.video_label = QLabel("웹캠 영상 연결 대기 중...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("border: 3px solid #3d3d3d; background-color: #000000; border-radius: 10px;")
        left_layout.addWidget(self.video_label)
        
        # AI 상태 표시 라벨
        self.status_label = QLabel("수동 조작 활성화 모드")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 13, QFont.Bold))
        self.status_label.setStyleSheet("color: #00adb5; padding: 10px;")
        left_layout.addWidget(self.status_label)
        main_layout.addLayout(left_layout)

        # [우측 영역] 제어 패널 디자인
        right_layout = QVBoxLayout()
        
        # 1. AI 제어 토글 스위치 그룹
        mode_group = QGroupBox("시스템 제어 제어 방식")
        mode_layout = QHBoxLayout()
        self.ai_checkbox = QCheckBox("MediaPipe AI 제어 활성화")
        self.ai_checkbox.stateChanged.connect(self.on_ai_toggle)
        self.btn_home = QPushButton("기본 홈 위치(Home) 복귀")
        self.btn_home.clicked.connect(self.send_home_position)
        
        mode_layout.addWidget(self.ai_checkbox)
        mode_layout.addWidget(self.btn_home)
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)

        # 2. 관절별 슬라이더 그룹
        slider_group = QGroupBox("수동 관절 제어 패널 (각도 조작)")
        slider_layout = QVBoxLayout()
        
        self.sliders = []
        self.slider_labels = []
        
        for i in range(6):
            h_layout = QHBoxLayout()
            limits = self.joint_limits[i]
            
            label = QLabel(f"관절 {i+1}: 0°")
            label.setFixedWidth(100)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(limits[0], limits[1])
            slider.setValue(0)
            slider.valueChanged.connect(self.on_slider_changed)
            
            h_layout.addWidget(label)
            h_layout.addWidget(slider)
            slider_layout.addLayout(h_layout)
            
            self.sliders.append(slider)
            self.slider_labels.append(label)

        # 그리퍼 슬라이더 추가
        grip_layout = QHBoxLayout()
        self.grip_label = QLabel("그리퍼: 50%")
        self.grip_label.setFixedWidth(100)
        self.grip_slider = QSlider(Qt.Horizontal)
        self.grip_slider.setRange(0, 100)
        self.grip_slider.setValue(50)
        self.grip_slider.valueChanged.connect(self.on_slider_changed)
        grip_layout.addWidget(self.grip_label)
        grip_layout.addWidget(self.grip_slider)
        slider_layout.addLayout(grip_layout)

        slider_group.setLayout(slider_layout)
        right_layout.addWidget(slider_group)
        main_layout.addLayout(right_layout)

    def spin_ros_once(self):
        # 메인 루프가 끊기지 않게 넌블로킹 스핀 구동
        rclpy.spin_once(self.ros_node, timeout_sec=0)

    def update_video_canvas(self, frame):
        # ROS 2 토픽 영상 이미지를 PyQt 화면 규격에 맞게 변환하여 렌더링
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(640, 480, Qt.KeepAspectRatio))

    def on_ai_toggle(self, state):
        self.ai_mode_enabled = (state == Qt.Checked)
        if self.ai_mode_enabled:
            self.status_label.setText("★ AI 카메라 관절 트래킹 제어 작동 중 ★")
            self.status_label.setStyleSheet("color: #ff2e63; font-weight: bold;")
            # 수동 조작 방지용 비활성화 처리
            for s in self.sliders: s.setEnabled(False)
            self.grip_slider.setEnabled(False)
        else:
            self.status_label.setText("수동 조작 활성화 모드")
            self.status_label.setStyleSheet("color: #00adb5;")
            for s in self.sliders: s.setEnabled(True)
            self.grip_slider.setEnabled(True)

    def on_slider_changed(self):
        # 수동 제어 시 변경된 슬라이더 조작 데이터를 취득 및 전송
        if not self.ai_mode_enabled:
            joints = []
            for i, s in enumerate(self.sliders):
                val = s.value()
                self.slider_labels[i].setText(f"관절 {i+1}: {val}°")
                joints.append(val)
                
            grip_val = self.grip_slider.value()
            self.grip_label.setText(f"그리퍼: {grip_val}%")
            self.ros_node.publish_commands(joints, grip_val)

    def sync_ai_joints(self, joint_data):
        # AI가 보낸 관절 각도를 안전 범위 안에서 수치화하고 슬라이더 막대를 고스트 작동시킴
        for i, s in enumerate(self.sliders):
            if i < len(joint_data):
                angle = int(joint_data[i])
                s.setValue(angle)
                self.slider_labels[i].setText(f"관절 {i+1}: {angle}° (AI)")
        
        # 즉시 로봇 노드로 송신 처리
        self.ros_node.publish_commands(joint_data, self.grip_slider.value())

    def sync_ai_gripper(self, grip_val):
        self.grip_slider.setValue(int(grip_val))
        self.grip_label.setText(f"그리퍼: {int(grip_val)}% (AI)")

    def send_home_position(self):
        # 모든 슬라이더를 0으로 맞추고 홈 위치 명령 전송
        if not self.ai_mode_enabled:
            for s in self.sliders:
                s.setValue(0)
            self.grip_slider.setValue(50)
            self.ros_node.publish_commands([0, 0, 0, 0, 0, 0], 50)

    def closeEvent(self, event):
        # 닫기 버튼 동작 시 ROS 2 동반 클린 종료
        self.ros_node.destroy_node()
        super().closeEvent(event)


def main():
    rclpy.init()
    app = QApplication(sys.argv)
    gui = MirobotGui()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()