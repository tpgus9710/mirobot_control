import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge

class EMAFilter:
    """모터 진동 최소화를 위한 스무딩 데이터 필터"""
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.last_val = None

    def apply(self, val):
        if self.last_val is None:
            self.last_val = val
        else:
            self.last_val = self.alpha * val + (1 - self.alpha) * self.last_val
        return self.last_val

class MirobotAiNode(Node):
    """
    웹캠으로부터 실시간 비디오를 입력받아 MediaPipe로 인체 관절 및 핸드 제스처를 분석합니다.
    분석된 가동 각도는 필터를 거쳐 정제된 후 ROS 2 토픽으로 송신됩니다.
    """
    def __init__(self):
        super().__init__('mirobot_ai_node')
        
        # 이미지 전송을 위한 CV Bridge 초기화
        self.bridge = CvBridge()
        
        # AI 결과값 발행용 퍼블리셔(Publisher)
        self.ai_joint_pub = self.create_publisher(Float32MultiArray, '/mirobot/ai_joints', 10)
        self.ai_gripper_pub = self.create_publisher(Float32, '/mirobot/ai_gripper', 10)
        self.image_pub = self.create_publisher(Image, '/mirobot/camera_image', 10)

        # MediaPipe 모델 로드
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils

        # 실시간 필터 설정 (어깨, 팔꿈치, 그리퍼 스무더)
        self.filters = {
            'joint1': EMAFilter(alpha=0.1),
            'joint2': EMAFilter(alpha=0.1),
            'joint3': EMAFilter(alpha=0.1),
            'gripper': EMAFilter(alpha=0.25)
        }

        # 카메라 장치 획득 (0번 웹캠)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("카메라 장치를 실행할 수 없습니다. 연결 상태를 확인하세요.")
            
        # 고속 처리를 위해 주기적 타이머 구동 (약 30 FPS)
        self.timer = self.create_timer(0.033, self.process_frame_loop)
        self.get_logger().info("Mirobot AI 분석 엔진이 성공적으로 시작되었습니다.")

    def calculate_angle(self, a, b, c):
        """세 점 사이의 각도를 수식 연산합니다."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(rad * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def process_frame_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # 연산 효율화를 위해 좌우 대칭 및 RGB 변환
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 추론 연산 가동
        pose_res = self.pose.process(image_rgb)
        hand_res = self.hands.process(image_rgb)
        
        # 기본 제어 데이터 프레임워크 초기화
        joints_to_send = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        gripper_to_send = 50.0 # 기본값 (절반 개방)
        
        # [A] 상체 자세 분석 영역 (오른쪽 어깨: 12, 팔꿈치: 14, 손목: 16)
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            shoulder = [lm[12].x, lm[12].y]
            elbow = [lm[14].x, lm[14].y]
            wrist = [lm[16].x, lm[16].y]
            
            # 1. 왼쪽/오른쪽 몸의 이동 -> Joint 1 (좌우 회전) 매핑
            # 어깨 중심과 손목의 상대 거리를 추적하여 좌우 150도 회전에 바인딩
            x_delta = wrist[0] - shoulder[0]
            target_j1 = np.interp(x_delta, [-0.4, 0.4], [-90.0, 90.0])
            joints_to_send[0] = self.filters['joint1'].apply(target_j1)
            
            # 2. 팔꿈치 접힘 각도 -> Joint 2 (상하 틸트) 매핑
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            # 팔꿈치 각도 40~160도를 미로봇 관절범위 -40~50도로 부드럽게 매핑
            target_j2 = np.interp(elbow_angle, [40, 160], [50.0, -40.0])
            joints_to_send[1] = self.filters['joint2'].apply(target_j2)

            # 3. 손 높이 변화 -> Joint 3 (전후 조절) 매핑
            y_delta = shoulder[1] - wrist[1]
            target_j3 = np.interp(y_delta, [-0.2, 0.3], [-45.0, 45.0])
            joints_to_send[2] = self.filters['joint3'].apply(target_j3)
            
            # 비디오 프레임에 뼈대 그리기
            self.mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # [B] 핸드 제스처 분석 영역 (집게 비례 매핑)
        if hand_res.multi_hand_landmarks:
            hlm = hand_res.multi_hand_landmarks[0].landmark
            
            # 랜드마크 4번(엄지 끝)과 8번(검지 끝) 간격 측정
            thumb_tip = [hlm[4].x, hlm[4].y]
            index_tip = [hlm[8].x, hlm[8].y]
            wrist_pos = [hlm[0].x, hlm[0].y]
            middle_base = [hlm[9].x, hlm[9].y]
            
            # 깊이에 의한 거리 왜곡 방지용 정규화 비율 적용 (손바닥 폭 크기 기준)
            palm_size = np.sqrt((wrist_pos[0]-middle_base[0])**2 + (wrist_pos[1]-middle_base[1])**2)
            dist = np.sqrt((thumb_tip[0]-index_tip[0])**2 + (thumb_tip[1]-index_tip[1])**2)
            ratio = dist / (palm_size + 1e-6)
            
            # 엄지와 검지가 가까워지면 집게를 닫고(0%), 펴면 연다(100%)
            target_grip = np.interp(ratio, [0.2, 1.0], [0.0, 100.0])
            gripper_to_send = self.filters['gripper'].apply(target_grip)
            
            self.mp_drawing.draw_landmarks(frame, hand_res.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)

        # ROS 2 토픽 발행 처리
        joint_msg = Float32MultiArray()
        joint_msg.data = [float(j) for j in joints_to_send]
        self.ai_joint_pub.publish(joint_msg)
        
        grip_msg = Float32()
        grip_msg.data = float(gripper_to_send)
        self.ai_gripper_pub.publish(grip_msg)

        # GUI 모니터링을 위해 이미지 데이터 전송
        try:
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"이미지 전송 역변환 실패: {e}")

    def destroy_node(self):
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()