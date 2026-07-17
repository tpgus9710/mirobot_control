import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Float32, Float32MultiArray, Bool
import cv2
import mediapipe as mp
import math
import time
from cv_bridge import CvBridge

# 카메라 프레임 분석, 3차원 인체 관절 매핑 및 임계 데드밴드 각도 산출 엔진 담당 노드 클래스
class MirobotAiNode(Node):
    # AI 추론기 객체, 프레임 디코더 및 가동 주기(250ms) 변수 초기화함
    def __init__(self):
        super().__init__('mirobot_ai_node')
        self.bridge = CvBridge()
        
        self.image_pub = self.create_publisher(ROSImage, '/mirobot/camera_image', 10)
        self.ai_joints_pub = self.create_publisher(Float32MultiArray, '/mirobot/ai_joint_commands', 10)
        self.gripper_pub = self.create_publisher(Float32, '/mirobot/ai_gripper_command', 10)
        
        # GUI로부터 연동 활성화 상태를 인가받을 구독자 정의
        self.ai_enable_sub = self.create_subscription(
            Bool,
            '/mirobot/ai_enable',
            self.ai_enable_callback,
            10
        )
        self.ai_enabled = False # 초기값은 비활성(송출 즉시 중단)
        
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        if not self.cap.isOpened():
            self.get_logger().error("원격 웹캠 장치 자원에 접근하지 못했습니다.")
            
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            max_num_hands=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 커스텀 상반신 관절망 연결 링크 구성 (양 어깨, 양 팔꿈치, 양 손목 링크 구성)
        self.ARM_CONNECTIONS = [
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16)
        ]
        
        self.last_published_joints = [0.0] * 6
        self.joint_threshold = 2.0 # 요청하신 움직임 각도 데드밴드 2.0도로 세팅
        
        self.last_published_gripper = 35.0
        self.gripper_threshold = 1.0
        
        # 250ms 완급 조율용 타임스탬프 변수 초기화
        self.last_joint_pub_time = 0.0
        self.last_gripper_pub_time = 0.0
        self.pub_interval = 0.250 # 250ms 전송 속도 제약 적용 (4Hz)
        
        self.timer = self.create_timer(0.05, self.process_frame)
        self.get_logger().info("데드밴드 필터 및 250ms 완급식 조절형 AI 엔진 준비 완료.")

    # GUI 토글스위치의 상태에 따라 원격 데이터 송출 스위치를 즉시 활성/비활성 처리함
    def ai_enable_callback(self, msg):
        self.ai_enabled = msg.data
        status_str = "연동 가동 (250ms 주기로 전송)" if self.ai_enabled else "가동 비활성 (관절/그리퍼 송출 즉시 전면 중단)"
        self.get_logger().info(f"MediaPipe AI 제어 상태 동적 업데이트: {status_str}")

    # 세 포인트(a, b, c)간의 물리적 3차원 내적 사이 각도를 라디안 디그리로 선형 산출함
    def calculate_angle_3d(self, a, b, c):
        v1 = [a.x - b.x, a.y - b.y, a.z - b.z]
        v2 = [c.x - b.x, c.y - b.y, c.z - b.z]
        
        dot_product = sum(i*j for i, j in zip(v1, v2))
        norm_v1 = math.sqrt(sum(i*i for i in v1))
        norm_v2 = math.sqrt(sum(i*i for i in v2))
        
        cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-6)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.acos(cos_theta) * 180.0 / math.pi

    # 비디오 캡처 장치로부터 이미지를 획득하여 랜드마크 분석, 데드밴드(2.0도) 필터링 및 토픽 송출을 실행함
    def process_frame(self):
        if not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        current_time = time.time()
        
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            
            sh = lm[12]
            el = lm[14]
            wr = lm[16]
            
            base_dx = el.x - sh.x
            target_j1 = max(-70.0, min(70.0, base_dx * -220.0))
            
            base_dy = el.y - sh.y
            target_j2 = max(-60.0, min(60.0, (base_dy - 0.3) * -180.0))
            
            arm_angle = self.calculate_angle_3d(sh, el, wr)
            target_j3 = max(-60.0, min(60.0, -60.0 + ((arm_angle - 50.0) / 130.0) * 120.0))
            
            proposed_angles = [target_j1, target_j2, target_j3, 0.0, 0.0, 0.0]
            
            # AI 모드가 ON이고 250ms 전송 주기가 만료되었을 때만 실제 ROS 토픽으로 송출
            if self.ai_enabled and (current_time - self.last_joint_pub_time >= self.pub_interval):
                joint_changed = False
                for i in range(6):
                    if abs(proposed_angles[i] - self.last_published_joints[i]) >= self.joint_threshold:
                        self.last_published_joints[i] = proposed_angles[i]
                        joint_changed = True
                
                if joint_changed:
                    joint_msg = Float32MultiArray()
                    joint_msg.data = [float(x) for x in self.last_published_joints]
                    self.ai_joints_pub.publish(joint_msg)
                    self.last_joint_pub_time = current_time
            
            # 비활성(OFF) 상태에서도 화면 상의 유저가 뼈대를 인지하도록 모니터링 드로잉은 상시 활성화 유지
            active_joints = [11, 12, 13, 14, 15, 16]
            for idx in active_joints:
                pt = lm[idx]
                if pt.visibility > 0.5:
                    cx, cy = int(pt.x * w), int(pt.y * h)
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
                    cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
            
            for start_idx, end_idx in self.ARM_CONNECTIONS:
                pt_start = lm[start_idx]
                pt_end = lm[end_idx]
                if pt_start.visibility > 0.5 and pt_end.visibility > 0.5:
                    x1, y1 = int(pt_start.x * w), int(pt_start.y * h)
                    x2, y2 = int(pt_end.x * w), int(pt_end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 150, 0), 4)

        if hand_results.multi_hand_landmarks:
            hlm = hand_results.multi_hand_landmarks[0].landmark
            
            wrist_pt = hlm[0]
            thumb_pt = hlm[4]
            index_pt = hlm[8]
            
            pinch_angle = self.calculate_angle_3d(thumb_pt, wrist_pt, index_pt)
            
            if pinch_angle < 12.0:
                proposed_gripper = 58.0
            elif pinch_angle > 36.0:
                proposed_gripper = 35.0
            else:
                proposed_gripper = 58.0 - ((pinch_angle - 12.0) / (36.0 - 12.0)) * (58.0 - 35.0)
                
            # AI 모드가 ON이고 250ms 전송 주기가 만료되었을 때만 그리퍼 토픽 전송
            if self.ai_enabled and (current_time - self.last_gripper_pub_time >= self.pub_interval):
                if abs(proposed_gripper - self.last_published_gripper) >= self.gripper_threshold:
                    self.last_published_gripper = proposed_gripper
                    
                    grip_msg = Float32()
                    grip_msg.data = float(proposed_gripper)
                    self.gripper_pub.publish(grip_msg)
                    self.last_gripper_pub_time = current_time
            
            self.mp_drawing.draw_landmarks(frame, hand_results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"Finger Angle: {pinch_angle:.1f}deg -> S{self.last_published_gripper:.1f}", (30, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        try:
            ros_img = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_pub.publish(ros_img)
        except Exception:
            pass

    # 종료 시 열려 있는 웹캠 카메라 디바이스 세션을 메모리에서 회수 및 해제함
    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


# rclpy 통신 모듈 초기화 및 AI 분석기 백그라운드 스핀 실행 수행함
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