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

# [GRU 랜드마크 보정] NumPy 순전파. PyTorch 없이 numpy만으로 동작한다.
#   모델 파일(.npz)이 없거나 import에 실패하면 보정 없이 원래대로 동작한다
#   (제어가 멈추는 것보다 보정을 포기하는 쪽이 안전).
try:
    from mirobot_ai_control.gru_runtime import LandmarkCorrector
except ImportError:
    try:
        from gru_runtime import LandmarkCorrector
    except ImportError:
        LandmarkCorrector = None
from collections import deque
from cv_bridge import CvBridge
from flask import Flask, Response, request
import websockets

# [학습 데이터 녹화] NLF_RECORD_DIR 환경변수가 설정된 경우에만 동작.
# 미설정 시 완전 비활성 (라이브 파이프라인 영향 0).
# ROS2 패키지로 설치된 경우와 같은 폴더에서 직접 실행하는 경우를 모두 지원.
try:
    from mirobot_ai_control.session_recorder import SessionRecorder
except ImportError:
    from session_recorder import SessionRecorder


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


# ══════════════════════════════════════════════════════════════════════════
#  Median Prefilter (One Euro Filter 앞단의 경량 이상값 제거기)
#
#  One Euro Filter는 "속도가 크면 스무딩을 약하게 푼다"는 적응형 로직이라,
#  MediaPipe가 한두 프레임만 랜드마크를 완전히 잘못 잡아 값이 순간적으로
#  튀면(occlusion, 모션 블러 등) 그 튐을 "빠른 실제 움직임"으로 오인해서
#  오히려 잘 통과시켜버리는 약점이 있음. 즉 적응형이라는 장점이 이상값에
#  대해서는 그대로 단점이 됨.
#
#  중앙값(median)은 정렬 후 가운데 값을 취하므로, 창(window) 절반 미만의
#  소수 이상값에는 거의 영향을 받지 않음. 그래서 One Euro Filter 앞에
#  짧은 창(3프레임)으로 한 번 걸러주면, "명백한 순간적 오탐지"는 여기서
#  대부분 사라지고, One Euro Filter는 원래 역할(속도 기반 스무딩)에만
#  집중할 수 있게 됨.
#
#  주의: 이미 있는 MIN_UPPERARM_LEN_M 등 벡터 길이 가드와는 역할이 다름.
#  가드는 "계산 자체가 무의미한 극단적 프레임"을 통째로 스킵하는 것이고,
#  이 필터는 "계산은 됐지만 값이 순간적으로 튄 경우"를 완만하게 눌러주는
#  것. 두 장치는 서로 대체가 아니라 보완 관계.
#
#  창을 3으로 짧게 잡은 이유: 창이 길수록 이상값에는 더 강해지지만 그만큼
#  지연(latency)도 늘어남. 여기서는 "한두 프레임짜리 순간적 튐"만 잡는 게
#  목적이라 최소한의 창(3)으로 지연을 거의 없앰(중앙값 자체는 새 값이
#  들어올 때마다 바로 갱신되므로 이동평균처럼 값을 뭉개지 않음).
# ══════════════════════════════════════════════════════════════════════════
class MedianPrefilter:
    def __init__(self, window=3):
        self.window = window
        self.buf = deque(maxlen=window)

    def filter(self, x):
        self.buf.append(x)
        # 버퍼가 아직 안 찼어도(첫 1~2프레임) 그 시점까지 모인 값들의
        # 중앙값을 그대로 반환 — len=1이면 그 값 자체, len=2면 둘 중
        # 정렬상 뒤쪽 값(정확한 중앙값 정의상 평균을 써도 되지만, 여기선
        # "튄 값 하나에 안 흔들린다"는 목적에는 이 방식으로 충분하고 더 단순함).
        return sorted(self.buf)[len(self.buf) // 2]

    def reset(self):
        # OneEuroFilter.reset()과 같은 시점(AI ON/OFF, 팔 전환, J5 stale-gap)에
        # 반드시 같이 호출해야 함. 안 그러면 리셋 전 버퍼에 남아있던 "이전 상태"
        # 값이 중앙값 계산에 섞여 들어가서, 재개 직후 몇 프레임 동안
        # 엉뚱한 값이 나올 수 있음.
        self.buf.clear()


class MirobotAiNode(Node):

    # ══════════════════════════════════════════════════════════════════════
    #  [v2 / IK 방식] 튜닝 상수
    #
    #  ★ 이전 버전(v1)과의 핵심 차이 ★
    #  v1: 사람 관절각(어깨각/팔꿈치각) → 로봇 관절각 을 선형보간으로 직접 매핑
    #      → 사람 팔(7자유도)과 로봇 팔(6자유도)의 구조가 달라서 자세가 어긋나고,
    #        arccos/atan2가 특이점 근처에서 발산하면서 값이 튀는 문제가 있었음.
    #  v2: 사람 "손목 위치"(3D) → 로봇 "손끝 위치"(3D) → 해석적 역기구학(IK)
    #      → 관절각이 아니라 위치를 맞추므로 물체 조작에 직접적으로 유리하고,
    #        IK에 넣기 "전에" 목표점을 도달 가능 영역으로 클램프하기 때문에
    #        애초에 arccos에 불량 입력이 들어가지 않아 튐이 구조적으로 사라짐.
    # ══════════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════
    #  1) 로봇 링크 파라미터 (mm)
    #
    #  출처: WLKATA 공식 ROS2 저장소의 URDF
    #        wlkata_mirobot_description/urdf/wlkata_mirobot_description.urdf
    #        (joint origin xyz 값을 그대로 mm로 환산)
    #
    #    joint1 : base_link -> link1   xyz = (0,        0, 0.127)
    #    joint2 : link1     -> link2   xyz = (0.029687, 0, 0)
    #    joint3 : link2     -> link3   xyz = (0.108,    0, 0)
    #    joint4 : link3     -> link4   xyz = (0.02,     0.16875, 0)   ← 전완 + 오프셋
    #
    #  ⚠ 이 값이 네 실물과 맞는지 확인하는 법 (1분):
    #     호밍 직후(J1~J6=0) 상태에서
    #       - 손목 중심(J5 축)의 높이가 베이스 바닥면에서 약 255mm
    #       - 손목 중심이 베이스 중심에서 앞쪽으로 약 198mm
    #     이면 이 값이 맞음. 크게 다르면 아래 5개 숫자만 실측값으로 고치면 됨.
    # ══════════════════════════════════════════════════════
    LINK_D1_MM   = 127.0      # J1축(베이스면) → J2축 높이
    LINK_A1_MM   = 29.687     # J1축 → J2축 수평 오프셋
    LINK_L2_MM   = 108.0      # J2축 → J3축 (상완)
    # 전완은 (앞 168.75mm, 위 20mm) 오프셋 구조라 "길이 + 고정각"으로 표현함
    LINK_L3_MM   = 169.932    # J3축 → 손목중심(J5축) 거리  = hypot(168.75, 20)
    LINK_PHI3_DEG = 83.2365   # 전완의 고정 오프셋각        = degrees(atan2(168.75, 20))

    # 그리퍼 길이 (J5축 → 집게 끝). Z_MIN 안전 여유 계산에만 쓰임.
    # 실측해서 넣으면 더 정확해짐. 몰라도 동작에는 지장 없음.
    LINK_TOOL_MM = 100.0

    # ══════════════════════════════════════════════════════
    #  2) 로봇 작업공간 한계 (mm, 베이스 좌표계)
    #     x = 정면 방향(+), y = 왼쪽 방향(+), z = 위쪽(+), 원점 = 베이스 중심 바닥
    #
    #  ★ 이 클램프는 반드시 IK "이전"에 적용됨 ★
    #     v1에서 값이 튀던 진짜 원인은 arccos/atan2에 도달 불가능한 입력이
    #     들어가서 미분값이 발산한 것이었음. 필터로 사후에 뭉개는 대신
    #     불량 입력 자체를 차단하는 구조로 바꿈.
    # ══════════════════════════════════════════════════════
    # J2축 중심에서 손목중심까지의 거리 D 의 허용 범위
    #   물리적 한계: |L2 - L3| = 61.9mm  ~  L2 + L3 = 277.9mm
    #   경계에 바짝 붙으면 arccos 인자가 ±1에 접근해 민감해지므로 여유를 둠
    #  ⚠ 최소값이 |L2-L3|=61.9mm 가 아니라 132mm인 이유:
    #     J3의 하드리밋이 +50°인데, J3=+50°일 때의 D가 이미 124.1mm임.
    #     즉 기구학적으로는 더 접힐 수 있어도 관절 리밋 때문에 못 접음.
    #     여기를 61.9로 잡으면 로봇이 도달 못 하는 목표가 만들어져서
    #     J3가 리밋에 걸린 채 위치 오차가 계속 남게 됨(전수검사로 확인).
    WS_D_MIN_MM = 132.0       # J3 리밋(+50°)에서의 실제 최소 도달거리 124.1mm + 여유
    WS_D_MAX_MM = 258.0       # 도달 한계(277.9mm)의 약 93%

    # J1 축(베이스 수직축) 바로 위 원기둥형 데드존.
    # r → 0 이면 atan2(y, x)의 방향이 정의되지 않아 J1이 노이즈로 미쳐 날뜀.
    WS_R_MIN_XY_MM = 60.0

    # 목표점을 로봇 정면(앞쪽 반공간)으로 제한.
    # [이유] 사용자가 팔을 몸 뒤로 당기면 매핑 결과 x가 음수가 될 수 있는데,
    # 그러면 atan2(y, x)가 ±180° 근처가 되고 y 부호가 바뀌는 순간 J1이
    # +180 ↔ -180 으로 불연속 점프함(로봇이 통째로 반 바퀴 돌아버림).
    # x를 항상 양수로 유지하면 방위각이 -90~+90 안에서만 움직여서 이 점프가
    # 원리적으로 발생할 수 없음.
    WS_X_MIN_MM = 70.0
    WS_YAW_MAX_DEG = 85.0   # J1 하드리밋(±100°)보다 여유 있게

    # 높이 한계. Z_MIN은 "손목 중심"의 최저 높이라서, 그리퍼가 아래로 향할 때는
    # 여기서 LINK_TOOL_MM 만큼 더 내려간다는 점을 감안해서 잡아야 함.
    # [현장 정보] 메카넘 바닥 → 미로봇 베이스 바닥면 = 약 85mm.
    #   → 지면 위 물체를 집으려면 Z_MIN을 음수로 낮춰야 하지만, 우선은
    #     "플랫폼 상판을 절대 찍지 않는" 안전값으로 시작하고 실측 후 조정 권장.
    WS_Z_MIN_MM = 40.0
    WS_Z_MAX_MM = 340.0

    # ══════════════════════════════════════════════════════
    #  3) 사람 → 로봇 매핑
    #
    #  사람 쪽 입력은 Pose world landmark(미터 단위 3D)의
    #    v = (손목 - 어깨) / 팔길이      ← 무차원. 체격/카메라 거리와 무관.
    #  세 성분(fwd/lat/up)을 로봇 (x, y, z)로 선형 매핑함.
    # ══════════════════════════════════════════════════════
    # 축 부호. 실기에서 반대로 움직이면 해당 값만 -1.0으로 뒤집으면 됨.
    # (화면 좌상단에 목표 좌표 mm가 표시되므로 그걸 보면서 맞추면 됨)
    SIGN_FWD = 1.0   # 팔을 카메라 쪽으로 뻗음 → 로봇 +x(앞)
    SIGN_LAT = -1.0   # 손을 화면 오른쪽으로   → 로봇 -y(오른쪽)
    SIGN_UP  = 1.0   # 손을 위로              → 로봇 +z

    # 캘리브레이션 전 기본 추정치 (무차원, 팔길이로 정규화된 값)
    #   팔을 곧게 아래로 내리면 up ≈ -1.0, 앞으로 수평이면 fwd ≈ 1.0
    # 캘리브레이션 중립 단계에서 측정해 고정하는 팔 길이(m).
    #  [왜 고정하는가] MediaPipe가 재는 팔 길이는 자세에 따라 크게 변한다.
    #  팔을 앞으로 뻗으면 투영 단축으로 짧게(0.30m), 옆으로 벌리면 길게(0.46m)
    #  나온다. 그런데 이 값은 정규화의 분모이므로, 프레임마다 바뀌면
    #    · GRU 입력 스케일이 매 프레임 달라져 학습 조건과 어긋나고
    #    · norm_fwd 값 자체가 자세가 아니라 분모 때문에 흔들린다.
    #  실제 사람 팔 길이는 변하지 않으므로 한 번 재서 쓰는 것이 옳다.
    CAL_ARM_LEN = 0.0          # 0이면 미측정 — 기존처럼 실시간 중앙값 사용
    CAL_FWD_NEUTRAL = 0.45
    CAL_FWD_MAX     = 0.95
    CAL_UP_NEUTRAL  = -0.50
    CAL_UP_MAX      = 0.60
    CAL_LAT_NEUTRAL = 0.0

    # 위 사람 쪽 기준점들이 대응되는 로봇 좌표 (mm)
    # [주의] 위의 WS_*_MM(작업공간 한계)과 헷갈리지 않도록 MAP_ 접두사를 씀.
    #        이 네 점은 모두 도달 가능 영역 안에 있어야 함(전수검사로 확인 완료).
    MAP_X_NEUTRAL = 198.0   # 중립 자세일 때 로봇이 있을 앞뒤 위치
    MAP_X_MAX     = 250.0   # 최대로 앞으로 뻗었을 때
    MAP_Z_NEUTRAL = 255.0   # 중립 자세일 때 높이
    MAP_Z_MAX     = 330.0   # 최대로 위로 들었을 때 (WS_Z_MAX_MM 와 동일)


    # ── 뒤쪽 x 매핑 ──────────────────────────────────────────────────────
    # 캘리브레이션이 중립과 '앞으로 최대' 두 점만 재므로, 뒤로 뺄 때 쓸
    # 기울기가 없다. 아래 MAP_Z_MIN 과 같은 이유·같은 방식이다.
    # 사람이 팔을 뒤로 빼는 범위는 앞으로 뻗는 범위보다 좁으므로 1.0 보다
    # 작은 값이 맞다. 뒤로 잘 안 가면 줄이고, 너무 예민하면 키운다.
    MAP_X_MIN           = 150.0   # 최대로 뒤로 뺐을 때 (WS_X_MIN_MM=70 보다 여유)
    MAP_BACK_SPAN_SCALE =   0.17   # 뒤쪽 사람 가동범위 = fwd_span * 이 값
    # ── 아래 방향 z 매핑 ──────────────────────────────────────────────────
    # [왜 따로 필요한가]
    #   MAP_Z_NEUTRAL 은 호밍 자세(전 관절 0°)의 손목 높이 255mm 에 맞춰져 있다.
    #   그래야 중립 자세를 취했을 때 로봇이 호밍 위치에 그 대로 머문다.
    #
    #   그런데 위쪽 게인 (MAP_Z_MAX - MAP_Z_NEUTRAL) / up_span 은 분자가
    #   15mm 뿐이라, 이 기울기를 아래쪽에도 그대로 쓰면 손을 배꼽까지 내려도
    #   z 가 거의 안 내려간다. 반대로 게인을 살리려고 MAP_Z_NEUTRAL 을 낮추면
    #   이번엔 중립 자세에서 로봇이 아래로 꺾여 내려간다.
    #   즉 하나를 고치면 다른 하나가 깨지는 구조였다.
    #
    #   그래서 J5 의 3점 보간과 같은 방식으로 위/아래 게인을 분리한다.
    #     norm_up >= 중립  →  중립 ~ MAP_Z_MAX     (위로 드는 동작)
    #     norm_up <  중립  →  MAP_Z_MIN ~ 중립     (아래로 내리는 동작)
    #
    # [아래쪽 범위를 왜 실측하지 않는가]
    #   캘리브레이션 3단계는 "위로 들어 유지"만 있고 아래로 내린 자세를 재지
    #   않는다. 단계를 늘리면 GUI 까지 손봐야 하므로, 우선은 위쪽 범위에
    #   배율을 곱해 추정한다. 사람은 팔을 위로 드는 범위가 아래로 내리는
    #   범위보다 넓으므로 1.0 보다 작은 값이 맞다.
    #   아래로 잘 안 내려가면 이 값을 키우고(0.8), 너무 예민하면 줄인다(0.4).
    MAP_Z_MIN            =  60.0   # 최대로 내렸을 때 (WS_Z_MIN_MM=40 보다 여유)
    MAP_DOWN_SPAN_SCALE  =   3.5   # 아래쪽 사람 가동범위 = up_span * 이 값
    # ── 좌우: J1 각도로 직접 매핑 ────────────────────────────────────────
    # [왜 y 거리가 아니라 각도인가]
    #   예전에는 raw_y 를 mm 로 더했다. 그러면 옆으로 갈수록 반경
    #   sqrt(x^2+y^2) 가 같이 커져서, J2축 기준 거리 D 가 WS_D_MAX(258)를
    #   넘어 구각 클램프에 걸렸다. 실측에서 J1 이 ±36° 에서 멈춘 원인이다.
    #   (관절 하드리밋은 ±100° 인데 3분의 1만 쓰고 있었다)
    #
    #   각도로 매핑하면 반경이 좌우 이동과 무관하게 유지된다. 로봇이
    #   회전만 하므로 D 가 변하지 않고, 구각 제한에 걸릴 이유가 없어진다.
    #   사람이 팔을 옆으로 벌리는 동작도 실제로는 어깨 축 회전이므로
    #   물리적으로도 이쪽이 맞다.
    #
    # [좌우 1:1 대칭]
    #   게인과 상한을 모두 부호 대칭으로 두었으므로 J1 이 양쪽으로 정확히
    #   같은 각도까지 돈다. 한쪽만 더 도는 일이 구조적으로 생기지 않는다.
    #
    # 실측 norm_lat 은 팔을 최대로 벌렸을 때 약 ±0.55 였다.
    #   게인 110 → ±60°,  게인 140 → ±77°
    # 예민하면 줄이고 답답하면 키운다.
    MAP_J1_GAIN_DEG = 110.0
    MAP_J1_MAX_DEG  =  70.0   # 대칭 상한. WS_YAW_MAX_DEG(85) 안쪽으로 둔다

    # 캘리브레이션 범위가 이보다 좁으면 자세를 잘못 취한 것으로 보고 실패 처리
    CALIB_MIN_RANGE_NORM = 0.18

    # ══════════════════════════════════════════════════════
    #  4) 툴 피치 (J5)
    #
    #  ★ URDF 순기구학으로 검증한 사실 ★
    #      툴이 향하는 각도 θ = J2 + J3 + J5   (도)
    #        θ =   0° → 그리퍼가 수직 아래를 향함
    #        θ = -90° → 그리퍼가 정면 수평을 향함
    #
    #  v1은 손목 각도를 J5에 "그대로" 넣었기 때문에, 팔을 움직여서 J2/J3가
    #  바뀌면 손목을 가만히 둬도 그리퍼 기울기가 같이 변했음(계속 손목으로
    #  보정해줘야 했던 원인). v2는 손목 각도로 "월드 기준 절대 각도 θ"를
    #  정하고, J5 = θ - (J2 + J3) 로 보상해서 팔 자세와 무관하게 유지함.
    # ══════════════════════════════════════════════════════
    # 기존 6단계 캘리브레이션의 손목 3점이 이제 "목표 툴 피치"로 해석됨.
    #
    # [방향 검증 노트 — 검수에서 반대로 들어간 것을 수정함]
    #   사람 손목의 물리적 대응:
    #     안쪽으로 꺾음(굴곡)  = 손끝이 아래로 향함  → 그리퍼 수직 아래 (θ=0)
    #     뒤로 꺾음(신전)      = 손끝이 위/앞으로 향함 → 그리퍼 정면 수평 (θ=-90)
    #   v1의 학습된 방향과도 일치함: v1은 뒤로 꺾을수록 J5가 가장 음수(-100)
    #   방향이었고, θ=J2+J3+J5 이므로 θ도 가장 음수(수평 쪽)였음.
    #   혹시 실기에서 그래도 반대로 느껴지면 아래 두 값을 서로 맞바꾸지 말고
    #   J5_BACK_BEND_SIGN을 -1.0으로 뒤집을 것(입력 부호만 반전, 캘리브레이션과 일관).
    TOOL_PITCH_BACK     = -90.0   # 손목을 뒤로 꺾음   → 정면 수평 (선반 옆에서 집기)
    TOOL_PITCH_STRAIGHT = -45.0   # 손목 일자          → 45° 사선
    TOOL_PITCH_FRONT    =   0.0   # 손목을 안쪽으로    → 수직 아래 (책상 위 집기)

    # True면 위 보상 사용, False면 v1처럼 J5에 직접 매핑(비교/폴백용)
    TOOL_PITCH_COMPENSATION = True

    # ── J5 손목 3점 보간 입력 기준각 (사람 손목) ─────────────────────────────
    # v1과 완전히 동일한 의미/동일한 캘리브레이션 절차. 출력만 관절각 → 툴피치로 바뀜.
    #
    # ★★ 반드시 손목 캘리브레이션(4~6단계)을 다시 수행할 것 ★★
    #   아래 세 값은 캘리브레이션에서 자동으로 덮어써지는 '자리 표시자'다.
    #   J5_HAND_VECTOR 를 palm3d 로 바꾸면서 각도의 정의 자체가 달라졌으므로,
    #   이전 캘리브레이션 값은 의미가 없다. 재캘리브레이션 없이 쓰면 손목을
    #   꺾는 정도와 그리퍼 기울기의 대응이 어긋난다.
    #
    #   palm3d 기준으로 각 값이 뜻하는 것:
    #     STRAIGHT  손이 전완과 일직선 → 사잇각이 0에 가까움
    #     BACK      손등 쪽으로 꺾음(신전) → 한쪽 부호로 커짐
    #     FRONT     손바닥 쪽으로 꺾음(굴곡) → 반대 부호로 커짐
    #   아래 ±35°는 캘리브레이션 전 임시값일 뿐 실측값이 아니다.
    WRIST_ANGLE_STRAIGHT =   0.0
    WRIST_ANGLE_BACK     =  35.0
    WRIST_ANGLE_FRONT    = -35.0
    J5_BACK_BEND_SIGN    =   1.0   # 뒤/앞이 반대로 반응하면 -1.0

    # ── J6 (툴 롤): 이번 버전에서는 비활성 ────────────────────────────────
    # 손등 너클 라인(검지 MCP↔새끼 MCP)의 롤 성분으로 계산하는 코드는 다 들어있고,
    # J1~J3 IK가 안정된 것을 확인한 뒤 True로 바꾸면 됨.
    # 한 번에 다 켜면 문제가 생겼을 때 원인 분리가 안 되므로 기본 False.
    ENABLE_J6 = False
    J6_SIGN = 1.0
    J6_MAX_DEG = 90.0
    MIN_KNUCKLE_VEC = 0.03   # 이보다 짧으면 foreshortening → 직전 값 홀드

    # ══════════════════════════════════════════════════════
    #  5) 안정성 가드
    # ══════════════════════════════════════════════════════
    MIN_UPPERARM_LEN_M   = 0.05   # 어깨→팔꿈치 3D 최소 길이 (m)
    MIN_FOREARM_LEN_J3_M = 0.05   # 팔꿈치→손목 3D 최소 길이 (m)
    MIN_ARM_LEN_M        = 0.15   # 팔 전체 길이 최소값 (정규화 분모 보호)
    MIN_FOREARM_VEC_J5   = 0.06   # J5용 2D 전완 벡터 최소 길이 (정규화 좌표)
    MIN_HAND_VEC_J5      = 0.02   # J5용 2D 손 벡터 최소 길이 (midtip 방식 전용)

    # ── J5 손 벡터 정의 ──────────────────────────────────────────────────
    # 'palm3d' : 3D 손바닥 장축 × 3D 전완의 사잇각.  ← 권장 (현재 기본값)
    #            pose_world_landmarks / multi_hand_world_landmarks(둘 다 미터
    #            단위, 카메라 정렬)를 쓴다. 아래 "왜 3D인가" 참고.
    # 'palm'   : 손목(0) → 너클선 중점((5+17)/2)의 2D 각도. 구 방식.
    #            이 세 점은 모두 손등 위에 있어 주먹을 쥐어도 가려지지 않지만,
    #            2D라서 손목 꺾임이 아닌 성분이 섞인다(아래 참고).
    # 'midtip' : 손목(0) → 중지 끝(12)의 2D 각도. 최초 방식.
    #            주먹을 쥐거나 손목을 꺾으면 중지 끝이 손바닥 안으로 말려
    #            벡터가 0에 수렴하고, MIN_HAND_VEC_J5 가드에 걸려 계산이
    #            중단된다(직전 값 홀드) → 손목을 꺾어도 J5가 반응하지 않음.
    #
    # ── 왜 3D인가 ────────────────────────────────────────────────────────
    # 화면상 손바닥 벡터의 방향은 두 가지에 동시에 반응한다.
    #   (1) 손목이 실제로 꺾일 때
    #   (2) 팔 전체 자세가 바뀌어 손이 회전할 때
    # 2D 투영에서는 이 둘이 구분되지 않는다. 'palm' 방식이 어깨→손목을
    # 기준선으로 삼아 (2)를 빼려 했지만, 기준선 자체가 2D 투영이라 완전히
    # 제거되지 않았다. 가동 범위가 실제 손목의 물리적 한계(±90°)를 훨씬
    # 넘겨 380~476°로 부풀고, ±360° 이어붙이기(unwrap) 방어 코드가 필요했던
    # 근본 원인이 이것이다.
    #
    # 3D에서는 팔 전체가 회전해도 전완과 손이 함께 회전하므로 사잇각이
    # 변하지 않는다. 합성 검증에서 팔을 임의 축으로 임의 각만큼 돌려도
    # 굴곡각이 소수점 둘째 자리까지 불변임을 확인했다.
    # 덤으로 ±180° 경계 문제가 사라져 unwrap 자체가 불필요해진다.
    #
    # ── 실측 (session2_C, 06_손목 동작 10개 구간) ────────────────────────
    #                        계산가능률   프레임간 점프   가동범위   정지std
    #   midtip                 48.3%        35.2°        219.7°    118.0°
    #   palm                   49.5%        30.9°        278.5°    117.3°
    #   palm3d (부호 없음)      66.0%        32.2°         56.2°     11.8°
    #   palm3d (부호 포함)      66.0%        38.4°        109.8°     19.5°
    #
    # 계산가능률이 오르고(J5가 얼어붙는 구간 감소), 정지 구간 잡음이
    # 6분의 1로 줄었다(손을 가만히 뒀을 때 그리퍼가 떠는 문제).
    J5_HAND_VECTOR = 'palm3d'
    MIN_PALM_VEC_J5 = 0.02   # 너클선 중점 벡터 최소 길이 (2D 'palm' 방식 전용)

    # ── palm3d 전용 ──────────────────────────────────────────────────────
    # world 랜드마크는 미터 단위다. 손바닥 폭·손목→너클 거리가 대략 0.07~0.10m
    # 이므로 0.015m는 "사실상 한 점으로 뭉개진 경우"만 걸러내는 값이다.
    MIN_VEC3_J5 = 0.015

    # 부호 판정 상태 기계.
    #   크기는 E(전완과 손바닥 장축의 사잇각, arccos이라 0~180°로 부호 없음),
    #   부호는 F(너클선을 회전축으로 삼아 투영해 잰 부호 있는 각)에서 가져온다.
    #   F는 회전축(너클선)이 MediaPipe 추정값이라 흔들려 크기가 불안정하지만,
    #   부호는 이산값이라 훨씬 둔감하다. 각각의 강점만 취하는 구성이다.
    #
    #   두 겹의 방어가 필요하다:
    #     불감대 — |F|가 이보다 작으면 부호가 사실상 무작위이므로 갱신하지
    #              않고 직전 부호를 유지한다.
    #     확정   — 반대 부호가 이만큼 연속으로 관측될 때만 실제로 바꾼다.
    #   불감대만으로는 부족하다. 합성 실험에서 잡음 25° 조건에 불감대 8°만
    #   쓰면 실제 방향 전환 1회에 33회 뒤집혔다.
    #
    #   실측 튜닝 (session2_C):
    #     불감대  확정   점프p95   되돌아간 비율
    #       8      3      51.7°       32%
    #      15      3      51.7°       29%
    #      15      5      38.4°       14%   ← 채택
    #   확정 5는 0.25초 지연을 뜻하지만, 실제 손목은 그보다 빠르게 방향을
    #   왕복하지 않으므로 지연이 비용이 아니라 순수 이득이었다.
    #   (손목 회전 구간 점프가 95.2° → 36.9°로 떨어짐)
    J5_SIGN_DEADBAND_DEG   = 15.0
    J5_SIGN_CONFIRM_FRAMES = 5

    # 왼손과 오른손은 거울상이므로, 같은 축을 기준으로 잰 부호 있는 각의
    # 부호가 서로 반대로 나온다. 위 실측은 왼팔(거울 보정 후 pose 인덱스
    # 14/16)에서 검증했다. 오른팔에서 앞/뒤가 반대로 반응하면 이 표의
    # 'right' 값을 +1.0으로 바꾼다.
    # [주의] 어느 쪽이든 팔을 바꾸면 손목 캘리브레이션을 다시 해야 한다.
    J5_FLEX_ARM_SIGN = {'left': 1.0, 'right': -1.0}

    # ── J5 기준축 (palm 방식 전용) ───────────────────────────────────────
    # palm 손 벡터의 각도를 무엇에 대해 재는가.
    #   'screen'   화면 수직축. 손목을 꺾지 않아도 팔 자세가 바뀌면 손등 방향이
    #              화면에서 크게 회전해 그것이 각도에 그대로 섞인다.
    #              (실측: '높이 이동' 구간 가동 범위 377°, '자유 조작' 342°)
    #   'hybrid'   어깨→손목 벡터를 기준으로 삼아 팔 자세 성분을 상쇄한다. ← 권장
    #              단, 팔을 카메라 쪽으로 뻗으면 이 벡터도 화면에서 짧아지므로,
    #              임계 미만이면 마지막 유효 기준 방향을 계속 사용한다.
    #              (화면 수직축으로 폴백하면 전환 지점에서 값이 불연속이 됨)
    #
    # 실측 비교 (session1_C, 10개 구간):
    #                 계산가능률   가동범위   정지std   범위과대 구간
    #   screen           88.4%     158.5°     4.5°       3/10
    #   어깨→손목만       81.2%     112.8°     8.6°       2/10
    #   hybrid           88.4%     112.2°     8.6°       2/10
    J5_REF_AXIS = 'hybrid'
    MIN_REF_VEC_J5 = 0.05    # 기준 벡터(어깨→손목) 최소 길이

    # ── unwrap (2D 'palm' 방식 전용) ─────────────────────────────────────
    # [주의] palm3d에서는 쓰이지 않는다. 3D 사잇각은 arccos이라 0~180°로
    #        닫혀 있어 ±180° 경계를 넘는 일 자체가 없기 때문이다.
    #        아래 상수와 j5_unwrap_* 상태는 J5_HAND_VECTOR='palm'으로
    #        되돌릴 때를 위해 남겨둔 것이다.
    #
    # palm 방식은 기준축에 대한 각도라 손목이 크게 돌면 ±180° 경계를 넘으며
    # 값이 360° 점프한다. 직전 각도와 이어붙여(unwrap) 연속적으로 만든다.
    J5_UNWRAP_MAX_JUMP = 180.0
    # [중요] 손이 잠깐 안 잡혔다가 다시 잡히면 그 사이 손목이 얼마나 돌았는지
    #   알 수 없다. 그런데도 직전 유효 각도와 이어붙이면 "한 바퀴 돌았다"고
    #   오판해 ±360°를 더하고, 그것이 누적되어 가동 범위가 부풀려진다.
    J5_UNWRAP_MAX_GAP_SEC = 0.15   # 3프레임(20Hz 기준)
    J5_UNWRAP_MAX_TURNS = 1        # 실제 손목은 한 바퀴 이상 돌지 않는다
    J5_STALE_GAP_SEC     = 0.3    # 이 시간 이상 끊겼다 재개되면 J5 필터 리셋

    # ══════════════════════════════════════════════════════
    #  GRU 랜드마크 보정
    # ══════════════════════════════════════════════════════
    #  MediaPipe 팔꿈치·손목 world 좌표의 관측 오차를 학습으로 보정한다.
    #  NLF(Neural Localizer Fields)를 교사로 삼아 오프라인 학습한 GRU를
    #  NumPy 순전파로 실행한다(프레임당 약 30µs, 20Hz 주기의 0.06%).
    #
    #  4인 데이터 leave-one-session-out 교차검증 결과(평균):
    #    손목 위치 오차 135.8 → 79.5mm (-41%)
    #    전완 방향     20.2° → 13.0°  (-38%)   ← J2/J3 관절각에 직결
    #    z축(깊이)     102.3 → 65.9mm (-37%)
    #    프레임간 떨림  17.3 → 11.8mm (-32%)
    #
    #  특히 z축 개선이 중요하다. MediaPipe의 z는 팔을 카메라 쪽으로 뻗을 때
    #  포화되어(중립 0.660 → 전방 0.726, 차이 0.066) 캘리브레이션 2단계가
    #  통과하지 못했는데, 보정 후에는 0.540 → 0.971(차이 0.430)로 정상화된다.
    ENABLE_GRU_CORRECTION = True
    GRU_MODEL_FILENAME = 'gru_4sess_val-s2_h64s40.npz'   # ai_node_new.py 와 같은 폴더
    #  hidden state가 0에서 출발한 직후 몇 프레임은 출력을 믿을 수 없다.
    #  이 구간에는 보정을 적용하지 않고 원본을 그대로 통과시킨다
    #  (리셋/호밍 직후 로봇이 튀는 것을 막는 안전장치).
    GRU_WARMUP_FRAMES = 10
    #  [진단] 보정 전후 값을 주기적으로 로그에 찍는다. 확인이 끝나면 False.
    GRU_DEBUG_LOG = True
    GRU_DEBUG_PERIOD_SEC = 1.0

    # 손/팔 추적이 끊겼을 때: 이 시간 동안은 마지막 유효 목표를 유지하고,
    # 그 이후에는 아예 명령 발행을 중단함(0을 보내면 원점으로 급이동하므로 절대 금지).
    TRACK_LOST_HOLD_SEC = 0.3

    # ── 직교공간 속도 제한 ────────────────────────────────────────────────
    # [주의] v1에서 제거했던 MAX_DELTA_J*(관절 변화량 제한)의 부활이 아님.
    # 그건 "관절 공간"에서 잘라내서 큰 동작을 계단식으로 쪼개는 부작용이 있었음.
    # 이건 "목표점의 이동 속도"를 mm/s로 제한하는 것이라, 궤적이 쪼개지지 않고
    # 매끄럽게 유지되면서 순간적인 급이동만 막아줌.
    MAX_CART_SPEED_MM_S = 250.0

    # ── 카테시안 One Euro Filter ─────────────────────────────────────────
    # [설계 변경] v1은 관절각(deg)에 필터를 걸었는데, 관절각은 비선형 매핑의
    # 결과라 특이점 근처에서 필터가 오히려 왜곡을 키움. v2는 선형 공간인
    # 카테시안 좌표(mm)에 걸어서 필터 특성이 예측 가능하게 만듦.
    # 단위가 deg → mm 로 바뀌었으므로 beta 값이 v1보다 훨씬 작아야 함
    # (속도가 mm/s 단위라 값 자체가 크기 때문).
    CART_MIN_CUTOFF = 1.2
    CART_BETA       = 0.015
    CART_D_CUTOFF   = 1.0

    # ── 카테시안 중앙값 사전필터 창 크기 ─────────────────────────────────
    # [측정 결과 — 지연의 최대 단일 원인]
    # 창=3인 중앙값은 "단조롭게 움직이는 신호"에 대해 정확히 1프레임 지연이다.
    # 정렬한 3개 중 가운데 = 항상 직전 프레임 값이기 때문. 20Hz면 그대로 50ms.
    # (원래 주석의 "지연을 거의 없앰"은 튀는 값 기준 설명이고, 실제 추종
    #  지연에는 해당하지 않는다.)
    #
    #   손 100mm/s 기준 추종 지연 실측(시뮬레이션):
    #     창=3, cutoff 1.2 →  91ms   ← 현재
    #     창=1, cutoff 1.2 →  41ms
    #     창=3, cutoff 3.0 →  80ms   (창을 두면 cutoff를 올려도 11ms만 준다)
    #     창=1, cutoff 3.0 →  30ms
    #
    # 즉 cutoff 튜닝보다 이 창을 없애는 쪽이 훨씬 크다. 다만 이 필터는
    # MediaPipe가 한두 프레임 튀는 것을 막는 역할도 하므로, 1로 내린 뒤
    # 떨림이 돌아오면 3으로 되돌릴 것. GRU 보정이 이 필터보다 앞단에서
    # 이미 랜드마크 튐을 잡고 있으므로 역할이 상당 부분 중복된다.
    #
    #   1 = 사실상 비활성(들어온 값을 그대로 통과)
    #   3 = 기존 동작
    CART_MEDIAN_WINDOW = 3

    # J5는 IK와 무관한 독립 축이라 v1 파라미터를 그대로 유지
    D_CUTOFF = 1.0
    J5_MIN_CUTOFF, J5_BETA = 2.0, 0.6
    J6_MIN_CUTOFF, J6_BETA = 2.0, 0.6

    # ── 데드밴드 ─────────────────────────────────────────────────────────
    # [설계 변경] v1의 관절별 데드밴드(DEADBAND_J1/J2/J3)는 제거함.
    # 특이점 근처에서는 손끝이 1mm 움직여도 관절이 20° 변할 수 있어서,
    # 관절 기준 데드밴드는 "미세한 손 떨림"과 "큰 관절 변화"를 구분하지 못함.
    # 대신 목표점(카테시안) 기준으로 판정하면 물리적 의미가 일정하게 유지됨.
    DEADBAND_CART_MM = 4.0
    DEADBAND_J5      = 1.5
    DEADBAND_J6      = 2.0
    DEADBAND_GRIPPER = 1.0

    # 핀치 판정 비율 (v1 그대로 유지)
    PINCH_CLOSE_RATIO = 0.38
    PINCH_OPEN_RATIO  = 0.70

    VIS_THRESHOLD = 0.5

    # ── 발행 주기 ────────────────────────────────────────────────────────
    # v1은 0.5초(2Hz)였음. 이건 노이즈 대책으로 낮춰뒀던 것인데, v2는
    # 사전 클램프로 튐의 원인 자체를 없앴고 카테시안 필터를 쓰기 때문에
    # 주기를 올려도 안전함. 2Hz에서는 로봇이 항상 반 박자 늦은 자세를 향해
    # 큰 걸음으로 움직여서 "내 팔과 안 맞는다"는 체감의 큰 원인이었음.
    # robot_node는 최신값 덮어쓰기 + ok 게이팅 구조라 시리얼이 넘치지 않음.
    # 혹시 불안정하면 0.2(5Hz)로 올려서 확인할 것.
    #
    # [0.1 → 0.05 변경 근거 — 시뮬레이션으로 확인함]
    # robot_node._try_send_pending()은 pending_joint_target 이라는 '단일 슬롯'에
    # 덮어쓰기만 하고, busy(=직전 명령의 'ok' 대기) 동안은 아무것도 안 보낸다.
    # 따라서 실제 시리얼 전송 횟수는 발행률이 아니라 GRBL 완료 속도가 결정한다.
    # 발행률을 올리면 '전송되는 값이 더 신선해질' 뿐 전송 횟수는 그대로다:
    #
    #   GRBL 1동작   발행률   실제 시리얼    전송된 값의 평균 노후도
    #     150ms      10Hz     6.1 회/초          39.2ms
    #     150ms      20Hz     6.1 회/초          15.0ms   ← 24ms 개선, 전송량 동일
    #     250ms      10Hz     3.8 회/초          44.7ms
    #     250ms      20Hz     3.8 회/초          20.7ms
    #     400ms      10Hz     2.4 회/초          38.8ms
    #     400ms      20Hz     2.4 회/초          21.0ms
    #
    # [주의] 이 이득은 손이 빠르게 움직일 때만 온전히 나온다. 저속에서는
    # DEADBAND_CART_MM(4.0mm)이 먼저 걸려서 발행 자체가 막히기 때문:
    #   손 10mm/s → 실효 발행간격 400ms,  20mm/s → 200ms  (발행률과 무관)
    # 미세 조작이 굼뜨게 느껴지면 PUB_INTERVAL이 아니라 DEADBAND_CART_MM을
    # 봐야 한다. 다만 데드밴드를 낮추면 정지 중 떨림이 그대로 로봇에 나가므로
    # 반드시 실기로 확인하고 내릴 것.
    PUB_INTERVAL = 0.05   # 20Hz (process_frame 타이머와 동일 — 게이트에서 버려지는 프레임 없음)

    # ── 관절 소프트 리밋 (robot_node.JOINT_HARD_LIMITS와 동일) ───────────
    # 상위에서 미리 막아서, 하드 리밋에 걸려 "6축 명령 전체가 무시되는" 일을 방지.
    JOINT_SOFT_LIMITS = [
        (-100.0, 100.0),  # J1
        ( -60.0,  90.0),  # J2
        (-180.0,  50.0),  # J3
        (-180.0, 180.0),  # J4
        (-180.0,  40.0),  # J5
        (-180.0, 180.0),  # J6
    ]

    # ══════════════════════════════════════════════════════
    #  6) 호밍 안전 (★ 절대 타협 금지 영역 ★)
    #
    #  과거 두 차례 발생한 사고 패턴:
    #    호밍 시작~완료 사이에 큐/버퍼에 낡은 목표값이 남아 있다가
    #    호밍 완료 직후 그대로 재생되어 로봇이 급이동.
    #
    #  v2에서 이 노드가 축적하는 상태가 v1보다 훨씬 많아졌으므로
    #  (목표점, 카테시안 필터, 클러치 원점, 팔길이 버퍼, 툴피치 홀드,
    #   직전 IK 해) 아래 3중 방어를 적용함. 자세한 것은
    #  _hard_reset_control_state() / homing_status_callback() 주석 참고.
    # ══════════════════════════════════════════════════════
    HOMING_SETTLE_SEC = 1.5   # 호밍 완료 후 이 시간 동안은 계산만 하고 발행 금지

    # ── 손 제스처 기반 AI ON/OFF 토글 (v1 그대로 유지) ─────────────────────
    GESTURE_START_FIST_HOLD_SEC = 0.15
    GESTURE_OPEN_HOLD_SEC       = 0.10
    GESTURE_FINAL_FIST_HOLD_SEC = 0.15
    GESTURE_MAX_TRANSITION_SEC  = 1.20
    GESTURE_COOLDOWN_SEC        = 2.00
    AI_ENABLE_SETTLE_SEC        = 0.80

    GESTURE_MIN_PALM_WIDTH_RATIO = 0.55
    GESTURE_MIN_FIST_WIDTH_RATIO = 0.35
    GESTURE_OPEN_MIN_FINGERS     = 4
    GESTURE_CLOSED_MAX_FINGERS   = 1

    # ── 캘리브레이션 ─────────────────────────────────────────────────────────
    # [v2 변경] 1~3단계의 의미가 바뀜.
    #   v1: 어깨각/팔꿈치각의 최소~최대를 재서 관절각 선형보간의 입력 범위로 사용
    #   v2: 관절각을 안 쓰므로 그 개념 자체가 사라짐. 대신 "이 사람이 편하게
    #       팔을 움직이는 실제 범위"를 재서 로봇 작업공간에 사상함.
    #       → 사람마다 다른 체격뿐 아니라 "무리하지 않는 편안한 범위"까지
    #         반영되므로, 팔을 억지로 끝까지 뻗지 않아도 로봇 작업공간 전체를 씀.
    #   4~6단계(손목)는 v1과 완전히 동일한 절차. 출력 해석만 툴피치로 바뀜.
    CALIB_REQUIRED_SAMPLES = 40   # 20Hz 처리 기준 약 2초 분량
    CALIB_MIN_RANGE_DEG = 10.0    # 손목 단계용(도 단위)
    CALIB_STATUS_PUB_INTERVAL = 0.2
    CALIB_PREP_DELAY_SEC = 3.0

    CALIB_STAGES = ['neutral', 'reach_forward', 'reach_up',
                    'wrist_straight', 'wrist_back', 'wrist_front']
    CALIB_INSTRUCTIONS = {
        'neutral':        "1/6 단계: 팔꿈치를 편하게 굽혀 몸 앞쪽에 손을 두는 '기본 자세'를 유지하세요. "
                          "(이 위치가 로봇의 기준 위치가 됩니다)",
        'reach_forward':  "2/6 단계: 팔을 카메라 쪽으로 최대한 앞으로 뻗은 자세를 유지하세요.",
        'reach_up':       "3/6 단계: 팔을 최대한 위로 들어올린 자세를 유지하세요.",
        # [중요] 손목 굴곡은 2D 화면 투영이라 팔의 위치에 따라 카메라에 보이는
        # 민감도가 달라짐 — 실제 조작 시의 자세(보통 몸 앞으로 내린 자세)와 다른
        # 자세에서 캘리브레이션하면 정작 그 자세에서 반응이 둔해짐.
        'wrist_straight': "4/6 단계: 팔을 몸 앞쪽에 편하게 둔 자세를 유지한 채, 손목만 전완과 일직선이 되도록 곧게 펴세요.",
        'wrist_back':     "5/6 단계: 팔은 그대로, 손등을 팔꿈치 쪽으로 최대한 꺾으세요. (그리퍼가 정면 수평을 향하게 됩니다)",
        'wrist_front':    "6/6 단계: 팔은 그대로, 손목만 안쪽으로 최대한 꺾으세요. (그리퍼가 수직 아래를 향하게 됩니다)",
    }

    # ── 미니멀 카메라 UI 스타일 (v1 그대로) ───────────────────────────────
    UI_PANEL_BG = (28, 28, 30)
    UI_PANEL_BORDER = (255, 255, 255)
    UI_TEXT_PRIMARY = (255, 255, 255)
    UI_TEXT_SECONDARY = (214, 214, 219)
    UI_ACCENT_BLUE = (255, 170, 90)
    UI_ACCENT_GREEN = (110, 214, 108)
    UI_ACCENT_RED = (110, 110, 255)
    UI_PANEL_ALPHA = 0.58

    # ══════════════════════════════════════════════════════
    def __init__(self):
        super().__init__('mirobot_ai_node')
        self.bridge = CvBridge()

        self.image_pub     = self.create_publisher(ROSImage,          '/mirobot/camera_image',       10)
        self.ai_joints_pub = self.create_publisher(Float32MultiArray, '/mirobot/ai_joint_commands',  10)
        self.gripper_pub   = self.create_publisher(Float32,           '/mirobot/ai_gripper_command', 10)

        # GUI 버튼뿐 아니라 카메라 손 제스처도 같은 토픽을 사용하도록
        # 이 노드가 /mirobot/ai_enable을 발행하고 동시에 구독함.
        self.ai_enable_pub = self.create_publisher(Bool, '/mirobot/ai_enable', 10)
        self.ai_enable_sub = self.create_subscription(
            Bool, '/mirobot/ai_enable', self.ai_enable_callback, 10
        )
        self.ai_enabled = False

        # 손 제스처 상태 머신: 시작 주먹 → 펼침 → 마지막 주먹 (v1 그대로)
        self.gesture_phase = 'wait_start_fist'
        self.gesture_pose_since = 0.0
        self.gesture_step_confirmed_at = 0.0
        self.gesture_cooldown_until = 0.0
        self.gesture_suspend_commands = False
        self.gesture_status_text = 'Gesture: START WITH FIST'
        self.gesture_debug_text = ''
        self.ai_publish_block_until = 0.0

        # GUI에서 선택한 제어 팔('left' or 'right'). 기본값: 왼팔.
        # [v2 변경] v1은 J1만 "반대쪽 팔"에서 뽑아 썼는데(한 팔로 묶으면 다른
        # 관절과 간섭해서 제어가 잘 안 됐던 임시 대응), v2는 J1이 IK에서
        # 손목 위치의 방위각으로 자동으로 나오므로 그럴 필요가 없음.
        # → 이제 제어팔 한쪽만 사용하고, 반대쪽 팔은 완전히 자유임.
        #   (나중에 모바일 주행 모드용으로 비워둔 상태)
        self.control_arm = 'left'
        self.create_subscription(
            String, '/mirobot/control_arm', self.control_arm_callback, 10
        )

        # ── 호밍 상태 구독 (★ 안전 핵심 ★) ────────────────────────────────
        self.create_subscription(
            Bool, '/mirobot/homing_status', self.homing_status_callback, 10
        )
        self.homing_active = False        # 호밍 진행 중이면 True → 계산·발행 전면 중단
        self.homing_lockout_until = 0.0   # 이 시각 전까지는 계산은 하되 발행 금지

        # ── 캘리브레이션 토픽 ────────────────────────────────────────────────
        self.calibration_status_pub = self.create_publisher(
            String, '/mirobot/calibration_status', 10)
        self.create_subscription(
            String, '/mirobot/calibrate_cmd', self.calibrate_cmd_callback, 10)

        # 캘리브레이션 결과값을 인스턴스 속성으로 복사
        # (클래스 기본값은 "초기화 시 되돌릴 원본"으로 보존)
        self.CAL_ARM_LEN = MirobotAiNode.CAL_ARM_LEN
        self.CAL_FWD_NEUTRAL = MirobotAiNode.CAL_FWD_NEUTRAL
        self.CAL_FWD_MAX     = MirobotAiNode.CAL_FWD_MAX
        self.CAL_UP_NEUTRAL  = MirobotAiNode.CAL_UP_NEUTRAL
        self.CAL_UP_MAX      = MirobotAiNode.CAL_UP_MAX
        self.CAL_LAT_NEUTRAL = MirobotAiNode.CAL_LAT_NEUTRAL
        self.WRIST_ANGLE_STRAIGHT = MirobotAiNode.WRIST_ANGLE_STRAIGHT
        self.WRIST_ANGLE_BACK     = MirobotAiNode.WRIST_ANGLE_BACK
        self.WRIST_ANGLE_FRONT    = MirobotAiNode.WRIST_ANGLE_FRONT

        self.calib_mode = None
        self.calib_session_active = False
        self.calib_next_stage_idx = 0
        self.calib_samples_pos   = []   # (fwd, lat, up) 정규화 좌표 샘플
        self.calib_samples_arm   = []   # 팔 길이 샘플(m) — 중립 단계에서만 사용
        self.calib_samples_wrist = []   # effective_bend(손목) 샘플
        self.calib_prep_until = 0.0
        self.last_calib_status_pub_time = 0.0

        # ── [학습 데이터 녹화] ──────────────────────────────────────────────
        # NLF_RECORD_DIR가 설정돼 있으면 그 아래 "시작시각" 하위 폴더에
        # MediaPipe 입력 프레임을 저장. 저장은 별도 스레드(큐)로 처리되며
        # 이미지·시각 정보만 다룬다 — 제어 상태(조인트 목표 등)와 완전 분리.
        self.recorder = SessionRecorder.from_env(self.get_logger())

        # 웹캠 초기화
        #
        # [녹화 모드] 촬영은 브라우저(record.html)가 올려주는 프레임만 사용한다.
        # 그런데 노트북 한 대로 촬영하면 이 노드가 /dev/video0 을 먼저 점유해서
        # 브라우저의 getUserMedia()가 NotAllowedError 로 실패한다(카메라는 보통
        # 한 프로세스만 열 수 있음). 그래서 녹화 모드에서는 로컬 웹캠을 아예
        # 열지 않고, 원격 프레임만 받는다.
        # (녹화가 아닐 때는 기존 동작 그대로 — 로컬 웹캠 폴백 유지)
        if self.recorder.enabled:
            self.cap = cv2.VideoCapture()   # 열지 않은 빈 객체 (isOpened() == False)
            self.get_logger().info(
                "[녹화] 로컬 웹캠을 열지 않습니다 — 촬영 GUI(브라우저)의 카메라만 사용합니다.")
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                self.get_logger().error("웹캠에 접근하지 못했습니다.")

        # 녹화 중 추론 솎아내기 주기 (N프레임에 1회만 Pose/Hands 실행).
        # 3이면 20Hz 저장 / 약 6.7Hz 인식 — 화면 확인과 프레임 체크에는 충분하고,
        # CPU 여유가 생겨 업로드 수신이 밀리지 않는다.
        self.REC_INFER_EVERY = int(os.environ.get('REC_INFER_EVERY', '3'))

        # ── [촬영 준비] 프레임 체크용 포즈 상태 (촬영 GUI가 폴링) ──────────
        # 제어 상태가 아니라 "화면 안에 팔이 들어오는지"만 담는 진단용 값이다.
        # 조인트 목표와는 무관하며, 호밍 안전 상태 레지스트리와도 분리된다.
        self.pose_status_latest = {'detected': False, 't': 0.0}
        self.pose_status_lock = threading.Lock()

        # ── 웹 GUI용 MJPEG 스트리밍 서버 (v1 그대로 — 카메라 파이프라인 동결) ──
        self.latest_jpeg = None
        self.jpeg_lock = threading.Lock()

        # ── 원격 카메라(GUI 접속 기기) 프레임 수신용 상태 (v1 그대로) ────────
        self.remote_frame = None
        self.remote_frame_time = 0.0
        self.remote_frame_lock = threading.Lock()
        self.REMOTE_FRAME_TIMEOUT = 2.0
        self.source_aspect = None

        self._start_mjpeg_server()
        self._start_upload_ws_server()

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose    = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            max_num_hands=2
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.ARM_CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16)]

        # ══════════════════════════════════════════════════════════════════
        #  [v2] 카테시안 파이프라인 상태
        #  필터를 관절각이 아니라 목표점(mm)에 거는 이유는 상수 섹션 주석 참고.
        # ══════════════════════════════════════════════════════════════════
        self.f_cx = OneEuroFilter(self.CART_MIN_CUTOFF, self.CART_BETA, self.CART_D_CUTOFF)
        self.f_cy = OneEuroFilter(self.CART_MIN_CUTOFF, self.CART_BETA, self.CART_D_CUTOFF)
        self.f_cz = OneEuroFilter(self.CART_MIN_CUTOFF, self.CART_BETA, self.CART_D_CUTOFF)
        self.m_cx = MedianPrefilter(window=self.CART_MEDIAN_WINDOW)
        self.m_cy = MedianPrefilter(window=self.CART_MEDIAN_WINDOW)
        self.m_cz = MedianPrefilter(window=self.CART_MEDIAN_WINDOW)

        # J5 / J6은 IK와 무관한 독립 축이므로 v1 파이프라인 유지
        self.f_j5 = OneEuroFilter(self.J5_MIN_CUTOFF, self.J5_BETA, self.D_CUTOFF)
        self.m_j5 = MedianPrefilter(window=3)
        self.f_j6 = OneEuroFilter(self.J6_MIN_CUTOFF, self.J6_BETA, self.D_CUTOFF)
        self.m_j6 = MedianPrefilter(window=3)

        # 필터링된 목표점 (mm, 로봇 베이스 좌표계)
        self.tgt_x = None
        self.tgt_y = None
        self.tgt_z = None
        self.cart_initialized = False

        # 팔 길이(m) 중앙값 버퍼 — 프레임마다 흔들리는 추정치를 안정화
        self.arm_len_buf = deque(maxlen=15)

        # ── GRU 랜드마크 보정기 ──────────────────────────────────────────
        # 모델 파일이 없거나 로드에 실패하면 corrector=None 으로 두고
        # 보정 없이 원래대로 동작한다. 제어가 멈추는 것보다 안전하다.
        self.corrector = None
        if self.ENABLE_GRU_CORRECTION and LandmarkCorrector is not None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), self.GRU_MODEL_FILENAME)
            try:
                self.corrector = LandmarkCorrector(
                    model_path, warmup=self.GRU_WARMUP_FRAMES)
                self.get_logger().info(
                    f"[GRU 보정] 활성화 — {self.GRU_MODEL_FILENAME} "
                    f"(워밍업 {self.GRU_WARMUP_FRAMES}프레임)")
            except Exception as e:
                self.get_logger().warn(
                    f"[GRU 보정] 모델을 불러오지 못해 비활성화합니다: {e}")
                self.corrector = None
        elif self.ENABLE_GRU_CORRECTION:
            self.get_logger().warn("[GRU 보정] gru_runtime 모듈이 없어 비활성화됩니다.")

        # 툴 피치 / J6 홀드값 (손이 안 보이면 마지막 값 유지)
        self.tool_pitch_deg = self.TOOL_PITCH_STRAIGHT
        self.j6_deg = 0.0
        self.j5_last_valid_time = 0.0
        # [상태 변수] 2D palm 방식 각도 unwrap 및 기준축 상태.
        #   모두 '직전 궤적의 기억'이므로 호밍/리셋 시 반드시 비운다.
        #   낡은 값이 남으면 리셋 직후 첫 프레임에서 ±360°가 잘못 더해지거나
        #   엉뚱한 기준 방향으로 각도를 재게 되어 그리퍼가 급회전한다.
        self.j5_unwrap_prev = None
        self.j5_unwrap_turns = 0
        self.j5_unwrap_time = 0.0
        self.j5_ref_dir = None
        # [상태 변수] palm3d 부호 확정 상태 기계.
        #   ★ 호밍 안전 ★ j5_sign 은 '직전에 손목이 어느 쪽으로 꺾여 있었나'
        #   라는 기억이다. 비우지 않으면 호밍 완료 직후 첫 유효 프레임에서
        #   E(크기)에 낡은 부호가 곱해져, 실제 손목 방향과 반대인 툴 피치가
        #   그대로 발행되고 그리퍼가 급회전한다. 크기가 클수록 위험하다.
        #   _hard_reset_control_state() 에 반드시 함께 등록되어 있어야 한다.
        self.j5_sign = None          # 확정된 부호 (+1.0 / -1.0 / None)
        self.j5_sign_cand = None     # 뒤집기 후보 부호
        self.j5_sign_cnt = 0         # 후보가 연속 관측된 프레임 수

        # 마지막 IK 해 (화면 표시 + 디버깅용)
        self.last_ik = None          # (j1, j2, j3) 도
        self.last_ik_ok = False
        self.ik_status_text = ''

        # 추적 상태
        self.last_track_time = 0.0
        self.last_cart_time  = 0.0   # 직교공간 속도 제한용 직전 프레임 시각   # 마지막으로 유효한 목표를 만든 시각

        # 마지막으로 실제 발행된 값 (데드밴드 비교 기준)
        self.last_published_joints  = [0.0] * 6
        self.last_published_gripper = 35.0
        self.last_published_cart    = None   # (x, y, z) mm — 카테시안 데드밴드용

        self.last_joint_pub_time   = 0.0
        self.last_gripper_pub_time = 0.0

        # ══════════════════════════════════════════════════════════════════
        #  [지연 계측] 프레임 수신 → 관절명령 발행까지의 파이 내부 지연
        #
        #  왜 이 계측이 필요한가:
        #    체감 지연의 원인이 네트워크인지, 필터인지, 발행 게이트인지,
        #    GRBL인지를 추측으로 구분할 수 없다. remote_frame_time(업로드
        #    웹소켓이 프레임을 받은 시각)과 발행 시각은 '같은 시계'라서
        #    클럭 오차 없이 파이 내부 구간만 정확히 뗄 수 있다.
        #    (폰→파이 편도 지연은 여기 안 잡힘 — 그건 별도 왕복 측정 필요)
        #
        #  [호밍 안전 — 명시적 확인]
        #    이 세 변수는 오직 로그 출력에만 쓰이는 통계 누적기다. 제어
        #    경로(목표값·관절각·그리퍼)에서 읽는 곳이 한 곳도 없고, 큐/버퍼/
        #    지연 전송 구조가 아니므로 낡은 값이 재생될 경로 자체가 없다.
        #    따라서 _hard_reset_control_state()에 등록하지 않는다. 오히려
        #    호밍마다 초기화하면 통계가 끊겨 계측 목적에 반한다.
        # ══════════════════════════════════════════════════════════════════
        self.lat_samples = []          # 최근 구간의 지연 샘플(초)
        self.lat_last_report = 0.0     # 마지막 요약 출력 시각
        self.LAT_REPORT_SEC = 2.0      # 요약 출력 주기
        self.LAT_LOG_ENABLE = True     # 튜닝이 끝나면 False로 끌 것

        # 리셋 직후 첫 유효 프레임은 "기준점 캡처"에만 쓰고 명령을 내리지 않음
        self.first_frame_after_reset = True

        self.timer = self.create_timer(0.05, self.process_frame)  # 20Hz 처리
        self.get_logger().info(
            "AI 노드(v2) 준비 완료 — 해석적 역기구학(IK) 기반 위치 제어. "
            f"링크: d1={self.LINK_D1_MM} a1={self.LINK_A1_MM} "
            f"L2={self.LINK_L2_MM} L3={self.LINK_L3_MM} phi={self.LINK_PHI3_DEG}"
        )
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
                        # 표시용 크롭에 쓸 원본 비율 기록 (계산엔 미사용)
                        self.source_aspect = decoded.shape[1] / decoded.shape[0]
                    return ('', 204)
                return ('decode failed', 400)
            except Exception as e:
                return (str(e), 500)

        # ── [학습 데이터 녹화] 촬영 GUI(record.html)용 엔드포인트 ────────────
        # 브라우저가 "다음 블록" 버튼을 누르면 여기로 요청이 오고,
        # 서버가 자기 시계(monotonic)로 마커를 찍는다. 브라우저 시계를 쓰지
        # 않기 때문에 폰으로 촬영해도 프레임 타임스탬프와 정확히 정렬된다.
        @flask_app.route('/mark', methods=['POST', 'OPTIONS'])
        def mark():
            if request.method == 'OPTIONS':
                return ('', 204)
            if not getattr(self, 'recorder', None) or not self.recorder.enabled:
                return ('recording disabled', 409)
            data = request.get_json(force=True, silent=True) or {}
            action = str(data.get('action', 'add'))
            if action == 'undo':
                m = self.recorder.undo_marker()
                return ({'ok': m is not None, 'undone': m}, 200)
            event = str(data.get('event', ''))[:200]
            kind = str(data.get('kind', 'block'))[:20]
            if not event:
                return ('event required', 400)
            m = self.recorder.add_marker(event, kind, data.get('note'))
            return ({'ok': True, 'marker': m}, 200)

        # 촬영 GUI가 "지금 정말 녹화 중인지"를 확인하는 용도
        @flask_app.route('/rec_status')
        def rec_status():
            r = getattr(self, 'recorder', None)
            if not r or not r.enabled:
                return ({'recording': False}, 200)
            return ({'recording': True,
                     'session_dir': r.session_dir,
                     'saved': r.saved_count,
                     'submitted': r.frame_idx,
                     'dropped': r.dropped_count,
                     'dup': r.dup_count,
                     'stalled': r._dup_run >= 10,   # 0.5초 이상 프레임이 안 바뀜
                     'markers': len(r.markers)}, 200)

        # ── [촬영 준비] 프레임 체크용 포즈 상태 ────────────────────────────
        # 촬영 GUI가 "팔을 뻗었을 때 손목이 화면 밖으로 나가는지"를
        # 실시간으로 판정하는 데 쓴다. 서버가 실제로 돌리는 MediaPipe 결과를
        # 그대로 돌려주므로, 브라우저에서 따로 추정할 때와 달리 실제와 일치한다.
        @flask_app.route('/pose_status')
        def pose_status():
            with self.pose_status_lock:
                s = dict(self.pose_status_latest)
            s['age'] = round(time.time() - s.get('t', 0), 2) if s.get('t') else 999
            return (s, 200)

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
                    # [진단] 폰이 실제로 보내는 원본 해상도를 1초에 한 번만 찍음.
                    # 여기 로그의 w×h가 예: 480×640(세로)이면 크롭이 동작해야 정상,
                    # 640×480(가로)이면 폰이 이미 가로로 보내는 것이라 크롭할 게 없음.
                    now_dbg = time.time()
                    if now_dbg - getattr(self, '_last_src_log', 0) > 1.0:
                        self._last_src_log = now_dbg
                        self.get_logger().info(
                            f"[진단] 업로드된 원본 프레임 해상도: "
                            f"{decoded.shape[1]}x{decoded.shape[0]} "
                            f"(aspect={decoded.shape[1]/decoded.shape[0]:.3f})")
                    with self.remote_frame_lock:
                        self.remote_frame = decoded
                        self.remote_frame_time = time.time()
                        # 표시용 크롭에 쓸 원본 비율 기록 (계산엔 미사용)
                        self.source_aspect = decoded.shape[1] / decoded.shape[0]
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

    # ══════════════════════════════════════════════════════════════════════
    #  ★★★ 제어 상태 전체 리셋 — 호밍 안전의 핵심 ★★★
    #
    #  [왜 이 함수가 따로 있어야 하는가]
    #  과거 두 차례, "호밍 시작~완료 사이에 큐/버퍼에 낡은 목표값이 남아 있다가
    #  호밍 완료 직후 재생되어 로봇이 급이동"하는 사고가 있었음.
    #  v1의 ai_node는 축적 상태가 필터 몇 개뿐이라 위험이 작았지만, v2는
    #  훨씬 많은 상태를 들고 있음:
    #      tgt_x/y/z            (필터링된 목표점)
    #      f_cx/f_cy/f_cz       (카테시안 One Euro 내부 x_prev, dx_prev, t_prev)
    #      m_cx/m_cy/m_cz       (중앙값 버퍼 3프레임)
    #      arm_len_buf          (팔 길이 중앙값 15프레임)
    #      corrector            (GRU hidden state + 속도용 직전 좌표 + 워밍업)
    #      j5_unwrap_*          (2D palm 방식 ±360° 이어붙이기 상태)
    #      j5_sign / j5_sign_*  (palm3d 부호 확정 상태 — 직전 꺾임 방향의 기억)
    #      tool_pitch_deg / j6_deg  (손이 안 보일 때 유지되는 홀드값)
    #      last_ik              (직전 IK 해)
    #      last_published_*     (데드밴드 비교 기준)
    #  이 중 하나라도 호밍 전 값이 남으면 같은 사고가 재발함.
    #  그래서 "상태를 지우는 곳"을 이 함수 하나로 단일화하고,
    #  호밍 시작·호밍 완료·AI 토글·제어팔 변경 모두 여기만 호출하게 만듦.
    #  (여러 곳에 흩어져 있으면 나중에 하나를 빠뜨리는 게 과거 사고의 원인이었음)
    # ══════════════════════════════════════════════════════════════════════
    def _hard_reset_control_state(self, reason=''):
        # 1) 목표점 소거 — 낡은 목표가 남아 재생되는 것을 원천 차단
        self.tgt_x = None
        self.tgt_y = None
        self.tgt_z = None
        self.cart_initialized = False

        # 2) 카테시안 필터 내부 상태 소거
        self.f_cx.reset(); self.f_cy.reset(); self.f_cz.reset()
        self.m_cx.reset(); self.m_cy.reset(); self.m_cz.reset()

        # 3) 손목/툴 축 필터 및 홀드값 소거
        self.f_j5.reset(); self.m_j5.reset()
        self.f_j6.reset(); self.m_j6.reset()
        self.tool_pitch_deg = self.TOOL_PITCH_STRAIGHT
        self.j6_deg = 0.0
        self.j5_last_valid_time = 0.0
        # [상태 변수] 2D palm 방식 각도 unwrap 및 기준축 상태.
        #   모두 '직전 궤적의 기억'이므로 호밍/리셋 시 반드시 비운다.
        #   낡은 값이 남으면 리셋 직후 첫 프레임에서 ±360°가 잘못 더해지거나
        #   엉뚱한 기준 방향으로 각도를 재게 되어 그리퍼가 급회전한다.
        self.j5_unwrap_prev = None
        self.j5_unwrap_turns = 0
        self.j5_unwrap_time = 0.0
        self.j5_ref_dir = None
        # [상태 변수] palm3d 부호 확정 상태 기계. 위와 같은 이유로 함께 비운다.
        #   낡은 부호가 남으면 리셋 직후 첫 유효 프레임에서 실제 손목과 반대
        #   방향의 툴 피치가 발행되어 그리퍼가 급회전한다.
        self.j5_sign = None
        self.j5_sign_cand = None
        self.j5_sign_cnt = 0

        # 4) 팔 길이 추정 버퍼 소거 (사람이 바뀌었을 수도 있음)
        self.arm_len_buf.clear()

        # 4-1) GRU 보정기 상태 소거
        #  [호밍 안전] GRU hidden state와 속도 계산용 직전 좌표는 '직전 궤적의
        #  기억'이다. 비우지 않으면 호밍 완료 직후 첫 프레임이 호밍 전 자세
        #  쪽으로 보정되어, 그 값이 IK 목표점으로 나가 로봇이 급이동한다.
        #  reset()은 워밍업 카운터도 함께 되돌리므로, 리셋 후 GRU_WARMUP_FRAMES
        #  동안은 보정 없이 원본이 통과된다(이중 안전장치).
        if self.corrector is not None:
            self.corrector.reset()

        # 5) IK 해 / 추적 상태 소거
        self.last_ik = None
        self.last_ik_ok = False
        self.ik_status_text = ''
        self.last_track_time = 0.0
        self.last_cart_time = 0.0

        # 6) 데드밴드 기준값(앵커) 소거.
        #    [정확한 의미] 이 두 값은 "발행할 값"이 아니라 "변화량 비교 기준"임.
        #    - 호밍 직후: 로봇이 실제로 전 관절 0°이므로 0 앵커가 정확히 맞음.
        #    - AI 토글/캘리브 완료 등 로봇이 0이 아닐 때: 앵커가 0이면 첫 판정에서
        #      '변했다'로 나와 리셋 후 첫 명령이 무조건 발행됨 — 이는 의도된 동작.
        #      발행되는 값 자체는 항상 현재 프레임의 IK 결과이므로 위험하지 않음.
        self.last_published_joints = [0.0] * 6
        self.last_published_cart = None

        # 7) 리셋 후 첫 유효 프레임은 기준점 캡처 전용 — 명령을 내리지 않음.
        #    (리셋 직후 사용자의 손이 어디에 있든, 그 위치를 '현재'로 받아들이고
        #     거기서부터 시작해야 함. 바로 명령을 내면 그 위치로 급이동함)
        self.first_frame_after_reset = True

        if reason:
            self.get_logger().info(f"[제어상태 전체 리셋] {reason}")

    # ══════════════════════════════════════════════════════════════════════
    #  [지연 계측] 프레임 수신 → 관절명령 발행 구간 요약
    #
    #  읽는 법 (수치는 '파이 내부'만. 폰 카메라·업로드 구간은 미포함):
    #    - avg 가 100ms를 넘으면 필터/발행게이트가 원인일 가능성이 큼
    #      → CART_MEDIAN_WINDOW를 1로, 그래도 크면 CART_MIN_CUTOFF를 올림
    #    - max 만 크고 avg는 작으면 CPU 스파이크(추론 밀림)
    #    - n(초당 발행수)이 기대치보다 훨씬 작으면 DEADBAND_CART_MM에 막힌 것
    #      (저속 미세조작 구간에서 정상적으로 발생함)
    #
    #  이 함수는 로그만 출력하며 어떤 제어 상태도 바꾸지 않는다.
    # ══════════════════════════════════════════════════════════════════════
    def _record_latency(self, dt_sec, now):
        self.lat_samples.append(dt_sec)

        if self.lat_last_report == 0.0:
            self.lat_last_report = now
            return
        if now - self.lat_last_report < self.LAT_REPORT_SEC:
            return

        s = self.lat_samples
        self.lat_samples = []
        span = now - self.lat_last_report
        self.lat_last_report = now
        if not s:
            return

        s_sorted = sorted(s)
        p50 = s_sorted[len(s_sorted) // 2] * 1000.0
        self.get_logger().info(
            f"[지연] 수신→발행  평균 {sum(s)/len(s)*1000:5.1f}ms  "
            f"중앙 {p50:5.1f}ms  최소 {s_sorted[0]*1000:5.1f}ms  "
            f"최대 {s_sorted[-1]*1000:5.1f}ms  |  "
            f"발행 {len(s)/span:4.1f}회/초  "
            f"(median창={self.CART_MEDIAN_WINDOW} "
            f"cutoff={self.CART_MIN_CUTOFF} "
            f"pub={1.0/self.PUB_INTERVAL:.0f}Hz "
            f"deadband={self.DEADBAND_CART_MM}mm)")

    def _reset_motion_filters(self):
        """v1과의 호환용 별칭. 실제 동작은 전체 리셋과 동일."""
        self._hard_reset_control_state()

    def _apply_ai_enabled(self, enabled, source='topic'):
        """GUI와 손 제스처가 공통으로 사용하는 실제 AI 상태 변경 함수."""
        enabled = bool(enabled)

        # [안전] 호밍 중에는 어떤 경로로도 AI를 켤 수 없음
        if enabled and self.homing_active:
            self.get_logger().warn("호밍 진행 중이라 AI 제어를 켜지 않습니다.")
            return

        if enabled == self.ai_enabled:
            return

        self.ai_enabled = enabled
        now = time.time()

        if enabled:
            # ★ AI를 켜는 순간이 곧 "클러치 재잡기" 시점 ★
            #   상태를 전부 지우고 현재 손 위치를 새 기준점으로 삼기 때문에,
            #   팔이 피로하면 제스처로 껐다 켜서 편한 자세에서 다시 시작할 수 있음.
            #   (별도 클러치 제스처를 만들면 기존 '주먹→펼침→주먹' 토글과
            #    충돌하므로, 토글 자체를 클러치로 겸용하는 설계)
            self._hard_reset_control_state('AI ON — 현재 손 위치를 새 기준점으로 사용')
            self.ai_publish_block_until = now + self.AI_ENABLE_SETTLE_SEC
            self.last_joint_pub_time = now
            self.last_gripper_pub_time = now
            self.get_logger().info(
                f"AI 제어 활성화({source}) — {self.AI_ENABLE_SETTLE_SEC:.1f}초 후 명령 발행, "
                f"제어 팔: {self.control_arm}")
        else:
            self.ai_publish_block_until = 0.0
            # 끌 때도 지워둠 — 다음에 켤 때 낡은 상태가 섞이지 않게
            self._hard_reset_control_state()
            self.get_logger().info(f"AI 제어 비활성화({source})")

    def ai_enable_callback(self, msg):
        self._apply_ai_enabled(msg.data, source='GUI/topic')

    def control_arm_callback(self, msg):
        """GUI에서 팔 선택이 바뀔 때 호출. 상태도 리셋해서 전환 직후 급이동 방지."""
        arm = msg.data.strip().lower()
        if arm in ('left', 'right') and arm != self.control_arm:
            self.control_arm = arm
            self._hard_reset_control_state(f'제어 팔 변경 → {arm}')
            self._reset_toggle_gesture('Gesture: START WITH FIST')
            self.get_logger().info(f"제어 팔 변경: {arm}")

    # ══════════════════════════════════════════════════════════════════════
    #  ★★★ 호밍 상태 처리 — 3중 방어 ★★★
    #
    #  robot_node는 호밍 "시작"에 True, "완료"에 False를 발행함.
    #  v1은 True(시작)만 처리하고 False(완료)는 그냥 return 했는데,
    #  그게 위험함: 호밍이 진행되는 35초 동안에도 이 노드는 계속 프레임을
    #  처리하면서 필터/목표점을 채우고 있으므로, 완료 시점에는 이미
    #  "호밍 전과 무관한, 그러나 낡은" 상태가 다시 쌓여 있을 수 있음.
    #
    #  [방어 1] 호밍 시작 → AI 강제 OFF + 전체 리셋 + homing_active=True
    #           homing_active인 동안 process_frame은 IK 계산 자체를 하지 않음.
    #           (결과를 버리는 게 아니라 아예 계산·저장하지 않음 → 축적 불가)
    #  [방어 2] 호밍 완료 → 전체 리셋을 한 번 더 수행
    #           호밍 진행 중 어떤 경로로든 상태가 채워졌을 가능성을 마지막으로 차단.
    #  [방어 3] 호밍 완료 후 HOMING_SETTLE_SEC 동안 발행 금지
    #           그 사이에 필터가 "사용자의 실제 현재 자세"로 다시 채워지므로,
    #           첫 발행 명령은 반드시 현재 손 위치를 반영한 값이 됨.
    #
    #  여기에 더해 AI가 강제로 꺼지므로, 사용자가 제스처나 GUI로 명시적으로
    #  다시 켜기 전까지는 어떤 관절 명령도 발행되지 않음.
    #  (robot_node의 is_homing 큐 비우기, GUI onAiJoints의 isHoming 체크까지
    #   합치면 총 3개 노드에서 독립적으로 막는 구조)
    # ══════════════════════════════════════════════════════════════════════
    def homing_status_callback(self, msg):
        homing = bool(msg.data)

        if homing:
            # ── 방어 1: 호밍 시작 ──────────────────────────────────────────
            self.homing_active = True
            self.homing_lockout_until = float('inf')   # 완료 신호 전까지 무기한 발행 금지

            # AI를 강제로 끔. _apply_ai_enabled 내부에서도 리셋이 돌지만,
            # 이미 꺼져 있던 경우에는 early return 되므로 아래에서 명시적으로 한 번 더 호출.
            if self.ai_enabled:
                self.ai_enabled = False
                self.ai_publish_block_until = 0.0
                # GUI가 화면 상태를 갱신할 수 있도록 토픽으로도 알림
                try:
                    m = Bool()
                    m.data = False
                    self.ai_enable_pub.publish(m)
                except Exception:
                    pass

            self._hard_reset_control_state('호밍 시작 — 목표값/필터/홀드값 전체 소거, AI 강제 OFF')
            self._reset_toggle_gesture('Gesture: START WITH FIST')

            # 캘리브레이션도 기본값으로 초기화 (v1과 동일 — 사람이 바뀌는 시점으로 간주)
            was_calibrating = (self.calib_mode is not None) or self.calib_session_active
            self._reset_calibration_to_defaults()

            self.get_logger().info("호밍 시작 감지 — 제어 상태와 캘리브레이션을 초기화했습니다.")
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
        else:
            # ── 방어 2 + 3: 호밍 완료 ──────────────────────────────────────
            self.homing_active = False
            self._hard_reset_control_state(
                '호밍 완료 — 진행 중 쌓였을 수 있는 상태를 한 번 더 소거')
            self.homing_lockout_until = time.time() + self.HOMING_SETTLE_SEC
            self.get_logger().info(
                f"호밍 완료 감지 — {self.HOMING_SETTLE_SEC:.1f}초 안정화 후 명령 발행이 가능해집니다. "
                f"(AI는 꺼진 상태이므로 제스처 또는 GUI로 직접 켜야 합니다)")

    def _reset_calibration_to_defaults(self):
        self.CAL_ARM_LEN = MirobotAiNode.CAL_ARM_LEN
        self.CAL_FWD_NEUTRAL = MirobotAiNode.CAL_FWD_NEUTRAL
        self.CAL_FWD_MAX     = MirobotAiNode.CAL_FWD_MAX
        self.CAL_UP_NEUTRAL  = MirobotAiNode.CAL_UP_NEUTRAL
        self.CAL_UP_MAX      = MirobotAiNode.CAL_UP_MAX
        self.CAL_LAT_NEUTRAL = MirobotAiNode.CAL_LAT_NEUTRAL
        self.WRIST_ANGLE_STRAIGHT = MirobotAiNode.WRIST_ANGLE_STRAIGHT
        self.WRIST_ANGLE_BACK     = MirobotAiNode.WRIST_ANGLE_BACK
        self.WRIST_ANGLE_FRONT    = MirobotAiNode.WRIST_ANGLE_FRONT

        self.calib_mode = None
        self.calib_session_active = False
        self.calib_next_stage_idx = 0
        self.calib_samples_pos = []
        self.calib_samples_arm = []
        self.calib_samples_wrist = []
        self.calib_prep_until = 0.0

    def calibrate_cmd_callback(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == 'start':
            if self.homing_active:
                self._publish_calib_status(
                    force=True, override_text="호밍 중에는 캘리브레이션을 시작할 수 없습니다.")
                return

            if not self.calib_session_active:
                self.calib_session_active = True
                self.calib_next_stage_idx = 0

            if self.calib_next_stage_idx < len(self.CALIB_STAGES):
                self.calib_mode = self.CALIB_STAGES[self.calib_next_stage_idx]
                self.calib_samples_pos = []
                self.calib_samples_arm = []
                self.calib_samples_wrist = []
                # 버튼을 누른 직후는 아직 자세를 잡는 중이므로, 이 시각까지는
                # 샘플을 모으지 않고 "준비 시간"으로만 사용함.
                self.calib_prep_until = time.time() + self.CALIB_PREP_DELAY_SEC
                self.get_logger().info(f"캘리브레이션 단계 진행: {self.calib_mode}")
                self._publish_calib_status(force=True)
            else:
                self.calib_session_active = False

        elif cmd == 'cancel':
            if self.calib_mode is not None or self.calib_session_active:
                self.get_logger().info("캘리브레이션 취소됨")
            self.calib_mode = None
            self.calib_session_active = False
            self.calib_next_stage_idx = 0
            self.calib_samples_pos = []
            self.calib_samples_arm = []
            self.calib_samples_wrist = []
            self.calib_prep_until = 0.0
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

    def _j5_palm3d_geometry(self, world_lm, hand_world_lm, active_arm):
        """palm3d 방식의 순수 기하 계산. (크기E, 부호원F) 또는 None.

        상태를 전혀 건드리지 않는다. 부호 확정 상태 기계는 호출부에 둬서
        '상태를 만드는 곳'을 한눈에 보이게 했다(호밍 안전 검수 편의).

        정의:
            P0  = 손목,  P5 = 검지 MCP,  P17 = 새끼 MCP   (hand world, 미터)
            k   = P17 - P5          너클선 = 굴곡 회전축
            m   = (P5+P17)/2 - P0   손바닥 장축 (손목 → 너클선 중점)
            f   = 손목 - 팔꿈치      전완            (pose world, 미터)

            E = angle(f, m)                      0~180°, 부호 없음. 안정적.
            F = f와 m을 k에 수직인 평면으로 투영해 k를 축으로 잰 부호 있는 각.
                투영이 핵심이다. 회전축 방향 성분을 빼야 '그 축을 중심으로
                얼마나 돌았는가'만 남고, 팔 전체 자세 변화가 섞이지 않는다.

        반환: (e_mag, f_signed) / 계산 불가면 None
              f_signed 에는 좌우 거울상 보정(J5_FLEX_ARM_SIGN)이 이미 적용됨.
        """
        if world_lm is None or hand_world_lm is None:
            return None

        el_i = 13 if active_arm == 'right' else 14
        wr_i = 15 if active_arm == 'right' else 16
        try:
            ew, ww = world_lm[el_i], world_lm[wr_i]
            p0, p5, p17 = hand_world_lm[0], hand_world_lm[5], hand_world_lm[17]
        except (IndexError, TypeError):
            return None

        # 전완 (팔꿈치 → 손목)
        fx, fy, fz = ww.x - ew.x, ww.y - ew.y, ww.z - ew.z
        # 너클선 (검지 MCP → 새끼 MCP) = 굴곡 회전축
        kx, ky, kz = p17.x - p5.x, p17.y - p5.y, p17.z - p5.z
        # 손바닥 장축 (손목 → 너클선 중점)
        mx = (p5.x + p17.x) * 0.5 - p0.x
        my = (p5.y + p17.y) * 0.5 - p0.y
        mz = (p5.z + p17.z) * 0.5 - p0.z

        fm = math.sqrt(fx * fx + fy * fy + fz * fz)
        km = math.sqrt(kx * kx + ky * ky + kz * kz)
        mm = math.sqrt(mx * mx + my * my + mz * mz)
        t = self.MIN_VEC3_J5
        if fm < t or km < t or mm < t:
            return None

        # ── E: 부호 없는 3D 사잇각 ────────────────────────────────────────
        e_mag = self._angle_between_3d(fx, fy, fz, mx, my, mz)

        # ── F: 너클선을 축으로 투영해 잰 부호 있는 각 ─────────────────────
        nx, ny, nz = kx / km, ky / km, kz / km        # 단위 회전축
        fd = fx * nx + fy * ny + fz * nz
        md = mx * nx + my * ny + mz * nz
        fpx, fpy, fpz = fx - fd * nx, fy - fd * ny, fz - fd * nz
        mpx, mpy, mpz = mx - md * nx, my - md * ny, mz - md * nz
        fpm = math.sqrt(fpx * fpx + fpy * fpy + fpz * fpz)
        mpm = math.sqrt(mpx * mpx + mpy * mpy + mpz * mpz)
        if fpm < 1e-9 or mpm < 1e-9:
            # 전완이나 손바닥 장축이 회전축과 거의 나란함 → 방향이 정의되지
            # 않는다. 크기는 살아 있지만 부호를 새로 정할 근거가 없으므로
            # None을 돌려 호출부가 직전 부호를 유지하게 한다.
            return (e_mag, None)

        cx = fpy * mpz - fpz * mpy
        cy = fpz * mpx - fpx * mpz
        cz = fpx * mpy - fpy * mpx
        f_signed = math.degrees(math.atan2(
            cx * nx + cy * ny + cz * nz,
            fpx * mpx + fpy * mpy + fpz * mpz))

        f_signed *= self.J5_FLEX_ARM_SIGN.get(active_arm, 1.0)
        return (e_mag, f_signed)

    @staticmethod
    def _joint_angle_2d(a, b, c):
        """a-b-c 세 점에서 b를 꼭짓점으로 하는 2D 각도(0~180°)."""
        bax, bay = a.x - b.x, a.y - b.y
        bcx, bcy = c.x - b.x, c.y - b.y
        return MirobotAiNode._angle_between_2d(bax, bay, bcx, bcy)

    def _classify_toggle_hand(self, hlm):
        """선택 손을 open/closed/other로 분류하고 (상태, 펴진 손가락 수, 손바닥 비율) 반환."""
        if hlm is None or len(hlm) < 21:
            return 'missing', 0, 0.0

        wrist = hlm[0]
        palm_length = self._dist2d(wrist, hlm[9]) + 1e-6
        palm_width = self._dist2d(hlm[5], hlm[17])
        palm_ratio = palm_width / palm_length
        palm_front = palm_ratio >= self.GESTURE_MIN_PALM_WIDTH_RATIO

        extended = 0

        # 검지/중지/약지/소지: PIP 관절이 거의 펴졌고 손끝이 PIP보다 손목에서 멀면 펼침.
        for mcp_idx, pip_idx, tip_idx in ((5, 6, 8), (9, 10, 12),
                                          (13, 14, 16), (17, 18, 20)):
            angle = self._joint_angle_2d(hlm[mcp_idx], hlm[pip_idx], hlm[tip_idx])
            tip_farther = self._dist2d(wrist, hlm[tip_idx]) > self._dist2d(wrist, hlm[pip_idx]) * 1.08
            if angle >= 150.0 and tip_farther:
                extended += 1

        # 엄지: IP가 펴져 있고 엄지끝이 검지 MCP에서 충분히 멀어졌을 때 펼침.
        thumb_angle = self._joint_angle_2d(hlm[2], hlm[3], hlm[4])
        thumb_away = self._dist2d(hlm[5], hlm[4]) > self._dist2d(hlm[5], hlm[3]) * 1.08
        if thumb_angle >= 145.0 and thumb_away:
            extended += 1

        if palm_front and extended >= self.GESTURE_OPEN_MIN_FINGERS:
            return 'open', extended, palm_ratio

        # 주먹에서는 손가락이 손바닥을 가리면서 palm_ratio가 작아질 수 있으므로
        # 펼친 손보다 느슨한 폭 기준을 사용함. 그래도 손가락 수가 1개 이하일 때만
        # 인정하므로 단순한 반쯤 접힌 손은 주먹으로 처리되지 않음.
        fist_visible = palm_ratio >= self.GESTURE_MIN_FIST_WIDTH_RATIO
        if fist_visible and extended <= self.GESTURE_CLOSED_MAX_FINGERS:
            return 'closed', extended, palm_ratio
        return 'other', extended, palm_ratio

    def _reset_toggle_gesture(self, status='Gesture: START WITH FIST'):
        self.gesture_phase = 'wait_start_fist'
        self.gesture_pose_since = 0.0
        self.gesture_step_confirmed_at = 0.0
        self.gesture_suspend_commands = False
        self.gesture_status_text = status

    def _publish_gesture_toggle(self, enabled):
        """AI 상태를 즉시 내부 반영하고, 웹 GUI/다른 ROS 노드에도 같은 상태를 알림."""
        self._apply_ai_enabled(enabled, source='hand gesture')
        msg = Bool()
        msg.data = bool(enabled)
        self.ai_enable_pub.publish(msg)
        self.get_logger().info(
            f"손 제스처 토글 완료 — /mirobot/ai_enable={msg.data} 발행")

    def _update_ai_toggle_gesture(self, hlm, now):
        """주먹 → 빠르게 펼침 → 다시 주먹 순서를 인식하는 비블로킹 상태 머신."""
        # 캘리브레이션 자세와 손 제스처가 겹치지 않도록 세션 중에는 완전히 비활성화.
        if self.calib_mode is not None or self.calib_session_active:
            self._reset_toggle_gesture('Gesture: DISABLED DURING CALIBRATION')
            self.gesture_debug_text = ''
            return

        hand_state, finger_count, palm_ratio = self._classify_toggle_hand(hlm)
        self.gesture_debug_text = ''

        if now < self.gesture_cooldown_until:
            remain = self.gesture_cooldown_until - now
            self.gesture_suspend_commands = False
            self.gesture_status_text = f'Gesture: COOLDOWN {remain:.1f}s'
            return

        # 1단계: 주먹으로 시작. AI가 이미 켜져 있다면 주먹 후보가 잡히는 순간부터
        # 명령을 잠시 막아, 끄기 위해 취하는 동작이 로봇 명령으로 중계되지 않게 함.
        if self.gesture_phase == 'wait_start_fist':
            if hand_state == 'closed':
                self.gesture_suspend_commands = True
                if self.gesture_pose_since == 0.0:
                    self.gesture_pose_since = now
                held = now - self.gesture_pose_since
                self.gesture_status_text = (
                    f'Gesture: START FIST {held:.1f}/{self.GESTURE_START_FIST_HOLD_SEC:.1f}s')
                if held >= self.GESTURE_START_FIST_HOLD_SEC:
                    self.gesture_phase = 'wait_open'
                    self.gesture_step_confirmed_at = now
                    self.gesture_pose_since = 0.0
                    self.gesture_status_text = 'Gesture: OPEN HAND NOW'
            else:
                self.gesture_suspend_commands = False
                self.gesture_pose_since = 0.0
                self.gesture_status_text = 'Gesture: START WITH FIST'
            return

        # 시작 주먹이 확정된 뒤에는 전체 토글 동작이 끝날 때까지 명령을 차단함.
        self.gesture_suspend_commands = True
        elapsed = now - self.gesture_step_confirmed_at
        if elapsed > self.GESTURE_MAX_TRANSITION_SEC:
            self._reset_toggle_gesture('Gesture: TOO SLOW - START WITH FIST')
            return

        # 2단계: 제한 시간 안에 손을 빠르게 펼침.
        if self.gesture_phase == 'wait_open':
            if hand_state == 'open':
                if self.gesture_pose_since == 0.0:
                    self.gesture_pose_since = now
                held = now - self.gesture_pose_since
                self.gesture_status_text = (
                    f'Gesture: OPEN {held:.1f}/{self.GESTURE_OPEN_HOLD_SEC:.1f}s')
                if held >= self.GESTURE_OPEN_HOLD_SEC:
                    self.gesture_phase = 'wait_final_fist'
                    self.gesture_step_confirmed_at = now
                    self.gesture_pose_since = 0.0
                    self.gesture_status_text = 'Gesture: CLOSE FIST NOW'
            else:
                self.gesture_pose_since = 0.0
                remain = max(0.0, self.GESTURE_MAX_TRANSITION_SEC - elapsed)
                self.gesture_status_text = f'Gesture: OPEN HAND ({remain:.1f}s)'
            return

        # 3단계: 펼친 손이 확정된 뒤 다시 주먹을 쥐면 실제 AI 상태를 토글함.
        if hand_state == 'closed':
            if self.gesture_pose_since == 0.0:
                self.gesture_pose_since = now
            held = now - self.gesture_pose_since
            self.gesture_status_text = (
                f'Gesture: FINAL FIST {held:.1f}/{self.GESTURE_FINAL_FIST_HOLD_SEC:.1f}s')
            if held >= self.GESTURE_FINAL_FIST_HOLD_SEC:
                new_state = not self.ai_enabled
                self._publish_gesture_toggle(new_state)
                self.gesture_cooldown_until = now + self.GESTURE_COOLDOWN_SEC
                self._reset_toggle_gesture(
                    'Gesture: AI ON' if new_state else 'Gesture: AI OFF')
        else:
            self.gesture_pose_since = 0.0
            remain = max(0.0, self.GESTURE_MAX_TRANSITION_SEC - elapsed)
            self.gesture_status_text = f'Gesture: CLOSE FIST ({remain:.1f}s)'

    # ══════════════════════════════════════════════════════════════════════
    #  해석적 순기구학 / 역기구학  (J1, J2, J3 → 손목중심 위치)
    #
    #  [왜 반복 솔버(DLS, 야코비안)를 안 쓰는가]
    #  실제로 제어하는 축이 J1/J2/J3 세 개이고 결정할 값도 3D 위치 세 개라
    #  여유자유도가 0임. 자유도와 구속이 같으면 닫힌 해가 존재하고,
    #  닫힌 해가 있으면 반복 솔버보다 항상 유리함:
    #    - 빠름(수십 마이크로초), 수렴 실패가 없음, 결과가 결정론적임.
    #  대신 여유자유도가 없다는 것은 "손끝 위치를 정하면 팔꿈치 위치도 자동
    #  결정된다"는 뜻이라, 사람 팔꿈치를 따라가게 만들 여지는 없음.
    #  → 대신 팔꿈치 분기를 항상 같은 쪽으로 고정해서 자세가 튀지 않게 함.
    #
    #  [기구 모델] (URDF에서 도출, r-z 평면)
    #      r = a1 + L2*sin(J2) + L3*sin(J2 + J3 + phi)
    #      z = d1 + L2*cos(J2) + L3*cos(J2 + J3 + phi)
    #      x = r*cos(J1),  y = r*sin(J1)
    #  검증: J1=J2=J3=0 → (x, z) = (198.4, 255.0) mm  ← URDF FK와 일치
    # ══════════════════════════════════════════════════════════════════════

    def _fk_wrist(self, j1_deg, j2_deg, j3_deg):
        """관절각(도) → 손목중심 위치 (mm). 검증/표시용."""
        j1 = math.radians(j1_deg)
        j2 = math.radians(j2_deg)
        a3 = math.radians(j2_deg + j3_deg + self.LINK_PHI3_DEG)
        r = self.LINK_A1_MM + self.LINK_L2_MM * math.sin(j2) + self.LINK_L3_MM * math.sin(a3)
        z = self.LINK_D1_MM + self.LINK_L2_MM * math.cos(j2) + self.LINK_L3_MM * math.cos(a3)
        return r * math.cos(j1), r * math.sin(j1), z

    def _clamp_to_workspace(self, x, y, z):
        """
        ★ IK "이전"에 목표점을 도달 가능 영역 안으로 밀어 넣는 함수 ★

        v1에서 값이 튀던 근본 원인은 arccos/atan2에 도달 불가능하거나 특이한
        입력이 들어가서 미분값이 발산한 것이었음. 필터로 사후에 뭉개는 대신
        여기서 불량 입력 자체를 만들지 않으면 튐이 구조적으로 발생하지 않음.

        경계에 닿았을 때 '정지'가 아니라 '경계를 따라 미끄러지도록(투영)' 함 —
        사용자가 팔을 더 뻗어도 로봇이 갑자기 멈추지 않아 이질감이 적음.

        반환: (x, y, z, clamped_bool)
        """
        clamped = False

        # 1) 높이 한계
        if z < self.WS_Z_MIN_MM:
            z = self.WS_Z_MIN_MM; clamped = True
        elif z > self.WS_Z_MAX_MM:
            z = self.WS_Z_MAX_MM; clamped = True

        # 2) 앞쪽 반공간 강제 — J1의 ±180° 불연속 점프를 원리적으로 차단
        if x < self.WS_X_MIN_MM:
            x = self.WS_X_MIN_MM
            clamped = True

        # 3) 방위각(J1) 제한 — 위에서 x>0을 보장했으므로 여기서 각도가 감기지 않음
        yaw = math.atan2(y, x)
        yaw_max = math.radians(self.WS_YAW_MAX_DEG)
        if yaw > yaw_max or yaw < -yaw_max:
            yaw = max(-yaw_max, min(yaw_max, yaw))
            rr = math.hypot(x, y)
            x, y = rr * math.cos(yaw), rr * math.sin(yaw)
            clamped = True

        # 4) J1 축 데드존 — r이 0에 가까우면 방위각이 정의되지 않아 J1이 폭주함
        r = math.hypot(x, y)
        if r < self.WS_R_MIN_XY_MM:
            if r < 1e-6:
                # 완전히 축 위 → 방향 정보가 없으므로 정면(+x)으로 밀어냄
                x, y = self.WS_R_MIN_XY_MM, 0.0
            else:
                s = self.WS_R_MIN_XY_MM / r
                x, y = x * s, y * s
            r = self.WS_R_MIN_XY_MM
            clamped = True

        # 5) J2 축 중심의 구각(shell) 안으로 투영
        #    J2 축은 (r = a1, z = d1) 위치에 있고, 손목중심은 그 점에서
        #    |L2 - L3| ~ (L2 + L3) 범위의 거리에만 존재할 수 있음.
        R = r - self.LINK_A1_MM
        Z = z - self.LINK_D1_MM
        D = math.hypot(R, Z)

        if D < 1e-6:
            # 정확히 J2 축 위 — 방향이 없으므로 정면 최소거리로 밀어냄
            R, Z, D = self.WS_D_MIN_MM, 0.0, self.WS_D_MIN_MM
            clamped = True
        elif D < self.WS_D_MIN_MM:
            s = self.WS_D_MIN_MM / D
            R, Z, D = R * s, Z * s, self.WS_D_MIN_MM
            clamped = True
        elif D > self.WS_D_MAX_MM:
            s = self.WS_D_MAX_MM / D
            R, Z, D = R * s, Z * s, self.WS_D_MAX_MM
            clamped = True

        if clamped:
            r_new = R + self.LINK_A1_MM
            z = Z + self.LINK_D1_MM
            # 방위각은 유지한 채 반지름만 갱신
            if r > 1e-6:
                x, y = x / r * r_new, y / r * r_new
            else:
                x, y = r_new, 0.0

            # 투영 결과가 다시 높이 한계를 넘을 수 있으므로 한 번만 더 확인.
            # (여기서 또 벗어나면 억지로 맞추지 않고 그대로 둠 — 아래 IK가
            #  arccos 인자를 다시 클램프하므로 발산하지 않음)
            z = max(self.WS_Z_MIN_MM, min(self.WS_Z_MAX_MM, z))

        return x, y, z, clamped

    def _ik_position(self, x, y, z):
        """
        손목중심 목표 위치(mm) → (J1, J2, J3) 도.
        입력은 _clamp_to_workspace()를 통과한 값이어야 함.
        실패할 수 없는 구조이지만, 안전을 위해 arccos 인자를 한 번 더 클램프함.

        반환: (j1, j2, j3, ok)
        """
        # J1: 손목중심의 방위각. 로봇 매뉴얼상 +y가 왼쪽이고, URDF FK로
        #     J1=+90° → +y 임을 확인했으므로 부호 변환이 필요 없음.
        j1 = math.degrees(math.atan2(y, x))

        r = math.hypot(x, y)
        R = r - self.LINK_A1_MM
        Z = z - self.LINK_D1_MM
        D2 = R * R + Z * Z
        L2 = self.LINK_L2_MM
        L3 = self.LINK_L3_MM

        # 두 링크 벡터 사이의 상대 회전각 delta:
        #   D^2 = L2^2 + L3^2 + 2*L2*L3*cos(delta)
        cos_d = (D2 - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
        # 클램프 이후라면 |cos_d| <= 1 이지만, 부동소수 오차 대비 방어
        cos_d = max(-1.0, min(1.0, cos_d))
        delta = math.acos(cos_d)   # 0 ~ pi (양의 분기)

        # ── 팔꿈치 분기 고정 ────────────────────────────────────────────────
        # acos는 항상 양수 해만 주므로 delta > 0 분기로 고정됨.
        # 이 분기가 홈 자세(J3=0 ⇒ delta=phi≈83.2°)와 같은 쪽이라
        # 자세가 자연스럽고, J3 범위도 [-83.2°, +50°]로 좁게 유지되어
        # 하드 리밋(-180~+50)에 절대 걸리지 않음.
        # → 프레임마다 분기가 뒤집혀 팔꿈치가 위아래로 튀는 현상이 원천적으로 없음.
        j3 = math.degrees(delta) - self.LINK_PHI3_DEG

        # J2: 전체 방향각에서 링크 삼각형의 내부각을 뺌
        psi = math.atan2(R, Z)   # +z에서 +r 방향으로 잰 각도
        j2 = math.degrees(psi - math.atan2(L3 * math.sin(delta),
                                           L2 + L3 * math.cos(delta)))

        # 관절 소프트 리밋 확인 (여기서 벗어나면 위치를 정확히 못 맞춘다는 뜻)
        ok = True
        lo, hi = self.JOINT_SOFT_LIMITS[0]
        if j1 < lo or j1 > hi:
            j1 = max(lo, min(hi, j1)); ok = False
        lo, hi = self.JOINT_SOFT_LIMITS[1]
        if j2 < lo or j2 > hi:
            j2 = max(lo, min(hi, j2)); ok = False
        lo, hi = self.JOINT_SOFT_LIMITS[2]
        if j3 < lo or j3 > hi:
            j3 = max(lo, min(hi, j3)); ok = False

        return j1, j2, j3, ok

    def _solve_j5(self, target_tool_pitch_deg, j2_deg, j3_deg):
        """
        목표 툴 피치(월드 기준, 도) → J5 관절각.

        [원리]  URDF 순기구학으로 검증한 관계식:
                  툴이 향하는 각도 theta = J2 + J3 + J5
                    theta =   0° → 그리퍼가 수직 아래
                    theta = -90° → 그리퍼가 정면 수평
                따라서  J5 = theta - (J2 + J3)

        [왜 필요한가]  v1은 손목 각도를 J5에 그대로 넣었기 때문에, 팔을 앞으로
        뻗어서 J2/J3가 바뀌면 손목을 가만히 둬도 그리퍼가 같이 기울어졌음.
        물건을 집을 때마다 손목으로 계속 보정해줘야 했던 이유가 이것.
        보상을 넣으면 "손목으로 정한 각도"가 팔 자세와 무관하게 유지됨.

        반환: (j5_deg, saturated_bool)
        """
        if not self.TOOL_PITCH_COMPENSATION:
            # 폴백: v1처럼 툴피치 값을 그대로 관절각으로 사용
            j5 = target_tool_pitch_deg
        else:
            j5 = target_tool_pitch_deg - (j2_deg + j3_deg)

        lo, hi = self.JOINT_SOFT_LIMITS[4]
        if j5 < lo:
            return lo, True
        if j5 > hi:
            return hi, True
        return j5, False
    def _publish_calib_status(self, force=False, override_text=None):
        """캘리브레이션 진행 상태를 GUI에 표시하기 위한 String 토픽 발행.

        [주의] index.html의 onCalibStatus()가 아래 키워드로 버튼 상태를 판정하므로,
        문구를 바꿔도 이 키워드들은 반드시 포함되어야 함:
            '캘리브레이션 완료!' / '취소' / '초기화' / '다음 단계로 진행' / '다시 시도'
        """
        now = time.time()
        if not force and (now - self.last_calib_status_pub_time < self.CALIB_STATUS_PUB_INTERVAL):
            return
        self.last_calib_status_pub_time = now

        if override_text is not None:
            text = override_text
        elif self.calib_mode is None:
            text = ""
        elif now < self.calib_prep_until:
            remaining = max(0.0, self.calib_prep_until - now)
            instr = self.CALIB_INSTRUCTIONS[self.calib_mode]
            text = f"{instr}  [자세 잡는 중... {remaining:.1f}초 후 측정 시작]"
        else:
            instr = self.CALIB_INSTRUCTIONS[self.calib_mode]
            if self.calib_mode in ('neutral', 'reach_forward', 'reach_up'):
                count = len(self.calib_samples_pos)
            else:
                count = len(self.calib_samples_wrist)
            text = f"{instr}  ({min(count, self.CALIB_REQUIRED_SAMPLES)}/{self.CALIB_REQUIRED_SAMPLES})"

        msg = String()
        msg.data = text
        self.calibration_status_pub.publish(msg)

    def _calib_fail(self, current_idx, reason):
        """단계 실패 처리 — 값은 되돌리고 같은 단계를 재시도할 수 있게 함."""
        self.get_logger().warn(f"캘리브레이션 실패: {reason}")
        self.calib_mode = None
        self.calib_next_stage_idx = current_idx
        self.calib_samples_pos = []
        self.calib_samples_arm = []
        self.calib_samples_wrist = []
        self._publish_calib_status(
            force=True,
            override_text=(f"캘리브레이션 실패: {reason}\n"
                           f"준비되면 '이 단계 재시도' 버튼을 눌러 다시 시도하세요."))

    def _finish_calib_stage(self):
        """현재 단계의 샘플을 평균 내어 반영하고, 다음 단계는 자동으로 넘어가지 않고
        사용자가 버튼을 다시 누를 때까지 대기함 (자세를 바꿀 시간을 주기 위함)."""
        stage = self.calib_mode
        current_idx = self.CALIB_STAGES.index(stage)

        # ── 1~3단계: 사람 팔의 실제 이동 범위 측정 ────────────────────────
        # v1은 여기서 어깨각/팔꿈치각을 쟀지만, v2는 관절각 매핑을 쓰지 않으므로
        # 대신 "정규화된 손목 위치"의 기준점과 최대치를 잰다.
        if stage in ('neutral', 'reach_forward', 'reach_up'):
            n = len(self.calib_samples_pos)
            avg_fwd = sum(s[0] for s in self.calib_samples_pos) / n
            avg_lat = sum(s[1] for s in self.calib_samples_pos) / n
            avg_up  = sum(s[2] for s in self.calib_samples_pos) / n

            if stage == 'neutral':
                self.CAL_FWD_NEUTRAL = avg_fwd
                self.CAL_LAT_NEUTRAL = avg_lat
                self.CAL_UP_NEUTRAL  = avg_up

                # 팔 길이를 여기서 확정한다. 이후 모든 정규화(및 GRU 입력)가
                # 이 고정값을 분모로 쓰므로, 자세에 따라 값이 흔들리지 않는다.
                # 중앙값을 쓰는 이유: 수집 구간에 자세를 잡는 과도기가 섞여
                # 몇 프레임이 크게 튈 수 있는데, 평균은 그 영향을 받는다.
                if len(self.calib_samples_arm) >= 5:
                    arr = sorted(self.calib_samples_arm)
                    med = arr[len(arr) // 2]
                    if med >= self.MIN_ARM_LEN_M:
                        self.CAL_ARM_LEN = med
                        self.get_logger().info(
                            f"[캘리브레이션] 팔 길이 확정 {med*1000:.0f}mm "
                            f"(샘플 {len(arr)}개, {arr[0]*1000:.0f}~{arr[-1]*1000:.0f}mm)")
                    else:
                        self.get_logger().warn(
                            f"[캘리브레이션] 팔 길이 측정값이 너무 작습니다"
                            f"({med*1000:.0f}mm) — 실시간 추정으로 대체합니다.")

                self.get_logger().info(
                    f"[캘리브레이션] 중립 fwd={avg_fwd:.3f} lat={avg_lat:.3f} up={avg_up:.3f}")

            elif stage == 'reach_forward':
                self.CAL_FWD_MAX = avg_fwd
                self.get_logger().info(f"[캘리브레이션] 최대 전방 fwd={avg_fwd:.3f}")
                if (self.CAL_FWD_MAX - self.CAL_FWD_NEUTRAL) < self.CALIB_MIN_RANGE_NORM:
                    self.CAL_FWD_MAX = MirobotAiNode.CAL_FWD_MAX
                    self._calib_fail(current_idx,
                                     "앞으로 뻗은 거리가 기본 자세와 거의 같습니다. "
                                     "팔을 카메라 쪽으로 더 확실히 뻗어주세요.")
                    return

            elif stage == 'reach_up':
                self.CAL_UP_MAX = avg_up
                self.get_logger().info(f"[캘리브레이션] 최대 상방 up={avg_up:.3f}")
                if (self.CAL_UP_MAX - self.CAL_UP_NEUTRAL) < self.CALIB_MIN_RANGE_NORM:
                    self.CAL_UP_MAX = MirobotAiNode.CAL_UP_MAX
                    self._calib_fail(current_idx,
                                     "들어올린 높이가 기본 자세와 거의 같습니다. "
                                     "팔을 더 높이 들어주세요.")
                    return

        # ── 4~6단계: 손목 3점 (v1과 동일한 절차) ──────────────────────────
        elif stage == 'wrist_straight':
            avg_wrist = sum(self.calib_samples_wrist) / len(self.calib_samples_wrist)
            self.WRIST_ANGLE_STRAIGHT = avg_wrist
            self.get_logger().info(f"[캘리브레이션] WRIST_ANGLE_STRAIGHT={avg_wrist:.1f}°")

        elif stage == 'wrist_back':
            avg_wrist = sum(self.calib_samples_wrist) / len(self.calib_samples_wrist)
            self.WRIST_ANGLE_BACK = avg_wrist
            self.get_logger().info(f"[캘리브레이션] WRIST_ANGLE_BACK={avg_wrist:.1f}°")
            if abs(self.WRIST_ANGLE_BACK - self.WRIST_ANGLE_STRAIGHT) < self.CALIB_MIN_RANGE_DEG:
                self.WRIST_ANGLE_BACK = MirobotAiNode.WRIST_ANGLE_BACK
                self._calib_fail(current_idx, "손목 뒤로 꺾기 범위가 너무 좁습니다.")
                return

        elif stage == 'wrist_front':
            avg_wrist = sum(self.calib_samples_wrist) / len(self.calib_samples_wrist)
            self.WRIST_ANGLE_FRONT = avg_wrist
            self.get_logger().info(f"[캘리브레이션] WRIST_ANGLE_FRONT={avg_wrist:.1f}°")
            if abs(self.WRIST_ANGLE_STRAIGHT - self.WRIST_ANGLE_FRONT) < self.CALIB_MIN_RANGE_DEG:
                self.WRIST_ANGLE_FRONT = MirobotAiNode.WRIST_ANGLE_FRONT
                self._calib_fail(current_idx, "손목 안쪽 꺾기 범위가 너무 좁습니다.")
                return

        # 이번 단계 수집 종료 (자동으로 다음 단계 시작하지 않고 대기)
        self.calib_mode = None
        self.calib_samples_pos = []
        self.calib_samples_arm = []
        self.calib_samples_wrist = []

        if current_idx + 1 < len(self.CALIB_STAGES):
            self.calib_next_stage_idx = current_idx + 1
            next_instr = self.CALIB_INSTRUCTIONS[self.CALIB_STAGES[self.calib_next_stage_idx]]
            self._publish_calib_status(
                force=True,
                override_text=(
                    f"{current_idx + 1}단계 완료! \n"
                    f"준비되면 '다음 단계 시작' 버튼을 눌러 다음 단계로 진행하세요. \n"
                    f"({next_instr})"
                )
            )
        else:
            self.calib_session_active = False
            self.calib_next_stage_idx = 0
            # 캘리브레이션이 끝나면 기준점이 통째로 바뀌므로, 낡은 목표점과
            # 필터 상태가 남아 있으면 첫 프레임에서 로봇이 튈 수 있음 → 전체 리셋.
            self._hard_reset_control_state('캘리브레이션 완료 — 기준점 변경')
            self._publish_calib_status(
                force=True,
                override_text=(
                    f"캘리브레이션 완료! "
                    f"전방 {self.CAL_FWD_NEUTRAL:.2f}~{self.CAL_FWD_MAX:.2f}, "
                    f"상방 {self.CAL_UP_NEUTRAL:.2f}~{self.CAL_UP_MAX:.2f}, "
                    f"손목 안쪽{self.WRIST_ANGLE_FRONT:.0f}°/일자{self.WRIST_ANGLE_STRAIGHT:.0f}°/"
                    f"뒤{self.WRIST_ANGLE_BACK:.0f}° 로 반영되었습니다."
                )
            )

    # ── 메인 처리 루프 ────────────────────────────────────────────────────────────
    # ── 메인 처리 루프 ────────────────────────────────────────────────────────────


    @staticmethod
    def _draw_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.55, radius=16,
                           border_color=None, border_thickness=1):
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        if x2 <= x1 or y2 <= y1:
            return
        radius = max(1, min(int(radius), (x2 - x1) // 2, (y2 - y1) // 2))

        overlay = img.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

        if border_color is not None and border_thickness > 0:
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), border_color, border_thickness, cv2.LINE_AA)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), border_color, border_thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), border_color, border_thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), border_color, border_thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, border_color, border_thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, border_color, border_thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, border_color, border_thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, border_color, border_thickness, cv2.LINE_AA)

    @staticmethod
    def _put_text(img, text, x, y, scale, color, thickness=1):
        cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def _put_text_shadowed(self, img, text, x, y, scale, color, thickness=1,
                           shadow=(0, 0, 0), shadow_alpha=0.35, shadow_offset=(0, 1)):
        overlay = img.copy()
        sx = int(x + shadow_offset[0])
        sy = int(y + shadow_offset[1])
        cv2.putText(overlay, text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, shadow, max(1, thickness + 1), cv2.LINE_AA)
        cv2.addWeighted(overlay, shadow_alpha, img, 1.0 - shadow_alpha, 0, img)
        self._put_text(img, text, x, y, scale, color, thickness)

    @staticmethod
    def _lerp_color(c1, c2, t):
        t = max(0.0, min(1.0, float(t)))
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def _draw_capsule_progress(self, frame, x1, y1, x2, y2, progress,
                               fill_color, bg_color=(70, 70, 74), alpha=0.65):
        progress = max(0.0, min(1.0, float(progress)))
        radius = max(1, (y2 - y1) // 2)
        self._draw_rounded_rect(frame, x1, y1, x2, y2, bg_color, alpha=alpha,
                                radius=radius, border_color=None, border_thickness=0)
        fill_x2 = x1 + int((x2 - x1) * progress)
        if fill_x2 > x1 + 2:
            self._draw_rounded_rect(frame, x1, y1, fill_x2, y2, fill_color, alpha=0.9,
                                    radius=radius, border_color=None, border_thickness=0)

    def _gesture_progress_info(self, now):
        phase = self.gesture_phase
        if now < self.gesture_cooldown_until:
            remain = self.gesture_cooldown_until - now
            total = max(0.01, self.GESTURE_COOLDOWN_SEC)
            return 1.0 - min(1.0, remain / total), self.UI_ACCENT_GREEN
        if phase == 'wait_start_fist' and self.gesture_pose_since > 0:
            held = max(0.0, now - self.gesture_pose_since)
            return min(1.0, held / max(0.01, self.GESTURE_START_FIST_HOLD_SEC)), self.UI_ACCENT_BLUE
        if phase == 'wait_open' and self.gesture_pose_since > 0:
            held = max(0.0, now - self.gesture_pose_since)
            return min(1.0, held / max(0.01, self.GESTURE_OPEN_HOLD_SEC)), self.UI_ACCENT_BLUE
        if phase == 'wait_final_fist' and self.gesture_pose_since > 0:
            held = max(0.0, now - self.gesture_pose_since)
            return min(1.0, held / max(0.01, self.GESTURE_FINAL_FIST_HOLD_SEC)), self.UI_ACCENT_BLUE
        return None, None

    def _draw_joint_dot(self, frame, x, y, radius, fill_color, ring_color=(255, 255, 255),
                        alpha=0.95, ring_thickness=1):
        overlay = frame.copy()
        cv2.circle(overlay, (int(x), int(y)), int(radius), fill_color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.circle(frame, (int(x), int(y)), int(radius + 2), ring_color, ring_thickness, cv2.LINE_AA)

    def _draw_soft_line(self, frame, p1, p2, color, thickness=3, shadow_alpha=0.20):
        overlay = frame.copy()
        cv2.line(overlay, p1, p2, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.addWeighted(overlay, shadow_alpha, frame, 1.0 - shadow_alpha, 0, frame)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    def _draw_pose_overlay(self, frame, pose_landmarks, active_arm):
        if pose_landmarks is None:
            return
        lm = pose_landmarks.landmark
        active_joint_color = (255, 188, 110)
        active_line_color = (255, 170, 92)
        inactive_joint_color = (132, 132, 138)
        inactive_line_color = (92, 92, 96)
        core_line_color = (110, 110, 116)
        joint_sets = {'right': {11, 13, 15}, 'left': {12, 14, 16}}
        active_set = joint_sets.get(active_arm, set())

        for s_idx, e_idx in self.ARM_CONNECTIONS:
            ps, pe = lm[s_idx], lm[e_idx]
            if ps.visibility <= self.VIS_THRESHOLD or pe.visibility <= self.VIS_THRESHOLD:
                continue
            x1, y1 = int(ps.x * frame.shape[1]), int(ps.y * frame.shape[0])
            x2, y2 = int(pe.x * frame.shape[1]), int(pe.y * frame.shape[0])
            if {s_idx, e_idx} == {11, 12}:
                color, thickness = core_line_color, 2
            elif s_idx in active_set and e_idx in active_set:
                color, thickness = active_line_color, 4
            else:
                color, thickness = inactive_line_color, 2
            self._draw_soft_line(frame, (x1, y1), (x2, y2), color, thickness=thickness)

        for idx in [11, 12, 13, 14, 15, 16]:
            pt = lm[idx]
            if pt.visibility <= self.VIS_THRESHOLD:
                continue
            cx, cy = int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])
            visibility_t = max(0.0, min(1.0, float(pt.visibility)))
            if idx in active_set:
                fill = self._lerp_color((110, 130, 255), active_joint_color, visibility_t)
                self._draw_joint_dot(frame, cx, cy, 4, fill, ring_color=(255, 255, 255), alpha=0.96)
            else:
                fill = self._lerp_color((84, 84, 88), inactive_joint_color, visibility_t)
                self._draw_joint_dot(frame, cx, cy, 3, fill, ring_color=(210, 210, 214), alpha=0.85)

    def _draw_hand_overlay(self, frame, hand_landmarks):
        if hand_landmarks is None:
            return
        hlm = hand_landmarks.landmark
        fingertip_ids = {4, 8, 12, 16, 20}
        palm_ids = {0, 1, 2, 5, 9, 13, 17}
        for s_idx, e_idx in self.mp_hands.HAND_CONNECTIONS:
            p1, p2 = hlm[s_idx], hlm[e_idx]
            x1, y1 = int(p1.x * frame.shape[1]), int(p1.y * frame.shape[0])
            x2, y2 = int(p2.x * frame.shape[1]), int(p2.y * frame.shape[0])
            if s_idx in palm_ids and e_idx in palm_ids:
                line_color, thickness = (205, 205, 210), 1
            else:
                line_color, thickness = (255, 194, 116), 1
            self._draw_soft_line(frame, (x1, y1), (x2, y2), line_color, thickness=thickness, shadow_alpha=0.14)

        for idx, pt in enumerate(hlm):
            cx, cy = int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])
            if idx in fingertip_ids:
                self._draw_joint_dot(frame, cx, cy, 3, (255, 194, 116), ring_color=(255, 255, 255), alpha=0.98)
            elif idx in palm_ids:
                self._draw_joint_dot(frame, cx, cy, 2, (225, 225, 229), ring_color=(255, 255, 255), alpha=0.92)
            else:
                self._draw_joint_dot(frame, cx, cy, 1, (198, 198, 203), ring_color=(240, 240, 244), alpha=0.90)

    def _draw_badge(self, frame, text, x, y, accent_color, align='left'):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        dot_r = 5
        pad_x = 16
        pad_y = 11
        gap = 10
        bw = tw + pad_x * 2 + dot_r * 2 + gap
        bh = max(34, th + pad_y * 2)

        if align == 'right':
            x1 = int(x - bw)
            x2 = int(x)
        else:
            x1 = int(x)
            x2 = int(x + bw)
        y1 = int(y)
        y2 = int(y + bh)

        self._draw_rounded_rect(
            frame, x1, y1, x2, y2,
            self.UI_PANEL_BG,
            alpha=self.UI_PANEL_ALPHA,
            radius=bh // 2,
            border_color=(255, 255, 255),
            border_thickness=1,
        )

        dot_cx = x1 + pad_x + dot_r
        dot_cy = y1 + bh // 2
        cv2.circle(frame, (dot_cx, dot_cy), dot_r, accent_color, -1, cv2.LINE_AA)
        self._put_text_shadowed(frame, text, dot_cx + dot_r + gap, y1 + bh // 2 + th // 2 - 2,
                                scale, self.UI_TEXT_PRIMARY, thickness)

    def _humanize_gesture_overlay(self):
        status = self.gesture_status_text or ''
        clean = status.replace('Gesture:', '').strip()

        if self.calib_mode is not None:
            return 'Calibration in progress', 'Follow the guide in the calibration panel'
        if clean.startswith('COOLDOWN'):
            return 'Please wait a moment', 'System is stabilizing'
        if 'TOO SLOW' in clean:
            return 'Too slow — try again', '주먹 > 손바닥펴기 → 주먹'
        if 'DISABLED DURING CALIBRATION' in clean:
            return 'Calibration in progress', 'Gesture toggle is temporarily disabled'
        if 'START WITH FIST' in clean or clean.startswith('START FIST'):
            return 'Make a fist to start', '주먹 > 손바닥펴기 → 주먹'
        if 'OPEN HAND NOW' in clean or clean.startswith('OPEN '):
            return 'Open your hand', 'Hold it briefly'
        if 'CLOSE FIST NOW' in clean or clean.startswith('FINAL FIST') or clean.startswith('CLOSE FIST'):
            return 'Close your hand', 'Hold the fist briefly'
        if clean == 'AI ON':
            return 'AI control enabled', 'Repeat the gesture to turn it off'
        if clean == 'AI OFF':
            return 'AI control disabled', 'Repeat the gesture to turn it on'
        return clean or 'Ready', 'Keep your hand inside the frame'

    def _draw_minimal_overlay(self, frame, frame_w, frame_h, now):
        arm_label = f"{self.control_arm.upper()} ARM"
        self._draw_badge(frame, arm_label, 16, 16, self.UI_ACCENT_BLUE, align='left')

        ai_text = 'AI ON' if self.ai_enabled else 'AI OFF'
        ai_accent = self.UI_ACCENT_GREEN if self.ai_enabled else self.UI_ACCENT_RED
        self._draw_badge(frame, ai_text, 16, 58, ai_accent, align='left')

        
    def process_frame(self):
        # ── 프레임 소스 결정 (v1 그대로 — 카메라 파이프라인 동결 구간) ──────────
        frame = None
        frame_src_t = -1.0   # [녹화] 원격 프레임 도착 시각 (로컬 웹캠이면 -1)
        with self.remote_frame_lock:
            if (self.remote_frame is not None and
                    time.time() - self.remote_frame_time < self.REMOTE_FRAME_TIMEOUT):
                frame = self.remote_frame.copy()
                frame_src_t = self.remote_frame_time

        if frame is None:
            if not self.cap.isOpened():
                # [녹화 모드] 로컬 웹캠을 열지 않으므로, 브라우저에서 프레임이
                # 올라오기 전에는 여기로 들어온다. 촬영 GUI를 아직 연결하지
                # 않았거나 업로드가 끊긴 상황을 알 수 있도록 주기적으로 알림.
                if self.recorder.enabled:
                    now_w = time.time()
                    if now_w - getattr(self, '_last_nocam_log', 0) > 5.0:
                        self._last_nocam_log = now_w
                        self.get_logger().warn(
                            "[녹화] 아직 프레임이 들어오지 않습니다 — "
                            "촬영 GUI(record.html)에서 '촬영 시작'을 눌렀는지 확인하세요.")
                return
            ret, frame = self.cap.read()
            if not ret:
                return

        if frame.shape[1] != 480 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (480, 480))

        # 좌우 반전(거울 모드): 이 때문에 MediaPipe 핸드니스가 실제와 반대
        frame     = cv2.flip(frame, 1)

        # [학습 데이터 녹화] MediaPipe가 보는 것과 "동일한" 프레임을 저장.
        # 위치가 중요함: 480×480 리사이즈·좌우반전 이후, 오버레이 그리기 이전.
        # (submit은 논블로킹 — 실제 저장은 별도 스레드가 수행)
        self.recorder.submit(frame, src_t=frame_src_t)

        # ── [녹화 모드] 추론 솎아내기 ──────────────────────────────────────
        # 녹화에 필요한 건 원본 프레임뿐이고, Pose/Hands 추론은 화면 표시와
        # 프레임 체크용일 뿐이다. 그런데 빠른 동작 구간에서는 Hands가 추적을
        # 놓쳐 매 프레임 손바닥 검출을 다시 돌리기 때문에 CPU가 포화되고,
        # 그 여파로 업로드 수신 스레드가 밀려 remote_frame이 갱신되지 않는다.
        # → 화면이 멈춘 것처럼 보이고, 같은 프레임이 반복 저장된다.
        #
        # 그래서 녹화 중에는 N프레임에 한 번만 추론한다. 저장은 매 프레임 그대로.
        if self.recorder.enabled:
            self._rec_infer_ctr = getattr(self, '_rec_infer_ctr', 0) + 1
            if self._rec_infer_ctr % self.REC_INFER_EVERY != 0:
                return   # 저장은 이미 끝남. 인식·렌더링만 건너뜀.
        h, w, _   = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)

        # ── [촬영 준비] 프레임 여백 계산 ────────────────────────────────────
        # 정규화 좌표(0~1)에서 각 관절이 화면 경계로부터 얼마나 떨어져 있는지를
        # 잰다. MediaPipe는 화면 밖으로 나간 부위도 좌표를 외삽해서 내놓기 때문에
        # 값이 0 미만이나 1 초과로 나올 수 있는데, 그게 곧 "프레임 이탈" 신호다.
        # 이 값이 학습 데이터 품질을 좌우하므로(화면 밖은 NLF도 복원 못 함)
        # 촬영 전에 반드시 확인해야 한다.
        try:
            if pose_results.pose_landmarks:
                _lm = pose_results.pose_landmarks.landmark
                _sh = 11 if self.control_arm == 'right' else 12
                _el = 13 if self.control_arm == 'right' else 14
                _wr = 15 if self.control_arm == 'right' else 16

                def _margin(p):
                    # 경계까지의 최소 거리(음수면 화면 밖)
                    return min(p.x, 1.0 - p.x, p.y, 1.0 - p.y)

                _st = {
                    'detected': True,
                    't': time.time(),
                    'wrist':    {'x': round(_lm[_wr].x, 4), 'y': round(_lm[_wr].y, 4),
                                 'vis': round(_lm[_wr].visibility, 3),
                                 'margin': round(_margin(_lm[_wr]), 4)},
                    'elbow':    {'x': round(_lm[_el].x, 4), 'y': round(_lm[_el].y, 4),
                                 'vis': round(_lm[_el].visibility, 3),
                                 'margin': round(_margin(_lm[_el]), 4)},
                    'shoulder': {'x': round(_lm[_sh].x, 4), 'y': round(_lm[_sh].y, 4),
                                 'vis': round(_lm[_sh].visibility, 3),
                                 'margin': round(_margin(_lm[_sh]), 4)},
                }
                _st['min_margin'] = round(min(_st['wrist']['margin'],
                                              _st['elbow']['margin']), 4)
                _st['min_vis'] = round(min(_st['wrist']['vis'], _st['elbow']['vis']), 3)
            else:
                _st = {'detected': False, 't': time.time()}
            with self.pose_status_lock:
                self.pose_status_latest = _st
        except Exception:
            pass

        world_lm = None
        if pose_results.pose_world_landmarks:
            world_lm = pose_results.pose_world_landmarks.landmark

        current_time = time.time()
        active_arm   = None

        # ══════════════════════════════════════════════════════════════════
        #  ★ 호밍 중이면 여기서 계산을 통째로 중단 ★
        #
        #  결과를 계산해두고 버리는 게 아니라 아예 계산·저장하지 않음.
        #  이렇게 해야 호밍이 진행되는 동안 목표점·필터·홀드값이 다시
        #  채워지는 경로가 원천적으로 존재하지 않게 됨.
        #  (화면 송출은 계속해야 사용자가 상황을 볼 수 있으므로 렌더링만 유지)
        # ══════════════════════════════════════════════════════════════════
        if self.homing_active:
            if pose_results.pose_landmarks:
                self._draw_pose_overlay(frame, pose_results.pose_landmarks, None)
            self._draw_minimal_overlay(frame, w, h, current_time)
            self._draw_badge(frame, 'HOMING', 16, 100, self.UI_ACCENT_RED, align='left')
            self._emit_frame(frame)
            return

        # ── STEP 1: 사용할 팔 결정 (GUI 선택값 고정 사용) ────────────────────
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            sh_idx = 11 if self.control_arm == 'right' else 12
            wr_idx = 15 if self.control_arm == 'right' else 16

            if (lm[sh_idx].visibility > self.VIS_THRESHOLD and
                    lm[wr_idx].visibility > self.VIS_THRESHOLD):
                active_arm = self.control_arm

        # ══════════════════════════════════════════════════════════════════
        #  STEP 2: 사람 손목 위치를 무차원 좌표로 변환
        #
        #  v1은 여기서 어깨각/팔꿈치각(스칼라 2개)을 뽑았지만, v2는
        #  "어깨를 원점으로 한 손목의 3D 위치"를 그대로 씀.
        #      v = (손목 - 어깨) / 팔길이
        #  팔길이로 나누기 때문에 체격·카메라 거리와 무관한 값이 되고,
        #  팔을 곧게 폈을 때 |v| ≈ 1 이 되어 해석이 직관적임.
        #
        #  깊이(앞뒤)는 이미지 z가 아니라 pose_world_landmarks의 z를 씀 —
        #  v1의 J2/J3가 이 좌표계로 이미 잘 동작했으므로 검증된 신호원임.
        # ══════════════════════════════════════════════════════════════════
        norm_fwd = norm_lat = norm_up = None
        pos_valid = False
        raw_arm_len = None      # 이번 프레임에서 관측된 팔 길이(캘리브 수집용)

        if active_arm is not None and world_lm is not None:
            lm = pose_results.pose_landmarks.landmark
            sh_idx = 11 if active_arm == 'right' else 12
            el_idx = 13 if active_arm == 'right' else 14
            wr_idx = 15 if active_arm == 'right' else 16

            sh_w = world_lm[sh_idx]
            el_w = world_lm[el_idx]
            wr_w = world_lm[wr_idx]

            # 좌표를 튜플로 고정 — 이후 보정 결과로 교체될 수 있으므로
            # 랜드마크 객체가 아니라 값으로 다룬다.
            sh_p = (sh_w.x, sh_w.y, sh_w.z)
            el_p = (el_w.x, el_w.y, el_w.z)
            wr_p = (wr_w.x, wr_w.y, wr_w.z)

            # ── GRU 보정 ─────────────────────────────────────────────────
            # 위치: MediaPipe world 좌표를 얻은 직후, 팔 길이·정규화 계산 전.
            #   여기서 보정해야 이후의 모든 단계(정규화, 캘리브레이션 샘플,
            #   IK 목표점)가 일관되게 보정된 좌표를 쓴다.
            # 워밍업 중이거나 모델이 없으면 원본이 그대로 반환된다.
            if self.corrector is not None:
                try:
                    # 학습 시에는 세션당 하나의 고정된 팔 길이로 정규화했다.
                    # 배포에서도 같은 조건을 만들어야 하므로, 캘리브레이션에서
                    # 측정한 고정값을 쓴다. 프레임마다 바뀌는 중앙값을 쓰면
                    # 입력 스케일이 흔들려 보정량이 들쭉날쭉해진다.
                    if self.CAL_ARM_LEN >= self.MIN_ARM_LEN_M:
                        self.corrector.set_arm_len(self.CAL_ARM_LEN)
                    elif len(self.arm_len_buf) >= 5:
                        self.corrector.set_arm_len(
                            sorted(self.arm_len_buf)[len(self.arm_len_buf) // 2])
                    el_c, wr_c = self.corrector.correct(sh_p, el_p, wr_p)
                    el_raw, wr_raw = el_p, wr_p          # 진단용 원본 보관
                    el_p = (float(el_c[0]), float(el_c[1]), float(el_c[2]))
                    wr_p = (float(wr_c[0]), float(wr_c[1]), float(wr_c[2]))

                    # ── [진단] 보정 전후 비교 ────────────────────────────
                    # 같은 프레임의 원본과 보정값을 나란히 찍어, 보정이 실제로
                    # 적용되고 있는지 눈으로 확인한다. 워밍업 중에는 두 값이
                    # 같아야 정상이다(설계상 원본을 그대로 통과시킴).
                    if self.GRU_DEBUG_LOG:
                        if current_time - getattr(self, '_gru_dbg_t', 0.0) >= \
                                self.GRU_DEBUG_PERIOD_SEC:
                            self._gru_dbg_t = current_time
                            dw = math.sqrt(sum((wr_p[i]-wr_raw[i])**2 for i in range(3)))
                            de = math.sqrt(sum((el_p[i]-el_raw[i])**2 for i in range(3)))
                            warm = self.corrector.rt.warm
                            wu = self.corrector.warmup
                            self.get_logger().info(
                                f"[GRU 진단] 손목z {wr_raw[2]*1000:+7.1f} → {wr_p[2]*1000:+7.1f}mm "
                                f"(Δ손목 {dw*1000:5.1f}mm, Δ팔꿈치 {de*1000:5.1f}mm) "
                                f"팔길이 {self.corrector.arm_len:.3f}m"
                                f"{'(고정)' if self.CAL_ARM_LEN >= self.MIN_ARM_LEN_M else '(추정)'} "
                                f"{'[워밍업 '+str(warm)+'/'+str(wu)+']' if warm < wu else ''}")
                except Exception as e:
                    # 보정 실패가 제어를 멈추게 해서는 안 된다.
                    if not getattr(self, '_gru_err_logged', False):
                        self._gru_err_logged = True
                        self.get_logger().warn(f"[GRU 보정] 실행 오류 — 이후 비활성화: {e}")
                    self.corrector = None

            ua = math.sqrt((el_p[0] - sh_p[0]) ** 2 + (el_p[1] - sh_p[1]) ** 2 + (el_p[2] - sh_p[2]) ** 2)
            fa = math.sqrt((wr_p[0] - el_p[0]) ** 2 + (wr_p[1] - el_p[1]) ** 2 + (wr_p[2] - el_p[2]) ** 2)

            if (lm[el_idx].visibility > self.VIS_THRESHOLD and
                    ua >= self.MIN_UPPERARM_LEN_M and fa >= self.MIN_FOREARM_LEN_J3_M):
                # 팔 길이는 프레임마다 흔들리므로 중앙값으로 안정화.
                # (분모라서 한 프레임만 튀어도 목표점 전체가 크게 흔들림)
                self.arm_len_buf.append(ua + fa)
                raw_arm_len = sorted(self.arm_len_buf)[len(self.arm_len_buf) // 2]
                # 캘리브레이션에서 측정한 고정값이 있으면 그것을 쓴다.
                # 없을 때만(캘리브 전) 실시간 중앙값으로 대체한다.
                if self.CAL_ARM_LEN >= self.MIN_ARM_LEN_M:
                    arm_len = self.CAL_ARM_LEN
                else:
                    arm_len = sorted(self.arm_len_buf)[len(self.arm_len_buf) // 2]

                if arm_len >= self.MIN_ARM_LEN_M:
                    vx = (wr_p[0] - sh_p[0]) / arm_len
                    vy = (wr_p[1] - sh_p[1]) / arm_len
                    vz = (wr_p[2] - sh_p[2]) / arm_len

                    # MediaPipe world 좌표계: y는 아래로 +, z는 카메라에서 멀수록 +
                    #  → 팔을 카메라 쪽으로 뻗으면 z가 감소하므로 -vz가 "앞으로"
                    #  → 손을 위로 올리면 y가 감소하므로 -vy가 "위로"
                    norm_fwd = self.SIGN_FWD * (-vz)
                    norm_lat = self.SIGN_LAT * (vx)
                    norm_up  = self.SIGN_UP  * (-vy)
                    pos_valid = True

            # ── 캘리브레이션 샘플 수집 (1~3단계: 위치) ──────────────────────
            # AI ON/OFF와 무관하게 수집. 단, 버튼을 누른 직후 준비 시간 동안은
            # 자세를 잡는 과도기라 샘플을 모으지 않음.
            if (pos_valid and self.calib_mode in ('neutral', 'reach_forward', 'reach_up')
                    and current_time >= self.calib_prep_until):
                self.calib_samples_pos.append((norm_fwd, norm_lat, norm_up))
                # 팔 길이는 중립 자세에서만 모은다. 뻗거나 들어올린 자세는
                # 투영 단축 때문에 실제보다 짧게 측정되어 기준값으로 부적절하다.
                if self.calib_mode == 'neutral' and raw_arm_len is not None:
                    self.calib_samples_arm.append(raw_arm_len)
                if len(self.calib_samples_pos) >= self.CALIB_REQUIRED_SAMPLES:
                    self._finish_calib_stage()

            if self.calib_mode is not None:
                self._publish_calib_status()

        # [검수 수정] v1은 pose가 잡히기만 하면 스켈레톤을 그렸는데, 처음 조립본은
        # "제어팔이 유효할 때"만 그리도록 한 단계 깊게 들어가 있었음. 그러면 팔이
        # 잠깐 가려질 때 화면에서 뼈대가 통째로 사라져 사용자가 "시스템이 죽었나"
        # 하고 오해하게 됨 → v1과 같은 레벨로 복원.
        if pose_results.pose_landmarks:
            self._draw_pose_overlay(frame, pose_results.pose_landmarks, active_arm)

        # ══════════════════════════════════════════════════════════════════
        #  STEP 3: 활성 팔 쪽 손 매칭 → 툴 피치(J5) / J6 / 그리퍼
        #
        #  핸드니스 라벨("Left"/"Right")은 flip된 영상에서 신뢰할 수 없으므로
        #  Pose 손목과 Hand 손목(landmark[0])의 공간적 근접도로 매칭함 (v1 그대로).
        # ══════════════════════════════════════════════════════════════════
        MAX_HAND_MATCH_DIST = 0.15

        raw_tool_pitch   = None
        raw_j6           = None
        proposed_gripper = None
        matched_hand     = None

        if hand_results.multi_hand_landmarks and active_arm is not None:
            lm      = pose_results.pose_landmarks.landmark
            wr_idx  = 15 if active_arm == 'right' else 16
            wr_pose = lm[wr_idx]

            best_dist = float('inf')
            matched_hand_idx = -1
            for hand_i, hlm_item in enumerate(hand_results.multi_hand_landmarks):
                hw = hlm_item.landmark[0]
                dist = math.sqrt((hw.x - wr_pose.x) ** 2 + (hw.y - wr_pose.y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    matched_hand = hlm_item
                    matched_hand_idx = hand_i

            if best_dist > MAX_HAND_MATCH_DIST:
                matched_hand = None
                matched_hand_idx = -1

            if matched_hand is not None:
                hlm = matched_hand.landmark

                # ── 같은 손의 world 랜드마크 (palm3d 전용) ────────────────
                # multi_hand_world_landmarks 는 multi_hand_landmarks 와 같은
                # 순서로 오므로 인덱스를 그대로 쓴다. 손을 두 개까지 잡으므로
                # (max_num_hands=2) 인덱스를 맞추지 않으면 반대 손의 자세로
                # J5를 계산하게 된다.
                # 구버전 MediaPipe에는 이 속성이 아예 없을 수 있어 getattr로 접근한다.
                hand_world_lm = None
                _hwl = getattr(hand_results, 'multi_hand_world_landmarks', None)
                if _hwl is not None and 0 <= matched_hand_idx < len(_hwl):
                    hand_world_lm = _hwl[matched_hand_idx].landmark

                # 선택된 제어 손만 AI ON/OFF 제스처 판정에 사용
                self._update_ai_toggle_gesture(hlm, current_time)

                el_idx = 13 if active_arm == 'right' else 14
                el_lm  = lm[el_idx]
                wr_lm  = lm[wr_idx]

                forearm_dx = wr_lm.x - el_lm.x
                forearm_dy = wr_lm.y - el_lm.y
                forearm_mag = math.sqrt(forearm_dx ** 2 + forearm_dy ** 2)

                h_wrist_j5 = hlm[0]
                bend3d = None          # palm3d 결과 (부호 포함, 도)

                if self.J5_HAND_VECTOR == 'palm3d':
                    # ── 3D 손바닥 기준 (권장) ─────────────────────────────
                    # world 랜드마크가 없으면(구버전 MediaPipe, 또는 그 프레임에
                    # 미제공) 계산을 포기하고 직전 값을 홀드한다.
                    #
                    # [의도적으로 2D로 폴백하지 않는다]
                    #   2D와 3D는 각도의 '정의'가 다르고, 캘리브레이션은 둘 중
                    #   하나에 대해서만 맞춰져 있다. 프레임 단위로 오가면 전환
                    #   지점마다 그리퍼가 튄다. 홀드가 훨씬 안전하다.
                    geo = self._j5_palm3d_geometry(world_lm, hand_world_lm, active_arm)
                    if geo is not None:
                        e_mag, f_signed = geo

                        # ── 부호 확정 상태 기계 ───────────────────────────
                        # f_signed 가 None 이면(전완이 회전축과 거의 나란해
                        # 방향이 정의되지 않음) 부호를 갱신하지 않고 유지한다.
                        if f_signed is not None and \
                                abs(f_signed) >= self.J5_SIGN_DEADBAND_DEG:
                            new_sign = 1.0 if f_signed >= 0.0 else -1.0
                            if self.j5_sign is None:
                                # 첫 관측은 확정 없이 바로 채택한다. 여기서
                                # 기다리면 리셋 직후 몇 프레임 동안 부호가
                                # 없어 크기를 쓸 수 없다.
                                self.j5_sign = new_sign
                                self.j5_sign_cand = None
                                self.j5_sign_cnt = 0
                            elif new_sign != self.j5_sign:
                                if self.j5_sign_cand == new_sign:
                                    self.j5_sign_cnt += 1
                                else:
                                    self.j5_sign_cand = new_sign
                                    self.j5_sign_cnt = 1
                                if self.j5_sign_cnt >= self.J5_SIGN_CONFIRM_FRAMES:
                                    self.j5_sign = new_sign
                                    self.j5_sign_cand = None
                                    self.j5_sign_cnt = 0
                            else:
                                # 현재 부호와 같은 관측 → 후보 누적을 되돌린다.
                                self.j5_sign_cand = None
                                self.j5_sign_cnt = 0

                        if self.j5_sign is not None:
                            bend3d = e_mag * self.j5_sign

                    hand_ok = bend3d is not None
                    forearm_ok = True      # 2D 전완 벡터를 쓰지 않음

                elif self.J5_HAND_VECTOR == 'palm':
                    # ── 손등 기준 (권장) ──────────────────────────────────
                    # 손목(0)과 너클선 양끝(검지 MCP 5, 새끼 MCP 17)은 모두
                    # 손등 표면의 점이라 주먹을 쥐거나 손목을 꺾어도 가려지지
                    # 않는다. 벡터 길이가 손바닥 크기로 일정해 잘 죽지 않는다.
                    palm_dx = (hlm[5].x + hlm[17].x) * 0.5 - h_wrist_j5.x
                    palm_dy = (hlm[5].y + hlm[17].y) * 0.5 - h_wrist_j5.y
                    palm_mag = math.sqrt(palm_dx ** 2 + palm_dy ** 2)
                    hand_ok = palm_mag >= self.MIN_PALM_VEC_J5
                    forearm_ok = True          # 전완을 각도에 쓰지 않음
                else:
                    # ── 구 방식: 전완 × (손목→중지끝) ─────────────────────
                    h_mid_tip  = hlm[12]
                    hand_dx = h_mid_tip.x - h_wrist_j5.x
                    hand_dy = h_mid_tip.y - h_wrist_j5.y
                    hand_mag = math.sqrt(hand_dx ** 2 + hand_dy ** 2)
                    # [안정성 가드] 팔을 카메라 쪽으로 뻗으면 전완/손 벡터가
                    # 짧아지고, 0에 가까워질수록 atan2가 불안정해져 노이즈로 튐.
                    forearm_ok = forearm_mag >= self.MIN_FOREARM_VEC_J5
                    hand_ok    = hand_mag    >= self.MIN_HAND_VEC_J5

                if forearm_ok and hand_ok:
                    if self.J5_HAND_VECTOR == 'palm3d':
                        # 위에서 이미 부호까지 결정된 값. unwrap 불필요 —
                        # 3D 사잇각은 0~180°로 닫혀 있어 ±180° 경계를 넘지 않는다.
                        signed_bend_deg = bend3d
                    elif self.J5_HAND_VECTOR == 'palm':
                        # ── 기준축 결정 ──────────────────────────────────
                        # 팔 자세가 바뀌어도 "팔에 대한 손의 상대 각도"가
                        # 유지되도록 어깨→손목 방향을 기준으로 삼는다.
                        # 이 벡터가 짧아지면(팔을 카메라 쪽으로 뻗은 상태)
                        # 각도가 불안정해지므로 마지막 유효 방향을 재사용한다.
                        ref_x, ref_y = 0.0, 1.0        # 기본: 화면 수직축
                        if self.J5_REF_AXIS == 'hybrid':
                            sh_idx_j5 = 11 if active_arm == 'right' else 12
                            sh_lm_j5 = lm[sh_idx_j5]
                            rx = wr_lm.x - sh_lm_j5.x
                            ry = wr_lm.y - sh_lm_j5.y
                            rmag = math.sqrt(rx * rx + ry * ry)
                            if rmag >= self.MIN_REF_VEC_J5:
                                self.j5_ref_dir = (rx / rmag, ry / rmag)
                            if self.j5_ref_dir is not None:
                                ref_x, ref_y = self.j5_ref_dir

                        raw_angle = math.degrees(math.atan2(
                            ref_x * palm_dy - ref_y * palm_dx,
                            ref_x * palm_dx + ref_y * palm_dy))

                        # ±180° 경계를 넘을 때 값이 360° 튀므로 이어붙인다.
                        # 단, 직전 유효 프레임과 너무 벌어져 있으면 그 사이의
                        # 손목 회전을 알 수 없으므로 새로 시작한다.
                        gap = current_time - self.j5_unwrap_time
                        if (self.j5_unwrap_prev is None or
                                gap > self.J5_UNWRAP_MAX_GAP_SEC):
                            self.j5_unwrap_turns = 0
                        else:
                            d = raw_angle - self.j5_unwrap_prev
                            if d > self.J5_UNWRAP_MAX_JUMP:
                                self.j5_unwrap_turns -= 1
                            elif d < -self.J5_UNWRAP_MAX_JUMP:
                                self.j5_unwrap_turns += 1
                            self.j5_unwrap_turns = max(
                                -self.J5_UNWRAP_MAX_TURNS,
                                min(self.J5_UNWRAP_MAX_TURNS, self.j5_unwrap_turns))
                        self.j5_unwrap_prev = raw_angle
                        self.j5_unwrap_time = current_time
                        signed_bend_deg = raw_angle + 360.0 * self.j5_unwrap_turns
                    else:
                        cross = forearm_dx * hand_dy - forearm_dy * hand_dx
                        dot   = forearm_dx * hand_dx + forearm_dy * hand_dy
                        signed_bend_deg = math.degrees(math.atan2(cross, dot))
                    effective_bend = signed_bend_deg * self.J5_BACK_BEND_SIGN

                    # [안정성 가드] 오래 끊겼다 재개되면 One Euro가 그 시간차를
                    # "매우 빠른 움직임"으로 오인해 컷오프를 열어버리므로 필터 리셋.
                    if current_time - self.j5_last_valid_time > self.J5_STALE_GAP_SEC:
                        self.f_j5.reset()
                        self.m_j5.reset()
                        # 오래 끊긴 뒤에는 그 사이 손목이 얼마나 돌았는지,
                        # 팔 자세가 얼마나 바뀌었는지 알 수 없으므로 모두 버린다.
                        self.j5_unwrap_prev = None
                        self.j5_unwrap_turns = 0
                        self.j5_unwrap_time = 0.0
                        self.j5_ref_dir = None
                        # palm3d 부호도 같은 이유로 버린다. 오래 끊긴 사이에
                        # 손목이 반대로 꺾였을 수 있는데, 낡은 부호를 유지하면
                        # 재개 첫 프레임에서 반대 방향 툴 피치가 나간다.
                        #
                        # [주의] 이 블록은 signed_bend_deg 를 계산한 뒤에 실행되므로
                        # 이번 프레임 값은 이미 낡은 부호로 만들어져 있다. 그래서
                        # 아래에서 이번 프레임 결과도 함께 버린다.
                        self.j5_sign = None
                        self.j5_sign_cand = None
                        self.j5_sign_cnt = 0
                        stale_resumed = True
                    else:
                        stale_resumed = False
                    self.j5_last_valid_time = current_time

                    # [중요] 오래 끊겼다 재개된 첫 프레임의 값은 버린다.
                    #   위 리셋 블록은 signed_bend_deg 를 만든 '뒤'에 실행되므로,
                    #   이번 프레임 값은 이미 낡은 상태(palm3d의 j5_sign,
                    #   palm의 j5_ref_dir)로 계산돼 있다. 그대로 쓰면 리셋을
                    #   해놓고도 낡은 값이 한 번 발행되어 그리퍼가 튄다.
                    #   한 프레임 건너뛰어도 tool_pitch_deg 는 직전 값을 유지하므로
                    #   사용자 체감은 없다(0.05초).
                    if not stale_resumed:
                        # ── 캘리브레이션 샘플 수집 (4~6단계: 손목) ──────────
                        if (self.calib_mode in ('wrist_straight', 'wrist_back', 'wrist_front') and
                                current_time >= self.calib_prep_until):
                            self.calib_samples_wrist.append(effective_bend)
                            if len(self.calib_samples_wrist) >= self.CALIB_REQUIRED_SAMPLES:
                                self._finish_calib_stage()

                        # ── 3점 보간: 안쪽(FRONT) ↔ 일자(STRAIGHT) ↔ 뒤(BACK) ─
                        # [v2 변경] 보간 결과가 이제 "관절각 J5"가 아니라
                        # "월드 기준 목표 툴 피치 θ"임. 실제 J5는 _solve_j5()에서
                        # θ - (J2+J3) 로 계산됨 (팔 자세와 무관하게 각도 유지).
                        if effective_bend >= self.WRIST_ANGLE_STRAIGHT:
                            raw_tool_pitch = self._lerp_clamp(
                                effective_bend,
                                self.WRIST_ANGLE_STRAIGHT, self.WRIST_ANGLE_BACK,
                                self.TOOL_PITCH_STRAIGHT,   self.TOOL_PITCH_BACK)
                        else:
                            raw_tool_pitch = self._lerp_clamp(
                                effective_bend,
                                self.WRIST_ANGLE_FRONT,   self.WRIST_ANGLE_STRAIGHT,
                                self.TOOL_PITCH_FRONT,    self.TOOL_PITCH_STRAIGHT)

                # ── J6 (툴 롤): 손등 너클 라인 기준. 기본 비활성 ──────────────
                if self.ENABLE_J6:
                    k_dx = hlm[17].x - hlm[5].x     # 검지 MCP → 새끼 MCP
                    k_dy = hlm[17].y - hlm[5].y
                    k_mag = math.sqrt(k_dx ** 2 + k_dy ** 2)
                    # 손이 카메라 정면을 향하면 이 벡터가 짧아지면서 각도가
                    # 불안정해짐(J5에서 겪은 foreshortening과 같은 패턴)
                    # → 임계 이하이면 계산하지 않고 직전 값을 유지함.
                    if k_mag >= self.MIN_KNUCKLE_VEC:
                        roll = math.degrees(math.atan2(k_dy, k_dx)) * self.J6_SIGN
                        raw_j6 = max(-self.J6_MAX_DEG, min(self.J6_MAX_DEG, roll))

                # ── 그리퍼: 엄지끝↔검지끝 핀치 비율 (v1 그대로 유지) ──────────
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

                self._draw_hand_overlay(frame, matched_hand)

        # 선택 손이 이번 프레임에 매칭되지 않았어도 토글 상태 머신은 계속 진행
        if matched_hand is None:
            self._update_ai_toggle_gesture(None, current_time)

        # 손목/툴 축 값 갱신 (안 잡히면 직전 값 유지 — 0으로 되돌리면 급이동함)
        if raw_tool_pitch is not None:
            self.tool_pitch_deg = self.f_j5.filter(
                self.m_j5.filter(raw_tool_pitch), current_time)
        if raw_j6 is not None:
            self.j6_deg = self.f_j6.filter(self.m_j6.filter(raw_j6), current_time)

        # ══════════════════════════════════════════════════════════════════
        #  STEP 4: 사람 좌표 → 로봇 좌표 → 클램프 → IK → 발행
        #
        #  처리 순서가 중요함:
        #    필터 → 속도제한 → 작업공간 클램프 → IK
        #  클램프를 IK "직전"에 두는 이유는, 필터나 속도제한이 목표점을
        #  아주 살짝 작업공간 밖으로 밀어낼 수 있기 때문. IK에 들어가는
        #  값은 반드시 도달 가능한 값이어야 arccos가 발산하지 않음.
        # ══════════════════════════════════════════════════════════════════
        if pos_valid:
            # 사람 정규화 좌표 → 로봇 좌표(mm). 캘리브레이션으로 얻은
            # "이 사람이 편하게 움직이는 실제 범위"를 로봇 작업공간에 사상.
            fwd_span = max(self.CAL_FWD_MAX - self.CAL_FWD_NEUTRAL, 0.10)
            up_span  = max(self.CAL_UP_MAX  - self.CAL_UP_NEUTRAL,  0.10)
            g_fwd = (self.MAP_X_MAX - self.MAP_X_NEUTRAL) / fwd_span

            # ── 반경(전방)과 방위각(좌우)을 분리해서 계산 ──────────────────
            # 반경은 앞뒤 동작만으로 정하고, 좌우는 그 반경을 유지한 채
            # 회전만 시킨다. 이렇게 해야 옆으로 벌려도 구각 제한에 안 걸린다.
            # 앞/뒤 게인 분리 (z 와 같은 구조)
            d_fwd = norm_fwd - self.CAL_FWD_NEUTRAL
            if d_fwd >= 0.0:
                g_f = g_fwd
            else:
                back_span = max(fwd_span * self.MAP_BACK_SPAN_SCALE, 0.10)
                g_f = (self.MAP_X_NEUTRAL - self.MAP_X_MIN) / back_span
            r_fwd = self.MAP_X_NEUTRAL + g_f * d_fwd

            j1_cmd = self.MAP_J1_GAIN_DEG * (norm_lat - self.CAL_LAT_NEUTRAL)
            # x >= WS_X_MIN_MM 을 유지할 수 있는 각도 상한. 반경이 짧을수록
            # 더 일찍 걸리므로 반경에 따라 자동으로 좁힌다. 이걸 안 하면
            # 큰 각도에서 x 클램프가 발동해 방위각이 찌그러진다.
            j1_lim = self.MAP_J1_MAX_DEG
            if r_fwd > self.WS_X_MIN_MM:
                j1_lim = min(j1_lim,
                             math.degrees(math.acos(self.WS_X_MIN_MM / r_fwd)))
            else:
                j1_lim = 0.0
            # 부호 대칭 클램프 — 좌우가 정확히 1:1 이 되도록
            j1_cmd = max(-j1_lim, min(j1_lim, j1_cmd))

            raw_x = r_fwd * math.cos(math.radians(j1_cmd))
            raw_y = r_fwd * math.sin(math.radians(j1_cmd))

            # ── z: 중립을 기준으로 위/아래 게인을 따로 쓴다 ────────────────
            # 중립이 로봇 작업공간의 높은 쪽(255mm)에 있어서, 하나의 기울기로는
            # 위아래를 모두 커버할 수 없다. 상수 섹션의 MAP_Z_MIN 주석 참고.
            #
            # 두 구간이 중립점에서 같은 값(MAP_Z_NEUTRAL)을 갖도록 이어붙였으므로
            # 경계를 지날 때 값이 튀지 않는다. 기울기만 꺾인다.
            d_up = norm_up - self.CAL_UP_NEUTRAL
            if d_up >= 0.0:
                g_up = (self.MAP_Z_MAX - self.MAP_Z_NEUTRAL) / up_span
            else:
                # 아래쪽 사람 가동범위는 실측값이 없으므로 위쪽에 배율을 곱해 추정
                down_span = max(up_span * self.MAP_DOWN_SPAN_SCALE, 0.10)
                g_up = (self.MAP_Z_NEUTRAL - self.MAP_Z_MIN) / down_span
            raw_z = self.MAP_Z_NEUTRAL + g_up * d_up
            if self.GRU_DEBUG_LOG:
                if current_time - getattr(self, '_map_dbg_t', 0.0) >= 1.0:
                    self._map_dbg_t = current_time
                    self.get_logger().info(
                        f"[매핑] fwd {norm_fwd:+.3f} "
                        f"(중립{self.CAL_FWD_NEUTRAL:+.3f} 최대{self.CAL_FWD_MAX:+.3f}) "
                        f"→ x {r_fwd:5.0f}   "
                        f"up {norm_up:+.3f} "
                        f"(중립{self.CAL_UP_NEUTRAL:+.3f} 최대{self.CAL_UP_MAX:+.3f}) "
                        f"→ z {raw_z:5.0f}   "
                        f"lat {norm_lat:+.3f} → J1 {j1_cmd:+.0f}°")
            # 카테시안 필터 (중앙값 사전필터 → One Euro)
            fx = self.f_cx.filter(self.m_cx.filter(raw_x), current_time)
            fy = self.f_cy.filter(self.m_cy.filter(raw_y), current_time)
            fz = self.f_cz.filter(self.m_cz.filter(raw_z), current_time)

            # 직교공간 속도 제한.
            # [주의] v1에서 제거한 관절 변화량 제한(MAX_DELTA_J*)의 부활이 아님.
            # 그건 관절 공간에서 잘라내서 큰 동작을 계단식으로 쪼갰지만,
            # 이건 목표점의 이동 속도만 제한하므로 궤적 모양은 그대로 유지됨.
            if self.cart_initialized and self.tgt_x is not None:
                dt = max(current_time - self.last_cart_time, 1e-3)
                max_step = self.MAX_CART_SPEED_MM_S * dt
                dx, dy, dz = fx - self.tgt_x, fy - self.tgt_y, fz - self.tgt_z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist > max_step and dist > 1e-6:
                    s = max_step / dist
                    fx = self.tgt_x + dx * s
                    fy = self.tgt_y + dy * s
                    fz = self.tgt_z + dz * s

            # 작업공간 클램프 (IK 직전)
            cx, cy, cz, was_clamped = self._clamp_to_workspace(fx, fy, fz)

            self.tgt_x, self.tgt_y, self.tgt_z = cx, cy, cz
            self.last_cart_time = current_time
            self.cart_initialized = True
            self.last_track_time = current_time

            # 역기구학
            j1, j2, j3, ik_ok = self._ik_position(cx, cy, cz)
            j5, j5_sat = self._solve_j5(self.tool_pitch_deg, j2, j3)
            j6 = self.j6_deg if self.ENABLE_J6 else 0.0

            self.last_ik = (j1, j2, j3)
            self.last_ik_ok = ik_ok and not was_clamped
            if not ik_ok:
                self.ik_status_text = 'JOINT LIMIT'
            elif was_clamped:
                self.ik_status_text = 'WORKSPACE EDGE'
            elif j5_sat:
                self.ik_status_text = 'WRIST LIMIT'
            else:
                self.ik_status_text = ''

            # ── 리셋 직후 첫 유효 프레임: 기준점만 잡고 명령은 내리지 않음 ──
            # (호밍/AI ON/캘리브 완료 직후에 로봇이 사용자의 현재 손 위치로
            #  급이동하는 것을 막는 마지막 방어선)
            if self.first_frame_after_reset:
                self.first_frame_after_reset = False
                self.last_published_cart = (cx, cy, cz)
                self.get_logger().info(
                    f"기준점 캡처: target=({cx:.0f}, {cy:.0f}, {cz:.0f})mm "
                    f"→ J1={j1:.1f} J2={j2:.1f} J3={j3:.1f}")
            else:
                # ── 발행 게이트 ────────────────────────────────────────────
                publish_allowed = (
                    self.ai_enabled and
                    not self.homing_active and
                    current_time >= self.homing_lockout_until and
                    not self.gesture_suspend_commands and
                    current_time >= self.ai_publish_block_until and
                    # [검수 수정] calib_mode(측정 중)뿐 아니라 calib_session_active
                    # (단계 사이 대기 중)에도 발행을 막음. 단계 사이는 사용자가 다음
                    # 자세로 팔을 옮기는 시간인데, 이때 AI가 켜져 있으면 로봇이 그
                    # 이동을 그대로 쫓아가는 문제가 있었음 (v1부터 있던 빈틈).
                    self.calib_mode is None and
                    not self.calib_session_active
                )

                if publish_allowed and (current_time - self.last_joint_pub_time >= self.PUB_INTERVAL):
                    # 데드밴드는 관절이 아니라 목표점(mm) 기준으로 판정함.
                    # 특이점 근처에서는 손끝 1mm 이동이 관절 20° 변화가 될 수
                    # 있어서, 관절 기준 데드밴드로는 미세 떨림과 큰 변화를
                    # 구분할 수 없기 때문.
                    moved = True
                    if self.last_published_cart is not None:
                        px, py, pz = self.last_published_cart
                        moved = (math.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2)
                                 >= self.DEADBAND_CART_MM)

                    j5_changed = abs(j5 - self.last_published_joints[4]) >= self.DEADBAND_J5
                    j6_changed = (self.ENABLE_J6 and
                                  abs(j6 - self.last_published_joints[5]) >= self.DEADBAND_J6)

                    if moved or j5_changed or j6_changed:
                        out = [j1, j2, j3, 0.0, j5, j6]
                        # 최종 안전망: 소프트 리밋으로 한 번 더 클램프.
                        # (robot_node는 범위를 벗어나면 6축 명령 전체를 무시하므로,
                        #  여기서 미리 잘라야 명령이 통째로 사라지는 일이 없음)
                        for i, (lo, hi) in enumerate(self.JOINT_SOFT_LIMITS):
                            out[i] = max(lo, min(hi, out[i]))

                        self.last_published_joints = list(out)
                        self.last_published_cart = (cx, cy, cz)

                        joint_msg = Float32MultiArray()
                        joint_msg.data = [float(v) for v in out]
                        self.ai_joints_pub.publish(joint_msg)
                        self.last_joint_pub_time = current_time

                        # [지연 계측] 이 명령이 어느 프레임에서 비롯됐고,
                        # 그 프레임이 도착한 지 얼마나 지났는지 기록.
                        # frame_src_t < 0 이면 로컬 웹캠 소스라 측정 대상이 아님.
                        if self.LAT_LOG_ENABLE and frame_src_t > 0:
                            self._record_latency(current_time - frame_src_t, current_time)

                # ── 그리퍼 발행 ────────────────────────────────────────────
                if (proposed_gripper is not None and publish_allowed and
                        current_time - self.last_gripper_pub_time >= self.PUB_INTERVAL):
                    if abs(proposed_gripper - self.last_published_gripper) >= self.DEADBAND_GRIPPER:
                        self.last_published_gripper = proposed_gripper
                        grip_msg = Float32()
                        grip_msg.data = float(proposed_gripper)
                        self.gripper_pub.publish(grip_msg)
                        self.last_gripper_pub_time = current_time

        else:
            # 추적이 끊긴 경우: 마지막 유효 목표를 유지하되 새 명령은 발행하지 않음.
            # 0을 보내면 로봇이 원점으로 급이동하므로 절대 금지.
            if (self.last_track_time > 0.0 and
                    current_time - self.last_track_time > self.TRACK_LOST_HOLD_SEC):
                self.ik_status_text = 'TRACKING LOST'

        # ── STEP 5: 오버레이 렌더링 ───────────────────────────────────────
        self._draw_minimal_overlay(frame, w, h, current_time)
        self._draw_ik_overlay(frame, current_time)

        # ── STEP 6/7: ROS 이미지 + MJPEG 송출 ─────────────────────────────
        self._emit_frame(frame)

    def _draw_ik_overlay(self, frame, now):
        """목표 좌표(mm)와 IK 결과를 화면에 표시.

        SIGN_FWD / SIGN_LAT / SIGN_UP 을 맞출 때 이 숫자를 보면서 조정하면 됨:
        손을 앞으로 뻗었는데 X가 줄어들면 SIGN_FWD를 -1.0으로 뒤집는 식.
        """
        if self.tgt_x is None:
            return
        txt = f"T {self.tgt_x:.0f},{self.tgt_y:.0f},{self.tgt_z:.0f}mm"
        self._draw_badge(frame, txt, 16, 100, self.UI_ACCENT_BLUE, align='left')

        if self.last_ik is not None:
            j1, j2, j3 = self.last_ik
            jtxt = f"J {j1:.0f} {j2:.0f} {j3:.0f} / J5 {self.tool_pitch_deg:.0f}"
            color = self.UI_ACCENT_GREEN if self.last_ik_ok else self.UI_ACCENT_RED
            self._draw_badge(frame, jtxt, 16, 142, color, align='left')

        if self.ik_status_text:
            self._draw_badge(frame, self.ik_status_text, 16, 184,
                             self.UI_ACCENT_RED, align='left')

    def _emit_frame(self, frame):
        """카메라 이미지 ROS 토픽 발행 + 웹 GUI용 JPEG 인코딩 (v1 그대로)."""
        try:
            ros_img = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_pub.publish(ros_img)
        except Exception:
            pass

        try:
            ok_enc, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok_enc:
                with self.jpeg_lock:
                    self.latest_jpeg = jpeg_buf.tobytes()
        except Exception:
            pass
    def destroy_node(self):
        # [학습 데이터 녹화] 큐에 남은 프레임을 모두 저장하고 타임스탬프·메타 확정
        self.recorder.close()
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