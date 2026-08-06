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
from collections import deque
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
    WS_Z_MAX_MM = 290.0

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
    CAL_FWD_NEUTRAL = 0.45
    CAL_FWD_MAX     = 0.95
    CAL_UP_NEUTRAL  = -0.50
    CAL_UP_MAX      = 0.60
    CAL_LAT_NEUTRAL = 0.0

    # 위 사람 쪽 기준점들이 대응되는 로봇 좌표 (mm)
    # [주의] 위의 WS_*_MM(작업공간 한계)과 헷갈리지 않도록 MAP_ 접두사를 씀.
    #        이 네 점은 모두 도달 가능 영역 안에 있어야 함(전수검사로 확인 완료).
    MAP_X_NEUTRAL = 170.0   # 중립 자세일 때 로봇이 있을 앞뒤 위치
    MAP_X_MAX     = 235.0   # 최대로 앞으로 뻗었을 때
    MAP_Z_NEUTRAL = 140.0   # 중립 자세일 때 높이
    MAP_Z_MAX     = 270.0   # 최대로 위로 들었을 때
    # 좌우는 별도 캘리브레이션 단계가 없으므로 앞뒤 게인을 그대로 쓰되,
    # 배율 하나를 곱해서 감도를 따로 조절할 수 있게 함.
    # [왜 1.0이 아닌가] 앞뒤 게인만 그대로 쓰면 손을 옆으로 크게 움직여도
    # J1이 20°대밖에 안 돌아서 답답함(J1은 ±100°까지 쓸 수 있는데도).
    # 팔을 옆으로 벌리는 동작은 앞으로 뻗는 동작보다 가동 범위가 작기 때문.
    # 옆으로 너무 예민하면 1.0쪽으로, 둔하면 2.0쪽으로 조정.
    MAP_LAT_GAIN_SCALE = 1.6

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
    WRIST_ANGLE_STRAIGHT =   0.0
    WRIST_ANGLE_BACK     =  25.0
    WRIST_ANGLE_FRONT    = -20.0
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
    MIN_HAND_VEC_J5      = 0.02   # J5용 2D 손 벡터 최소 길이
    J5_STALE_GAP_SEC     = 0.3    # 이 시간 이상 끊겼다 재개되면 J5 필터 리셋

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
    PUB_INTERVAL = 0.1   # 10Hz

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
        self.calib_samples_wrist = []   # effective_bend(손목) 샘플
        self.calib_prep_until = 0.0
        self.last_calib_status_pub_time = 0.0

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠에 접근하지 못했습니다.")

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
        self.m_cx = MedianPrefilter(window=3)
        self.m_cy = MedianPrefilter(window=3)
        self.m_cz = MedianPrefilter(window=3)

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

        # 툴 피치 / J6 홀드값 (손이 안 보이면 마지막 값 유지)
        self.tool_pitch_deg = self.TOOL_PITCH_STRAIGHT
        self.j6_deg = 0.0
        self.j5_last_valid_time = 0.0

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

        # 4) 팔 길이 추정 버퍼 소거 (사람이 바뀌었을 수도 있음)
        self.arm_len_buf.clear()

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

        if frame.shape[1] != 480 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (480, 480))

        # 좌우 반전(거울 모드): 이 때문에 MediaPipe 핸드니스가 실제와 반대
        frame     = cv2.flip(frame, 1)
        h, w, _   = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)

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

        if active_arm is not None and world_lm is not None:
            lm = pose_results.pose_landmarks.landmark
            sh_idx = 11 if active_arm == 'right' else 12
            el_idx = 13 if active_arm == 'right' else 14
            wr_idx = 15 if active_arm == 'right' else 16

            sh_w = world_lm[sh_idx]
            el_w = world_lm[el_idx]
            wr_w = world_lm[wr_idx]

            ua = math.sqrt((el_w.x - sh_w.x) ** 2 + (el_w.y - sh_w.y) ** 2 + (el_w.z - sh_w.z) ** 2)
            fa = math.sqrt((wr_w.x - el_w.x) ** 2 + (wr_w.y - el_w.y) ** 2 + (wr_w.z - el_w.z) ** 2)

            # [DEBUG] 팔꿈치 world 랜드마크가 프레임 간 얼마나 튀는지 확인용.
            # 원인 진단 끝나면 이 블록은 삭제할 것.
            _prev_el_w = getattr(self, '_dbg_prev_el_w', None)
            if _prev_el_w is not None:
                _el_jump_mm = 1000.0 * math.sqrt(
                    (el_w.x - _prev_el_w[0]) ** 2 +
                    (el_w.y - _prev_el_w[1]) ** 2 +
                    (el_w.z - _prev_el_w[2]) ** 2)
                if _el_jump_mm > 15.0:  # 15mm/frame 이상 튀면 로그
                    self.get_logger().info(
                        f"[DBG-ELBOW] jump={_el_jump_mm:.1f}mm/frame  "
                        f"vis={lm[el_idx].visibility:.2f}  ua={ua*1000:.1f}mm  fa={fa*1000:.1f}mm")
            self._dbg_prev_el_w = (el_w.x, el_w.y, el_w.z)

            if (lm[el_idx].visibility > self.VIS_THRESHOLD and
                    ua >= self.MIN_UPPERARM_LEN_M and fa >= self.MIN_FOREARM_LEN_J3_M):
                # 팔 길이는 프레임마다 흔들리므로 중앙값으로 안정화.
                # (분모라서 한 프레임만 튀어도 목표점 전체가 크게 흔들림)
                self.arm_len_buf.append(ua + fa)
                arm_len = sorted(self.arm_len_buf)[len(self.arm_len_buf) // 2]

                if arm_len >= self.MIN_ARM_LEN_M:
                    vx = (wr_w.x - sh_w.x) / arm_len
                    vy = (wr_w.y - sh_w.y) / arm_len
                    vz = (wr_w.z - sh_w.z) / arm_len

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

                # 선택된 제어 손만 AI ON/OFF 제스처 판정에 사용
                self._update_ai_toggle_gesture(hlm, current_time)

                el_idx = 13 if active_arm == 'right' else 14
                el_lm  = lm[el_idx]
                wr_lm  = lm[wr_idx]

                forearm_dx = wr_lm.x - el_lm.x
                forearm_dy = wr_lm.y - el_lm.y
                forearm_mag = math.sqrt(forearm_dx ** 2 + forearm_dy ** 2)

                h_wrist_j5 = hlm[0]
                h_mid_tip  = hlm[12]
                hand_dx = h_mid_tip.x - h_wrist_j5.x
                hand_dy = h_mid_tip.y - h_wrist_j5.y
                hand_mag = math.sqrt(hand_dx ** 2 + hand_dy ** 2)

                # [안정성 가드] 팔을 카메라 쪽으로 뻗으면 전완/손 벡터가 짧아지고,
                # 0에 가까워질수록 atan2가 극도로 불안정해져 노이즈로 튐.
                forearm_ok = forearm_mag >= self.MIN_FOREARM_VEC_J5
                hand_ok    = hand_mag    >= self.MIN_HAND_VEC_J5

                if forearm_ok and hand_ok:
                    cross = forearm_dx * hand_dy - forearm_dy * hand_dx
                    dot   = forearm_dx * hand_dx + forearm_dy * hand_dy
                    signed_bend_deg = math.degrees(math.atan2(cross, dot))
                    effective_bend = signed_bend_deg * self.J5_BACK_BEND_SIGN

                    # [안정성 가드] 오래 끊겼다 재개되면 One Euro가 그 시간차를
                    # "매우 빠른 움직임"으로 오인해 컷오프를 열어버리므로 필터 리셋.
                    if current_time - self.j5_last_valid_time > self.J5_STALE_GAP_SEC:
                        self.f_j5.reset()
                        self.m_j5.reset()
                    self.j5_last_valid_time = current_time

                    # ── 캘리브레이션 샘플 수집 (4~6단계: 손목) ──────────────
                    if (self.calib_mode in ('wrist_straight', 'wrist_back', 'wrist_front') and
                            current_time >= self.calib_prep_until):
                        self.calib_samples_wrist.append(effective_bend)
                        if len(self.calib_samples_wrist) >= self.CALIB_REQUIRED_SAMPLES:
                            self._finish_calib_stage()

                    # ── 3점 보간: 안쪽(FRONT) ↔ 일자(STRAIGHT) ↔ 뒤(BACK) ────
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
            g_up  = (self.MAP_Z_MAX - self.MAP_Z_NEUTRAL) / up_span
            g_lat = g_fwd * self.MAP_LAT_GAIN_SCALE

            raw_x = self.MAP_X_NEUTRAL + g_fwd * (norm_fwd - self.CAL_FWD_NEUTRAL)
            raw_y = g_lat * (norm_lat - self.CAL_LAT_NEUTRAL)
            raw_z = self.MAP_Z_NEUTRAL + g_up  * (norm_up  - self.CAL_UP_NEUTRAL)

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

            # [DEBUG] 특이점 경계(WS_D_MAX_MM)까지 얼마나 가까운지 확인용.
            # 원인 진단 끝나면 이 블록은 삭제할 것.
            _r = math.hypot(cx, cy)
            _D = math.hypot(_r - self.LINK_A1_MM, cz - self.LINK_D1_MM)
            _ratio = _D / self.WS_D_MAX_MM
            if _ratio > 0.85:  # 경계의 85% 이상 접근했을 때만 로그
                self.get_logger().info(
                    f"[DBG-SINGULARITY] D={_D:.1f}mm  ratio={_ratio:.2f}  "
                    f"clamped={was_clamped}  target=({cx:.0f},{cy:.0f},{cz:.0f})")

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