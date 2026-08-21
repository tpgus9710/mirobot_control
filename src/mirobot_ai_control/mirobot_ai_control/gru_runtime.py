"""
gru_runtime.py — GRU 랜드마크 보정, NumPy 순전파 (Pi 배포용)

PyTorch 없이 numpy만으로 GRU를 한 프레임씩 실행한다.
학습된 가중치는 export_gru_npz.py 가 만든 .npz 에서 읽는다.

★ 상태 변수 주의 (호밍 안전) ★
    이 클래스는 hidden state와 이전 프레임 좌표를 내부에 들고 있다. 이는
    "직전 궤적의 기억"이므로, 호밍 완료 직후 낡은 상태로 첫 프레임을 처리하면
    호밍 전 자세 쪽으로 보정된 값이 나올 수 있다.
    → ai_node 의 _hard_reset_control_state() 에서 반드시 reset() 을 호출할 것.
    → 워밍업 구간(WARMUP 프레임) 동안은 보정을 적용하지 않고 raw 를 통과시킨다.

좌표 규약 (학습과 동일해야 함):
    · 어깨를 원점으로 하는 상대 벡터
    · 팔 길이(어깨→팔꿈치→손목 합)로 나눠 정규화
    · 입력 = [팔꿈치(3), 손목(3)] (+ vel 이면 각각의 1차 차분 6개 = 12차원)
    · 출력 = 잔차. 최종 = 입력 + 잔차
"""
import numpy as np


def _sigmoid(x):
    # 지수 폭주를 막기 위해 부호별로 나눠 계산
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class GRURuntime:
    """한 프레임씩 GRU를 실행한다 (seq_len=1, hidden state 이월)."""

    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.H = int(z["hidden"])
        self.in_dim = int(z["in_dim"])
        self.out_dim = int(z["out_dim"])
        self.with_elbow = bool(z["with_elbow"])
        self.use_vel = bool(z["vel"])
        self.arm_len_train = float(z["arm_len_train"])

        H = self.H
        # PyTorch nn.GRU 게이트 순서: (reset, update, new)
        W_ih, W_hh = z["W_ih"], z["W_hh"]
        b_ih, b_hh = z["b_ih"], z["b_hh"]
        self.Wir, self.Wiz, self.Win = W_ih[:H], W_ih[H:2*H], W_ih[2*H:]
        self.Whr, self.Whz, self.Whn = W_hh[:H], W_hh[H:2*H], W_hh[2*H:]
        self.bir, self.biz, self.bin_ = b_ih[:H], b_ih[H:2*H], b_ih[2*H:]
        self.bhr, self.bhz, self.bhn = b_hh[:H], b_hh[H:2*H], b_hh[2*H:]
        self.W_out, self.b_out = z["W_out"], z["b_out"]

        self.reset()

    # ── 상태 관리 ─────────────────────────────────────────────────────────
    def reset(self):
        """모든 상태를 초기화한다. 호밍 시작·완료 시 반드시 호출."""
        self.h = np.zeros(self.H, dtype=np.float32)
        self.prev_feat = None      # 속도(1차 차분) 계산용 직전 좌표
        self.warm = 0              # 워밍업 카운터

    # ── GRU 한 스텝 ───────────────────────────────────────────────────────
    def step(self, x):
        """입력 (in_dim,) → 출력 (out_dim,). hidden state를 갱신한다."""
        x = np.asarray(x, dtype=np.float32)
        h = self.h
        r = _sigmoid(self.Wir @ x + self.bir + self.Whr @ h + self.bhr)
        zg = _sigmoid(self.Wiz @ x + self.biz + self.Whz @ h + self.bhz)
        # 주의: reset 게이트는 (Whn @ h + bhn) 전체에 곱해진다 (bhn 포함)
        n = np.tanh(self.Win @ x + self.bin_ + r * (self.Whn @ h + self.bhn))
        h = (1.0 - zg) * n + zg * h
        self.h = h
        return self.W_out @ h + self.b_out


class LandmarkCorrector:
    """MediaPipe 랜드마크 → 정규화 → GRU 보정 → 원좌표 복원.

    ai_node 에서는 이 클래스만 쓰면 된다.

    사용:
        corr = LandmarkCorrector("gru.npz", warmup=10)
        el_c, wr_c = corr.correct(shoulder, elbow, wrist)   # 각각 (3,) 미터
        corr.reset()                                        # 호밍 시
    """

    WARMUP_DEFAULT = 10

    def __init__(self, npz_path, warmup=None, arm_len=None):
        self.rt = GRURuntime(npz_path)
        self.warmup = self.WARMUP_DEFAULT if warmup is None else int(warmup)
        # 배포 시에는 사용자별 실측 팔 길이를 쓰는 것이 맞다.
        # 지정하지 않으면 학습 시 값을 쓰되, 캘리브레이션에서 갱신할 것.
        self.arm_len = float(arm_len) if arm_len else self.rt.arm_len_train
        self.reset()

    def reset(self):
        """호밍 시작·완료 시 호출. GRU 상태와 속도 버퍼를 모두 비운다."""
        self.rt.reset()

    def set_arm_len(self, arm_len_m):
        """캘리브레이션에서 측정한 팔 길이(m)를 반영한다."""
        if arm_len_m and arm_len_m > 0.05:
            self.arm_len = float(arm_len_m)

    def correct(self, shoulder, elbow, wrist):
        """어깨 기준으로 정규화 → 보정 → 원래 좌표계로 복원.

        반환: (보정된 팔꿈치, 보정된 손목). 워밍업 중에는 입력을 그대로 돌려준다.
        """
        sh = np.asarray(shoulder, dtype=np.float32)
        el = np.asarray(elbow, dtype=np.float32)
        wr = np.asarray(wrist, dtype=np.float32)

        L = self.arm_len
        if not np.isfinite(L) or L <= 1e-6:
            return elbow, wrist

        # 어깨 원점 + 팔 길이 정규화 (학습과 동일)
        e = (el - sh) / L
        w = (wr - sh) / L
        feat = np.concatenate([e, w]).astype(np.float32)

        if self.rt.use_vel:
            if self.rt.prev_feat is None:
                vel = np.zeros_like(feat)
            else:
                vel = feat - self.rt.prev_feat
            self.rt.prev_feat = feat
            x = np.concatenate([feat, vel])
        else:
            x = feat

        res = self.rt.step(x)          # 잔차 예측

        # 워밍업: hidden state가 0에서 출발한 직후 몇 프레임은 신뢰할 수 없다.
        # 이 구간에서 보정을 적용하면 호밍 직후 로봇이 튈 수 있다.
        if self.rt.warm < self.warmup:
            self.rt.warm += 1
            return elbow, wrist

        out = feat + res
        el_c = out[:3] * L + sh
        wr_c = out[3:6] * L + sh
        return el_c, wr_c
