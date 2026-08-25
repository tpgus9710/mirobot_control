"""
gru_features.py — GRU 입력 보조 특징. 학습과 추론이 이 함수 하나를 공유한다.

왜 공유 모듈인가:
    특징을 학습 쪽에만 추가하고 런타임을 안 고치면 차원은 맞는데 값이 달라
    에러 없이 조용히 이상한 보정이 나간다. 반대도 마찬가지다. 계산을 한
    군데에 두고 train_gru / gru_runtime 이 같은 함수를 부르면 이 실수가
    구조적으로 불가능해진다.

왜 이 특징들인가 (mp_shrink.py 진단 결과):
    카메라 축과 팔이 나란해지는 0~15° 구간에서 MediaPipe는 팔을 30~35%
    짧게 본다. 15°를 넘으면 축소가 급격히 사라진다(세션2_C: 0.65 → 1.04).
    즉 보정에 필요한 양이 방위각에 따라 크게 달라진다.

    이 정보는 좌표 안에 들어 있지만 arccos·sqrt·atan2 같은 비선형 변환을
    거쳐야 나온다. GRU가 직접 배우기 어려운 형태라 명시적으로 넣어준다.

좌표 규약:
    입력 el, wr 은 어깨를 원점으로 하고 팔 길이로 나눈 정규화 좌표.
    MediaPipe world 좌표는 z가 음수일수록 카메라에 가깝다(실측 확인).
"""
import numpy as np

# 특징 세트 정의: 이름 → (차원 수, 설명)
FEATURE_SETS = {
    'off':   (0, '보조 특징 없음 (v7과 동일)'),
    'geo':   (5, '팔꿈치각 + 단축비 + 반지름 + 방위각(sin,cos)'),
    'geo+':  (7, 'geo + 앙각(sin,cos)'),
}


def n_extra(feats):
    """특징 세트의 추가 차원 수."""
    if feats not in FEATURE_SETS:
        raise ValueError(f'알 수 없는 특징 세트: {feats} (가능: {list(FEATURE_SETS)})')
    return FEATURE_SETS[feats][0]


def extra_feats(el, wr, feats='geo'):
    """보조 특징을 계산한다.

    el, wr : (..., 3) 어깨 기준 정규화 좌표 (팔꿈치, 손목)
    반환   : (..., n_extra) float32

    학습에서는 (n, 3) 배열이, 런타임에서는 (3,) 벡터가 들어온다.
    두 경우 모두 같은 코드로 처리된다.
    """
    if feats == 'off':
        shape = np.shape(el)[:-1] + (0,)
        return np.zeros(shape, np.float32)

    el = np.asarray(el, np.float64)
    wr = np.asarray(wr, np.float64)
    fwd = wr - el                                   # 전완 벡터
    eps = 1e-9

    # ── 1) 팔꿈치 각도 ─────────────────────────────────────────────────────
    # 깊이 모호성의 영향을 상대적으로 덜 받는 값. 진단에서 MP 팔꿈치각이
    # 방위각을 따라 109°→143°로 움직이는 것이 확인됐으므로, 지금 어느
    # 영역인지를 알려주는 직접적인 단서가 된다.
    # arccos 대신 코사인 자체를 쓴다. 미분이 매끄럽고 정규화도 필요 없다.
    u = -el                                         # 팔꿈치→어깨
    nu = np.linalg.norm(u, axis=-1)
    nf = np.linalg.norm(fwd, axis=-1)
    cos_elbow = (u * fwd).sum(-1) / (nu * nf + eps)

    # ── 2) 단축비 ─────────────────────────────────────────────────────────
    # 화면(xy)에 투영된 팔 길이 / 3D 팔 길이 = 카메라 축에서 벗어난 각의 sin.
    # 0에 가까울수록 팔이 카메라를 향해 있어 깊이 정보가 사라진다.
    # MediaPipe가 팔을 짧게 보는 정도와 가장 직접적으로 연결된 값이다.
    r3 = np.linalg.norm(wr, axis=-1)
    r2 = np.linalg.norm(wr[..., :2], axis=-1)
    fore = r2 / (r3 + eps)

    # ── 3) 반지름 ─────────────────────────────────────────────────────────
    # 팔 길이로 정규화했으므로 팔을 다 펴면 1.0 부근이어야 한다.
    # 실제로는 0.65~1.04로 흔들리며, 그 편차 자체가 보정해야 할 양이다.
    rad = r3

    # ── 4) 방위각 (sin, cos) ──────────────────────────────────────────────
    # 각도를 그대로 넣으면 0°와 360° 사이가 끊긴다. sin·cos 쌍으로 넣어
    # 연속성을 유지한다. z가 음수일수록 카메라 쪽이므로 정면은 -z 방향.
    x, z = wr[..., 0], wr[..., 2]
    az = np.arctan2(x, -z)
    cols = [cos_elbow, fore, rad, np.sin(az), np.cos(az)]

    # ── 5) 앙각 (geo+ 에서만) ─────────────────────────────────────────────
    if feats == 'geo+':
        y = -wr[..., 1]                             # MediaPipe는 y축 아래가 양수
        h = np.linalg.norm(wr[..., [0, 2]], axis=-1)
        ev = np.arctan2(y, h + eps)
        cols += [np.sin(ev), np.cos(ev)]

    out = np.stack(cols, axis=-1).astype(np.float32)
    assert out.shape[-1] == n_extra(feats), (out.shape, feats)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def append_feats(x, feats='geo', with_elbow=True):
    """정규화 좌표 배열에 보조 특징을 이어붙인다.

    x : (n, 6) = [팔꿈치3, 손목3]  또는 (n, 3) = [손목3]
        속도를 이미 붙인 (n, 12) / (n, 6) 도 받는다. 앞쪽 좌표만 사용한다.

    속도 뒤에 붙이는 이유: 보조 특징의 1차 차분은 의미가 약하고 차원만
    두 배로 늘린다. 좌표에만 속도를 주고 특징은 원본 값으로 둔다.
    """
    if feats == 'off':
        return x
    x = np.asarray(x, np.float32)
    if with_elbow:
        el, wr = x[..., 0:3], x[..., 3:6]
    else:
        # 팔꿈치가 없으면 팔이 곧게 펴졌다고 보고 중간점으로 대체한다.
        wr = x[..., 0:3]
        el = wr * 0.5
    return np.concatenate([x, extra_feats(el, wr, feats)], axis=-1)
