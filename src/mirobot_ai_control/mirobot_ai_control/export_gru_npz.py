"""
export_gru_npz.py — 학습된 GRU(.pt)를 Pi 배포용 .npz로 변환

왜 npz인가:
    Pi에 PyTorch를 설치하면 용량·기동 시간·의존성 부담이 크다. GRU는 순전파가
    단순해서 NumPy 20줄이면 되고, 그러면 ai_node가 이미 쓰는 numpy만으로 충분하다.

★ 게이트 순서 주의 ★
    PyTorch nn.GRU는 weight를 (reset, update, new) 순서로 이어붙여 저장한다.
    NumPy로 옮길 때 이 순서를 틀리면 에러 없이 조용히 이상한 값이 나온다.
    그래서 이 스크립트는 변환 후 반드시 PyTorch 출력과 대조 검증한다.

사용법:
    python3 export_gru_npz.py <모델.pt> [출력.npz]
"""
import os
import sys
import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="PyTorch 대조 검증 허용 오차")
    args = ap.parse_args()

    path = os.path.expanduser(args.model)
    d = torch.load(path, map_location="cpu")
    sd = d["state_dict"]

    H = int(d["hidden"])
    in_dim = int(d["in_dim"])
    out_dim = int(d["out_dim"])

    print(f"모델: {path}")
    print(f"  hidden={H}  in={in_dim}  out={out_dim}  "
          f"elbow={d['with_elbow']}  vel={d['vel']}  loss={d['loss']}")
    print(f"  학습 시 검증 오차 {d['val_mm']:.1f}mm (보정 전 {d['base_mm']:.1f}mm)")
    print(f"  학습 세션: {[os.path.basename(s.rstrip('/')) for s in d['train_sessions']]}")

    # ── 가중치 추출 (게이트 순서 r, z, n 유지) ─────────────────────────────
    W_ih = sd["gru.weight_ih_l0"].numpy().astype(np.float32)   # (3H, in)
    W_hh = sd["gru.weight_hh_l0"].numpy().astype(np.float32)   # (3H, H)
    b_ih = sd["gru.bias_ih_l0"].numpy().astype(np.float32)     # (3H,)
    b_hh = sd["gru.bias_hh_l0"].numpy().astype(np.float32)     # (3H,)
    W_out = sd["head.weight"].numpy().astype(np.float32)       # (out, H)
    b_out = sd["head.bias"].numpy().astype(np.float32)         # (out,)

    assert W_ih.shape == (3 * H, in_dim), W_ih.shape
    assert W_hh.shape == (3 * H, H), W_hh.shape

    out_path = args.out or os.path.splitext(path)[0] + ".npz"
    np.savez(
        out_path,
        W_ih=W_ih, W_hh=W_hh, b_ih=b_ih, b_hh=b_hh,
        W_out=W_out, b_out=b_out,
        hidden=np.int32(H), in_dim=np.int32(in_dim), out_dim=np.int32(out_dim),
        with_elbow=np.bool_(d["with_elbow"]), vel=np.bool_(d["vel"]),
        # 학습 시 정규화에 쓴 팔 길이. 배포 시에는 사용자별 실측값을 쓰지만,
        # 캘리브레이션 전 기본값으로 필요하다.
        arm_len_train=np.float32(d["arm_len"]),
        val_mm=np.float32(d["val_mm"]), base_mm=np.float32(d["base_mm"]),
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n저장 → {out_path}  ({size_kb:.0f} KB)")

    # ── 대조 검증 ─────────────────────────────────────────────────────────
    # 게이트 순서를 틀려도 실행은 되므로, 같은 입력에 두 구현의 출력이
    # 일치하는지 반드시 확인한다. 이 검증 없이 배포하면 로봇이 이상하게
    # 움직여도 원인을 찾기 어렵다.
    print("\nPyTorch ↔ NumPy 대조 검증...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gru_runtime import GRURuntime

    rt = GRURuntime(out_path)
    rng = np.random.default_rng(0)
    T = 50
    x = rng.normal(0, 1, (T, in_dim)).astype(np.float32)

    # PyTorch 쪽
    gru = torch.nn.GRU(in_dim, H, batch_first=True)
    gru.load_state_dict({k.replace("gru.", ""): v
                         for k, v in sd.items() if k.startswith("gru.")})
    head = torch.nn.Linear(H, out_dim)
    head.load_state_dict({k.replace("head.", ""): v
                          for k, v in sd.items() if k.startswith("head.")})
    gru.eval(); head.eval()
    with torch.no_grad():
        h_seq, _ = gru(torch.from_numpy(x).unsqueeze(0))
        ref = head(h_seq).squeeze(0).numpy()

    # NumPy 쪽 (한 프레임씩 — 실제 배포와 동일한 방식)
    got = np.stack([rt.step(x[t]) for t in range(T)])

    err = np.abs(ref - got)
    print(f"  최대 오차 {err.max():.2e}  평균 {err.mean():.2e}")
    if err.max() < args.tol:
        print("  ✓ 일치합니다. 배포해도 됩니다.")
    else:
        print(f"  ✗ 불일치! (허용 {args.tol:.0e}) 게이트 순서나 bias 처리를 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
