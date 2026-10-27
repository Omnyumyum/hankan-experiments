"""이미 잘 되던 그림(색칠 · 윤곽 그림)의 칸이 바뀌었는지 보는 검사.

    venv/bin/python -I eval_real/regions_keep.py --save 기준.npz   # 고치기 전에 한 번
    venv/bin/python -I eval_real/regions_keep.py --cmp 기준.npz    # 고친 뒤
그림: creature-picker/drawings (cases 포함) · animal-monster/drawings · drawings_quad 전부.
비교: 칸 수, 그리고 「칸 픽셀 중 예전과 같은 칸 짝에 들어간 비율」(예전 칸마다 가장 많이 겹친 새 칸, 반대로도 같은지).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rig_types as rt  # noqa: E402

FILES = sorted([*(HERE.parent / "drawings").rglob("*.png"), *(rt.AM / "drawings").glob("*.png"), *(rt.AM / "drawings_quad").glob("*.png")])


def labels(p):
    d = rt.Drawing(rt.sr.load_image(p))
    return rt.region_labels(d)[0]


def same(a, b):
    """두 칸 나누기가 같은 정도: 둘 다 칸인 픽셀 중 서로 1:1 짝인 칸에 든 비율 · 칸 ↔ 선이 바뀐 픽셀 비율."""
    both = (a > 0) & (b > 0)
    if not both.any():
        return 1.0 if not (a > 0).any() and not (b > 0).any() else 0.0, float(((a > 0) != (b > 0)).mean())
    pa, pb = a[both], b[both]
    ok = np.zeros(len(pa), bool)
    fwd, bwd = {}, {}
    for r in np.unique(pa):
        v, c = np.unique(pb[pa == r], return_counts=True)
        fwd[r] = v[np.argmax(c)]
    for r in np.unique(pb):
        v, c = np.unique(pa[pb == r], return_counts=True)
        bwd[r] = v[np.argmax(c)]
    for r, s in fwd.items():
        if bwd.get(s) == r:
            ok |= (pa == r) & (pb == s)
    return float(ok.mean()), float(((a > 0) != (b > 0)).sum() / max(1, (a > 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--cmp")
    a = ap.parse_args()
    if a.save:
        np.savez_compressed(a.save, **{str(p.relative_to(HERE.parent.parent)): labels(p) for p in FILES})
        print(f"{len(FILES)}장 저장")
        return
    old = np.load(a.cmp)
    worst = 1.0
    for p in FILES:
        k = str(p.relative_to(HERE.parent.parent))
        b, o = labels(p), old[k]
        n_o, n_b = len(np.unique(o)) - 1, len(np.unique(b)) - 1
        s, moved = same(o, b)
        worst = min(worst, s)
        flag = "" if s >= 0.98 and n_o == n_b else "  ← 달라짐"
        print(f"{k:60s} 칸 {n_o:3d} → {n_b:3d}  같은 칸 {s:6.1%}  칸↔선 {moved:5.1%}{flag}")
    print(f"\n가장 덜 같은 그림: {worst:.1%}")


if __name__ == "__main__":
    main()
