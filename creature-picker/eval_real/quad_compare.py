"""고치기 전 · 후 비교 (같은 그림끼리). 손으로 매긴 quad_labels.csv 로 나눠 본다.

    venv/bin/python -I eval_real/quad_compare.py quad_before.csv quad_after.csv --split check
- 망가짐(늘어남 > 5 또는 뒤집힘 > 1%) 전 → 후, 그림 종류(F 얼굴만 · Q 몸+다리 · O 기타)별
- 몸통만으로 돌린 것: F 중 몇 장 · Q 중 몇 장(= 진짜 네 발을 잘못 떨어뜨림)
- 진짜 네 발(Q)이고 둘 다 네 발로 돈 그림: 다리 움직임 전 · 후 (평균 · 중앙값, 20% 넘게 줄어든 장수 · 20% 넘게 는 장수)
"""
import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, split):
    return {(r["cat"], r["id"]): r for r in csv.DictReader(open(HERE / name, encoding="utf-8-sig")) if r["split"] == split}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--split", default="learn")
    a = ap.parse_args()
    B, A = load(a.before, a.split), load(a.after, a.split)
    lab = {k: r["label"] for k, r in load("quad_labels.csv", a.split).items()}
    print(f"\n## {a.split}: {a.before} → {a.after}")
    print("| 그림 종류 | 장수 | 망가짐 전 | 망가짐 후 | 몸통만으로 돌림 |")
    print("|---|---|---|---|---|")
    for L, name in (("F", "얼굴만 (F)"), ("Q", "몸 + 다리 (Q)"), ("O", "기타 (O)"), (None, "합계")):
        ks = [k for k in B if L is None or lab.get(k) == L]
        nb = sum(B[k]["bad"] == "1" for k in ks)
        na = sum(A[k]["bad"] == "1" for k in ks)
        ng = sum(A[k]["gate"] == "1" for k in ks)
        print(f"| {name} | {len(ks)} | {nb} ({100 * nb / max(1, len(ks)):.0f}%) | {na} ({100 * na / max(1, len(ks)):.0f}%) | {ng} |")
    q = [k for k in B if lab.get(k) == "Q" and A[k]["kind_used"] == "quadruped" and B[k]["anim_ok"] == "1" and A[k]["anim_ok"] == "1"]
    lb = np.array([float(B[k]["leg_move"]) for k in q])
    la = np.array([float(A[k]["leg_move"]) for k in q])
    r = la / np.maximum(lb, 1e-6)
    print(f"\n진짜 네 발(Q) {len(q)}장 다리 움직임(발 꼭짓점 폭 ÷ 긴 변): 평균 {lb.mean():.4f} → {la.mean():.4f}, "
          f"중앙값 {np.median(lb):.4f} → {np.median(la):.4f}, 20% 넘게 줄어든 그림 {int((r < 0.8).sum())}장, 20% 넘게 는 그림 {int((r > 1.2).sum())}장")
    fixed = [k for k in B if B[k]["bad"] == "1" and A[k]["bad"] != "1"]
    broke = [k for k in B if B[k]["bad"] != "1" and A[k]["bad"] == "1"]
    print(f"고쳐진 그림 {len(fixed)}장, 새로 망가진 그림 {len(broke)}장: {sorted(broke)}")
    down = sorted((round(float(x), 2), k) for x, k in zip(r, q) if x < 0.8)
    print("다리가 20% 넘게 덜 움직인 Q:", down)


if __name__ == "__main__":
    main()
