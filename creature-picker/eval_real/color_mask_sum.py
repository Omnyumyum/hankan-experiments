"""color_mask_eval.py 결과 CSV 둘(지금 방식 · 새 방식)을 그림마다 짝지어 비교한다.

    venv/bin/python -I eval_real/color_mask_sum.py A.csv B.csv
Meta 결과면 eval_real/color_sets.csv 의 「밝은 색칠」 비율로 색칠 많음 묶음을 더 나눈다 (밝은 색칠이 색칠의 절반 넘음).
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def key(r):
    return r["i"] if "i" in r else (r["cat"], r["id"], r["variant"])


def main():
    A = {key(r): r for r in load(sys.argv[1])}
    B = {key(r): r for r in load(sys.argv[2])}
    cs = {r["i"]: r for r in load(HERE / "color_sets.csv")}
    groups = {}
    for k, b in B.items():
        a = A[k]
        g = [b["cls"]]
        if "i" in b and b["cls"] == "heavy":
            c = cs[b["i"]]
            g.append("heavy-light" if float(c["light_frac"]) >= 0.5 * float(c["color_frac"]) else "heavy-dark")
        if "variant" in b:
            g.append(b["variant"])
        for gg in g + ["전체"]:
            groups.setdefault(gg, []).append((a, b))
    print("| 묶음 | 장수 | 실패 전 → 후 | IoU 중앙값 전 → 후 | 덜 잡음 전 → 후 | 더 잡음 전 → 후 | 0.1 넘게 좋아짐 / 나빠짐 | 평균 초 전 → 후 |")
    print("|---|---|---|---|---|---|---|---|")
    for g, ps in groups.items():
        n = len(ps)
        f = lambda rs, c: sum(c(r) for r in rs)  # noqa: E731
        a = [p[0] for p in ps]
        b = [p[1] for p in ps]
        d = np.array([float(y["iou"]) - float(x["iou"]) for x, y in ps])
        print(f"| {g} | {n} | {f(a, lambda r: r['fail'] == '1')} → **{f(b, lambda r: r['fail'] == '1')}** | "
              f"{np.median([float(r['iou']) for r in a]):.3f} → {np.median([float(r['iou']) for r in b]):.3f} | "
              f"{f(a, lambda r: float(r['recall']) < 0.7)} → {f(b, lambda r: float(r['recall']) < 0.7)} | "
              f"{f(a, lambda r: float(r['prec']) < 0.5)} → {f(b, lambda r: float(r['prec']) < 0.5)} | "
              f"{(d > 0.1).sum()} / {(d < -0.1).sum()} | "
              f"{np.mean([float(r['t_sec']) for r in a]):.2f} → {np.mean([float(r['t_sec']) for r in b]):.2f} |")


if __name__ == "__main__":
    main()
