"""망가진 그림(늘어남 > 5 또는 뒤집힘 > 1%)의 원인 나눠 세기.

    venv/bin/python -I eval_real/quad_causes.py --csv quad_before.csv --split learn
(a) 몸 없이 얼굴만: 손으로 매긴 quad_labels.csv 의 F (눈으로 봄)
그 밖의 망가진 그림은 「가장 늘어난 모서리」가 어디 있는지로 나눈다 (쉬는 자세 기준):
(c) 다리: 다리 기준점 줄(뿌리 → 무릎 → 발)에서 15px 안
(b) 가는 선 조각: 다리가 아니고, 둘레 10px 안 마스크 굵기(거리 최댓값)가 9px 미만 = 칠 없는 선 한 줄 (수염 · 털 · 갈기 · 꼬리 선)
(d) 기타: 칠해진(닫힌) 몸 · 머리 안
쉬는 자세의 기준점은 rig_types 로 다시 만든다(지금 코드 기준). 판정 그림은 quad_sheet.py worst 로 눈으로 확인.
"""
import argparse
import csv
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import rig_types as rt  # noqa: E402


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip((p - a) @ ab / max(ab @ ab, 1e-6), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def cause(r):
    png = HERE / "data" / "quad" / f"{r['cat']}_{r['split']}_{int(r['id']):03d}.png"
    d = rt.Drawing(cv2.imread(str(png)))
    pts = rt.snap_to_tips(d, rt.propose(d)["points"]["quadruped"])
    mask = d.mask_q
    p = np.array([float(r["worst_x"]), float(r["worst_y"])])
    if hasattr(rt, "walk_leg"):  # 고친 코드: 한 다리에 붙은 발을 합친 뒤의 다리
        feet = rt.merge_same_leg(mask, [rt.snap(mask, f) for f in pts["feet"]])
        top = np.nonzero(mask.any(1))[0][0]
        chains = [(l["root"], l["knee"], l["foot"]) for l in (rt.walk_leg(mask, f, top) for f in feet)]
    else:  # 예전 코드
        pins0, _ = rt.build_quadruped(mask, pts)
        chains = [pins0[1 + 3 * i:4 + 3 * i] for i in range(len(pts["feet"]))]
    leg = min(min(seg_dist(p, a, b), seg_dist(p, b, c)) for a, b, c in chains)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    x, y = int(p[0]), int(p[1])
    local = float(dt[max(0, y - 10):y + 11, max(0, x - 10):x + 11].max())
    if leg <= 15:
        return "c"
    return "b" if local < 9 else "d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="learn")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(HERE / a.csv, encoding="utf-8-sig")) if r["split"] == a.split]
    lab = {(r["cat"], r["id"]): r["label"] for r in csv.DictReader(open(HERE / "quad_labels.csv", encoding="utf-8-sig"))
           if r["split"] == a.split}
    bad = [r for r in rows if r["bad"] == "1"]
    with Pool() as p:
        cs = p.map(cause, bad)
    where = {(r["cat"], r["id"]): c for r, c in zip(bad, cs)}  # 얼굴만 그린 그림도 어디가 늘어났는지는 따로 적는다
    got = {k: ("a" if lab.get(k) == "F" else c) for k, c in where.items()}
    print("| 카테고리 | 장수 | 망가짐 | (a) 얼굴만 | (a) 중 늘어난 곳 b/c/d | (b) 가는 선 조각 | (c) 다리 | (d) 기타 |")
    print("|---|---|---|---|---|---|---|---|")
    for cat in ("dog", "cat", "horse", "pig", "합계"):
        rs = [r for r in rows if cat == "합계" or r["cat"] == cat]
        c = Counter(got[(r["cat"], r["id"])] for r in rs if r["bad"] == "1")
        wa = Counter(where[(r["cat"], r["id"])] for r in rs if r["bad"] == "1" and got[(r["cat"], r["id"])] == "a")
        print(f"| {cat} | {len(rs)} | {sum(c.values())} | {c['a']} | {wa['b']}/{wa['c']}/{wa['d']} | {c['b']} | {c['c']} | {c['d']} |")
    with open(HERE / "data" / f"causes_{Path(a.csv).stem}_{a.split}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cat", "id", "label", "cause", "where"])
        for (cat, i), c in sorted(got.items()):
            w.writerow([cat, i, lab.get((cat, i), ""), c, where[(cat, i)]])


if __name__ == "__main__":
    main()
