"""칸 나누기(region_labels) · 날개 칸 맞춤 · 날개 뒤집힘을 QuickDraw 로 재는 시험.

    venv/bin/python -I eval_real/regions_wings_eval.py --qd 받은폴더 --split train|check --out 이름 [--sheet]
받은폴더: run_eval.py 와 같은 {category}.ndjson 10개 (앞부분만 받아도 된다).
카테고리마다 섞은 뒤(seed 23) 앞쪽 = 학습용(train, 고칠 때만 봄), 그다음 = 확인용(check, 마지막에 한 번 잼). 둘은 겹치지 않는다.
- 날개 카테고리(bird · butterfly · bee): 학습용 100 · 확인용 220 장
- 나머지 7 카테고리: 학습용 50 · 확인용 100 장 (칸 개수만 잰다)
날개 그림은 점 두 가지로 움직인다:
- prop : propose 가 미리 놓은 점 그대로
- jit1, jit2 : 그 점을 하나하나 반경 10px 안에서 흔든 것 (check_wing_points.py 방식). QuickDraw 에는 정답 점이 없어서
  미리 놓은 점을 「정답」 자리로 본다.
결과: eval_real/rw_<out>.csv (그림 · 점마다 한 줄), 화면에 요약. 모아 보기는 rw_sheet.py.
「통째로」= 날개 무게 0.5 넘는 곳이 그림의 85% 이상 (몸까지 날개로 잡혀 그림 전체가 날개처럼 움직임).
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import check_wings as cw  # noqa: E402
import rig_types as rt  # noqa: E402
from run_eval import EXPECT, render  # noqa: E402

WINGS = ("bird", "butterfly", "bee")
N = {"wings": {"train": 100, "check": 220}, "other": {"train": 50, "check": 100}}
JIT_R, N_JIT = 10.0, 2


def load_split(path, cat, split, seed):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue  # 받다 만 마지막 줄
            if r.get("recognized"):
                rows.append(r["drawing"])
    n = N["wings" if cat in WINGS else "other"]
    idx = np.random.default_rng(seed).permutation(len(rows))
    a = 0 if split == "train" else n["train"]
    return [rows[i] for i in idx[a:a + n[split]]]


def n_regions(d):
    lab, _ = rt.region_labels(d)
    return int(len(np.unique(lab)) - (lab == 0).any())


def snap_reason(d, mask, pts):
    """칸이 있는데 점 3개로 떨어진 이유 (snap_points_to_regions 의 갈림길을 그대로 따라간다)."""
    lab, ink = rt.region_labels(d)
    k = 2 * rt.line_width(ink) + 1
    out = []
    for w in pts.get("wings") or []:
        if len(w) < 3:
            out.append("점 부족")
            continue
        A, B, T = (np.asarray(p, np.float32) for p in w[:3])
        if np.linalg.norm(B - A) < 4:
            out.append("경첩 너무 짧음")
            continue
        out.append("끝 근처 칸 없음" if not rt.region_at(d, *T, r=3 * k) else "칸 있음")
    return "/".join(out)


def run_wings(d, mask, pts, tag, base):
    row = dict(base, pts=tag, mode="", stretch="", flip="", big_flip=0, big_stretch=0, body_w="", wing_frac="", reason="", err="")
    try:
        rig, used = rt.resolve_wings(d, mask, pts)
        anim = rt.animate(d, "wings", pts)
        row["mode"] = used
        row["stretch"] = round(cw.stretch(anim), 2)
        row["flip"] = round(cw.flipped(anim), 4)
        row["big_flip"] = int(row["flip"] > 0.01)
        row["big_stretch"] = int(row["stretch"] > 5)
        wm = rig["wmap"]
        row["wing_frac"] = round(float(((wm > 0.5) & (mask > 0)).sum()) / max(1, int((mask > 0).sum())), 3)
        # 몸 가운데(양쪽 경첩 가운데의 가운데) 무게: 0.15 넘으면 몸통이 날개에 딸려 움직인다
        ws = [w for w in pts["wings"] if len(w) >= 3]
        if len(ws) >= 2:
            c = np.mean([(np.asarray(w[0]) + np.asarray(w[1])) / 2 for w in ws], 0)
            row["body_w"] = round(cw.weight_at(wm, c), 3)
        if used == "points":
            row["reason"] = "칸 0개" if base["n_regions"] == 0 else snap_reason(d, mask, pts)
    except Exception as e:
        row["err"] = repr(e)[:120]
    return row


def one(job):
    cat, split, k, strokes = job
    img = render(strokes)
    base = dict(cat=cat, split=split, id=k, n_regions="")
    try:
        d = rt.Drawing(img)
        base["n_regions"] = n_regions(d)
    except Exception as e:
        return [dict(base, pts="-", err=repr(e)[:120])]
    if cat not in WINGS:
        return [dict(base, pts="-")]
    try:
        pr = rt.propose(d)
    except Exception as e:
        return [dict(base, pts="-", err="propose " + repr(e)[:100])]
    mask, _ = d.masks("wings")
    p0 = pr["points"]["wings"]
    rows = [run_wings(d, mask, p0, "prop", base)]
    rng = np.random.default_rng(1000 * WINGS.index(cat) + k + (0 if split == "train" else 500000))
    lim = [mask.shape[1] - 1, mask.shape[0] - 1]
    for j in range(N_JIT):
        wj = [[np.clip(np.asarray(p, np.float32) + rng.uniform(-1, 1, 2) * JIT_R, 0, lim).tolist() for p in w] for w in p0["wings"]]
        rows.append(run_wings(d, mask, {"wings": wj, "style": p0.get("style")}, f"jit{j + 1}", base))
    return rows


def summary(rows):
    cats = list(EXPECT)
    print("\n## 칸 0개 비율")
    by = {}
    for r in rows:
        if r["pts"] in ("-", "prop"):
            by.setdefault(r["cat"], []).append(r["n_regions"])
    tot0 = tot = 0
    for c in cats:
        v = [x for x in by.get(c, []) if x != ""]
        z = sum(x == 0 for x in v)
        tot0 += z
        tot += len(v)
        print(f"{c:10s} {z:3d}/{len(v):3d}  ≤2칸 {sum(x <= 2 for x in v):3d}  중앙값 {np.median(v) if v else 0:.0f}")
    print(f"합계 칸 0개 {tot0}/{tot} ({tot0 / max(1, tot):.1%})")
    for tag in ("prop", "jit"):
        rr = [r for r in rows if r["pts"].startswith(tag)]
        if not rr:
            continue
        print(f"\n## 날개 — 점: {'미리 놓은 점 그대로' if tag == 'prop' else '흔든 점'} ({len(rr)}번)")
        print("| 카테고리 | 번 | 칸 맞춤 | 점 그대로 (칸 0개 / 칸 있음) | 뒤집힘>1% | 늘어남>5배 | 몸통 무게>0.15 | 통째로 | 오류 |")
        print("|---|---|---|---|---|---|---|---|---|")
        T = np.zeros(9, int)
        for c in WINGS:
            q = [r for r in rr if r["cat"] == c]
            pts_ = [r for r in q if r["mode"] == "points"]
            v = [len(q), sum(r["mode"] == "points-snapped" for r in q), len(pts_), sum(r["reason"] == "칸 0개" for r in pts_),
                 sum(r["big_flip"] for r in q), sum(r["big_stretch"] for r in q),
                 sum(r["body_w"] != "" and r["body_w"] > 0.15 for r in q),
                 sum(r["wing_frac"] != "" and r["wing_frac"] >= 0.85 for r in q), sum(bool(r["err"]) for r in q)]
            T += v
            print(f"| {c} | {v[0]} | {v[1]} | {v[2]} ({v[3]} / {v[2] - v[3]}) | {v[4]} | {v[5]} | {v[6]} | {v[7]} | {v[8]} |")
        print(f"| 합계 | {T[0]} | {T[1]} ({T[1] / T[0]:.1%}) | {T[2]} ({T[3]} / {T[2] - T[3]}) | {T[4]} ({T[4] / T[0]:.1%}) | {T[5]} | {T[6]} | {T[7]} | {T[8]} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--split", default="train", choices=["train", "check"])
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", default="now")
    ap.add_argument("--only", default="", help="쉼표로 카테고리만")
    a = ap.parse_args()
    jobs = []
    for cat in EXPECT:
        if a.only and cat not in a.only.split(","):
            continue
        for k, s in enumerate(load_split(Path(a.qd) / f"{cat}.ndjson", cat, a.split, a.seed)):
            jobs.append((cat, a.split, k, s))
    with Pool(4) as p:
        rows = [r for rs in p.imap(one, jobs, chunksize=2) for r in rs]
    keys = ["cat", "split", "id", "n_regions", "pts", "mode", "stretch", "flip", "big_flip", "big_stretch", "body_w", "wing_frac",
            "reason", "err"]
    with open(HERE / f"rw_{a.out}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    summary(rows)


if __name__ == "__main__":
    main()
