"""「네 발」로 움직였을 때 늘어남 · 뒤집힘을 재는 시험 (REPORT.md 실패 1번). dog · cat · horse · pig 만.

    ../venv/bin/python -I eval_real/quad_eval.py --qd 받은폴더 [--n 125] [--seed 23] [--split learn|check|all] [--out 이름] [--png]
받은폴더: run_eval.py 와 같은 {category}.ndjson (앞부분만 받아도 된다).
카테고리마다 섞은 뒤 앞 n장 = 학습용(learn, 고칠 때 봄), 다음 n장 = 확인용(check, 마지막에 한 번 잼).
기준은 REPORT.md 와 같다: 늘어남 > 5배 또는 뒤집힌 삼각형 > 1% 면 「나쁨」.
결과: eval_real/quad_<out>.csv (그림마다). --png 면 그림을 eval_real/data/quad/ 에 저장 (git 제외).

원인 나누기에 쓰는 칸:
- worst_part: 가장 많이 늘어난 모서리에 가장 가까운 기준점이 어느 부위인지 (leg · head · tail · centre)
- worst_thin: 그 모서리가 가는 선 조각(선 굵기 2배보다 가는 곳) 위인지
- leg_move: 다리(배 아래) 꼭짓점이 움직인 정도의 평균 ÷ 그림 긴 변 (고친 뒤 진짜 네 발 그림에서 줄면 안 된다)
- fallback: animate 가 네 발 대신 다른 방식으로 돌렸으면 그 이름 (안전 판정)
"""
import argparse
import base64
import csv
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
from run_eval import render  # noqa: E402

CATS = ("dog", "cat", "horse", "pig")
SAVE_PNG = False


def load(path, n, seed):
    import json
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("recognized"):
                rows.append(r["drawing"])
    idx = np.random.default_rng(seed).permutation(len(rows))
    if len(idx) < 2 * n:
        raise SystemExit(f"{path}: 인식된 그림 {len(rows)}장 < {2 * n}장")
    return {"learn": [rows[i] for i in idx[:n]], "check": [rows[i] for i in idx[n:2 * n]]}


def decode(anim):
    n = anim["n_vertices"]
    pos = np.frombuffer(base64.b64decode(anim["positions"]), np.float32).reshape(anim["frames"], n, 2)
    tris = np.frombuffer(base64.b64decode(anim["indices"]), np.uint32).reshape(-1, 3).astype(np.int64)
    uv = np.frombuffer(base64.b64decode(anim["uvs"]), np.float32).reshape(n, 2) * [anim["w"], anim["h"]]
    return pos, tris, uv


def worst_edge(anim):
    pos, tris, uv = decode(anim)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    l0 = np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1) + 1e-3
    best, at = 0.0, None
    for f in range(len(pos)):
        r = np.linalg.norm(pos[f, e[:, 0]] - pos[f, e[:, 1]], axis=1) / l0
        k = int(np.argmax(r))
        if r[k] > best:
            best, at = float(r[k]), (uv[e[k, 0]] + uv[e[k, 1]]) / 2
    return best, at


def one(job):
    cat, split, k, strokes = job
    row = dict(cat=cat, split=split, id=k, guess="", n_feet=0, stretch="", flipped="", bad="", fallback="",
               worst_x="", worst_y="", worst_part="", worst_thin="", leg_move="", err="")
    img = render(strokes)
    if SAVE_PNG:
        (HERE / "data/quad").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(HERE / f"data/quad/{cat}_{split}_{k:03d}.png"), img)
    try:
        d = rt.Drawing(img)
        pr = rt.propose(d)
        row["guess"] = pr["guess"] or "None"
        row["n_feet"] = int(getattr(d, "_guess_feat", {}).get("n_feet", 0))
        pts = pr["points"]["quadruped"]
        anim = rt.animate(d, "quadruped", pts)
        row["fallback"] = anim.get("fallback") or ""
        row["stretch"] = round(cw.stretch(anim), 2)
        row["flipped"] = round(cw.flipped(anim), 4)
        row["bad"] = int(row["stretch"] > 5 or row["flipped"] > 0.01)
        _, p = worst_edge(anim)
        row["worst_x"], row["worst_y"] = round(float(p[0]), 1), round(float(p[1]), 1)
        if row["fallback"]:
            return row
        # 어느 부위 기준점에 가장 가까운가 (animate 와 같은 점으로 다시 만든다)
        mask, _ = d.masks("quadruped")
        sp = rt.snap_to_tips(d, pts)
        pins0, pose = rt.build_quadruped(mask, sp)
        nfeet = len(pose.rig["legs"])  # 겹친 발은 build_quadruped 가 합친다
        part = ["centre"] + ["leg"] * (3 * nfeet) + ["head"] * 3 + ["tail"] * (len(pins0) - 1 - 3 * nfeet - 3)
        j = int(np.argmin(((pins0 - p) ** 2).sum(1)))
        row["worst_part"] = part[j]
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        lw = rt.line_width(d.ink > 0)
        x, y = int(round(p[0])), int(round(p[1]))
        win = dt[max(0, y - 3): y + 4, max(0, x - 3): x + 4]
        row["worst_thin"] = int(win.max() <= lw + 1)
        pos, _, uv = decode(anim)
        y_belly = int(min(pins0[1 + 3 * i][1] for i in range(nfeet)))
        legv = uv[:, 1] > y_belly
        if legv.any():
            mv = np.linalg.norm(pos[:, legv] - uv[legv][None], axis=2).max(0)
            row["leg_move"] = round(float(mv.mean()) / max(d.mask.shape), 4)
    except Exception as e:
        row["err"] = repr(e)[:160]
    return row


def main():
    global SAVE_PNG
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--n", type=int, default=125)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--split", choices=["learn", "check", "all"], default="learn")
    ap.add_argument("--out", default="now")
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    SAVE_PNG = a.png
    splits = ["learn", "check"] if a.split == "all" else [a.split]
    jobs = []
    for cat in CATS:
        sp = load(Path(a.qd) / f"{cat}.ndjson", a.n, a.seed)
        for s in splits:
            jobs += [(cat, s, k, st) for k, st in enumerate(sp[s])]
    with Pool() as p:
        rows = p.map(one, jobs, chunksize=4)
    with open(HERE / f"quad_{a.out}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s in splits:
        rs = [r for r in rows if r["split"] == s]
        print(f"\n## {s} ({len(rs)}장)")
        for cat in CATS:
            rc = [r for r in rs if r["cat"] == cat]
            bad = sum(1 for r in rc if r["bad"] == 1)
            err = sum(1 for r in rc if r["err"])
            fb = sum(1 for r in rc if r["fallback"])
            print(f"{cat:6s} 나쁨 {bad:3d}/{len(rc)}  돌려보냄 {fb:3d}  오류 {err}")
        bad = sum(1 for r in rs if r["bad"] == 1)
        print(f"합계   나쁨 {bad}/{len(rs)} ({100 * bad / max(1, len(rs)):.0f}%)")


if __name__ == "__main__":
    main()
