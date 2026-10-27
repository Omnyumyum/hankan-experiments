"""「네 발」로 움직였을 때 늘어남 · 뒤집힘 시험 (REPORT.md 실패 1번). dog · cat · horse · pig 만.

    venv/bin/python -I eval_real/quad_eval.py --qd 받은폴더 [--n 120] [--seed 27] [--split learn|check|all] [--out 이름]
카테고리마다 섞은 뒤 앞 n장 = 학습용(learn, 고칠 때 보는 것), 다음 n장 = 확인용(check, 마지막에 한 번 재는 것).
기준은 REPORT.md 와 같다: 늘어남 > 5배 또는 뒤집힌 삼각형 > 1% 면 「망가짐」.
- 다리 움직임: 발 점마다 가장 가까운 메시 꼭짓점이 프레임 동안 움직인 폭(최대 - 최소 거리) ÷ 그림 긴 변, 발 평균.
  고친 뒤 진짜 네 발 그림에서 이 값이 줄면 안 된다.
- 가장 늘어난 모서리가 어디인지(굵기 · 배 아래인지 · 다리 덩어리인지)도 적어 둔다 → 원인 나눠 세기(quad_causes.py)에 쓴다.
결과: eval_real/quad_<out>.csv. 그린 PNG 는 eval_real/data/quad/ (git 제외).
"""
import argparse
import base64
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import rig_types as rt  # noqa: E402
from run_eval import render  # noqa: E402

CATS = ("dog", "cat", "horse", "pig")
DATA = HERE / "data" / "quad"


def load_split(path, n, seed):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue  # 받다 만 마지막 줄
            if r.get("recognized"):
                rows.append(r["drawing"])
    idx = np.random.default_rng(seed).permutation(len(rows))
    return {s: [rows[i] for i in idx[k * n:(k + 1) * n]] for k, s in enumerate(("learn", "check"))}


def decode(anim):
    n = anim["n_vertices"]
    pos = np.frombuffer(base64.b64decode(anim["positions"]), np.float32).reshape(anim["frames"], n, 2)
    tris = np.frombuffer(base64.b64decode(anim["indices"]), np.uint32).reshape(-1, 3).astype(np.int64)
    uv = np.frombuffer(base64.b64decode(anim["uvs"]), np.float32).reshape(n, 2) * [anim["w"], anim["h"]]
    return pos, tris, uv


def measure(d, anim, feet):
    """늘어남 · 뒤집힘 · 가장 늘어난 곳 · 다리 움직임."""
    pos, tris, uv = decode(anim)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    l0 = np.linalg.norm(uv[e[:, 0]] - uv[e[:, 1]], axis=1) + 1e-3
    ratio = np.stack([np.linalg.norm(pos[f, e[:, 0]] - pos[f, e[:, 1]], axis=1) / l0 for f in range(len(pos))])
    fi, ei = np.unravel_index(np.argmax(ratio), ratio.shape)

    def area(p):
        a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])

    s0 = np.sign(area(uv))
    flip = max(float((np.sign(area(pos[f])) != s0).mean()) for f in range(len(pos)))
    size = max(anim["w"], anim["h"])
    amp = np.linalg.norm(pos.max(0) - pos.min(0), axis=1)  # 꼭짓점마다 움직인 폭
    leg = [float(amp[np.argmin(((uv - np.asarray(f)) ** 2).sum(1))]) / size for f in feet] if feet else [0.0]
    wx, wy = (uv[e[ei, 0]] + uv[e[ei, 1]]) / 2
    mask, _ = d.masks("quadruped")
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return {"stretch": round(float(ratio.max()), 2), "flipped": round(flip, 4), "leg_move": round(float(np.mean(leg)), 4),
            "worst_x": round(float(wx), 1), "worst_y": round(float(wy), 1),
            "worst_thick": round(float(dt[min(int(wy), mask.shape[0] - 1), min(int(wx), mask.shape[1] - 1)]), 1),
            "n_stretch5": int((ratio.max(0) > 5).sum())}


def one(job):
    cat, split, k, strokes = job
    row = dict(cat=cat, split=split, id=k, guess="", quad_why="", n_feet="", kind_used="quadruped", gate="", anim_ok=0,
               stretch="", flipped="", bad="", leg_move="", worst_x="", worst_y="", worst_thick="", n_stretch5="", t_sec="", err="")
    t0 = time.time()
    img = render(strokes)
    DATA.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DATA / f"{cat}_{split}_{k:03d}.png"), img)
    try:
        d = rt.Drawing(img)
        pr = rt.propose(d)
        row["guess"] = pr["guess"] or "None"
        row["quad_why"] = pr["why"]["quadruped"][:60]
        pts = pr["points"]["quadruped"]
        row["n_feet"] = len(pts["feet"])
        row["gate"] = int(bool(pts.get("no_body")))  # 얼굴만 판정 → animate 가 몸통만으로 돌린다
        anim = rt.animate(d, "quadruped", pts)
        row["kind_used"] = anim.get("used_kind") or "quadruped"
        feet = rt.snap_to_tips(d, pts)["feet"]
        row.update(measure(d, anim, feet))
        row["anim_ok"] = 1
        row["bad"] = int(row["stretch"] > 5 or row["flipped"] > 0.01)
    except Exception as e:
        row["err"] = repr(e)[:160]
    row["t_sec"] = round(time.time() - t0, 2)
    return row


def summary(rows, title):
    print(f"\n## {title} ({len(rows)}장)")
    print("| 카테고리 | 장수 | 망가짐 | 늘어남>5 | 뒤집힘>1% | 오류 | 몸통만으로 돌림 | 다리 움직임(평균, 네 발로 돈 것) | 시간(평균) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for cat in CATS + ("합계",):
        rs = [r for r in rows if cat == "합계" or r["cat"] == cat]
        if not rs:
            continue
        ok = [r for r in rs if r["anim_ok"]]
        bad = sum(r["bad"] == 1 for r in ok)
        s5 = sum(r["stretch"] > 5 for r in ok)
        f1 = sum(r["flipped"] > 0.01 for r in ok)
        lm = np.mean([r["leg_move"] for r in ok if r["kind_used"] == "quadruped"] or [0])
        gate = sum(r["gate"] == 1 for r in rs)
        print(f"| {cat} | {len(rs)} | {bad} ({100 * bad / len(rs):.0f}%) | {s5} | {f1} | {len(rs) - len(ok)} | {gate} | {lm:.4f} | "
              f"{np.mean([r['t_sec'] for r in rs]):.2f}초 |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--split", choices=["learn", "check", "all"], default="learn")
    ap.add_argument("--out", default="now")
    a = ap.parse_args()
    splits = ["learn", "check"] if a.split == "all" else [a.split]
    jobs = []
    for cat in CATS:
        sp = load_split(Path(a.qd) / f"{cat}.ndjson", a.n, a.seed)
        for s in splits:
            jobs += [(cat, s, k, st) for k, st in enumerate(sp[s])]
    with Pool() as p:
        rows = p.map(one, jobs, chunksize=2)
    with open(HERE / f"quad_{a.out}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s in splits:
        summary([r for r in rows if r["split"] == s], f"{dict(learn='학습용', check='확인용')[s]} {a.out}")


if __name__ == "__main__":
    main()
