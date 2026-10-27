"""긴 몸 똬리 · 꼬인 뱀 시험 (LONG_COIL.md). QuickDraw snake 로 늘어남 · 뒤집힘 · 뼈대가 몸을 덮는 비율을 잰다.

    venv/bin/python -I eval_real/long_coil_eval.py --qd 받은폴더 --out after
받은폴더: snake.ndjson (full/simplified, 앞 30MB). 결과: eval_real/long_coil_{out}.csv
그림 묶음 (모두 seed 23 으로 섞은 인식된 그림):
- rand_train / rand_test: 무작위 170장씩
- coil_train / coil_test: 나머지에서 한 획이 3.3π 넘게 한 방향으로 도는 그림(똬리 후보)을 섞인 순서대로 150장씩
모양(일자 L · S자 S · 똬리 C · 닿음/겹침 T · 기타 O)은 눈으로 매긴 eval_real/long_coil_labels.csv 를 붙인다.
"""
import argparse
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
import check_wings as cw  # noqa: E402
import rig_types as rt  # noqa: E402
from run_eval import render  # noqa: E402


def winding(stroke):
    p = np.array(stroke, float).T
    if len(p) < 4:
        return 0.0
    v = np.diff(p, axis=0)
    v = v[np.linalg.norm(v, axis=1) > 0]
    da = np.diff(np.arctan2(v[:, 1], v[:, 0]))
    return abs(((da + np.pi) % (2 * np.pi) - np.pi).sum())


def load_sets(path, n=170, n_coil=150, seed=23):
    rows = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue  # 받다 만 마지막 줄
        if r.get("recognized"):
            rows.append(r["drawing"])
    idx = np.random.default_rng(seed).permutation(len(rows))
    coil = [i for i in idx[2 * n:] if max(winding(s) for s in rows[i]) > 3.3 * np.pi]
    return {"rand_train": [rows[i] for i in idx[:n]], "rand_test": [rows[i] for i in idx[n:2 * n]],
            "coil_train": [rows[i] for i in coil[:n_coil]], "coil_test": [rows[i] for i in coil[n_coil:2 * n_coil]]}


def cover(mask, P):
    """기준점을 이은 선에서 (굵기 90% 값 × 2 + 3)px 안에 드는 마스크 비율."""
    dense = np.vstack([np.linspace(a, b, max(2, int(np.linalg.norm(b - a)))) for a, b in zip(P[:-1], P[1:])])
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    w = np.array([dt[int(y), int(x)] for x, y in dense])
    line = np.full(mask.shape, 255, np.uint8)
    pi = np.round(dense).astype(int)
    line[pi[:, 1].clip(0, mask.shape[0] - 1), pi[:, 0].clip(0, mask.shape[1] - 1)] = 0
    near = cv2.distanceTransform(line, cv2.DIST_L2, 5) <= 2 * float(np.percentile(w, 90)) + 3
    return float((near & (mask > 0)).sum() / max(1, (mask > 0).sum()))


def one(job):
    split, k, strokes = job
    row = dict(split=split, id=k, ok=0, stretch="", flip="", cover="", t="", err="", head="", tail="", how="")
    t0 = time.time()
    try:
        d = rt.Drawing(render(strokes))
        pts = rt.propose(d)["points"]["long"]
        row["head"], row["tail"] = pts["head"], pts["tail"]
        a = rt.animate(d, "long", pts)
        row["how"] = getattr(d, "_long_how", "")
        row["ok"] = 1
        row["stretch"], row["flip"] = round(cw.stretch(a), 2), round(cw.flipped(a), 4)
        row["cover"] = round(cover(d.masks("long")[0], rt.BUILD["long"](d.masks("long")[0], pts)[0]), 3)
    except Exception as e:
        row["err"] = repr(e)[:100]
    row["t"] = round(time.time() - t0, 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--out", default="after")
    a = ap.parse_args()
    sets = load_sets(Path(a.qd) / "snake.ndjson")
    labs = {(r["split"], int(r["id"])): r for r in csv.DictReader(open(HERE / "long_coil_labels.csv", encoding="utf-8-sig"))}
    jobs = [(s, i, st) for s, v in sets.items() for i, st in enumerate(v)]
    with Pool() as p:
        rows = p.map(one, jobs, chunksize=4)
    for r in rows:
        r["shape"] = labs.get((r["split"], r["id"]), {}).get("shape", "")
    with open(HERE / f"long_coil_{a.out}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    names = {"L": "일자", "S": "S자", "C": "똬리", "T": "닿음/겹침", "O": "기타"}
    for grp in (("rand_train", "coil_train"), ("rand_test", "coil_test")):
        print(f"\n## {' + '.join(grp)}")
        print("| 모양 | 장수 | 늘어남>5배 | 뒤집힘>1% | 늘어남 중앙값 | 마스크 바뀜 | 오류 |")
        print("|---|---|---|---|---|---|---|")
        for sh in "LSCTO":
            rs = [r for r in rows if r["split"] in grp and r["shape"] == sh]
            ok = [r for r in rs if r["ok"]]
            if not rs:
                continue
            st = sorted(r["stretch"] for r in ok)
            print(f"| {names[sh]} | {len(rs)} | {sum(r['stretch'] > 5 for r in ok)} | {sum(r['flip'] > 0.01 for r in ok)} | "
                  f"{st[len(st) // 2]:.2f} | {sum(r['how'] not in ('', '그대로') for r in ok)} | {len(rs) - len(ok)} |")


if __name__ == "__main__":
    main()
