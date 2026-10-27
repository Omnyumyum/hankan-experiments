"""두 「네 발」 판(ac1c266 · a29c661)을 같은 그림 · 같은 라벨로 재는 공통 측정 (INTEGRATE.md).
두 판의 quad_eval.py 는 각자 코드에만 있는 칸(no_body · used_kind / fallback · pose.rig)을 읽어서 서로 바꿔 돌릴 수 없어 따로 만들었다.
    ../venv/bin/python -I eval_real/integrate/quad_cross.py --code 잴_creature-picker_폴더 --qd 받은폴더(앞 30MiB) --set A|B --out x_이름_setA.csv
--set A: seed 27, n 120, 확인용 (ac1c266 라벨 quad_labels.csv)
--set B: seed 23, n 125, 확인용 (a29c661 라벨 quad_labels.txt)
"""
import argparse, base64, csv, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--qd", required=True)
ap.add_argument("--set", choices=["A", "B"], required=True)
ap.add_argument("--split", default="check")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.code)
sys.path.insert(0, a.code + "/eval_real")
import rig_types as rt  # noqa
import check_wings as cw  # noqa
from run_eval import render  # noqa

CATS = ("dog", "cat", "horse", "pig")
SEED, N = (27, 120) if a.set == "A" else (23, 125)


def load(path):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("recognized"):
                rows.append(r["drawing"])
    idx = np.random.default_rng(SEED).permutation(len(rows))
    k = 0 if a.split == "learn" else 1
    return [rows[i] for i in idx[k * N:(k + 1) * N]]


def one(job):
    cat, k, strokes = job
    row = dict(cat=cat, id=k, stretch="", flipped="", bad="", blob=0, leg_move="", err="")
    try:
        img = render(strokes)
        d = rt.Drawing(img)
        pr = rt.propose(d)
        pts = pr["points"]["quadruped"]
        anim = rt.animate(d, "quadruped", pts)
        row["blob"] = int(anim.get("used_kind") == "blob" or anim.get("fallback") == "blob")
        row["stretch"] = round(cw.stretch(anim), 2)
        row["flipped"] = round(cw.flipped(anim), 4)
        row["bad"] = int(row["stretch"] > 5 or row["flipped"] > 0.01)
        # 다리 움직임: 미리 놓은 발 자리(끝 맞춤 후)에 가장 가까운 꼭짓점의 움직인 폭 ÷ 긴 변
        n = anim["n_vertices"]
        pos = np.frombuffer(base64.b64decode(anim["positions"]), np.float32).reshape(anim["frames"], n, 2)
        uv = np.frombuffer(base64.b64decode(anim["uvs"]), np.float32).reshape(n, 2) * [anim["w"], anim["h"]]
        amp = np.linalg.norm(pos.max(0) - pos.min(0), axis=1)
        feet = rt.snap_to_tips(d, pts)["feet"]
        size = max(anim["w"], anim["h"])
        row["leg_move"] = round(float(np.mean([amp[np.argmin(((uv - np.asarray(f)) ** 2).sum(1))] for f in feet])) / size, 4)
    except Exception as e:
        row["err"] = repr(e)[:160]
    return row


if __name__ == "__main__":
    jobs = []
    for cat in CATS:
        jobs += [(cat, k, s) for k, s in enumerate(load(Path(a.qd) / f"{cat}.ndjson"))]
    with Pool(4) as p:
        rows = p.map(one, jobs, chunksize=4)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(a.code, a.set, "bad", sum(r["bad"] == 1 for r in rows), "/", len(rows), "blob", sum(r["blob"] for r in rows),
          "err", sum(bool(r["err"]) for r in rows))
