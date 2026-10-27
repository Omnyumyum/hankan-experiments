"""실제 사람이 그린 그림(Google QuickDraw)으로 비휴머노이드 코드 시험. 코드는 고치지 않고 재기만 한다.

    venv/bin/python eval_real/run_eval.py --qd 받은폴더 [--n 30] [--seed 7]
받은폴더: {category}.ndjson 10개 (storage.googleapis.com/quickdraw_dataset/full/simplified/). 그림 PNG 는 eval_real/data/ (git 제외).
결과: eval_real/results.csv, eval_real/confusion.csv, eval_real/summary.json, eval_real/sheet_*.png
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import check as ck  # noqa: E402
import check_wings as cw  # noqa: E402
import rig_types as rt  # noqa: E402

EXPECT = {"dog": "quadruped", "cat": "quadruped", "horse": "quadruped", "pig": "quadruped", "snake": "long",
          "bird": "wings", "butterfly": "wings", "bee": "wings", "cloud": "blob", "star": "blob"}
KINDS = ["quadruped", "long", "wings", "blob", "None"]


def load_qd(path, n, rng):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue  # 받다 만 마지막 줄
            if r.get("recognized"):
                rows.append(r)
    idx = rng.permutation(len(rows))[:n]
    return [rows[i] for i in idx], len(rows)


def render(strokes, size=400, thick=6):
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    sc = (size - 2 * thick - 4) / max(max(xs) - min(xs), max(ys) - min(ys), 1)
    ox = (size - sc * (max(xs) - min(xs))) / 2 - sc * min(xs)
    oy = (size - sc * (max(ys) - min(ys))) / 2 - sc * min(ys)
    img = np.full((size, size, 3), 255, np.uint8)
    for s in strokes:
        p = np.array([[x * sc + ox, y * sc + oy] for x, y in zip(*s)], np.float32)
        cv2.polylines(img, [np.round(p).astype(np.int32)], False, (0, 0, 0), thick, cv2.LINE_AA)
        if len(p) == 1:
            cv2.circle(img, tuple(np.round(p[0]).astype(int)), thick // 2, (0, 0, 0), -1)
    return img


def one(cat, i, png):
    row = dict(cat=cat, id=i, expect=EXPECT[cat], drawing_ok=0, guess="", match=0, anim_ok=0, stretch="", flipped="",
               big_stretch=0, big_flip=0, wing_mode="", n_regions="", t_sec="", err="")
    t0 = time.time()
    d = anim = None
    try:
        d = rt.Drawing(rt.sr.load_image(png))
        row["drawing_ok"] = 1
    except Exception as e:
        row["err"] = "Drawing: " + repr(e)[:120]
        row["t_sec"] = round(time.time() - t0, 2)
        return row, None, None, None
    try:
        lab, _ = rt.region_labels(d)
        row["n_regions"] = int(lab.max())
        pr = rt.propose(d)
        g = pr["guess"]
        row["guess"] = g or "None"
        row["match"] = int(g == row["expect"])
        pts = pr["points"][row["expect"]]
        anim = rt.animate(d, row["expect"], pts)
        row["anim_ok"] = 1
        row["stretch"] = round(cw.stretch(anim), 2)
        row["flipped"] = round(cw.flipped(anim), 4)
        row["big_stretch"] = int(row["stretch"] > 5)
        row["big_flip"] = int(row["flipped"] > 0.01)
        row["wing_mode"] = anim.get("wing_mode") or ""
    except Exception as e:
        row["err"] = (row["err"] + " " + repr(e)[:160]).strip()
        if not row["guess"]:
            row["guess"] = "(오류)"
    row["t_sec"] = round(time.time() - t0, 2)
    return row, d, anim, png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    rows, keep = [], {}
    pool = {}
    for cat in EXPECT:
        recs, tot = load_qd(Path(a.qd) / f"{cat}.ndjson", a.n, rng)
        pool[cat] = tot
        for k, r in enumerate(recs):
            png = data / f"{cat}_{k:02d}.png"
            cv2.imwrite(str(png), render(r["drawing"]))
            row, d, anim, p = one(cat, k, png)
            rows.append(row)
            keep[(cat, k)] = (d, anim)
            print(cat, k, row["guess"], row["anim_ok"], row["stretch"], row["flipped"], row["wing_mode"], row["err"][:60], flush=True)
    import csv
    with open(HERE / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # 혼동표: 기대 종류(행) × 추측(열)
    cm = {}
    for r in rows:
        if r["drawing_ok"]:
            cm.setdefault((r["cat"], r["expect"]), {}).setdefault(r["guess"], 0)
            cm[(r["cat"], r["expect"])][r["guess"]] += 1
    cols = KINDS + ["(오류)"]
    with open(HERE / "confusion.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["카테고리", "기대"] + cols)
        for (cat, ex), c in cm.items():
            w.writerow([cat, ex] + [c.get(k, 0) for k in cols])
    json.dump({"pool": pool, "n": len(rows)}, open(HERE / "summary.json", "w"))
    make_sheets(rows, keep, rng)


def make_sheets(rows, keep, rng):
    """종류별 모아 보기: 성공 · 실패 무작위 표본 각 4장, 칸마다 원본 작은 썸네일 + 움직임 3장면."""
    T = 150
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["expect"], []).append(r)
    for kind, rs in by_kind.items():
        def bad(r):
            return (not r["anim_ok"]) or r["big_stretch"] or r["big_flip"] or not r["match"]
        sets = [("성공", [r for r in rs if not bad(r)]), ("실패", [r for r in rs if bad(r)])]
        lines = []
        for name, lst in sets:
            pick = [lst[i] for i in rng.permutation(len(lst))[:4]]
            for r in pick:
                d, anim = keep[(r["cat"], r["id"])]
                cells = []
                src = cv2.resize(cv2.imread(str(HERE / "data" / f"{r['cat']}_{r['id']:02d}.png")), (T, T))
                cells.append(src)
                if anim is not None:
                    fr = ck.frames_of(d, kind, anim)
                    for f in (0, len(fr) // 3, 2 * len(fr) // 3):
                        im = fr[f]
                        s = T / max(im.shape[:2])
                        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
                        c = np.full((T, T, 3), 255, np.uint8)
                        c[:im.shape[0], :im.shape[1]] = im
                        cells.append(c)
                else:
                    cells += [np.full((T, T, 3), 235, np.uint8)] * 3
                line = np.hstack(cells)
                tag = f"{name} {r['cat']}#{r['id']} guess={r['guess']} str={r['stretch']} flip={r['flipped']} {r['wing_mode']}"
                cv2.putText(line, tag, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200) if name == "실패" else (0, 120, 0), 1, cv2.LINE_AA)
                lines.append(line)
        if lines:
            cv2.imwrite(str(HERE / f"sheet_{kind}.png"), np.vstack(lines))


if __name__ == "__main__":
    main()
