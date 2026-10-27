"""more_eval.py 결과에서 고른 그림을 눈으로 보기: 줄마다 원본 + 움직임 3장면 (가장 늘어난 장면 포함).

    ../venv/bin/python -I eval_real/more_sheet.py --qd 받은폴더 --csv eval_real/more_<이름>.csv --where "broken=1,kind=quadruped" \
        [--split train] [--n 16] [--seed 3] [--out 파일.png]
--where : 칸=값 을 쉼표로 (모두 맞는 줄만). 결과 PNG 기본 위치는 eval_real/data/sheets/ (git 제외).
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import check as ck  # noqa: E402
import check_wings as cw  # noqa: E402
import rig_types as rt  # noqa: E402
from guess_eval import load_split  # noqa: E402
from run_eval import render  # noqa: E402

T = 160


def cell(im, tag=""):
    s = T / max(im.shape[:2])
    im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
    c = np.full((T, T, 3), 255, np.uint8)
    c[:im.shape[0], :im.shape[1]] = im
    cv2.rectangle(c, (0, 0), (T - 1, T - 1), (210, 210, 210), 1)
    if tag:
        cv2.putText(c, tag, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 200), 1, cv2.LINE_AA)
    return c


def line(img, kind, tag):
    d = rt.Drawing(img)
    pr = rt.propose(d)
    cells = [cell(img, tag)]
    try:
        anim = rt.animate(d, kind, pr["points"][kind])
        fr = ck.frames_of(d, anim["used_kind"], anim)
        st = cw.stretch(anim)
        worst = int(np.argmax([cw.stretch({**anim, "frames": 1, "positions": _one(anim, f)}) for f in range(anim["frames"])]))
        for f in (0, len(fr) // 3, worst):
            cells.append(cell(fr[f], f"f{f}" + (f" str {st:.1f}" if f == worst else "")))
    except Exception as e:
        cells += [cell(np.full((T, T, 3), 235, np.uint8), repr(e)[:30])] * 3
    return np.hstack(cells)


def _one(anim, f):
    import base64
    n = anim["n_vertices"]
    pos = np.frombuffer(base64.b64decode(anim["positions"]), np.float32).reshape(anim["frames"], n, 2)
    return base64.b64encode(np.ascontiguousarray(pos[f:f + 1]).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--where", default="")
    ap.add_argument("--split")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    cond = [kv.split("=", 1) for kv in a.where.split(",") if kv]
    rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8-sig"))
            if all(r[k] == v for k, v in cond) and (not a.split or r["split"] == a.split)]
    pick = [rows[i] for i in sorted(np.random.default_rng(a.seed).permutation(len(rows))[:a.n])]
    cache, lines = {}, []
    for r in pick:
        if r["cat"] not in cache:
            cache[r["cat"]] = load_split(Path(a.qd) / f"{r['cat']}.ndjson", 100, 11)
        img = render(cache[r["cat"]][r["split"]][int(r["id"])])
        lines.append(line(img, r["kind"], f"{r['cat']}#{r['split'][:2]}{r['id']} {r['guess']} f{r['n_feet']}"))
        print(r["cat"], r["id"], r["stretch"], r["flipped"], flush=True)
    out = Path(a.out) if a.out else HERE / "data" / "sheets" / "more.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(lines) if lines else np.full((T, T, 3), 255, np.uint8))
    print(out)


if __name__ == "__main__":
    main()
