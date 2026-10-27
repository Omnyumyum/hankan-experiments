"""regions_wings_eval 결과에서 골라 모아 보기 그림을 만든다 (눈으로 원인 확인용).

    venv/bin/python -I eval_real/rw_sheet.py --qd 받은폴더 --csv eval_real/rw_이름.csv --what zero|snapfail|flip --out 그림.png [--max 24]
한 줄 = 원본 | 칸(색) | 날개 무게(빨강) + 점 | 움직임 1장면. 제목에 카테고리 · 번호 · 칸 수 · 방식 · 뒤집힘.
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
import check  # noqa: E402
import rig_types as rt  # noqa: E402
from regions_wings_eval import WINGS, load_split  # noqa: E402
from run_eval import render  # noqa: E402

S = 200


def colored(lab):
    rng = np.random.default_rng(3)
    pal = rng.integers(60, 255, (int(lab.max()) + 1, 3)).astype(np.uint8)
    pal[0] = 0
    return pal[lab]


def row_img(strokes, r):
    d = rt.Drawing(render(strokes))
    lab, _ = rt.region_labels(d)
    tiles = [cv2.resize(d.crop, (S, S)), cv2.resize(colored(lab), (S, S), interpolation=cv2.INTER_NEAREST)]
    if r["cat"] in WINGS:
        pr = rt.propose(d)
        pts = pr["points"]["wings"]
        mask, _ = d.masks("wings")
        rig, used = rt.resolve_wings(d, mask, pts)
        a = (rig["wmap"] * 0.6)[..., None]
        heat = np.zeros_like(d.crop, np.float32)
        heat[..., 2] = 255
        img = (d.crop * (1 - a) + heat * a).astype(np.uint8)
        for w in pts["wings"]:
            for j, p in enumerate(w):
                cv2.circle(img, (int(p[0]), int(p[1])), 5, ((0, 160, 0), (0, 160, 0), (255, 0, 0))[j], -1)
        tiles.append(cv2.resize(img, (S, S)))
        anim = rt.animate(d, "wings", pts)
        fr = check.frames_of(d, "wings", anim)
        tiles.append(cv2.resize(fr[8], (S, S)))
    row = np.hstack(tiles + [np.full((S, S, 3), 255, np.uint8)] * (4 - len(tiles)))
    cv2.putText(row, f"{r['cat']}#{r['id']} regions={r['n_regions']} {r.get('mode', '')} flip={r.get('flip', '')}", (4, 14), 0, 0.42,
                (0, 0, 200), 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--what", required=True, choices=["zero", "snapfail", "flip", "ids"])
    ap.add_argument("--ids", default="", help="cat:id,cat:id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=24)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8-sig")))
    if a.what == "zero":
        sel = [r for r in rows if r["pts"] in ("-", "prop") and r["n_regions"] == "0"]
    elif a.what == "snapfail":
        sel = [r for r in rows if r["pts"] == "prop" and r["mode"] == "points" and r["n_regions"] != "0"]
    elif a.what == "flip":
        sel = [r for r in rows if r["pts"] == "prop" and r["big_flip"] == "1"]
    else:
        want = {tuple(x.split(":")) for x in a.ids.split(",")}
        sel = [r for r in rows if r["pts"] in ("-", "prop") and (r["cat"], r["id"]) in want]
    sel = sel[: a.max]
    cache = {}
    out = []
    for r in sel:
        key = (r["cat"], r["split"])
        if key not in cache:
            cache[key] = load_split(Path(a.qd) / f"{r['cat']}.ndjson", r["cat"], r["split"], 23)
        out.append(row_img(cache[key][int(r["id"])], r))
        print(r["cat"], r["id"], r["n_regions"], r.get("mode"), r.get("reason"), r.get("flip"), flush=True)
    cv2.imwrite(a.out, np.vstack(out))


if __name__ == "__main__":
    main()
