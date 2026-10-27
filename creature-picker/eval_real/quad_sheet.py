"""눈으로 보기 도우미 (결과 그림은 eval_real/data/ 아래, git 제외).

    venv/bin/python -I eval_real/quad_sheet.py grid  --csv quad_before.csv --split learn [--cat cat] [--per 48]
        → 그림 원본 썸네일 격자 (그림 종류 손으로 매기기: 얼굴만 · 몸 있는 네 발 · 기타)
    venv/bin/python -I eval_real/quad_sheet.py worst --csv quad_before.csv --split learn [--only-bad]
        → 망가진 그림마다 원본 + 가장 늘어난 장면(빨간 점 = 가장 늘어난 모서리)
    venv/bin/python -I eval_real/quad_sheet.py compare --old 예전_rig_types.py --split check
        → eval_real/sheet_quad_stretch.png: 줄마다 원본 | 예전 두 장면 | 지금 두 장면 (확인용에서 무작위:
          고쳐진 진짜 네 발 · 몸통만으로 돌린 얼굴 · 몸통만으로 잘못 돌린 진짜 네 발 · 아직 망가진 것 · 원래 괜찮던 진짜 네 발)
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
OUT = HERE / "data" / "sheets"


def thumb(path, T, tag):
    im = cv2.resize(cv2.imread(str(path)), (T, T), interpolation=cv2.INTER_AREA)
    cv2.rectangle(im, (0, 0), (T - 1, T - 1), (200, 200, 200), 1)
    cv2.putText(im, tag, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 220), 1, cv2.LINE_AA)
    return im


def grid(rows, T, cols, name):
    tiles = [thumb(HERE / "data" / "quad" / f"{r['cat']}_{r['split']}_{int(r['id']):03d}.png", T,
                   f"{r['cat'][:1]}{int(r['id'])}") for r in rows]
    while len(tiles) % cols:
        tiles.append(np.full((T, T, 3), 255, np.uint8))
    img = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / name), img)
    print(OUT / name)


def worst(rows, name, T=200):
    import check as ck
    import rig_types as rt
    lines = []
    for r in rows:
        png = HERE / "data" / "quad" / f"{r['cat']}_{r['split']}_{int(r['id']):03d}.png"
        d = rt.Drawing(cv2.imread(str(png)))
        pts = rt.propose(d)["points"]["quadruped"]
        anim = rt.animate(d, "quadruped", pts)
        fr = ck.frames_of(d, "quadruped", anim)
        src = d.crop.copy()
        cv2.circle(src, (int(float(r["worst_x"])), int(float(r["worst_y"]))), 8, (0, 0, 255), 2)
        cells = [cv2.resize(src, (T, T))]
        for f in (len(fr) // 4, 3 * len(fr) // 4):
            im = fr[f]
            s = T / max(im.shape[:2])
            im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
            c = np.full((T, T, 3), 255, np.uint8)
            c[:im.shape[0], :im.shape[1]] = im
            cells.append(c)
        line = np.hstack(cells)
        cv2.putText(line, f"{r['cat']}#{r['id']} str={r['stretch']} flip={r['flipped']} thick={r['worst_thick']}", (3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)
        lines.append(line)
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(lines), 12):
        p = OUT / f"{name}_{i // 12}.png"
        cv2.imwrite(str(p), np.vstack(lines[i:i + 12]))
        print(p)


def compare(old_path, split, seed=3, T=170):
    import importlib.util
    import check as ck
    import rig_types as rt
    spec = importlib.util.spec_from_file_location("rt_old", old_path)
    rt_old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt_old)
    B = {(r["cat"], r["id"]): r for r in csv.DictReader(open(HERE / "quad_before.csv", encoding="utf-8-sig")) if r["split"] == split}
    A = {(r["cat"], r["id"]): r for r in csv.DictReader(open(HERE / "quad_after.csv", encoding="utf-8-sig")) if r["split"] == split}
    lab = {(r["cat"], r["id"]): r["label"] for r in csv.DictReader(open(HERE / "quad_labels.csv", encoding="utf-8-sig")) if r["split"] == split}
    groups = [("fixed Q", [k for k in B if lab[k] == "Q" and B[k]["bad"] == "1" and A[k]["bad"] == "0" and A[k]["gate"] == "0"]),
              ("face->blob", [k for k in B if lab[k] == "F" and A[k]["gate"] == "1" and B[k]["bad"] == "1"]),
              ("Q->blob (wrong)", [k for k in B if lab[k] == "Q" and A[k]["gate"] == "1"]),
              ("still bad", [k for k in B if A[k]["bad"] == "1"]),
              ("ok Q", [k for k in B if lab[k] == "Q" and B[k]["bad"] == "0" and A[k]["bad"] == "0" and A[k]["gate"] == "0"])]
    rng = np.random.default_rng(seed)

    def cells(mod, d, r):
        pts = mod.propose(d)["points"]["quadruped"]
        anim = mod.animate(d, "quadruped", pts)
        kind = anim.get("used_kind") or "quadruped"
        fr = ck.frames_of(d, kind, anim)
        out = []
        for f in (len(fr) // 4, 3 * len(fr) // 4):
            im = fr[f]
            s = T / max(im.shape[:2])
            im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
            c = np.full((T, T, 3), 255, np.uint8)
            c[:im.shape[0], :im.shape[1]] = im
            out.append(c)
        return out

    lines = []
    for name, ks in groups:
        for k in [ks[i] for i in rng.permutation(len(ks))[:4]]:
            png = HERE / "data" / "quad" / f"{k[0]}_{split}_{int(k[1]):03d}.png"
            img = cv2.imread(str(png))
            src = cv2.resize(img, (T, T))
            old = cells(rt_old, rt_old.Drawing(img), B[k])
            new = cells(rt, rt.Drawing(img), A[k])
            sep = np.full((T, 6, 3), 120, np.uint8)
            line = np.hstack([src, sep] + old + [sep] + new)
            tag = (f"{name} {k[0]}#{k[1]} [{lab[k]}]  before str={B[k]['stretch']} flip={B[k]['flipped']}  "
                   f"after str={A[k]['stretch']} flip={A[k]['flipped']} {A[k]['kind_used']}")
            cv2.putText(line, tag, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1, cv2.LINE_AA)
            lines.append(line)
    cv2.imwrite(str(HERE / "sheet_quad_stretch.png"), np.vstack(lines))
    print(HERE / "sheet_quad_stretch.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["grid", "worst", "compare"])
    ap.add_argument("--csv", default="quad_after.csv")
    ap.add_argument("--old", help="compare: 예전 rig_types.py (git show 로 꺼낸 것)")
    ap.add_argument("--split", default="learn")
    ap.add_argument("--cat")
    ap.add_argument("--per", type=int, default=48)
    ap.add_argument("--only-bad", action="store_true")
    ap.add_argument("--ids", help="cat:id,cat:id ...")
    a = ap.parse_args()
    if a.what == "compare":
        return compare(a.old, a.split)
    rows = list(csv.DictReader(open(HERE / a.csv, encoding="utf-8-sig")))
    rows = [r for r in rows if r["split"] == a.split and (not a.cat or r["cat"] == a.cat)]
    if a.only_bad:
        rows = [r for r in rows if r["bad"] == "1"]
    if a.ids:
        want = {tuple(x.split(":")) for x in a.ids.split(",")}
        rows = [r for r in rows if (r["cat"], r["id"]) in want]
    if a.what == "grid":
        for i in range(0, len(rows), a.per):
            grid(rows[i:i + a.per], 130, 8, f"grid_{a.split}_{a.cat or 'all'}_{i // a.per}.png")
    else:
        worst(rows, f"worst_{a.split}_{a.cat or 'all'}")


if __name__ == "__main__":
    main()
