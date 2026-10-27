"""날개 「점 3개」 흔들기 검사: 점이 조금 빗나가도 결과가 괜찮은가.

그림마다 맞는 점(뿌리 위 · 뿌리 아래 · 날개 끝)을 정해 두고, 점마다 반경 R(잘라 낸 그림 px) 안에서 무작위로 흔들어 N 번 돌린다.
점 3개 그대로(raw)와 칸에 맞춘 것(snap)을 check_wings.py 와 같은 기준으로 판정해 통과율을 낸다.
(움직여야 할 점 ≥ 0.6 · 가만히 있어야 할 점 ≤ 0.15 · 늘어남 ≤ 5배 · 뒤집힌 삼각형 ≤ 1%)

그림 묶음: old = 예전 8장(아래 TRUTH), train · test = gen_wing_point_cases.py 의 학습용 · 확인용 42장씩.
실패는 그림 · 점이 떨어진 곳(날개 칸 / 몸 칸 / 선 위 / 종이) · 쓴 방식(칸 맞춤 / 점 그대로)별로 센다.

    ../venv/bin/python check_wing_points.py                                   # old, R=10, N=8
    ../venv/bin/python check_wing_points.py --set test --r 10 16 24 --n 24 --raw --csv out/wp.csv
    ../venv/bin/python check_wing_points.py --set train --which tip           # 날개 끝 점만 흔들기
"""
import argparse
import csv
import json
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

import check_wings as cw
import gen_wing_point_cases as gp
import rig_types as rt

# 그림: 날개마다 [뿌리 위, 뿌리 아래, 끝] (좌표계는 check_wings 와 같다), 날개짓
TRUTH = {
    "front_bird": ([[(240, 200), (235, 245), (35, 140)], [(320, 200), (325, 245), (525, 140)]], "updown"),
    "top_butterfly": ([[(244, 130), (244, 305), (70, 150)], [(276, 130), (276, 305), (450, 150)]], "fold"),
    "side_bird": ([[(130, 96), (232, 106), (105, 50)]], "updown"),
    # 비스듬한 나비는 한쪽에 날개가 두 장(앞 · 뒤)이라 날개 끝을 둘 찍는다 (같은 뿌리)
    "coloring_butterfly_3q": ([[(205, 240), (165, 330), (40, 30)], [(205, 240), (165, 330), (25, 340)]], "fold"),
    "lineart_butterfly": ([[(242, 130), (242, 310), (65, 150)], [(278, 130), (278, 310), (455, 150)]], "fold"),
    "fairy": ([[(200, 215), (208, 250), (75, 180)], [(320, 215), (312, 250), (445, 180)]], "updown"),
    "crayon_wings": ([[(240, 140), (240, 250), (70, 175)], [(280, 140), (280, 250), (450, 175)]], "fold"),
    "dragonfly": ([[(266, 105), (266, 180), (55, 140)], [(294, 105), (294, 180), (505, 140)]], "updown"),
}
POINT_NAMES = ("뿌리 위", "뿌리 아래", "끝")


def case_info(name):
    if name in cw.EXISTING:
        path, coord, _t, _s, move, stay, _st = cw.EXISTING[name]
    else:
        _fn, _t, _s, move, stay, _st, _e = cw.CASES[name]
        path, coord = cw.CASE_DIR / f"{name}.png", "orig"
    return path, coord, move, stay


def load_case(name):
    """→ (원래 그림, 날개 점, 날개짓, 움직여야 할 점, 가만히 있어야 할 점, 좌표계)"""
    if name in TRUTH:
        path, coord, move, stay = case_info(name)
        wings, style = TRUTH[name]
        return rt.sr.load_image(path), wings, style, move, stay, coord
    img, info = gp.by_name(name)
    return img, info["wings"], info["style"], info["move"], info["stay"], "orig"


def judge(d, mask, f, pts, move, stay):
    """반환: (통과 여부, 이유, 쓴 방식)"""
    try:
        rig, used = rt.resolve_wings(d, mask, pts)
        anim = rt.animate(d, "wings", pts)
    except Exception as e:
        return False, f"오류 {e}", "-"
    wm = rig["wmap"]
    why = []
    for p in move:
        if cw.weight_at(wm, f(p)) < cw.MOVE_MIN:
            why.append("날개 안 움직임")
            break
    for p in stay:
        if cw.weight_at(wm, f(p)) > cw.STAY_MAX:
            why.append("몸통 딸려 옴")
            break
    if cw.stretch(anim) > 5:
        why.append("찢김")
    if cw.flipped(anim) > 0.01:
        why.append("뭉개짐")
    return not why, ",".join(why), used


def landing(d, mask, p, wing_ids, body_ids):
    """점이 떨어진 곳: 날개 칸 · 몸 칸 · 다른 칸 · 선 위(그림 안이지만 칸 아님) · 종이(그림 밖)."""
    lab, _ = rt.region_labels(d)
    x, y = int(round(p[0])), int(round(p[1]))
    r = int(lab[y, x])
    if r == 0:
        return "선 위" if mask[y, x] else "종이"
    if r in wing_ids:
        return "날개 칸"
    if r in body_ids:
        return "몸 칸"
    return "다른 칸"


def run_case(job):
    name, radii, n, which, raw = job
    img, wings, style, move, stay, coord = load_case(name)
    d = rt.Drawing(img)
    mask, _ = d.masks("wings")
    f = (lambda p: p) if coord == "crop" else cw.to_crop(d, img)
    base = [[np.array(f(p), np.float32) for p in w] for w in wings]
    # 정답 칸: 흔들지 않은 점으로 칸에 맞춘 결과 + 움직여야 할 점이 든 칸 / 몸 칸: 가만히 있어야 할 점이 든 칸
    lab, _ = rt.region_labels(d)
    snapped = rt.snap_points_to_regions(d, mask, {"wings": [[p.tolist() for p in w] for w in base]}) or []
    wing_ids = set(snapped) | {rt.region_at(d, *f(p)) for p in move}
    body_ids = {rt.region_at(d, *f(p)) for p in stay} - {0}
    wing_ids -= body_ids | {0}
    rows = []
    for R in radii:
        rng = np.random.default_rng(zlib.crc32(f"{name}/{R}/{which}".encode()))
        for t in range(n + 1):
            jit = []
            for w in base:
                ww = []
                for j, p in enumerate(w):
                    shake = t > 0 and (which == "all" or (which == "tip") == (j == 2))
                    q = p + (rng.uniform(-1, 1, 2) * R if shake else 0)
                    ww.append(np.clip(q, 0, [mask.shape[1] - 1, mask.shape[0] - 1]).tolist())
                jit.append(ww)
            row = {"name": name, "r": R, "t": t, "pts": json.dumps([[[round(v, 1) for v in p] for p in w] for w in jit])}
            for wi, w in enumerate(jit):
                for j, p in enumerate(w):
                    row[f"w{wi}_{j}"] = landing(d, mask, p, wing_ids, body_ids)
            good, why, used = judge(d, mask, f, {"wings": jit, "style": style, "snap": True}, move, stay)
            row.update(snap_ok=int(good), snap_why=why, used=used)
            if raw:
                good_r, why_r, _ = judge(d, mask, f, {"wings": jit, "style": style, "snap": False}, move, stay)
                row.update(raw_ok=int(good_r), raw_why=why_r)
            rows.append(row)
    return rows


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def report(rows, raw):
    radii = sorted({r["r"] for r in rows})
    names = list(dict.fromkeys(r["name"] for r in rows))
    print(f"\n== 그림별 통과 (칸에 맞춤{' / 점 그대로' if raw else ''}) ==")
    print(f"{'그림':28s} " + " ".join(f"{'R=' + str(int(R)):>11s}" for R in radii) + "   실패 이유 (칸에 맞춤, 전체 반경)")
    for nm in names:
        rr = [r for r in rows if r["name"] == nm]
        cells = []
        for R in radii:
            x = [r for r in rr if r["r"] == R]
            s = f"{sum(r['snap_ok'] for r in x)}/{len(x)}"
            if raw:
                s += f"·{sum(r['raw_ok'] for r in x)}"
            cells.append(f"{s:>11s}")
        why = Counter(w for r in rr if not r["snap_ok"] for w in r["snap_why"].split(","))
        print(f"{nm:28s} " + " ".join(cells) + "   " + "; ".join(f"{k} {v}" for k, v in why.most_common()))
    print("\n== 반경별 전체 ==")
    for R in radii:
        x = [r for r in rows if r["r"] == R]
        s = f"R={R:>4.0f}: 칸에 맞춤 {pct(sum(r['snap_ok'] for r in x), len(x))}"
        if raw:
            s += f" · 점 그대로 {pct(sum(r['raw_ok'] for r in x), len(x))}"
        print(s)
    print("\n== 쓴 방식별 (칸에 맞춤 경로) ==")
    by = defaultdict(list)
    for r in rows:
        by[r["used"]].append(r["snap_ok"])
    for k, v in by.items():
        print(f"  {k:16s} {pct(sum(v), len(v))} 통과")
    print("\n== 점이 떨어진 곳별 실패율 (칸에 맞춤, 흔든 것만) ==")
    for j, pn in enumerate(POINT_NAMES):
        by = defaultdict(list)
        for r in rows:
            if r["t"] == 0:
                continue
            for wi in range(2):
                k = f"w{wi}_{j}"
                if k in r:
                    by[r[k]].append(r["snap_ok"])
        print(f"  {pn:6s} " + " · ".join(f"{k} 실패 {pct(len(v) - sum(v), len(v))}" for k, v in sorted(by.items())))
    tot = sum(r["snap_ok"] for r in rows)
    print(f"\n전체 칸에 맞춤 {pct(tot, len(rows))}" + (f" · 점 그대로 {pct(sum(r['raw_ok'] for r in rows), len(rows))}" if raw else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, nargs="+", default=[10], help="흔드는 반경 (잘라 낸 그림 px), 여러 개 가능")
    ap.add_argument("--n", type=int, default=8, help="그림 · 반경마다 몇 번 (+ 흔들지 않은 1번)")
    ap.add_argument("--set", default="old", help="old · train · test · all, 또는 그림 이름(쉼표)")
    ap.add_argument("--which", default="all", choices=("all", "tip", "roots"), help="흔들 점")
    ap.add_argument("--raw", action="store_true", help="점 3개 그대로도 판정 (느림)")
    ap.add_argument("--csv", help="한 번마다 결과를 CSV 로")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    if args.set == "old":
        names = list(TRUTH)
    elif args.set in ("train", "test"):
        names = gp.names(args.set)
    elif args.set == "all":
        names = list(TRUTH) + gp.names()
    else:
        names = args.set.split(",")
    print(f"그림 {len(names)}장 · 반경 {args.r}px · 그림마다 {args.n}번 (+ 흔들지 않은 1번) · 흔든 점 {args.which}")
    jobs = [(nm, args.r, args.n, args.which, args.raw) for nm in names]
    with Pool(args.jobs) as pool:
        rows = [r for rr in pool.imap(run_case, jobs) for r in rr]
    report(rows, args.raw)
    if args.csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
