"""아이가 칠한 종이 그림 사진에서 무엇이 깨지는지 잰다. 코드는 고치지 않고 재기만 한다.

    QuickDraw 를 사진처럼 바꿔서:  venv/bin/python -I eval_real/colored_eval.py qd --qd 받은폴더 --set learn|check [--n 20]
    Meta Amateur Drawings 사진:     venv/bin/python -I eval_real/colored_eval.py meta --meta 받은폴더 --set learn|check
받은폴더(qd): {category}.ndjson 10개. 카테고리마다 seed 27 로 섞어 앞 n장 = 학습용(learn, 고칠 때 봄), 그다음 n장 = 확인용(check, 마지막에 한 번).
받은폴더(meta): amateur_drawings/ 와 ann_part.json(annotation 중 받은 그림 것만). 받은 순서 앞 200장 = 학습용, 나머지 = 확인용.
결과: eval_real/colored_<qd|meta>_<set>.csv, 화면에 표.
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
import kid_photo as kp  # noqa: E402
import rig_types as rt  # noqa: E402
from run_eval import EXPECT  # noqa: E402

sr = rt.sr


def crop_box(img):
    """shape_rig.crop_and_mask 가 자른 곳(원본 좌표). 마스크를 원본 좌표로 되돌려 정답과 비교하려고 같은 계산을 한다."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    ink = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    area = st[:, cv2.CC_STAT_AREA]
    edge = {k for k in range(1, n) if area[k] >= 30 and (st[k, 0] <= 1 or st[k, 1] <= 1 or st[k, 0] + st[k, 2] >= w - 1 or st[k, 1] + st[k, 3] >= h - 1)}
    inner = sum(area[k] for k in range(1, n) if area[k] >= 30 and k not in edge)
    full = inner < sum(area[k] for k in edge)
    for k in range(1, n):
        if area[k] < 30 or (k in edge and (not full or sr.is_straight(lab, k))):
            ink[lab == k] = 0
    ys, xs = np.nonzero(ink)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.15 * max(x1 - x0, y1 - y0))
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def mask_iou(img, d, gt):
    if hasattr(rt, "paper_only"):  # Drawing 이 종이 밖을 먼저 지운다 → 자른 곳도 지운 그림에서 계산
        img = rt.paper_only(img)
    x0, y0, x1, y1 = crop_box(img)
    m = ((d.mask > 0) | (d.alpha > 0)).astype(np.uint8)
    s = sr.MAX_SIDE / max(y1 - y0, x1 - x0)
    if (int((x1 - x0) * s), int((y1 - y0) * s)) != (m.shape[1], m.shape[0]):
        return float("nan")  # 자른 곳 계산이 어긋남 (shape_rig 가 바뀜)
    full = np.zeros(gt.shape, bool)
    full[y0:y1, x0:x1] = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return float((full & gt).sum() / max(1, (full | gt).sum()))


def run_one(img, gt, expect):
    """expect: 기대 종류 (None 이면 사람 그림: 추측한 종류로, 모르겠음이면 몸통만으로 움직여 봄)."""
    row = dict(drawing_ok=0, iou="", mask_fail=1, guess="", n_regions="", anim_kind="", anim_ok=0, stretch="", flipped="",
               big_stretch=0, big_flip=0, wing_mode="", t_sec="", err="")
    t0 = time.time()
    try:
        d = rt.Drawing(img)
        row["drawing_ok"] = 1
    except Exception as e:
        row["err"] = "Drawing: " + repr(e)[:120]
        row["t_sec"] = round(time.time() - t0, 2)
        return row, None, None
    anim = None
    try:
        iou = mask_iou(img, d, gt)
        row["iou"] = round(iou, 3)
        row["mask_fail"] = int(not iou >= 0.5)
        lab, _ = rt.region_labels(d)
        row["n_regions"] = int(lab.max())
        pr = rt.propose(d)
        g = pr["guess"]
        row["guess"] = g or "None"
        kind = expect or g or "blob"
        row["anim_kind"] = kind
        anim = rt.animate(d, kind, pr["points"][kind])
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
    return row, d, anim


def qd_job(job):
    cat, k, variant, strokes, other, seed = job
    img, gt = kp.make(strokes, variant, seed, other)
    row, _, _ = run_one(img, gt, EXPECT[cat])
    return dict(cat=cat, id=k, variant=variant, expect=EXPECT[cat], **row)


def meta_job(job):
    k, path, segs, scale = job
    img = cv2.imread(path)
    if scale != 1:  # 받은 그림은 긴 변 512px 로 줄어 있다. 휴대폰 원본 해상도에서 선 굵기가 어떤지 보려고 키워 본다
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gt = np.zeros(img.shape[:2], np.uint8)
    for s in segs:
        cv2.fillPoly(gt, [np.round(np.array(s).reshape(-1, 2) * scale).astype(np.int32)], 1)
    row, _, _ = run_one(img, gt > 0, None)
    return dict(cat="human", id=k, variant="meta", expect="", file=Path(path).name, h=img.shape[0], w=img.shape[1], **row)


def load_qd(path, n, seed=27):
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
    return {"learn": [rows[i] for i in idx[:n]], "check": [rows[i] for i in idx[n:2 * n]], "other": [rows[i] for i in idx[-50:]]}


def pct(a, b):
    return f"{a} ({100 * a / max(1, b):.0f}%)"


def summarize(rows, variants):
    """변환마다 한 줄."""
    out = []
    base = None
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        if not rs:
            continue
        n = len(rs)
        dfail = sum(1 - r["drawing_ok"] for r in rs)
        mfail = sum(r["mask_fail"] for r in rs)
        ious = [r["iou"] for r in rs if r["iou"] != "" and r["iou"] == r["iou"]]
        ok = sum(r["guess"] == r["expect"] for r in rs)
        none = sum(r["guess"] in ("None", "(오류)", "") for r in rs)
        bad = n - ok - none
        nreg = [r["n_regions"] for r in rs if r["n_regions"] != ""]
        r0 = sum(x == 0 for x in nreg)
        aerr = sum(r["drawing_ok"] and not r["anim_ok"] for r in rs)
        bs = sum(r["big_stretch"] for r in rs)
        bf = sum(r["big_flip"] for r in rs)
        ts = [r["t_sec"] for r in rs]
        s = dict(v=v, n=n, dfail=dfail, mfail=mfail, iou=np.median(ious) if ious else float("nan"), ok=ok, bad=bad, none=none,
                 reg=np.median(nreg) if nreg else float("nan"), reg0=r0, aerr=aerr, bs=bs, bf=bf, t=np.mean(ts), t90=np.percentile(ts, 90))
        if base is None:
            base = s
        out.append(s)
    return out, base


def print_table(summ, base, title):
    print(f"\n## {title}")
    print("| 변환 | 장수 | Drawing 실패 | 마스크 실패 (IoU<0.5) | IoU 중앙값 | 추측 맞음 | 추측 **틀림** | 모르겠음 | 칸 중앙값 | 칸 0개 | animate 오류 | 늘어남>5배 | 뒤집힘>1% | 평균 초 | 90% 초 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summ:
        n = s["n"]
        dbad = s["bad"] - base["bad"]
        print(f"| {kp.NAME.get(s['v'], s['v'])} | {n} | {pct(s['dfail'], n)} | {pct(s['mfail'], n)} | {s['iou']:.2f} | {pct(s['ok'], n)} | "
              f"**{pct(s['bad'], n)}**{'' if s is base else f' ({dbad:+d})'} | {pct(s['none'], n)} | {s['reg']:.0f} | {pct(s['reg0'], n)} | "
              f"{s['aerr']} | {pct(s['bs'], n)} | {pct(s['bf'], n)} | {s['t']:.2f} | {s['t90']:.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["qd", "meta"])
    ap.add_argument("--qd")
    ap.add_argument("--meta")
    ap.add_argument("--set", choices=["learn", "check"], default="learn")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--variants", default=",".join(kp.VARIANTS))
    ap.add_argument("--scale", type=float, default=1.0, help="meta: 그림을 이만큼 키워서 넣는다")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    variants = a.variants.split(",")
    name = a.out or f"colored_{a.mode}_{a.set}"
    if a.mode == "qd":
        jobs = []
        for ci, cat in enumerate(EXPECT):
            sp = load_qd(Path(a.qd) / f"{cat}.ndjson", a.n)
            for k, st in enumerate(sp[a.set]):
                seed = 1000 * ci + k + (0 if a.set == "learn" else 500)
                for v in variants:
                    jobs.append((cat, k, v, st, sp["other"][k % 50], seed))
        fn = qd_job
    else:
        ann = json.load(open(Path(a.meta) / "ann_part.json"))
        pairs = list(zip(ann["images"], ann["annotations"]))
        pairs = pairs[:200] if a.set == "learn" else pairs[200:]
        jobs = [(k, str(Path(a.meta) / im["file_name"]), an["segmentation"], a.scale) for k, (im, an) in enumerate(pairs)]
        jobs = [j for j in jobs if cv2.imread(j[1]) is not None]  # tar 앞부분만 받아서 마지막 그림은 잘려 있다
        fn = meta_job
    t0 = time.time()
    with Pool() as p:
        rows = p.map(fn, jobs, chunksize=2)
    print(f"{len(rows)}개 {time.time() - t0:.0f}초", flush=True)
    with open(HERE / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if a.mode == "qd":
        summ, base = summarize(rows, variants)
        print_table(summ, base, f"{name} (카테고리 10 × {a.n}장, 변환마다 같은 그림)")
    else:
        n = len(rows)
        c = {}
        for r in rows:
            c[r["guess"]] = c.get(r["guess"], 0) + 1
        ious = [r["iou"] for r in rows if r["iou"] != "" and r["iou"] == r["iou"]]
        print(f"\n## {name} ({n}장, 사람 그림)")
        print(f"- Drawing 실패 {pct(sum(1 - r['drawing_ok'] for r in rows), n)}, 마스크 실패(IoU<0.5) {pct(sum(r['mask_fail'] for r in rows), n)}, IoU 중앙값 {np.median(ious):.2f}")
        print("- 추측: " + ", ".join(f"{k} {pct(v, n)}" for k, v in sorted(c.items(), key=lambda x: -x[1])))
        for kind in rt.TYPES:
            rs = [r for r in rows if r["anim_kind"] == kind]
            if rs:
                print(f"- {kind} 로 움직임 {len(rs)}장: 오류 {sum(not r['anim_ok'] for r in rs)}, 늘어남>5배 {pct(sum(r['big_stretch'] for r in rs), len(rs))}, 뒤집힘>1% {pct(sum(r['big_flip'] for r in rs), len(rs))}")
        nreg = [r["n_regions"] for r in rows if r["n_regions"] != ""]
        ts = [r["t_sec"] for r in rows]
        print(f"- 칸 중앙값 {np.median(nreg):.0f}, 칸 0개 {pct(sum(x == 0 for x in nreg), n)}, 평균 {np.mean(ts):.2f}초, 90% {np.percentile(ts, 90):.2f}초, 최대 {max(ts):.1f}초")


if __name__ == "__main__":
    main()
