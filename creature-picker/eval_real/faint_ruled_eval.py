"""연한 연필 · 줄 종이 그림에서 마스크를 지금 clean_photo(old) 와 faint_ruled.clean_photo(new) 로 나란히 잰다.

    Meta:      ../venv/bin/python -I eval_real/faint_ruled_eval.py meta --meta eval_real/data/meta --impl old|new --set learn|check [--out 이름]
    QuickDraw: ../venv/bin/python -I eval_real/faint_ruled_eval.py qd --qd eval_real/data/qd --impl old|new --set learn|check [--n 30] [--out 이름]
Meta 묶음: faint_ruled_pick.py 가 고른 faint · ruled, 그리고 대조(control) = 고르지 않은 그림 중 index 25장마다 1장.
  학습 · 확인은 파일 이름 md5 첫 글자 짝 · 홀 (faint_ruled_pick.split_of).
QuickDraw 변환: kid_photo 의 orig · fill · lined · mix_mild · mix_heavy 와 여기서 만든 faint(연한 연필) · faint_lined(연한 연필 + 줄 · 모눈).
  카테고리마다 seed 27 로 섞어 앞 n장 학습, 그다음 n장 확인 (colored_eval.load_qd 와 같음).
결과: --out 을 주면 eval_real/<이름>.csv (커밋용), 안 주면 eval_real/data/fr_<...>.csv. 화면에 묶음별 요약.
재는 것: Drawing 성공, 마스크 IoU(정답과), 마스크 실패(IoU<0.5), 사진 정리 시간 · Drawing 전체 시간.
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

cv2.setNumThreads(1)  # 여러 프로세스로 돌리므로 OpenCV 자체 스레드는 끈다 (안 끄면 서로 잡아먹어 몇 배 느려진다)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import colored_eval as ce  # noqa: E402
import faint_ruled_pick as fp  # noqa: E402
import kid_photo as kp  # noqa: E402
import rig_types as rt  # noqa: E402
from run_eval import EXPECT  # noqa: E402

OLD = rt.clean_photo
IMPL = {"old": OLD}
QD_VARIANTS = ["orig", "fill", "lined", "mix_mild", "mix_heavy", "faint", "faint_lined"]


def use(impl):
    if impl == "new":
        import faint_ruled as fr
        IMPL["new"] = fr.clean_photo
    rt.clean_photo = IMPL[impl]  # Drawing · colored_eval.mask_iou 가 모듈에서 찾아 쓴다


# ---------- QuickDraw 연한 연필 변환 ----------
def faint_make(strokes, variant, seed):
    """연한 연필: 가는 회색 선(종이보다 12~40 어두움), 연필 결 · 끊김, 종이 결 · 조명 · 흐림 · jpeg. faint_lined 는 줄 · 모눈 공책 위."""
    polys = kp._geom(strokes, np.random.default_rng(seed))
    rng = np.random.default_rng([seed, 100 + QD_VARIANTS.index(variant)])
    inside, ln6 = kp._interior(polys)
    paper_col = (250, 250, 250) if rng.random() < 0.6 else (rng.uniform(205, 235), rng.uniform(225, 245), rng.uniform(235, 252))
    img = kp._paper(rng, paper_col, rng.uniform(3, 7), variant == "faint_lined")
    thick = int(rng.integers(2, 5))
    lines = kp._lines(polys, thick)
    gt = inside | (lines > 0.5) | ln6
    depth = rng.uniform(12, 40)  # 종이보다 이만큼 어두운 연필
    cov = lines * kp._tooth(rng, strength=rng.uniform(0.3, 0.6))
    # 손 힘이 고르지 않아 선이 군데군데 더 연하다
    cov *= np.clip(0.55 + 0.6 * kp._smooth_noise(rng, int(rng.integers(20, 60))), 0, 1)
    grey = np.array(paper_col, np.float32).mean()
    img = img - cov[..., None] * depth * (img / max(grey, 1))
    if rng.random() < 0.5:
        img, gt = kp._tilt(img, gt, rng, 6, 0.02)
    img = kp._light(img, rng, rng.uniform(0.1, 0.35))
    if rng.random() < 0.3:
        img = kp._shadow(img, rng, rng.uniform(0.2, 0.4))
    img = kp._blur(img, rng, 0.5)
    img = kp._jpeg(img, rng, (50, 85), 2.5)
    return np.clip(img, 0, 255).astype(np.uint8), gt


def make(strokes, variant, seed, other):
    if variant.startswith("faint"):
        return faint_make(strokes, variant, seed)
    return kp.make(strokes, variant, seed, other)


# ---------- 재기 ----------
def measure(img, gt):
    row = dict(drawing_ok=0, iou="", mask_fail=1, box_iou="", box_fail=1, t_clean="", t_sec="", notes="", err="")
    t0 = time.time()
    rt.clean_photo(img)
    row["t_clean"] = round(time.time() - t0, 3)
    t0 = time.time()
    try:
        d = rt.Drawing(img)
    except Exception as e:
        row["t_sec"] = round(time.time() - t0, 3)
        row["err"] = repr(e)[:120]
        return row
    row["t_sec"] = round(time.time() - t0, 3)
    row["drawing_ok"] = 1
    row["notes"] = " / ".join(d.notes)
    iou = ce.mask_iou(img, d, gt)
    row["iou"] = round(iou, 3)
    row["mask_fail"] = int(not iou >= 0.5)
    bi = box_iou(img, d, gt)
    row["box_iou"] = round(bi, 3)
    row["box_fail"] = int(not bi >= 0.5)
    return row


def box_iou(img, d, gt):
    """잡은 마스크와 정답의 둘레 상자 IoU. 정답 다각형이 선을 따라 납작하게 접힌 그림(연한 그림에 많다)에도 「그림 자리를 잡았나」를 본다."""
    x0, y0, x1, y1 = ce.crop_box(rt.clean_photo(img))
    m = ((d.mask > 0) | (d.alpha > 0)).astype(np.uint8)
    ys, xs = np.nonzero(m)
    gy, gx = np.nonzero(gt)
    if len(xs) == 0 or len(gx) == 0:
        return 0.0
    sx, sy = (x1 - x0) / m.shape[1], (y1 - y0) / m.shape[0]
    a = (x0 + xs.min() * sx, y0 + ys.min() * sy, x0 + (xs.max() + 1) * sx, y0 + (ys.max() + 1) * sy)
    b = (gx.min(), gy.min(), gx.max() + 1, gy.max() + 1)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def gt_quality():
    """faint_ruled_gt.txt: 정답 마스크를 눈으로 본 판정. 돌아오는 것 {(파일 이름, 묶음): 'good'|'bad'}."""
    q = {}
    for ln in open(HERE / "faint_ruled_gt.txt", encoding="utf-8"):
        if ln.strip() and not ln.startswith("#"):
            f, g, v = ln.split()[:3]
            q[(f, g)] = v
    return q


def meta_job(job):
    impl, group, it, root, gq = job
    use(impl)
    img = cv2.imread(str(Path(root) / "x" / it["file"]))
    gt = fp.gt_mask(it["segs"], img.shape[:2]) > 0
    name = Path(it["file"]).name
    return dict(group=group, i=it["i"], file=name, split=fp.split_of(it["file"]), gt_ok=int(gq.get((name, group), "good") == "good"), **measure(img, gt))


def qd_job(job):
    impl, cat, k, variant, strokes, other, seed = job
    use(impl)
    img, gt = make(strokes, variant, seed, other)
    return dict(group=variant, cat=cat, id=k, **measure(img, gt))


def meta_groups(root):
    idx = json.load(open(root / "index.json"))
    pick = json.load(open(root / "pick.json"))
    by = {it["file"]: it for it in idx}
    picked = set(pick["faint"]) | set(pick["ruled"])
    ctrl = [it for it in idx[::25] if it["file"] not in picked]
    return {"faint": [by[f] for f in pick["faint"]], "ruled": [by[f] for f in pick["ruled"]], "control": ctrl}


def summary(rows, groups):
    print("| 묶음 | 장수 | Drawing 실패 | 마스크 실패 (IoU<0.5) | IoU 중앙값 | 자리 실패 (상자 IoU<0.5) | 사진 정리 평균 · 90% 초 | Drawing 평균 초 |")
    print("|---|---|---|---|---|---|---|---|")
    for g in groups + [g + "/정답 깨끗" for g in groups if g in ("faint", "ruled")]:
        rs = [r for r in rows if r["group"] == g.split("/")[0] and (not g.endswith("깨끗") or r.get("gt_ok") == 1)]
        if not rs:
            continue
        n = len(rs)
        ious = [r["iou"] if r["iou"] != "" and r["iou"] == r["iou"] else 0.0 for r in rs]
        tc = [r["t_clean"] for r in rs]
        print(f"| {g} | {n} | {sum(1 - r['drawing_ok'] for r in rs)} | {sum(r['mask_fail'] for r in rs)} ({100 * sum(r['mask_fail'] for r in rs) / n:.0f}%) | "
              f"{np.median(ious):.3f} | {sum(r['box_fail'] for r in rs)} | {np.mean(tc):.2f} · {np.percentile(tc, 90):.2f} | {np.mean([r['t_sec'] for r in rs]):.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["meta", "qd"])
    ap.add_argument("--meta")
    ap.add_argument("--qd")
    ap.add_argument("--impl", choices=["old", "new"], default="old")
    ap.add_argument("--set", choices=["learn", "check"], default="learn")
    ap.add_argument("--groups", default="")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.mode == "meta":
        root = Path(a.meta)
        G = meta_groups(root)
        groups = a.groups.split(",") if a.groups else list(G)
        gq = gt_quality()
        jobs = [(a.impl, g, it, str(root), gq) for g in groups for it in G[g] if fp.split_of(it["file"]) == a.set]
        fn = meta_job
    else:
        groups = a.groups.split(",") if a.groups else QD_VARIANTS
        jobs = []
        for ci, cat in enumerate(EXPECT):
            sp = ce.load_qd(Path(a.qd) / f"{cat}.ndjson", a.n)
            for k, st in enumerate(sp[a.set]):
                seed = 1000 * ci + k + (0 if a.set == "learn" else 500)
                for v in groups:
                    jobs.append((a.impl, cat, k, v, st, sp["other"][k % 50], seed))
        fn = qd_job
    t0 = time.time()
    with Pool(a.procs) as p:
        rows = p.map(fn, jobs, chunksize=2)
    path = HERE / f"{a.out}.csv" if a.out else HERE / "data" / f"fr_{a.mode}_{a.set}_{a.impl}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{path.name}: {len(rows)}개 {time.time() - t0:.0f}초 (impl={a.impl}, set={a.set})")
    summary(rows, groups)


if __name__ == "__main__":
    main()
