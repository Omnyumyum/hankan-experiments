"""4차 합치기 — Meta 사람 그림 찾기 상자 + 「자동 먼저 → 틀리면 누름」 을 지금 방식과 나란히 잰다.

세 단계 (creature-picker 폴더에서, 중간 파일은 eval_real/data/boxcache/ — git 제외):
  1) base : 그림마다 정답 · 찾기 상자(find_box) · 지금 마스크(사진 전체 → Drawing) · 상자 안 마스크(여백 몇 가지) ·
            Meta segment() 마스크 · 누른 점(tap_segment_eval.taps 와 같은 점) · 종이 상자
       ../venv/bin/python -I eval_real/box_eval.py base --set tap_learn [--margins 0,0.08,0.2] [--procs 4]
  2) sam  : MobileSAM(clean_photo 그림 입력, TAP_SEGMENT 에서 고른 것)으로 누름 후보 — 사진 전체 · 상자 안 각각
       ../venv/bin/python -I eval_real/box_eval.py sam --set tap_learn [--keys 파일]
       (qd_check 5400장은 누름 결과가 쓰이는 그림 — 누름을 묻거나 자동 마스크가 틀린 그림 — 만 --keys 로 돌렸다)
  3) score: box_score.py 가 읽는다.
묶음 (--set):
  nh_learn · nh_check   — 사람이 아닌 그림 (box_nonhuman_labels.csv, 파일 이름 md5 첫 글자 짝 = learn · 홀 = check)
  tap_learn · tap_check — TAP_SEGMENT 의 학습용 300 · 확인용 833 (tap_meta_learn.csv · tap_meta_check.csv 의 파일)
  fr_learn · fr_check   — FAINT_RULED 의 연한 연필 · 줄 종이 · 대조 (faint_ruled_meta_{learn,check}_after.csv 의 파일)
  qd_learn · qd_check   — QuickDraw 사진 변환 16가지(kid_photo) + 연한 연필 2가지(faint_ruled_eval), 카테고리마다 5장 · 30장
Meta Amateur Drawings 는 CC BY-NC 4.0 → 재기에만. 그림 · 모아 보기는 올리지 않는다.
"""
import argparse
import csv
import gzip
import hashlib
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import colored_eval as ce  # noqa: E402
import kid_photo as kp  # noqa: E402
import rig_types as rt  # noqa: E402
import tap_segment_eval as te  # noqa: E402
from run_eval import EXPECT  # noqa: E402

DATA = HERE / "data"
CACHE = DATA / "boxcache"
META = DATA / "meta"
QD = DATA / "qd30m"
QD_VARIANTS = kp.VARIANTS + ["faint", "faint_lined"]
SAM_MARGIN = 0.1  # 누름 모델에 넣는 상자 안 그림의 여백 (상자 너비 · 높이의 비율)
pack, unpack = te.pack, te.unpack


# ---------- 묶음 ----------
def md5_split(name):
    return "learn" if int(hashlib.md5(name.encode()).hexdigest()[0], 16) % 2 == 0 else "check"


def _meta_index():
    idx = json.load(open(META / "index.json"))
    return {Path(it["file"]).name: it for it in idx}


def items(name):
    grp, _, split = name.partition("_")
    if grp == "qd":
        n = 5 if split == "learn" else 30
        out = []
        for ci, cat in enumerate(EXPECT):
            sp = ce.load_qd(QD / f"{cat}.ndjson", 30)
            for k, st in enumerate(sp[split][:n]):
                seed = 1000 * ci + k + (0 if split == "learn" else 500)
                for v in QD_VARIANTS:
                    out.append(dict(key=f"{cat}{k}_{v}", group=v, cat=cat, variant=v, strokes=st, other=sp["other"][k % 50], seed=seed))
        return out
    by = _meta_index()
    if grp == "nh":
        rows = [r for r in csv.DictReader(open(HERE / "box_nonhuman_labels.csv", encoding="utf-8-sig")) if md5_split(r["file"]) == split]
        pairs = [(r["kind"], r["file"]) for r in rows]
    elif grp == "tap":
        pairs = [("tap", r["file"]) for r in csv.DictReader(open(HERE / f"tap_meta_{split}.csv", encoding="utf-8-sig"))]
    elif grp == "fr":
        pairs = [(r["group"], r["file"]) for r in csv.DictReader(open(HERE / f"faint_ruled_meta_{split}_after.csv", encoding="utf-8-sig"))]
    else:
        raise ValueError(name)
    return [dict(key=f"meta{by[f]['i']}", group=g, file=by[f]["file"], segs=by[f]["segs"]) for g, f in pairs]


def load_img(it):
    if "file" in it:
        return te.load_img(it)
    import faint_ruled_eval as fe
    img, gt = fe.make(it["strokes"], it["variant"], it["seed"], it["other"])
    return img, gt > 0


# ---------- 마스크 ----------
def drawing_mask(img):
    """지금 방식(clean_photo → Drawing) 마스크를 img 좌표로. 실패하면 빈 마스크. (마스크, 시간, notes, 오류)"""
    t0 = time.time()
    try:
        d = rt.Drawing(img)
        t = time.time() - t0
        m, _ = te.base_full(img, d)
        return m, t, " / ".join(d.notes), ""
    except Exception as e:
        return np.zeros(img.shape[:2], bool), time.time() - t0, "", repr(e)[:80]


def expand(box, shape, f):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    h, w = shape
    return (max(0, int(np.floor(x0 - f * bw))), max(0, int(np.floor(y0 - f * bh))),
            min(w, int(np.ceil(x1 + f * bw))), min(h, int(np.ceil(y1 + f * bh))))


def in_box(img, box, f, fn):
    """상자(여백 f)로 자른 그림에 fn 을 돌리고 결과 마스크를 원본 좌표로."""
    x0, y0, x1, y1 = expand(box, img.shape[:2], f)
    out = fn(np.ascontiguousarray(img[y0:y1, x0:x1]))
    full = np.zeros(img.shape[:2], bool)
    full[y0:y1, x0:x1] = out[0]
    return (full,) + tuple(out[1:])


def meta_segment(img):
    """AnimatedDrawings examples/image_to_annotations.py 의 segment() (MIT, Meta). 상자로 자른 그림 → 마스크."""
    from scipy import ndimage
    from skimage import measure
    t0 = time.time()
    g = np.min(img, axis=2)
    g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 8)
    g = cv2.bitwise_not(g)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k, iterations=2)
    g = cv2.morphologyEx(g, cv2.MORPH_DILATE, k, iterations=2)
    mask = np.zeros([g.shape[0] + 2, g.shape[1] + 2], np.uint8)
    mask[1:-1, 1:-1] = g.copy()
    ff = np.full(g.shape, 255, np.uint8)
    h, w = g.shape[:2]
    for x in range(0, w - 1, 10):
        cv2.floodFill(ff, mask, (x, 0), 0)
        cv2.floodFill(ff, mask, (x, h - 1), 0)
    for y in range(0, h - 1, 10):
        cv2.floodFill(ff, mask, (0, y), 0)
        cv2.floodFill(ff, mask, (w - 1, y), 0)
    ff[0, :] = 0
    ff[-1, :] = 0
    ff[:, 0] = 0
    ff[:, -1] = 0
    m2 = cv2.bitwise_not(ff)
    best, big = None, 0
    for c in measure.find_contours(m2, 0.0):
        x = np.zeros(m2.T.shape, np.uint8)
        cv2.fillPoly(x, [np.int32(c)], 1)
        s = int(x.sum())
        if s > big:
            best, big = x, s
    if best is None:
        return np.zeros(img.shape[:2], bool), time.time() - t0
    return ndimage.binary_fill_holes(best).T, time.time() - t0


# ---------- 1) base ----------
_DET = None


def _init(margins):
    global _DET, MARGINS
    import torch
    torch.set_num_threads(1)
    import find_box as fb
    _DET = fb.load()
    MARGINS = margins


def base_job(it):
    import find_box as fb
    img, gt = load_img(it)
    seed = int.from_bytes(it["key"].encode()[:8].ljust(8, b"\0"), "little") % (2 ** 31)
    t0 = time.time()
    box, score, allb = fb.detect_top(_DET, img)
    det_t = time.time() - t0
    full, full_t, notes, err = drawing_mask(img)
    r = dict(key=it["key"], group=it["group"], file=Path(it.get("file", "")).name, gt=pack(gt), h=img.shape[0], w=img.shape[1],
             box=box, score=score, boxes=[b for b in allb[:10]], det_t=det_t,
             full=pack(full), full_t=full_t, notes=notes, err=err, taps=te.taps(gt, seed), pbox=te.paper_box(img), inbox={})
    if box is not None:
        for f in MARGINS:
            m, t, nt, er = in_box(img, box, f, drawing_mask)
            r["inbox"][f] = dict(mask=pack(m), t=t, notes=nt, err=er)
        m, t = in_box(img, box, 0.0, meta_segment)
        r["metaseg"] = dict(mask=pack(m), t=t)
    return r


def cpath(name, what):
    return CACHE / f"{name}_{what}.pkl.gz"


def run_base(a):
    CACHE.mkdir(parents=True, exist_ok=True)
    its = items(a.set)
    margins = [float(x) for x in a.margins.split(",")]
    t0 = time.time()
    rows = []
    with Pool(a.procs, initializer=_init, initargs=(margins,)) as p:
        for k, r in enumerate(p.imap(base_job, its, chunksize=2)):
            rows.append(r)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(its)} {time.time() - t0:.0f}초", flush=True)
    out = cpath(a.set, "base")
    pickle.dump(rows, gzip.open(str(out) + ".part", "wb"))
    Path(str(out) + ".part").replace(out)
    print(f"base {a.set}: {len(rows)}장 {time.time() - t0:.0f}초, 상자 찾음 {sum(r['box'] is not None for r in rows)}")


# ---------- 2) sam ----------
_SAM = None


def _init_sam(threads):
    global _SAM
    import torch
    torch.set_num_threads(threads)
    import tap_models as tm
    _SAM = tm.load("mobile_sam")


def _cands(img, pts, box=None):
    if not pts:  # 상자만 (점 없음)
        import tap_models as tm
        import torch
        with torch.inference_mode():
            m, s, _ = _SAM.p.predict(point_coords=None, point_labels=None, box=np.array(box, np.float32), multimask_output=True)
        ms, sc = tm._order(m, s)
        return [pack(x) for x in ms], [float(s) for s in sc]
    ms, sc = _SAM.predict(pts, box)
    return [pack(x) for x in ms], [float(s) for s in sc]


def sam_job(job):
    b, it = job
    img, _ = load_img(it)
    clean = rt.clean_photo(img)
    t0 = time.time()
    _SAM.set_image(clean)
    enc = time.time() - t0
    tp = b["taps"]
    out = dict(key=b["key"], enc=enc, full={})
    out["full"]["in"] = _cands(img, [tp["i"]])
    out["full"]["in2"] = _cands(img, [tp["i"], tp["f"]])
    if b["box"] is not None:
        out["full"]["dbox"] = _cands(img, [], list(b["box"]))  # 누르지 않고 찾기 상자만 (참고)
        out["full"]["in+dbox"] = _cands(img, [tp["i"]], list(b["box"]))
        x0, y0, x1, y1 = expand(b["box"], img.shape[:2], SAM_MARGIN)
        t0 = time.time()
        _SAM.set_image(np.ascontiguousarray(clean[y0:y1, x0:x1]))
        out["enc_box"] = time.time() - t0
        sh = lambda p: [p[0] - x0, p[1] - y0]  # noqa: E731
        out["crop"] = dict(xy=(x0, y0, x1, y1), **{"in": _cands(img, [sh(tp["i"])]), "in2": _cands(img, [sh(tp["i"]), sh(tp["f"])])})
    return out


def run_sam(a):
    base = pickle.load(gzip.open(cpath(a.set, "base"), "rb"))
    if a.keys:  # 이 그림들만 (누름이 쓰이는 그림만 돌려 시간을 줄일 때)
        keep = set(open(a.keys).read().split())
        base = [b for b in base if b["key"] in keep]
    its = {it["key"]: it for it in items(a.set)}
    t0 = time.time()
    rows = []
    with Pool(a.procs, initializer=_init_sam, initargs=(a.threads,)) as p:
        for k, r in enumerate(p.imap(sam_job, [(b, its[b["key"]]) for b in base], chunksize=2)):
            rows.append(r)
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(base)} {time.time() - t0:.0f}초", flush=True)
    out = cpath(a.set, "sam")
    pickle.dump(rows, gzip.open(str(out) + ".part", "wb"))
    Path(str(out) + ".part").replace(out)
    print(f"sam {a.set}: {len(rows)}장 {time.time() - t0:.0f}초")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["base", "sam"])
    ap.add_argument("--set", required=True)
    ap.add_argument("--margins", default="0,0.08,0.2")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--keys", default="", help="sam: 이 파일에 적힌 key 의 그림만")
    a = ap.parse_args()
    {"base": run_base, "sam": run_sam}[a.mode](a)


if __name__ == "__main__":
    main()
