"""그림을 한 번 누르면 그 그림만 잘라 내기 — SAM 계열 점 프롬프트 분할을 지금 clean_photo 마스크와 나란히 잰다.

세 단계 (creature-picker 폴더에서, 결과 중간 파일은 eval_real/data/tapcache/ — git 제외):
  1) base   : 그림마다 정답 마스크 · 지금 마스크(clean_photo → Drawing) · 누른 점 · 종이 상자 · 그림 상자를 만든다
       ../venv/bin/python -I eval_real/tap_segment_eval.py base --data meta --set learn --n 300
       ../venv/bin/python -I eval_real/tap_segment_eval.py base --data qd --set learn --n 20 [--variants orig,lined,...]
  2) model  : 모델 하나로 후보 마스크 3개씩을 만든다 (이미지 인코더는 그림마다 한 번, 시간 · 메모리도 잰다)
       ../venv/bin/python -I eval_real/tap_segment_eval.py model --data meta --set learn --model mobile_sam
  3) score  : 후처리(마스크 고르기 · 안쪽 채우기 · clean_photo 와 섞기)를 바꿔 가며 숫자를 낸다 (모델을 다시 돌리지 않는다)
       ../venv/bin/python -I eval_real/tap_segment_eval.py score --data meta --set learn --models mobile_sam,sam2_t [--csv 이름]

데이터:
  meta — Meta Amateur Drawings (CC BY-NC 4.0 → 재기에만). photo_mask_eval.py prep 이 만든 index.json 의 learn / check 나누기를 그대로 쓴다.
  qd   — QuickDraw 사진 변환(kid_photo.make, colored_eval.py 와 같은 그림 · 같은 seed). 앞 30,000,000 바이트.
「누른 점」 흉내 (그림마다 seed 고정, 정답 마스크로만 고름 — 모델 · 후처리와 상관없이 같은 점):
  in   : 정답 마스크 안, 거리 변환이 큰 쪽(가장 깊은 곳의 절반 이상) 무작위 한 점
  edge : 정답 마스크 안, 가장자리 쪽(가장 깊은 곳의 15% 이하) 무작위 한 점
  둘째 점(in2): 정답 안 깊은 쪽(25% 이상) 중 첫 점에서 먼 쪽(가장 먼 거리의 절반 이상) 무작위 한 점
  손떨림: 모든 점에 표준편차 = 긴 변의 2% 정규 잡음 (정답 밖으로 나갈 수도 있다 — 실제로도 그렇다)
  종이 상자: rig_types 의 paper_only + _find_paper 가 찾은 종이의 둘레 상자 (못 찾으면 사진 전체)
  그림 상자(cbox): clean_photo 로 정리한 그림에서 crop_and_mask 가 자르는 상자 (지금 마스크의 자르기와 같음)
"""
import argparse
import csv
import gzip
import json
import pickle
import resource
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import colored_eval as ce  # noqa: E402
import kid_photo as kp  # noqa: E402
import rig_types as rt  # noqa: E402
from run_eval import EXPECT  # noqa: E402

DATA = HERE / "data"
CACHE = DATA / "tapcache"
META = DATA / "meta"
QD = DATA / "qd30m"
JITTER = 0.02
PROMPTS = ["in", "edge", "in+paper", "edge+paper", "in2", "in+cbox", "edge+cbox"]


# ---------- 그림 ----------
def meta_items(split, n):
    idx = json.load(open(META / "index.json"))
    items = [it for it in idx if it["split"] == split][: n or None]
    return [dict(key=f"meta{it['i']}", file=it["file"], segs=it["segs"]) for it in items]


def qd_items(split, n, variants):
    out = []
    for ci, cat in enumerate(EXPECT):
        sp = ce.load_qd(QD / f"{cat}.ndjson", n)
        for k, st in enumerate(sp[split]):
            seed = 1000 * ci + k + (0 if split == "learn" else 500)
            for v in variants:
                out.append(dict(key=f"{cat}{k}_{v}", cat=cat, id=k, variant=v, strokes=st, other=sp["other"][k % 50], seed=seed))
    return out


def load_img(it):
    """(BGR 그림, 정답 bool 마스크)"""
    if "file" in it:
        img = cv2.imread(str(META / "x" / it["file"]))
        gt = np.zeros(img.shape[:2], np.uint8)
        for s in it["segs"]:
            cv2.fillPoly(gt, [np.round(np.array(s).reshape(-1, 2)).astype(np.int32)], 1)
        return img, gt > 0
    img, gt = kp.make(it["strokes"], it["variant"], it["seed"], it["other"])
    return img, gt > 0


# ---------- 누른 점 · 상자 ----------
def taps(gt, seed):
    rng = np.random.default_rng(seed)
    h, w = gt.shape
    sd = JITTER * max(h, w)
    dt = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 5)
    mx = dt.max()

    def pick(cond):
        ys, xs = np.nonzero(cond)
        if len(xs) == 0:
            ys, xs = np.nonzero(gt)
        k = rng.integers(len(xs))
        return np.array([xs[k], ys[k]], np.float32)

    def jit(p):
        q = p + rng.normal(0, sd, 2)
        return [float(np.clip(q[0], 0, w - 1)), float(np.clip(q[1], 0, h - 1))]

    p_in = pick(dt >= 0.5 * mx)
    p_edge = pick((dt > 0) & (dt <= max(1.5, 0.15 * mx)))
    deep = dt >= 0.25 * mx
    ys, xs = np.nonzero(deep)
    far = np.hypot(xs - p_in[0], ys - p_in[1])
    p_far = pick(deep & (np.hypot(*np.meshgrid(np.arange(w) - p_in[0], np.arange(h) - p_in[1])) >= 0.5 * far.max()))
    return dict(i=jit(p_in), e=jit(p_edge), f=jit(p_far))


def paper_box(img):
    im = rt.paper_only(img)
    h, w = im.shape[:2]
    s = min(1.0, 512 / max(h, w))
    sm = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else im
    paper, frac = rt._find_paper(sm)
    if frac >= 0.97 or not paper.any():
        return [0, 0, w - 1, h - 1]
    ys, xs = np.nonzero(paper)
    return [float(xs.min() / s), float(ys.min() / s), float(xs.max() / s), float(ys.max() / s)]


def prompt(name, tp, pbox, cbox):
    pts = [tp["e"]] if name.startswith("edge") else [tp["i"]]
    if name == "in2":
        pts = [tp["i"], tp["f"]]
    box = pbox if name.endswith("+paper") else cbox if name.endswith("+cbox") else None
    return pts, box


# ---------- 지금 마스크 (clean_photo → Drawing) 를 원본 좌표로 ----------
def base_full(img, d):
    im = rt.clean_photo(img)
    x0, y0, x1, y1 = ce.crop_box(im)
    m = ((d.mask > 0) | (d.alpha > 0)).astype(np.uint8)
    full = np.zeros(img.shape[:2], bool)
    full[y0:y1, x0:x1] = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0
    return full, [float(x0), float(y0), float(x1 - 1), float(y1 - 1)]


def pack(m):
    return (m.shape, np.packbits(m.astype(bool), axis=None))


def unpack(p):
    shape, b = p
    return np.unpackbits(b, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def base_job(it):
    img, gt = load_img(it)
    seed = int.from_bytes(it["key"].encode()[:8].ljust(8, b"\0"), "little") % (2 ** 31)
    t0 = time.time()
    try:
        d = rt.Drawing(img)
        t = time.time() - t0
        bm, cbox = base_full(img, d)
    except Exception as e:  # 선을 못 찾음
        t = time.time() - t0
        bm, cbox = np.zeros(gt.shape, bool), [0, 0, gt.shape[1] - 1, gt.shape[0] - 1]
        it = dict(it, err=repr(e)[:80])
    keep = {k: v for k, v in it.items() if k not in ("strokes", "other", "segs")}
    return dict(keep, gt=pack(gt), base=pack(bm), base_t=round(t, 3), taps=taps(gt, seed), pbox=paper_box(img), cbox=cbox)


def cache_path(data, split, what, tag=""):
    return CACHE / f"{data}_{split}{tag}_{what}.pkl.gz"


def items_for(a):
    if a.data == "meta":
        return meta_items(a.set, a.n)
    return qd_items(a.set, a.n, a.variants.split(","))


def tag_of(a):
    return "" if a.data == "meta" else f"_n{a.n}_" + ("all" if a.variants == ",".join(kp.VARIANTS) else a.variants.replace(",", "-"))


def run_base(a):
    CACHE.mkdir(parents=True, exist_ok=True)
    items = items_for(a)
    t0 = time.time()
    with Pool(a.procs) as p:
        rows = p.map(base_job, items, chunksize=4)
    pickle.dump(rows, gzip.open(cache_path(a.data, a.set, "base", tag_of(a)), "wb"))
    print(f"base {a.data} {a.set}: {len(rows)}장 {time.time() - t0:.0f}초")


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_model(a):
    import torch
    import tap_models as tm
    torch.set_num_threads(a.threads)
    base = pickle.load(gzip.open(cache_path(a.data, a.set, "base", tag_of(a)), "rb"))
    items = {it["key"]: it for it in items_for(a)}
    m0 = rss_mb()
    t0 = time.time()
    m = tm.load(a.model)
    load_t = time.time() - t0
    m1 = rss_mb()
    rows = []
    for b in base:
        img, _ = load_img(items[b["key"]])
        t0 = time.time()
        if a.clean:  # clean_photo 로 정리한 그림(종이 밖 · 줄 · 다른 그림을 지우고 선을 진하게)을 모델에 넣는다
            img = rt.clean_photo(img)
        m.set_image(img)
        enc = time.time() - t0
        out = {}
        dec = []
        for pn in PROMPTS:
            pts, box = prompt(pn, b["taps"], b["pbox"], b["cbox"])
            t0 = time.time()
            ms, sc = m.predict(pts, box)
            dec.append(time.time() - t0)
            out[pn] = ([pack(x) for x in ms], sc.tolist())
        rows.append(dict(key=b["key"], enc=enc, dec=float(np.mean(dec)), out=out))
        if len(rows) % 50 == 0:
            print(f"  {len(rows)}/{len(base)} 인코더 평균 {np.mean([r['enc'] for r in rows]):.2f}초", flush=True)
    info = dict(model=a.model + ("_clean" if a.clean else ""), threads=a.threads, load_t=load_t, rss_before=m0, rss_loaded=m1, rss_peak=rss_mb(), n=len(rows))
    out = cache_path(a.data, a.set, a.model + ("_clean" if a.clean else ""), tag_of(a))
    pickle.dump(dict(info=info, rows=rows), gzip.open(str(out) + ".part", "wb"))
    Path(str(out) + ".part").replace(out)  # 다 쓴 뒤에 이름을 바꾼다 (쓰는 중에 score 가 읽지 않게)
    print(info)


# ---------- 후처리 ----------
def fill_closed(m, k):
    """선으로 닫힌 안쪽 채우기 (k px 까지 끊긴 틈은 닫힌 것으로 봄). 선만 잡은 마스크용."""
    mm = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, rt._disk(k))
    return (rt._fill_holes(mm) > 0) | m


def paper_like(m, pbox):
    """종이 한 장(또는 사진 전체)을 통째로 잡은 후보: 넓이가 종이 상자의 60% 넘거나, 상자 너비 · 높이의 90% 넘게 퍼짐."""
    x0, y0, x1, y1 = pbox
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if m.sum() > 0.6 * bw * bh:
        return True
    ys, xs = np.nonzero(m)
    return len(xs) > 0 and (xs.max() - xs.min()) > 0.9 * bw and (ys.max() - ys.min()) > 0.9 * bh


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


BORDER = 0.01


def touches_border(m, frac=None):
    """사진 테두리에 닿은 화소가 테두리 길이의 frac 넘음 — 책상 · 종이 · 배경 쪽 후보. 그림은 보통 사진 안쪽에 있다."""
    frac = BORDER if frac is None else frac
    h, w = m.shape
    edge = m[0].sum() + m[-1].sum() + m[:, 0].sum() + m[:, -1].sum()
    return edge > frac * 2 * (h + w)


def choose(cands, sc, how, pbox, base, tp):
    """cands: 작은 것 → 큰 것 3개.
    how: score(모델 점수) · big(가장 큰 것) · guard(종이 같은 후보를 빼고 점수) · guardbig(종이 같은 후보를 빼고 가장 큰 것)
         · mix(clean_photo 마스크가 누른 곳을 담으면 그것과 가장 닮은 후보).
    끝에 b 가 붙으면(guardb · guardbbig · mixb) 사진 테두리에 닿은 후보도 뺀다.
    @base 가 붙으면 남는 후보가 없을 때 지금 마스크로 돌아간다 (없으면 가장 작은 후보)."""
    how, _, fb = how.partition("@")
    if how == "score":
        return cands[int(np.argmax(sc))]
    if how == "big":
        return cands[2]
    border = how.endswith("b") and how != "guardbig" or how == "guardbbig"
    core = how.replace("guardbbig", "guardbig").rstrip("b") if how != "guardbig" else how
    ok = [k for k in range(3) if cands[k].any() and not paper_like(cands[k], pbox) and not (border and touches_border(cands[k]))]
    if not ok:
        return base if fb == "base" else cands[0]
    if core == "guard":
        return cands[max(ok, key=lambda k: sc[k])]
    if core == "guardbig":  # 종이가 아닌 것 중 가장 큰 것
        return cands[max(ok)]
    if core == "mix":  # clean_photo 마스크가 누른 곳을 담고 있으면 그것과 가장 닮은 후보, 아니면 guard
        x, y = int(round(tp[0])), int(round(tp[1]))
        r = max(3, int(0.03 * max(base.shape)))
        if base[max(0, y - r): y + r + 1, max(0, x - r): x + r + 1].any():
            return cands[max(ok, key=lambda k: iou(cands[k], base))]
        return cands[max(ok, key=lambda k: sc[k])]
    raise ValueError(how)


STRATS = ["score", "guard", "guardbig", "guardb", "guardbbig", "score+fill", "guardbig+fill", "guardbbig+fill", "guardbbig@base+fill", "mixb+fill", "mixb@base+fill"]


def final_mask(strat, cands, sc, pbox, base, tp, fill_k):
    how, _, post = strat.partition("+")
    m = choose(cands, sc, how, pbox, base, tp)
    if post.startswith("fill"):
        m = fill_closed(m, fill_k)
    if post.endswith("|base"):  # 지금 마스크와 많이 닮았으면 지금 마스크(칸 · 선 굵기에 맞춘 것)를 그대로 쓴다
        if iou(m, base) > 0.8:
            m = base
    return m


def run_score(a):
    global BORDER
    BORDER = a.border
    base = pickle.load(gzip.open(cache_path(a.data, a.set, "base", tag_of(a)), "rb"))
    rows = []
    for b in base:
        gt, bm = unpack(b["gt"]), unpack(b["base"])
        r = dict(key=b["key"], file=Path(b.get("file", "")).name, variant=b.get("variant", "meta"), base_iou=round(iou(bm, gt), 3), base_t=b["base_t"],
                 tap_in_base=int(bm[int(round(b["taps"]["i"][1])), int(round(b["taps"]["i"][0]))]))
        rows.append(r)
    infos = {}
    for model in a.models.split(","):
        c = pickle.load(gzip.open(cache_path(a.data, a.set, model, tag_of(a)), "rb"))
        infos[model] = c["info"]
        by = {r["key"]: r for r in c["rows"]}
        for b, r in zip(base, rows):
            gt, bm = unpack(b["gt"]), unpack(b["base"])
            fill_k = max(5, int(a.fill * max(gt.shape)) | 1)
            mr = by[b["key"]]
            r[f"{model}:enc"] = round(mr["enc"], 3)
            for pn in a.prompts.split(","):
                cands, sc = mr["out"][pn]
                cands = [unpack(x) for x in cands]
                pts, _ = prompt(pn, b["taps"], b["pbox"], b["cbox"])
                for st in a.strats.split(","):
                    m = final_mask(st, cands, sc, b["pbox"], bm, pts[0], fill_k)
                    r[f"{model}:{pn}:{st}"] = round(iou(m, gt), 3)
    variants = sorted({r["variant"] for r in rows}, key=lambda v: kp.VARIANTS.index(v) if v in kp.VARIANTS else -1)
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        n = len(rs)
        bi = np.array([r["base_iou"] for r in rs])
        print(f"\n## {a.data} {a.set} {v} ({n}장)")
        print(f"지금 clean_photo: 마스크 실패 {int((bi < 0.5).sum())} ({100 * (bi < 0.5).mean():.1f}%) · IoU 중앙값 {np.median(bi):.3f} · "
              f"누른 점(in)이 지금 마스크 안 {sum(r['tap_in_base'] for r in rs)}/{n}")
        print("| 모델 | 프롬프트 | " + " | ".join(a.strats.split(",")) + " |")
        print("|---|---|" + "---|" * len(a.strats.split(",")))
        for model in a.models.split(","):
            for pn in a.prompts.split(","):
                cells = []
                for st in a.strats.split(","):
                    x = np.array([r[f"{model}:{pn}:{st}"] for r in rs])
                    cells.append(f"{int((x < 0.5).sum())} · {np.median(x):.2f}")
                print(f"| {model} | {pn} | " + " | ".join(cells) + " |")
    print("\n(칸: 마스크 실패 IoU<0.5 장수 · IoU 중앙값)")
    # 화면 흐름 흉내: 사람이 결과를 보고 틀렸으면(IoU<0.5) 다음 단계로 간다 (사람 눈이 IoU 0.5 를 정확히 가른다고 가정)
    st = a.main
    print(f"\n## 흐름별 마스크 실패 (후처리 {st}; 「→」 = 앞 단계가 틀렸을 때만 다음 단계)")
    print("| 모델 | 데이터 | 자동만 | 한 번 누름 | 두 번 누름 | 한 번 → 틀리면 한 번 더 | 자동 → 틀리면 누름 | 자동 → 누름 → 한 번 더 | 자동 · 누름 결과 중 사람이 고름 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for model in a.models.split(","):
        for v in variants:
            rs = [r for r in rows if r["variant"] == v]
            fb = np.array([r["base_iou"] < 0.5 for r in rs])
            f1 = np.array([r[f"{model}:in:{st}"] < 0.5 for r in rs])
            f2 = np.array([r[f"{model}:in2:{st}"] < 0.5 for r in rs])
            n = len(rs)
            cells = [fb, f1, f2, f1 & f2, fb & f1, fb & f1 & f2, fb & f1]
            print(f"| {model} | {v} | " + " | ".join(f"{int(c.sum())} ({100 * c.mean():.0f}%)" for c in cells) + f" |  (n={n})")
    for model, info in infos.items():
        enc = [r[f"{model}:enc"] for r in rows]
        print(f"- {model}: 불러오기 {info['load_t']:.1f}초, 인코더 평균 {np.mean(enc):.2f} · 90% {np.percentile(enc, 90):.2f}초 (스레드 {info['threads']}), "
              f"메모리 torch 만 {info['rss_before']:.0f} → 불러온 뒤 {info['rss_loaded']:.0f} → 최대 {info['rss_peak']:.0f} MB")
    bt = [r["base_t"] for r in rows]
    print(f"- 지금 clean_photo → Drawing: 평균 {np.mean(bt):.2f} · 90% {np.percentile(bt, 90):.2f}초 (동시 {a.procs})")
    if a.csv:
        with open(HERE / f"{a.csv}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["base", "model", "score"])
    ap.add_argument("--data", choices=["meta", "qd"], default="meta")
    ap.add_argument("--set", choices=["learn", "check"], default="learn")
    ap.add_argument("--n", type=int, default=0, help="meta: 앞 n장 (0 = 전부), qd: 카테고리마다 n장")
    ap.add_argument("--variants", default=",".join(kp.VARIANTS))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--model", default="mobile_sam")
    ap.add_argument("--models", default="mobile_sam")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--strats", default=",".join(STRATS))
    ap.add_argument("--border", type=float, default=0.01, help="score: 사진 테두리에 닿은 화소가 테두리 길이의 이 비율을 넘는 후보는 뺀다")
    ap.add_argument("--main", default="guardbbig@base+fill", help="score: 흐름 표에 쓸 후처리")
    ap.add_argument("--fill", type=float, default=0.03, help="안쪽 채우기: 이 비율(긴 변)까지 끊긴 틈은 닫힌 것으로")
    ap.add_argument("--clean", action="store_true", help="model: clean_photo 로 정리한 그림을 넣는다 (score 에서는 모델 이름 뒤에 _clean)")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    {"base": run_base, "model": run_model, "score": run_score}[a.mode](a)


if __name__ == "__main__":
    main()
