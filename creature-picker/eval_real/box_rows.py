"""box_eval.py 중간 파일 → 그림마다 숫자 한 줄 (모든 여백 · 누름 방식). 규칙 고르기를 빠르게 하려고 한 번만 계산해 둔다.

    ../venv/bin/python -I eval_real/box_rows.py nh_learn tap_learn ...   → eval_real/data/boxcache/<묶음>_rows.pkl
누름 방식: tap_full = 사진 전체 MobileSAM + 점, tap_crop = 상자(여백 SAM_MARGIN) 안 그림만 MobileSAM + 점, tap_prompt = 사진 전체 + 점 + 찾기 상자.
각각 TAP_SEGMENT 최종 후처리(guardbbig@base+fill), base(후보가 다 빠질 때 돌아갈 마스크)는 그 갈래의 자동 마스크.
"""
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import box_score as bs  # noqa: E402

MARGINS = [0.0, 0.1, 0.2, 0.35, 0.5]
unpack, iou = bs.unpack, bs.iou


def one(job):
    b, s = job
    gt = unpack(b["gt"])
    full = unpack(b["full"])
    gb = bs.mask_box(gt)
    r = dict(key=b["key"], file=b["file"], group=b["group"], h=b["h"], w=b["w"], score=b["score"], has_box=b["box"] is not None,
             nbox=sum(x[4] >= 0.5 for x in b["boxes"]), err=b["err"], notes=b["notes"], det_t=b["det_t"], full_t=b["full_t"],
             iou_now=iou(full, gt), full_frac=full.sum() / gt.size, full_empty=not full.any(),
             box_iou_gt=bs.bbox_iou(b["box"], gb) if b["box"] is not None and gb else 0.0)
    if s:
        r["enc"] = s["enc"]
        r["tap_full@now"] = iou(bs.tap_full(b, s, "in", full), gt)
        r["tap2_full@now"] = iou(bs.tap_full(b, s, "in2", full), gt)
    if b["box"] is not None:
        bx = b["box"]
        r["box_frac"] = (bx[2] - bx[0]) * (bx[3] - bx[1]) / gt.size
        fbb = bs.mask_box(full)
        r["fullbox_vs_det"] = bs.bbox_iou(fbb, bx) if fbb else 0.0
        r["iou_meta"] = iou(unpack(b["metaseg"]["mask"]), gt)
        for m in MARGINS:
            ib = unpack(b["inbox"][m]["mask"])
            r[f"box{m}"] = iou(ib, gt)
            r[f"agree{m}"] = iou(ib, full)
            r[f"empty{m}"] = not ib.any()
            r[f"t{m}"] = b["inbox"][m]["t"]
            r[f"frac{m}"] = ib.sum() / gt.size
            if s and m in (0.1, 0.2):
                r[f"tap_crop@{m}"] = iou(bs.tap_crop(b, s, ib), gt)
                r[f"tap_full@{m}"] = iou(bs.tap_full(b, s, "in", ib), gt)
                r[f"tap_prompt@{m}"] = iou(bs.tap_full(b, s, "in+dbox", ib), gt)
                c, _ = s["full"]["dbox"]
                r[f"sam_agree{m}"] = max(iou(unpack(x), ib) for x in c)
        if s:
            r["enc_box"] = s.get("enc_box", 0.0)
            c, _ = s["full"]["dbox"]
            r["sam_agree_now"] = max(iou(unpack(x), full) for x in c)
    if not s:  # 누름 후보를 안 만든 그림 (누름도 안 묻고 자동도 맞은 그림) — 누름 결과가 쓰이지 않으므로 자동 결과로 채운다
        r["tap_full@now"] = r["iou_now"]
        for m in (0.1, 0.2):
            if f"box{m}" in r:
                r[f"tap_prompt@{m}"] = r[f"tap_crop@{m}"] = r[f"tap_full@{m}"] = r[f"box{m}"]
        r["no_sam"] = True
    return r


def main():
    for name in sys.argv[1:]:
        base, sam = bs.load(name)
        with Pool(4) as p:
            rows = p.map(one, [(b, sam.get(b["key"])) for b in base], chunksize=4)
        pickle.dump(rows, open(bs.CACHE / f"{name}_rows.pkl", "wb"))
        print(name, len(rows), "sam" if sam else "no sam")


if __name__ == "__main__":
    main()
