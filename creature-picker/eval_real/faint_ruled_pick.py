"""Meta Amateur Drawings 에서 (가) 연한 연필 그림 · (나) 줄 · 모눈 종이 그림을 자동으로 고른다 (재기 · 확인용, CC BY-NC 4.0).

    ../venv/bin/python -I eval_real/faint_ruled_pick.py feat --meta eval_real/data/meta      # 그림마다 숫자 → data/meta/pick_feat.csv
    ../venv/bin/python -I eval_real/faint_ruled_pick.py pick --meta eval_real/data/meta      # 고르기 → data/meta/pick.json · 눈으로 볼 모아 보기
받은폴더는 photo_mask_eval.py prep 로 만든 index.json 이 있는 곳.
(가) 선 대비(dk95): 정답 마스크 안에서 가장 진한 K 픽셀(K = 마스크 둘레 × 2)의 「둘레 중앙값보다 어두운 정도」 중앙값.
     15 미만이면 연하다. 마스크 안 색이 진하거나(ch90 ≥ 25) 넓게 칠했으면(filled ≥ 0.3) 뺀다.
(나) 줄: 정답 마스크를 뺀 종이에서 가는 어두운 선 응답을 (조금씩 돌려 가며) 한 방향으로 모아,
     고른 간격의 봉우리가 사진 전체에 있는지(주기 세기 · 봉우리가 퍼진 정도)를 본다.
(나) 주기 세기 rule_s > 0.6 이고 주기마다 봉우리가 있는 비율 > 0.8.
눈으로 보고 뺀 그림은 eval_real/faint_ruled_drop.txt (파일 이름 · 묶음), 정답 마스크 판정은 eval_real/faint_ruled_gt.txt.
학습 · 확인 나눔: 파일 이름의 md5 첫 글자가 짝수면 학습용, 홀수면 확인용.
"""
import argparse
import csv
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def gt_mask(segs, shape):
    gt = np.zeros(shape, np.uint8)
    for s in segs:
        cv2.fillPoly(gt, [np.round(np.array(s).reshape(-1, 2)).astype(np.int32)], 1)
    return gt


def contrast_feat(img, gt):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    ring = (cv2.dilate(gt, np.ones((25, 25), np.uint8)) > 0) & (cv2.dilate(gt, np.ones((7, 7), np.uint8)) == 0)
    if ring.sum() < 50 or gt.sum() < 50:
        return None
    paper = float(np.median(L[ring]))
    inside = gt > 0
    p5 = float(np.percentile(L[inside], 5))
    # 선 픽셀: 둘레(넓은 중앙값)보다 8 넘게 어두운 곳
    bg = cv2.medianBlur(L.astype(np.uint8), 21).astype(np.float32)
    dk = np.maximum(bg - L, 0)
    lines = inside & (dk > 8)
    line_dark = float(np.median(dk[lines])) if lines.sum() > 30 else 0.0
    # 선 대비: 마스크 안에서 가장 진한 K 픽셀(K = 마스크 둘레 길이 × 2, 윤곽선 한 바퀴쯤)의 중앙 어두움
    cs, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    K = int(2 * sum(cv2.arcLength(c, True) for c in cs))
    near = cv2.dilate(gt, np.ones((5, 5), np.uint8)) > 0
    dks = np.sort(dk[near])[::-1]
    dk95 = float(np.median(dks[:max(20, K)]))
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    pc = float(np.median(chroma[ring]))
    ch90 = float(np.percentile(np.abs(chroma[inside] - pc), 90))
    # 마스크 안에서 종이보다 밝기가 크게 다른(칠한) 넓이
    filled = float((np.abs(L[inside] - paper) > 25).mean())
    return dict(paper=round(paper, 1), contrast=round(paper - p5, 1), dk95=round(dk95, 1), line_dark=round(line_dark, 1), line_frac=round(float(lines.sum() / inside.sum()), 3),
                ch90=round(ch90, 1), filled=round(filled, 3))


def period_score(resp, valid):
    """resp 를 줄마다 모아 주기 세기. 돌아온 값: (세기, 주기 px, 봉우리가 있는 줄 비율)."""
    rows = (resp * valid).sum(1) / np.maximum(valid.sum(1), 1)
    ok = valid.sum(1) > 0.3 * valid.shape[1]
    if ok.sum() < 64:
        return 0.0, 0.0, 0.0
    p = rows[ok] - gaussian_filter1d(rows[ok], 6)
    p = p - p.mean()
    n = len(p)
    F = np.abs(np.fft.rfft(p * np.hanning(n))) ** 2
    fr = np.fft.rfftfreq(n)
    band = (fr > 1 / 60) & (fr < 1 / 7)
    if not band.any() or F[1:].sum() == 0:
        return 0.0, 0.0, 0.0
    k = int(np.argmax(np.where(band, F, 0)))
    s = float(F[max(0, k - 1):k + 2].sum() / F[1:].sum())
    per = 1 / fr[k]
    # 고른가: 주기마다 나눈 칸 중 봉우리(위 75%)가 있는 칸 비율
    m = int(n // per)
    if m < 4:
        return s, per, 0.0
    thr = np.percentile(p, 75)
    hits = [p[int(i * per):int((i + 1) * per)].max() > thr for i in range(m)]
    return s, float(per), float(np.mean(hits))


def ruled_feat(img, gt):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, np.ones((7, 7), np.uint8))
    valid = (cv2.dilate(gt, np.ones((15, 15), np.uint8)) == 0).astype(np.float32)
    h, w = L.shape
    best = (0.0, 0.0, 0.0, 0, 0)
    for axis in (0, 1):
        r0, v0 = (bh, valid) if axis == 0 else (bh.T.copy(), valid.T.copy())
        hh, ww = r0.shape
        for ang in range(-8, 9, 2):
            M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0)
            r = cv2.warpAffine(r0, M, (ww, hh))
            v = cv2.warpAffine(v0, M, (ww, hh), flags=cv2.INTER_NEAREST)
            s, per, even = period_score(r, v)
            if s > best[0]:
                best = (s, per, even, axis, ang)
    # 반대 방향(모눈)도 같은 각에서
    s2 = 0.0
    other = 1 - best[3]
    r0, v0 = (bh, valid) if other == 0 else (bh.T.copy(), valid.T.copy())
    hh, ww = r0.shape
    M = cv2.getRotationMatrix2D((ww / 2, hh / 2), best[4], 1.0)
    s2, per2, _ = period_score(cv2.warpAffine(r0, M, (ww, hh)), cv2.warpAffine(v0, M, (ww, hh), flags=cv2.INTER_NEAREST))
    return dict(rule_s=round(best[0], 3), rule_per=round(best[1], 1), rule_even=round(best[2], 2), rule_axis=best[3], rule_ang=best[4],
                rule_s2=round(s2, 3), rule_per2=round(per2, 1))


def split_of(name):
    return "learn" if int(hashlib.md5(Path(name).name.encode()).hexdigest()[0], 16) % 2 == 0 else "check"


def feat_one(job):
    it, root = job
    img = cv2.imread(str(Path(root) / "x" / it["file"]))
    row = dict(i=it["i"], file=it["file"], pm_split=it["split"], split=split_of(it["file"]))
    if img is None:
        return None
    gt = gt_mask(it["segs"], img.shape[:2])
    c = contrast_feat(img, gt)
    if c is None:
        return None
    row.update(c)
    row.update(ruled_feat(img, gt))
    return row


def feat(a):
    root = Path(a.meta)
    idx = json.load(open(root / "index.json"))
    with Pool(a.procs) as p:
        rows = [r for r in p.map(feat_one, [(it, str(root)) for it in idx], chunksize=16) if r]
    with open(root / "pick_feat.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)}장 → {root / 'pick_feat.csv'}")


def is_faint(r):
    return r["dk95"] < 15 and r["ch90"] < 25 and r["filled"] < 0.3


def is_ruled(r):
    return r["rule_s"] > 0.6 and r["rule_even"] > 0.8


def pick(a):
    root = Path(a.meta)
    rows = list(csv.DictReader(open(root / "pick_feat.csv")))
    for r in rows:
        for k, v in r.items():
            if k not in ("file", "pm_split", "split"):
                r[k] = float(v)
    drop = set()
    dp = Path(__file__).resolve().parent / "faint_ruled_drop.txt"
    if dp.exists():  # 눈으로 보고 뺀 것 (파일 이름 · 묶음)
        drop = {(ln.split()[0], ln.split()[1]) for ln in open(dp) if ln.strip() and not ln.startswith("#")}
    faint = [r for r in rows if is_faint(r) and (Path(r["file"]).name, "faint") not in drop]
    ruled = [r for r in rows if is_ruled(r) and (Path(r["file"]).name, "ruled") not in drop]
    out = dict(faint=[r["file"] for r in faint], ruled=[r["file"] for r in ruled])
    json.dump(out, open(root / "pick.json", "w"), indent=0)
    for k, rs in (("faint", faint), ("ruled", ruled)):
        print(f"{k}: {len(rs)}장 (학습 {sum(r['split'] == 'learn' for r in rs)} · 확인 {sum(r['split'] == 'check' for r in rs)})")
    if a.sheet:
        for k, rs in (("faint", faint), ("ruled", ruled)):
            sheet(root, rs, root / f"pick_{k}")


def sheet(root, rs, stem, per=48):
    for p in range(0, len(rs), per):
        tiles = []
        for r in rs[p:p + per]:
            img = cv2.imread(str(root / "x" / r["file"]))
            t = np.full((200, 160, 3), 255, np.uint8)
            s = min(160 / img.shape[1], 180 / img.shape[0])
            im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            t[:im.shape[0], :im.shape[1]] = im
            cv2.putText(t, f"{int(r['i'])} d{r['dk95']:.0f} r{r['rule_s']:.2f}", (2, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            tiles.append(t)
        while len(tiles) % 8:
            tiles.append(np.full((200, 160, 3), 255, np.uint8))
        rows_ = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
        cv2.imwrite(f"{stem}_{p // per}.jpg", np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["feat", "pick"])
    ap.add_argument("--meta", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    feat(a) if a.mode == "feat" else pick(a)


if __name__ == "__main__":
    sys.exit(main())
