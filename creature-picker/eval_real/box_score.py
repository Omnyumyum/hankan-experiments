"""box_eval.py 의 중간 파일로 네 갈래(지금 · 상자만 · 누름만 · 상자 + 누름)를 나란히 잰다. 모델은 다시 돌리지 않는다.

    ../venv/bin/python -I eval_real/box_score.py learn            # 학습용: 여백 · 상자 버리기 · 누름 묻기 규칙을 고르는 표
    ../venv/bin/python -I eval_real/box_score.py check [--csv 이름] # 확인용: 고른 규칙(아래 RULE)으로 한 번만
갈래:
  지금      = 사진 전체 → Drawing (clean_photo 는 FAINT_RULED 갈래를 붙인 것)
  상자만    = 찾기 상자가 있고 「상자 버리기」 규칙에 안 걸리면 상자(여백 RULE["margin"])로 자른 그림 → Drawing, 아니면 지금 방식
  누름만    = 지금 마스크를 보여 주고, 「누름 묻기」 신호가 켜지면 어른이 그림 가운데를 한 번 누름 → MobileSAM(clean_photo 그림) 후보 중
              TAP_SEGMENT 의 최종 후처리(guardbbig@base+fill: 종이 같은 · 테두리 닿은 후보 빼고 가장 큰 것, 없으면 자동 마스크, 닫힌 안쪽 채우기)
  상자 + 누름 = 상자만 결과를 보여 주고, 신호가 켜지면 누름. 상자를 쓴 그림은 상자 안 그림(여백 SAM_MARGIN)만 MobileSAM 에 넣는다
누른 점은 TAP_SEGMENT 와 같은 흉내(정답 안 깊은 곳 무작위 + 손떨림 긴 변 2%). 「틀렸나」 신호는 정답을 보지 않는 값만 쓴다.
참고로 사람이 화면을 보고 틀린 것(IoU<0.5)만 누르는 경우(TAP_SEGMENT 의 흐름)도 같이 적는다.
"""
import argparse
import csv
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import tap_segment_eval as te  # noqa: E402

CACHE = HERE / "data" / "boxcache"
unpack, iou = te.unpack, te.iou
GROUPS = {"nh": ["quadruped", "wings", "blob", "long"], "tap": ["tap"], "fr": ["faint", "ruled", "control"], "qd": None}
SETS = ["nh", "tap", "fr", "qd"]

# 학습용에서 고른 규칙 (learn 표를 보고 정함 — 보고서 INTEGRATE4.md 3절)
RULE = dict(margin=0.2, min_score=0.0, min_agree=0.0, ask="none", ask_t=0.0, tap_box="crop")


def bbox_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def mask_box(m):
    ys, xs = np.nonzero(m)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)] if len(xs) else None


def load(name):
    base = pickle.load(gzip.open(CACHE / f"{name}_base.pkl.gz", "rb"))
    p = CACHE / f"{name}_sam.pkl.gz"
    sam = {r["key"]: r for r in pickle.load(gzip.open(p, "rb"))} if p.exists() else {}
    return base, sam


def tap_full(b, s, prompt, auto):
    gt_shape = (b["h"], b["w"])
    fill_k = max(5, int(0.03 * max(gt_shape)) | 1)
    cands, sc = s["full"][prompt]
    cands = [unpack(x) for x in cands]
    return te.final_mask("guardbbig@base+fill", cands, sc, b["pbox"], auto, b["taps"]["i"], fill_k)


def tap_crop(b, s, auto):
    x0, y0, x1, y1 = s["crop"]["xy"]
    cands, sc = s["crop"]["in"]
    cands = [unpack(x) for x in cands]
    sub = auto[y0:y1, x0:x1]
    fill_k = max(5, int(0.03 * max(y1 - y0, x1 - x0)) | 1)
    tp = [b["taps"]["i"][0] - x0, b["taps"]["i"][1] - y0]
    m = te.final_mask("guardbbig@base+fill", cands, sc, [0, 0, x1 - x0 - 1, y1 - y0 - 1], sub, tp, fill_k)
    out = auto.copy()
    out[y0:y1, x0:x1] = m
    out[:y0] = False
    out[y1:] = False
    out[:, :x0] = False
    out[:, x1:] = False
    return out


def features(b, s, margin):
    """정답을 보지 않는 값만."""
    full = unpack(b["full"])
    f = dict(score=b["score"], has_box=b["box"] is not None, nbox=sum(x[4] >= 0.5 for x in b["boxes"]),
             full_empty=not full.any(), full_err=bool(b["err"]))
    area = b["h"] * b["w"]
    f["full_frac"] = full.sum() / area
    if b["box"] is not None:
        bx = b["box"]
        f["box_frac"] = (bx[2] - bx[0]) * (bx[3] - bx[1]) / area
        ib = unpack(b["inbox"][margin]["mask"])
        f["inbox_empty"] = not ib.any()
        f["agree"] = iou(ib, full)  # 상자 안 마스크 ↔ 사진 전체 마스크
        fb = mask_box(full)
        f["fullbox_vs_det"] = bbox_iou(fb, bx) if fb else 0.0
        ibb = mask_box(ib)
        f["inbox_vs_det"] = bbox_iou(ibb, bx) if ibb else 0.0
        if s and "dbox" in s["full"]:  # 누르지 않고 찾기 상자만 MobileSAM 에 (참고 신호)
            c, sc = s["full"]["dbox"]
            f["sam_box_agree"] = max(iou(unpack(x), ib) for x in c)
    else:
        f.update(box_frac=0.0, inbox_empty=True, agree=0.0, fullbox_vs_det=0.0, inbox_vs_det=0.0)
    return f


def use_box(f, rule):
    return f["has_box"] and not f["inbox_empty"] and f["score"] >= rule["min_score"] and (f["agree"] >= rule["min_agree"] or rule["min_agree"] <= 0)


def ask(f, rule, boxed):
    """누름을 물을지 (정답을 보지 않는 신호)."""
    k, t = rule["ask"], rule["ask_t"]
    if k == "none":
        return False
    if k == "all":
        return True
    empty = f["full_empty"] if not boxed else f["inbox_empty"]
    if empty:
        return True
    if k == "agree":  # 상자 안 · 사진 전체 두 마스크가 서로 다르면
        return not f["has_box"] or f["agree"] < t
    if k == "score":
        return f["score"] < t
    if k == "agree|score":
        return not f["has_box"] or f["agree"] < t or f["score"] < 0.9
    if k == "sam":
        return f.get("sam_box_agree", 0.0) < t
    raise ValueError(k)


def run_rows(name, rule, oracle=False):
    base, sam = load(name)
    out = []
    for b in base:
        s = sam.get(b["key"])
        gt = unpack(b["gt"])
        full = unpack(b["full"])
        f = features(b, s, rule["margin"])
        boxed = use_box(f, rule)
        auto_box = unpack(b["inbox"][rule["margin"]]["mask"]) if boxed else full
        r = dict(key=b["key"], file=b["file"], group=b["group"], score=round(b["score"], 3), has_box=int(f["has_box"]), used_box=int(boxed),
                 agree=round(f["agree"], 3), iou_now=round(iou(full, gt), 3), iou_box=round(iou(auto_box, gt), 3))
        gb = mask_box(gt)
        r["box_iou_gt"] = round(bbox_iou(b["box"], gb), 3) if b["box"] is not None and gb else 0.0
        r["iou_metaseg"] = round(iou(unpack(b["metaseg"]["mask"]), gt), 3) if "metaseg" in b else 0.0
        if s:
            tf = tap_full(b, s, "in", full)
            if boxed and rule["tap_box"] == "crop" and "crop" in s:
                tb = tap_crop(b, s, auto_box)
            elif boxed and rule["tap_box"] == "prompt" and "in+dbox" in s["full"]:
                tb = tap_full(b, s, "in+dbox", auto_box)
            else:
                tb = tap_full(b, s, "in", auto_box)
            r["iou_tap"], r["iou_boxtap"] = round(iou(tf, gt), 3), round(iou(tb, gt), 3)
            a_now, a_box = ask(f, rule, False), ask(f, rule, boxed)
            r["ask_now"], r["ask_box"] = int(a_now), int(a_box)
            r["iou_tap_flow"] = r["iou_tap"] if a_now else r["iou_now"]
            r["iou_boxtap_flow"] = r["iou_boxtap"] if a_box else r["iou_box"]
            # 참고: 사람이 보고 틀린 것만 누름
            r["iou_tap_human"] = r["iou_tap"] if r["iou_now"] < 0.5 else r["iou_now"]
            r["iou_boxtap_human"] = r["iou_boxtap"] if r["iou_box"] < 0.5 else r["iou_box"]
        r["t_det"], r["t_now"] = round(b["det_t"], 3), round(b["full_t"], 3)
        r["t_box"] = round(b["inbox"][rule["margin"]]["t"], 3) if boxed else ""
        if s:
            r["t_sam"] = round(s["enc_box"] if boxed and "enc_box" in s else s["enc"], 3)
        out.append(r)
    return out


def fail(rs, k):
    x = np.array([r[k] for r in rs], float)
    return int((x < 0.5).sum()), float(np.median(x)) if len(x) else float("nan")


def table(name, rs, groups):
    has_sam = "iou_tap" in rs[0]
    print(f"\n### {name} ({len(rs)}장)")
    print("| 묶음 | 장수 | 상자 찾음 / 씀 | 지금 | 상자만 | 누름만 (물음) | 상자 + 누름 (물음) | 참고: 사람이 보고 누름 — 누름만 · 상자 + 누름 |")
    print("|---|---|---|---|---|---|---|---|")
    for g in (groups or sorted({r["group"] for r in rs})) + ["전체"]:
        x = rs if g == "전체" else [r for r in rs if r["group"] == g]
        if not x:
            continue
        n = len(x)

        def c(k):
            f, m = fail(x, k)
            return f"{f} ({100 * f / n:.0f}%) · {m:.2f}"
        cells = [c("iou_now"), c("iou_box")]
        if has_sam:
            cells += [c("iou_tap_flow") + f" ({sum(r['ask_now'] for r in x)})", c("iou_boxtap_flow") + f" ({sum(r['ask_box'] for r in x)})",
                      f"{fail(x, 'iou_tap_human')[0]} · {fail(x, 'iou_boxtap_human')[0]}"]
        print(f"| {g} | {n} | {sum(r['has_box'] for r in x)} / {sum(r['used_box'] for r in x)} | " + " | ".join(cells) + " |")
    print("(칸: 마스크 실패 IoU<0.5 장수 (%) · IoU 중앙값, 괄호 = 누름을 물은 장수)")


def learn(a):
    import itertools
    sets = a.sets.split(",")
    # 1) 여백: 상자를 늘 쓸 때 실패 수
    print("## 1. 상자 여백 (상자가 있으면 늘 씀, 상자 없으면 지금 방식) — 학습용 마스크 실패 · IoU 중앙값")
    print("| 묶음 | 지금 | " + " | ".join(f"여백 {m}" for m in a.margins) + " | Meta segment() |")
    print("|---|---|" + "---|" * (len(a.margins) + 1))
    for name in sets:
        base, _ = load(f"{name}_learn")
        gts = [unpack(b["gt"]) for b in base]
        now = [iou(unpack(b["full"]), g) for b, g in zip(base, gts)]
        cells = [f"{sum(v < 0.5 for v in now)} · {np.median(now):.2f}"]
        for m in a.margins:
            v = [iou(unpack(b["inbox"][m]["mask"]) if b["box"] is not None else unpack(b["full"]), g) for b, g in zip(base, gts)]
            cells.append(f"{sum(x < 0.5 for x in v)} · {np.median(v):.2f}")
        v = [iou(unpack(b["metaseg"]["mask"]), g) if b["box"] is not None else 0.0 for b, g in zip(base, gts)]
        cells.append(f"{sum(x < 0.5 for x in v)} · {np.median(v):.2f}")
        print(f"| {name}_learn ({len(base)}) | " + " | ".join(cells) + " |")
    # 2) 상자 버리기 규칙
    print("\n## 2. 상자 버리기 규칙 (여백 %s) — 학습용 마스크 실패 합" % a.margin)
    print("| 점수 하한 | 두 마스크 닮음 하한 | " + " | ".join(sets) + " | 합 | 상자 씀 |")
    print("|---|---|" + "---|" * (len(sets) + 2))
    for ms, ma in itertools.product([0.0, 0.7, 0.9, 0.97], [0.0, 0.1, 0.2, 0.3]):
        rule = dict(RULE, margin=a.margin, min_score=ms, min_agree=ma, ask="none")
        cells, tot, used = [], 0, 0
        for name in sets:
            rs = run_rows(f"{name}_learn", rule)
            f, _ = fail(rs, "iou_box")
            cells.append(str(f))
            tot += f
            used += sum(r["used_box"] for r in rs)
        print(f"| {ms} | {ma} | " + " | ".join(cells) + f" | {tot} | {used} |")


def check(a):
    for name in a.sets.split(","):
        rs = run_rows(f"{name}_{a.split}", RULE)
        table(f"{name}_{a.split}", rs, GROUPS.get(name))
        if a.csv:
            with open(HERE / f"{a.csv}_{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rs[0]))
                w.writeheader()
                w.writerows(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["learn", "check"])
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--split", default="check")
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.35, 0.5])
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    learn(a) if a.mode == "learn" else check(a)


if __name__ == "__main__":
    main()
