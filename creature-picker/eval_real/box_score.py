"""box_eval.py 의 중간 파일로 네 갈래(지금 · 상자만 · 누름만 · 상자 + 누름)를 나란히 잰다. 모델은 다시 돌리지 않는다.

    ../venv/bin/python -I eval_real/box_rows.py nh_check tap_check fr_check qd_check   # 먼저 그림마다 숫자 한 줄
    ../venv/bin/python -I eval_real/box_score.py report --split learn|check [--csv 이름]  # 고른 규칙(RULE)으로 네 갈래 표
갈래:
  지금      = 사진 전체 → Drawing (clean_photo 는 FAINT_RULED 갈래를 붙인 것)
  상자만    = 찾기 상자가 있고, 상자 안 마스크와 사진 전체 마스크가 IoU ≥ RULE["min_agree"] 이면 상자(여백 RULE["margin"])로 자른 그림 → Drawing,
              아니면 상자를 버리고 지금 방식
  누름만    = 지금 마스크를 보여 주고, 「누름 묻기」 신호가 켜지면 어른이 그림 가운데를 한 번 누름 → MobileSAM(clean_photo 그림) 후보 중
              TAP_SEGMENT 의 최종 후처리(guardbbig@base+fill: 종이 같은 · 테두리 닿은 후보 빼고 가장 큰 것, 없으면 자동 마스크, 닫힌 안쪽 채우기)
  상자 + 누름 = 상자만 결과를 보여 주고, 상자를 썼는데 두 마스크 IoU < RULE["ask_t"] 이면(상자를 안 썼으면 마스크가 빈 경우만) 누름을 묻는다.
              누름은 사진 전체 MobileSAM 에 「누른 점 + 찾기 상자」를 같이 넣는다 (학습용에서 상자 안만 넣기 · 점만보다 좋았음)
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
RULE = dict(margin=0.1, min_agree=0.1, ask_t=0.4)  # 여백 10%, 두 마스크 IoU<0.1 이면 상자 버림, 상자를 썼고 IoU<0.4 면 누름 물음


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


def fail(rs, k):
    x = np.array([r[k] for r in rs], float)
    return int((x < 0.5).sum()), float(np.median(x)) if len(x) else float("nan")


def decide(r, rule=None):
    """고른 규칙 (학습용에서 정함). r: box_rows.py 한 줄. 반환 (상자 씀, 누름 물음) — 정답을 보지 않는 값만 쓴다."""
    rule = rule or RULE
    m = rule["margin"]
    used = r["has_box"] and not r[f"empty{m}"] and r[f"agree{m}"] >= rule["min_agree"]
    ask = (r[f"agree{m}"] < rule["ask_t"]) if used else r["full_empty"]
    return used, ask


def final(r, rule=None):
    rule = rule or RULE
    m = rule["margin"]
    used, ask = decide(r, rule)
    auto = r[f"box{m}"] if used else r["iou_now"]
    tap = r[f"tap_prompt@{m}"] if used else r["tap_full@now"]
    return dict(used=used, ask=ask, now=r["iou_now"], box=auto, tap_only=r["iou_now"], tap_only_human=r["tap_full@now"] if r["iou_now"] < 0.5 else r["iou_now"],
                box_tap=tap if ask else auto, box_tap_human=tap if auto < 0.5 else auto, meta=r.get("iou_meta", 0.0), box_hit=r["box_iou_gt"] >= 0.5)


def report(a):
    import pickle
    for name in a.sets.split(","):
        rows = pickle.load(open(CACHE / f"{name}_{a.split}_rows.pkl", "rb"))
        out = [dict(key=r["key"], file=r["file"], group=r["group"], score=r["score"], **final(r)) for r in rows]
        groups = GROUPS.get(name) or sorted({r["group"] for r in out}, key=lambda g: list(dict.fromkeys(r["group"] for r in out)).index(g))
        print(f"\n### {name}_{a.split} ({len(out)}장)")
        print("| 묶음 | 장수 | 상자 찾음 · 정답 상자와 맞음 · 씀 | 지금 | 상자만 | 누름만 (신호) | 상자 + 누름 (물음) | 참고: 사람이 보고 누름 — 누름만 · 상자 + 누름 | 참고: 상자 + Meta segment() |")
        print("|---|---|---|---|---|---|---|---|---|")
        for g in groups + ["전체"]:
            x = out if g == "전체" else [r for r in out if r["group"] == g]
            if not x:
                continue
            n = len(x)

            def c(k):
                f, m = fail(x, k)
                return f"{f} ({100 * f / n:.0f}%) · {m:.2f}"
            rx = [r for r in rows if g == "전체" or r["group"] == g]
            print(f"| {g} | {n} | {sum(r['has_box'] for r in rx)} · {sum(r['box_hit'] for r in x)} · {sum(r['used'] for r in x)} | {c('now')} | {c('box')} | {c('tap_only')} | "
                  f"{c('box_tap')} ({sum(r['ask'] for r in x)}, {100 * sum(r['ask'] for r in x) / n:.0f}%) | "
                  f"{fail(x, 'tap_only_human')[0]} ({sum(r['now'] < 0.5 for r in x)}번 누름) · {fail(x, 'box_tap_human')[0]} ({sum(r['box'] < 0.5 for r in x)}번) | {c('meta')} |")
        if a.csv:
            with open(HERE / "integrate4" / f"{a.csv}_{name}_{a.split}.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(out[0]))
                w.writeheader()
                w.writerows([{k: (round(v, 3) if isinstance(v, float) else int(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in r.items()} for r in out])
    print("(칸: 마스크 실패 IoU<0.5 장수 (%) · IoU 중앙값. 누름만은 찾기 모델 없이 쓸 수 있는 신호가 학습용에서 「안 물음」보다 나은 것이 없어 묻지 않음 = 지금과 같음)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["report"])
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--split", default="check")
    ap.add_argument("--csv", default="")
    a = ap.parse_args()
    report(a)


if __name__ == "__main__":
    main()
