"""숫자 모으기. out/base/*.jsonl (전체 실행), out/jitter_wide.jsonl (흔들기 넓은 표본),
results/grade_500.csv (눈 매기기) → results/*.csv 와 화면 출력.

  results/found_rate.csv  종류 x 꼴: 마스크 만듦 · 「찾음」 비율 (전체 / 몸 옆모습으로 고른 것)
  results/jitter_rate.csv 종류 x 꼴: 흔들기마다 관절이 몸 상자 긴 변의 5% 안에 머문 비율
  results/yield.csv       종류마다 쓸 만한 정답 어림 (전체 Quick, Draw! 크기로)
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import qd
from pick_rule import body_side

V = ["flip", "rot+5", "rot-5", "s0.8", "s1.2", "width"]


def load_base():
    rows = []
    for c in qd.CATS:
        p = Path(f"out/base/{c}.jsonl")
        if p.exists():
            rows += [json.loads(x) for x in open(p)]
    return rows


def found_table(base):
    out = []
    for c in qd.CATS + ["ALL"]:
        rs = [r for r in base if c == "ALL" or r["cat"] == c]
        if not rs:
            continue
        sel = [r for r in rs if body_side(r)]
        row = {"cat": c, "n": len(rs), "body_side_auto": round(len(sel) / len(rs), 3)}
        for st in qd.STYLES:
            row[f"{st}_ok"] = round(np.mean([r[st]["ok"] for r in rs]), 3)
            row[f"{st}_found"] = round(np.mean([r[st]["found"] for r in rs]), 3)
            row[f"{st}_found_in_body"] = round(np.mean([r[st]["found"] for r in sel]), 3) if sel else ""
        out.append(row)
    return out


def jitter_table(path="out/jitter_wide.jsonl"):
    if not Path(path).exists():
        return []
    recs = [json.loads(x) for x in open(path)]
    out = []
    groups = defaultdict(list)
    for r in recs:
        if not r["found"]:
            continue
        groups[(r["cat"], r["style"])].append(r)
        groups[("ALL", r["style"])].append(r)
        groups[(r["cat"], "ALL")].append(r)
        groups[("ALL", "ALL")].append(r)
    for (c, st), rs in sorted(groups.items(), key=lambda kv: ((qd.CATS + ["ALL"]).index(kv[0][0]), kv[0][1])):
        row = {"cat": c, "style": st, "n": len(rs)}
        allv = []
        for v in V:
            fr = [0.0 if r.get(v) is None else float(np.mean(np.array(r[v]) <= 0.05)) for r in rs]
            row[v] = round(float(np.mean(fr)), 3)                 # 관절 단위 비율
            row[v + "_all18"] = round(float(np.mean([f == 1.0 for f in fr])), 3)   # 18개 다 머문 그림 비율
            row[v + "_lost"] = round(float(np.mean([r.get(v) is None for r in rs])), 3)  # 흔든 뒤 「못 찾음」
            allv += fr
        row["mean"] = round(float(np.mean(allv)), 3)
        out.append(row)
    return out


def yield_table(base, found):
    """쓸 만한 정답 어림 = 전체 recognized 수 x 몸 옆모습 고르기 통과 x 「찾음」 x 자동 거르기 통과 x 정밀도.
    꼴은 굵은 선(가장 좋음) 기준. 거르기 통과율 · 정밀도는 눈 매기기 표본(확인용)에서."""
    size = {r["cat"]: int(r["est_recognized"]) for r in csv.DictReader(open("results/qd_size.csv"))}
    g = list(csv.DictReader(open("results/grade_500.csv")))
    chk = [r for r in g if r["split"] == "check"]
    keep = [r for r in chk if r["auto_keep"] == "1"]
    prec = np.mean([r["usable"] == "1" for r in keep])
    out = []
    for f in found:
        c = f["cat"]
        if c == "ALL":
            continue
        gc = [r for r in g if r["cat"] == c and r["style"] == "thick"]
        keep_rate = np.mean([r["auto_keep"] == "1" for r in gc]) if gc else 0
        raw_usable = np.mean([r["usable"] == "1" for r in gc]) if gc else 0
        n_sel_found = size[c] * f["body_side_auto"] * f["thick_found_in_body"]
        out.append({"cat": c, "est_recognized": size[c], "body_side_auto": f["body_side_auto"],
                    "thick_found_in_body": f["thick_found_in_body"],
                    "est_selected_found": round(n_sel_found),
                    "usable_rate_no_filter(thick,eye)": round(float(raw_usable), 3),
                    "auto_keep_rate(thick)": round(float(keep_rate), 3),
                    "est_after_filter": round(n_sel_found * keep_rate),
                    "est_usable_after_filter": round(n_sel_found * keep_rate * prec)})
    return out, prec


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    base = load_base()
    print("전체 실행 그림 수", len(base))
    found = found_table(base)
    write("results/found_rate.csv", found)
    for r in found:
        print(r)
    jt = jitter_table()
    write("results/jitter_rate.csv", jt)
    for r in jt:
        if r["cat"] == "ALL" or r["style"] == "ALL":
            print({k: r[k] for k in ["cat", "style", "n", "mean"] + V})
    if len(found) > len(qd.CATS):
        y, prec = yield_table(base, found)
        write("results/yield.csv", y)
        print("확인용 정밀도", round(float(prec), 3))
        for r in y:
            print(r)
        print("합", sum(r["est_after_filter"] for r in y), sum(r["est_usable_after_filter"] for r in y))


if __name__ == "__main__":
    main()
