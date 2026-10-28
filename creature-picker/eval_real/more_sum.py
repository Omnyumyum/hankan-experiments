"""more_eval.py 결과 CSV → 카테고리별 · 종류별 표 (MORE_ANIMALS.md 에 붙인 표).

    ../venv/bin/python -I eval_real/more_sum.py eval_real/more_<이름>.csv [--split test]
"""
import argparse
import csv
from collections import Counter

NAME = {"quadruped": "네 발", "long": "긴 몸", "wings": "날개", "blob": "몸통만", "모름": "모름"}
KINDS = ["quadruped", "long", "wings", "blob", "None"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--split")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.csv, encoding="utf-8-sig")) if not a.split or r["split"] == a.split]
    cats = list(dict.fromkeys(r["cat"] for r in rows))
    print("| 그림 | 기대 | 장수 | 네 발 | 긴 몸 | 날개 | 몸통만 | None | 맞음 | 틀림 | 모르겠음 | 움직인 수 | 늘어남>5 | 뒤집힘>1% | **망가짐** |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot = Counter()
    by_kind = {}
    for cat in sorted(cats, key=lambda c: (list(NAME).index(next(r["expect"] for r in rows if r["cat"] == c)), c)):
        rs = [r for r in rows if r["cat"] == cat]
        ex = rs[0]["expect"]
        c = Counter(r["guess"] or "(오류)" for r in rs)
        an = [r for r in rs if r["broken"] != ""]
        bs, bf, br = (sum(int(r[k]) for r in an) for k in ("big_stretch", "big_flip", "broken"))
        if ex == "모름":
            ok = bad = "-"
            none = c["None"] + c["(오류)"]
        else:
            ok = c[ex]
            none = c["None"] + c["(오류)"]
            bad = len(rs) - ok - none
            tot.update(ok=ok, bad=bad, none=none)
        k = by_kind.setdefault(ex, Counter())
        k.update(n=len(rs), anim=len(an), bs=bs, bf=bf, br=br, guessed=len(rs) - none)
        if ex != "모름":
            k.update(ok=ok, bad=bad)
        cells = [f"**{c[g]}**" if g == ex else str(c[g]) for g in KINDS]
        print(f"| {cat} | {NAME[ex]} | {len(rs)} | " + " | ".join(cells) + f" | {ok} | {bad} | {none} | {len(an)} | {bs} | {bf} | **{br}** |")
    n = max(1, tot["ok"] + tot["bad"] + tot["none"])
    print(f"\n기대 종류가 있는 카테고리 합계: 맞음 {tot['ok']} ({100 * tot['ok'] / n:.0f}%) · 틀림 {tot['bad']} ({100 * tot['bad'] / n:.1f}%) · 모르겠음 {tot['none']}")
    print("\n| 기대 종류 | 장수 | 맞음 | 틀림 | 추측 낸 수 | 움직인 수 | 늘어남>5 | 뒤집힘>1% | 망가짐 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for ex, k in by_kind.items():
        print(f"| {NAME[ex]} | {k['n']} | {k['ok'] if ex != '모름' else '-'} | {k['bad'] if ex != '모름' else '-'} | {k['guessed']} | {k['anim']} | {k['bs']} | {k['bf']} | {k['br']} ({100 * k['br'] / max(1, k['anim']):.1f}%) |")
    err = [r for r in rows if r["err"]]
    print(f"\n오류 {len(err)}장", Counter(r["err"][:50] for r in err).most_common(5))


if __name__ == "__main__":
    main()
