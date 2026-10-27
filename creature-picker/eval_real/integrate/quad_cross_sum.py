"""quad_cross.py 결과 요약: 라벨(F/Q/O)별 망가짐 · 몸통만 · Q 다리 움직임.
    ../venv/bin/python eval_real/integrate/quad_cross_sum.py eval_real/integrate   (x_{base,A,B,M}_set{A,B}.csv 를 읽음)
라벨: A 묶음 = ac1c266 의 quad_labels.csv, B 묶음 = a29c661 의 quad_labels.txt (git show 로 읽음)."""
import csv, subprocess, sys
import numpy as np

S = sys.argv[1]
REPO = "/home/user/hankan-experiments"


def labels_A():
    txt = subprocess.check_output(["git", "-C", REPO, "show", "ac1c266:creature-picker/eval_real/quad_labels.csv"]).decode("utf-8-sig")
    return {(r["cat"], int(r["id"])): r["label"] for r in csv.DictReader(txt.splitlines()) if r["split"] == "check"}


def labels_B():
    txt = subprocess.check_output(["git", "-C", REPO, "show", "a29c661:creature-picker/eval_real/quad_labels.txt"]).decode()
    out = {}
    for line in txt.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        split, cat, rng = p[0], p[1], p[2]
        if split != "check":
            continue
        lo, hi = map(int, rng.split("-"))
        lists = {}
        for t in p[3:]:
            k, v = t.split(":")
            lists[k] = [int(x) for x in v.split(",") if x]
        default = "F" if "Q" in lists else "Q"
        for i in range(lo, hi + 1):
            out[(cat, i)] = default
        for k, v in lists.items():
            for i in v:
                out[(cat, i)] = k
    return out


def load(c, s):
    return {(r["cat"], int(r["id"])): r for r in csv.DictReader(open(f"{S}/x_{c}_set{s}.csv"))}


for s, lab in (("A", labels_A()), ("B", labels_B())):
    base = load("base", s)
    print(f"\n### 그림 묶음 {s} (확인용 {len(base)}장)  라벨 수", {k: list(lab.values()).count(k) for k in "FQO"})
    print("| 코드 | 망가짐 | F 망가짐 | Q 망가짐 | O 망가짐 | 몸통만 (F/Q/O) | Q 다리 움직임 평균 | Q 다리 20%↓ | 오류 |")
    for c in ("base", "A", "B", "M"):
        try:
            R = load(c, s)
        except FileNotFoundError:
            continue
        n = len(R)
        bad = lambda L: sum(1 for k, r in R.items() if r["bad"] == "1" and (L is None or lab.get(k) == L))
        cnt = lambda L: sum(1 for k in R if lab.get(k) == L)
        blob = {L: sum(1 for k, r in R.items() if r["blob"] == "1" and lab.get(k) == L) for L in "FQO"}
        qk = [k for k in R if lab.get(k) == "Q" and R[k]["blob"] != "1" and base[k]["leg_move"] and R[k]["leg_move"]]
        lm0 = np.mean([float(base[k]["leg_move"]) for k in qk])
        lm1 = np.mean([float(R[k]["leg_move"]) for k in qk])
        drop = sum(1 for k in qk if float(R[k]["leg_move"]) < 0.8 * float(base[k]["leg_move"]))
        err = sum(1 for r in R.values() if r["err"])
        print(f"| {c} | {bad(None)} ({100*bad(None)/n:.1f}%) | {bad('F')}/{cnt('F')} | {bad('Q')}/{cnt('Q')} | {bad('O')}/{cnt('O')} | "
              f"{blob['F']}/{blob['Q']}/{blob['O']} | {lm0:.4f} → {lm1:.4f} | {drop}/{len(qk)} | {err} |")
