"""종류 추측(propose 의 guess)만 따로 재는 시험. 카테고리마다 같은 파일에서 학습용 · 확인용 그림을 겹치지 않게 뽑는다.

    venv/bin/python -I eval_real/guess_eval.py --qd 받은폴더 [--n 100] [--seed 11] [--split train|test|fresh|all] [--old] [--out 이름]
받은폴더: run_eval.py 와 같은 {category}.ndjson 10개 (앞부분만 받아도 된다).
섞은 뒤 앞 n장 = 학습용(train, 모델을 배울 때만 씀), 다음 n장 = 확인용(test, 모델 종류 · 기준을 비교할 때 봄),
그다음 n장 = 마지막 확인용(fresh, 다 정한 뒤 한 번만 잼).
--old : 예전 규칙(guess_rules)으로 잰다.
결과: eval_real/guess_<out>.csv (그림마다 추측), 화면에 카테고리별 혼동표 · 맞음/틀림/모르겠음.
"""
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import rig_types as rt  # noqa: E402
from run_eval import EXPECT, KINDS, render  # noqa: E402


def load_split(path, n, seed):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue  # 받다 만 마지막 줄
            if r.get("recognized"):
                rows.append(r["drawing"])
    idx = np.random.default_rng(seed).permutation(len(rows))
    if len(idx) < 3 * n:
        raise SystemExit(f"{path}: 인식된 그림 {len(rows)}장 < {3 * n}장 — 더 받아야 한다")
    return {s: [rows[i] for i in idx[k * n:(k + 1) * n]] for k, s in enumerate(("train", "test", "fresh"))}


def one(job):
    cat, split, k, strokes = job
    img = render(strokes)
    try:
        d = rt.Drawing(img)
        g = rt.propose(d)["guess"] or "None"
    except Exception as e:
        g = "(오류)"
        print(cat, split, k, repr(e)[:100], flush=True)
    return dict(cat=cat, split=split, id=k, expect=EXPECT[cat], guess=g)


def table(rows, title):
    print(f"\n## {title} ({len(rows)}장)")
    print("| 그림 | 네 발 | 긴 몸 | 날개 | 몸통만 | None | 맞음 | 틀림 | 모르겠음 |")
    print("|---|---|---|---|---|---|---|---|---|")
    name = {"quadruped": "네 발", "long": "긴 몸", "wings": "날개", "blob": "몸통만"}
    tot = [0, 0, 0]
    for cat in EXPECT:
        rs = [r for r in rows if r["cat"] == cat]
        if not rs:
            continue
        c = {k: sum(r["guess"] == k for r in rs) for k in KINDS + ["(오류)"]}
        ok = c[EXPECT[cat]]
        none = c["None"] + c["(오류)"]
        bad = len(rs) - ok - none
        tot = [tot[0] + ok, tot[1] + bad, tot[2] + none]
        cells = [f"**{c[k]}**" if k == EXPECT[cat] else str(c[k]) for k in KINDS]
        print(f"| {cat} → {name[EXPECT[cat]]} | " + " | ".join(cells) + f" | {ok} | {bad} | {none} |")
    n = max(1, sum(tot))
    print(f"| **합계** | | | | | | **{tot[0]} ({100 * tot[0] / n:.0f}%)** | **{tot[1]} ({100 * tot[1] / n:.0f}%)** | **{tot[2]} ({100 * tot[2] / n:.0f}%)** |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qd", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--split", choices=["train", "test", "fresh", "all"], default="test")
    ap.add_argument("--old", action="store_true")
    ap.add_argument("--out", default="now")
    a = ap.parse_args()
    splits = ["train", "test", "fresh"] if a.split == "all" else [a.split]
    if a.old:
        rt.GUESS_MODEL = Path("/없음")  # 모델 파일이 없으면 guess_kind 가 예전 규칙을 쓴다
    jobs = []
    for cat in EXPECT:
        sp = load_split(Path(a.qd) / f"{cat}.ndjson", a.n, a.seed)
        for s in splits:
            jobs += [(cat, s, k, st) for k, st in enumerate(sp[s])]
    with Pool() as p:
        rows = p.map(one, jobs, chunksize=4)
    with open(HERE / f"guess_{a.out}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s in splits:
        table([r for r in rows if r["split"] == s], f"{dict(train='학습용', test='확인용', fresh='마지막 확인용')[s]} {a.out}")


if __name__ == "__main__":
    main()
