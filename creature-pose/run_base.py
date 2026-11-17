"""종류마다 앞 3000장(recognized)을 꼴 세 가지로 그려 규칙을 돌린다. 몸 모양 숫자(features)도 같이.
출력: out/base/{cat}.jsonl  (한 줄 = 그림 한 장)
쓰기: venv/bin/python run_base.py [종류 ...]"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import features
import qd
import rule

OUT = Path("out/base")
N = 3000


def one(args):
    cat, i, d = args
    rec = {"cat": cat, "idx": i, "key": d["key"], **features.feats(d)}
    for st in qd.STYLES:
        img, _ = qd.render(d, st)
        o = rule.run(img)
        rec[st] = {"ok": o["ok"], "found": o["found"], "box": o["box"],
                   "joints": None if o["joints"] is None else [[round(x, 1), round(y, 1)] for x, y in o["joints"]]}
    return rec


def main(cats):
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(4) as pool:
        for cat in cats:
            path = OUT / f"{cat}.jsonl"
            if path.exists() and sum(1 for _ in open(path)) >= N:
                continue
            ds = qd.load(cat, N)
            with open(path, "w") as f:
                for rec in pool.imap(one, [(cat, i, d) for i, d in enumerate(ds)], chunksize=8):
                    f.write(json.dumps(rec) + "\n")
            print(cat, "끝", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or qd.CATS)
