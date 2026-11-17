"""흔들어도 같은가. 고른 그림마다 6가지로 흔들어 다시 그리고 규칙을 돌려, 관절을 기준 그림 좌표로 되돌려 비교한다.

흔들기: flip(좌우 뒤집기) · rot+5 · rot-5 · s0.8 · s1.2 (선 굵기도 같이 바뀜, 사진 크기 바꾸기와 같음) · width(선 굵기만 바꿈)
뒤집기: 규칙의 다리 이름(right/left)은 실제 가까운 쪽/먼 쪽이 아니라 머리 쪽을 기준으로 한 「안쪽/바깥쪽」이라
        (quadruped.find_joints 334~340줄) 뒤집어도 같은 다리는 같은 이름이어야 한다. 그래서 이름을 바꾸지 않고 비교한다.
거리: 몸 상자(기준 그림 마스크 상자) 긴 변으로 나눈 값. 흔든 그림에서 다리를 못 찾으면 거리 = 무한대.
입력: out/pick.json ([{cat, idx, style}...])   출력: out/jitter.jsonl
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import qd
import rule

VARIANTS = {"flip": dict(flip=True), "rot+5": dict(rot=5), "rot-5": dict(rot=-5),
            "s0.8": dict(scale=0.8), "s1.2": dict(scale=1.2), "width": dict()}

_cache: dict = {}


def drawing(cat, idx):
    if cat not in _cache:
        _cache.clear()
        _cache[cat] = qd.load(cat, 3000)
    return _cache[cat][idx]


def one(item):
    cat, idx, st = item["cat"], item["idx"], item["style"]
    d = drawing(cat, idx)
    img, _ = qd.render(d, st)
    base = rule.run(img)
    rec = {"cat": cat, "idx": idx, "style": st, "found": base["found"]}
    if not base["found"]:
        return rec
    l, t, r, b = base["box"]
    size = max(r - l, b - t)
    J = base["joints"]
    for name, kw in VARIANTS.items():
        if name == "width":
            kw = dict(width=qd.WIDTH_ALT[st])
        im2, (_, to_base) = qd.render(d, st, **kw)
        o = rule.run(im2)
        if not o["found"]:
            rec[name] = None
            continue
        back = to_base(o["joints"])
        rec[name] = [round(float(v), 4) for v in np.linalg.norm(back - J, axis=1) / size]
    return rec


def main(pick_path="out/pick.json", out_path="out/jitter.jsonl"):
    items = json.load(open(pick_path))
    done = set()
    if Path(out_path).exists():
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["cat"], r["idx"], r["style"]))
    items = [it for it in items if (it["cat"], it["idx"], it["style"]) not in done]
    items.sort(key=lambda it: (it["cat"], it["idx"]))
    with Pool(4) as pool, open(out_path, "a") as f:
        for k, rec in enumerate(pool.imap(one, items, chunksize=4)):
            f.write(json.dumps(rec) + "\n")
            if k % 500 == 0:
                f.flush()
                print(k, len(items), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
