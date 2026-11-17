"""흔들기 넓은 표본: 종류 x 꼴마다, 몸 옆모습으로 고르고 「찾음」인 그림에서 40장 (고정 씨앗).
출력: out/pick_wide.json"""
import json
import random

import qd
from pick_rule import body_side

items = []
for c in qd.CATS:
    base = [json.loads(x) for x in open(f"out/base/{c}.jsonl")]
    for st in qd.STYLES:
        pool = [r["idx"] for r in base if body_side(r) and r[st]["found"]]
        for i in random.Random(f"wide-{c}-{st}").sample(pool, min(40, len(pool))):
            items.append({"cat": c, "idx": i, "style": st})
json.dump(items, open("out/pick_wide.json", "w"))
print(len(items))
