"""눈으로 매길 그림 뽑기: 종류 x 꼴(51칸)마다, 자동 고르기를 통과하고 규칙이 「찾음」이라 한 그림에서
볼 것 6장 + 확인용 4장. 몸 표본(sheet_body)에 쓴 그림은 뺀다. 뒤 볼 것 300 · 확인용 200 으로 맞춘다.
출력: out/grade_pick.json"""
import csv
import json
import random

import features
import qd
import rule
from pick_rule import body_side

used = {(r["cat"], int(r["idx"])) for r in csv.DictReader(open("results/body_side_labels.csv"))}
picks = []
for cat in qd.CATS:
    ds = qd.load(cat, 3000)
    order = list(range(len(ds)))
    random.Random(f"grade-{cat}").shuffle(order)
    pos = 0
    for st in qd.STYLES:
        got = []
        while len(got) < 10 and pos < len(order):
            i = order[pos]; pos += 1
            if (cat, i) in used or not body_side(features.feats(ds[i])):
                continue
            img, _ = qd.render(ds[i], st)
            if rule.run(img)["found"]:
                got.append(i)
        for k, i in enumerate(got):
            picks.append({"cat": cat, "idx": i, "style": st, "split": "see" if k < 6 else "check"})
rng = random.Random("trim")
see = [p for p in picks if p["split"] == "see"]; chk = [p for p in picks if p["split"] == "check"]
see = rng.sample(see, 300) if len(see) > 300 else see
chk = rng.sample(chk, 200) if len(chk) > 200 else chk
picks = see + chk
rng.shuffle(picks)
for n, p in enumerate(picks):
    p["gid"] = n
json.dump(picks, open("out/grade_pick.json", "w"))
print(len(see), len(chk))
