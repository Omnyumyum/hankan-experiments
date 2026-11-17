"""눈으로 매기기용 모아 보기: 뽑은 500장에 규칙 뼈대를 겹쳐 12장씩 한 판으로.
출력: out/grade_sheets/{see|check}_NN.png, out/grade_joints.json"""

import json
from pathlib import Path

import cv2
import numpy as np

import overlay
import qd
import rule

OUT = Path("out/grade_sheets")
T = 320


def tile(p, cache):
    if p["cat"] not in cache:
        cache.clear(); cache[p["cat"]] = qd.load(p["cat"], 3000)
    d = cache[p["cat"]][p["idx"]]
    img, _ = qd.render(d, p["style"])
    o = rule.run(img)
    p["found"] = o["found"]; p["joints"] = o["joints"].round(1).tolist(); p["box"] = o["box"]
    l, t, r, b = o["box"]
    # 몸 상자 둘레로 잘라 크게 본다
    s = max(r - l, b - t) * 0.62
    cx, cy = (l + r) / 2, (t + b) / 2
    x0, y0 = int(max(0, cx - s)), int(max(0, cy - s))
    x1, y1 = int(min(qd.SIZE, cx + s)), int(min(qd.SIZE, cy + s))
    ov = overlay.draw(img, o["joints"], o["found"])[y0:y1, x0:x1]
    canvas = np.full((T, T, 3), 255, np.uint8)
    k = T / max(ov.shape[:2])
    ov = cv2.resize(ov, (int(ov.shape[1] * k), int(ov.shape[0] * k)), interpolation=cv2.INTER_AREA)
    canvas[:ov.shape[0], :ov.shape[1]] = ov
    cv2.putText(canvas, f"{p['gid']} {p['cat'][:5]} {p['style']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    cv2.rectangle(canvas, (0, 0), (T - 1, T - 1), (160, 160, 160), 1)
    return canvas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    picks = json.load(open("out/grade_pick.json"))
    cache = {}
    tiles = {}
    for p in sorted(picks, key=lambda p: p["cat"]):
        tiles[p["gid"]] = tile(p, cache)
    for split in ("see", "check"):
        ps = sorted([p for p in picks if p["split"] == split], key=lambda p: p["gid"])
        for n in range(0, len(ps), 12):
            group = [tiles[p["gid"]] for p in ps[n:n + 12]]
            group += [np.full((T, T, 3), 255, np.uint8)] * (12 - len(group))
            grid = np.vstack([np.hstack(group[r * 4:(r + 1) * 4]) for r in range(3)])
            cv2.imwrite(str(OUT / f"{split}_{n // 12:02d}.png"), grid)
    json.dump(picks, open("out/grade_joints.json", "w"))


if __name__ == "__main__":
    main()
