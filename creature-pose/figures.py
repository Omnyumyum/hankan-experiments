"""커밋하는 모아 보기 그림 (figures/*.jpg, 장당 300KB 이하). 그림 출처: Quick, Draw! (Google, CC BY 4.0).

  fig01_usable.jpg       눈으로 「쓸 만함」이라 매긴 것
  fig02_head_wrong.jpg   머리 쪽을 반대로 짐작 (앞 · 뒷다리 이름이 뒤바뀜)
  fig03_tail_wrong.jpg   꼬리를 다리로 쓰거나 꼬리를 놓침
  fig04_nonsense.jpg     뼈대가 엉뚱함 (얼굴 · 갈기를 다리로, 다리 못 나눔 등)
  fig05_styles.jpg       같은 그림 세 꼴 (선만 · 굵은 선 · 채움)
  fig06_body_side.jpg    몸 전체 옆모습 표본 판 (고양이 · 말)
  fig07_jitter.jpg       흔들기: 기준 · 뒤집기 · +5도 · 0.8배 · 선 굵기
"""

import csv
import random
from pathlib import Path

import cv2
import numpy as np

import overlay
import qd
import rule

OUT = Path("figures")
CREDIT = "Drawings: Quick, Draw! (Google), CC BY 4.0. Skeleton: drawtale-ai quadruped rule @4c2147f"
T = 240
_cache: dict = {}


def drawing(cat, idx):
    if cat not in _cache:
        _cache.clear()
        _cache[cat] = qd.load(cat, 3000)
    return _cache[cat][idx]


def crop_tile(img, J, found, box, label):
    l, t, r, b = box
    s = max(r - l, b - t) * 0.62
    cx, cy = (l + r) / 2, (t + b) / 2
    x0, y0 = int(max(0, cx - s)), int(max(0, cy - s))
    x1, y1 = int(min(img.shape[1], cx + s)), int(min(img.shape[0], cy + s))
    ov = (overlay.draw(img, J, found) if J is not None else img)[y0:y1, x0:x1]
    k = T / max(ov.shape[:2])
    ov = cv2.resize(ov, (int(ov.shape[1] * k), int(ov.shape[0] * k)), interpolation=cv2.INTER_AREA)
    canvas = np.full((T, T, 3), 255, np.uint8)
    canvas[:ov.shape[0], :ov.shape[1]] = ov
    cv2.putText(canvas, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (T - 1, T - 1), (170, 170, 170), 1)
    return canvas


def tile_for(cat, idx, style, label, **kw):
    img, _ = qd.render(drawing(cat, idx), style, **kw)
    o = rule.run(img)
    box = o["box"] or (0, 0, img.shape[1], img.shape[0])
    return crop_tile(img, o["joints"], o["found"], box, label)


def save(name, tiles, cols=4, title=""):
    tiles = tiles + [np.full((T, T, 3), 255, np.uint8)] * ((-len(tiles)) % cols)
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)])
    bar = np.full((44, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, CREDIT, (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    img = np.vstack([grid, bar])
    for q in (85, 75, 65, 55):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if len(buf) <= 290_000:
            break
    (OUT / name).write_bytes(buf.tobytes())
    print(name, len(buf) // 1024, "KB")


def main():
    OUT.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open("results/grade_500.csv")))
    rng = random.Random("fig")

    def pick(cond, n=12):
        s = [r for r in rows if cond(r)]
        s = rng.sample(s, min(n, len(s)))
        return sorted(s, key=lambda r: r["cat"])

    def tiles(rs):
        return [tile_for(r["cat"], int(r["qd_index"]), r["style"], f"{r['gid']} {r['cat']} {r['style']}") for r in rs]

    save("fig01_usable.jpg", tiles(pick(lambda r: r["usable"] == "1")),
         title="Usable (graded by eye). red/orange=back legs outer/inner, blue/cyan=front inner/outer, green=tail")
    save("fig02_head_wrong.jpg", tiles(pick(lambda r: r["skeleton_nonsense"] == "0" and r["head_side_ok"] == "0"
                                            and r["axis_ok"] == "1")),
         title="Head side guessed wrong (arrow at neck = rule's head side): front/back legs swapped")
    save("fig03_tail_wrong.jpg", tiles(pick(lambda r: r["skeleton_nonsense"] == "0" and r["tail_ok"] == "0"
                                            and r["head_side_ok"] == "1")),
         title="Tail wrong: tail used as a back leg, tail missed, or ear/trunk/neck taken as tail")
    save("fig04_nonsense.jpg", tiles(pick(lambda r: r["skeleton_nonsense"] == "1")),
         title="Skeleton nonsense although rule said 'found' (face/mane/hump/neck taken as legs etc.)")
    # 같은 그림 세 꼴
    t = []
    for r in pick(lambda r: r["usable"] == "1", 4):
        for st in qd.STYLES:
            t.append(tile_for(r["cat"], int(r["qd_index"]), st, f"{r['cat']} {st}"))
    save("fig05_styles.jpg", t, cols=3, title="Same drawing, three styles: line / thick / fill")
    # 몸 옆모습 표본
    lab = list(csv.DictReader(open("results/body_side_labels.csv")))
    t = []
    for cat in ("horse", "cat"):
        for r in [x for x in lab if x["cat"] == cat][:8]:
            img, _ = qd.render(drawing(cat, int(r["idx"])), "thick")
            ys, xs = np.where(img.min(2) < 250)
            tag = "body side" if r["body_side"] == "1" else "not"
            t.append(crop_tile(img, None, True, (xs.min(), ys.min(), xs.max(), ys.max()), f"{cat} {r['k']}: {tag}"))
    save("fig06_body_side.jpg", t, title="Full body side view? (eye labels, results/body_side_labels.csv)")
    # 흔들기
    t = []
    for r in pick(lambda r: r["skeleton_nonsense"] == "0", 3):
        d = (r["cat"], int(r["qd_index"]), r["style"])
        t.append(tile_for(*d, f"{r['cat']} base"))
        t.append(tile_for(*d, "flip", flip=True))
        t.append(tile_for(*d, "rot+5", rot=5))
        t.append(tile_for(*d, "x0.8", scale=0.8))
        t.append(tile_for(*d, "width", width=qd.WIDTH_ALT[r["style"]]))
    save("fig07_jitter.jpg", t, cols=5, title="Jitter: base / mirror / +5 deg / 0.8x / stroke width")


if __name__ == "__main__":
    main()
