"""Meta analyze 결과(관절 15개)를 그림 위에 겹쳐 한 장으로 모은다."""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
BONES = [("hip", "torso"), ("torso", "neck"), ("torso", "right_shoulder"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_hand"), ("torso", "left_shoulder"), ("left_shoulder", "left_elbow"),
         ("left_elbow", "left_hand"), ("hip", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_foot"),
         ("hip", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_foot")]


def color(name):
    if name.startswith("right"):
        return (0, 0, 255)
    if name.startswith("left"):
        return (255, 0, 0)
    return (0, 160, 0)


tiles = []
rows = []
for jf in sorted((HERE / "meta").glob("*.json")):
    d = json.loads(jf.read_text(encoding="utf-8"))
    img = cv2.imread(str(HERE / "drawings" / f"{jf.stem}.png"))
    if d.get("success"):
        j = {p["name"]: p for p in d["joints"]}
        for a, b in BONES:
            cv2.line(img, (j[a]["x"], j[a]["y"]), (j[b]["x"], j[b]["y"]), color(b), 3)
        for p in d["joints"]:
            cv2.circle(img, (p["x"], p["y"]), 7, color(p["name"]), -1)
        scores = [p["score"] for p in d["joints"]]
        label = f"mean {np.mean(scores):.2f} min {min(scores):.2f}"
        rows.append((jf.stem, round(float(np.mean(scores)), 2), round(min(scores), 2), d.get("confidence")))
    else:
        label = d.get("message", "fail")[:40]
        rows.append((jf.stem, None, None, d.get("message")))
    h, w = img.shape[:2]
    s = 300 / max(h, w)
    t = cv2.resize(img, (int(w * s), int(h * s)))
    tile = np.full((330, 300, 3), 255, np.uint8)
    tile[30:30 + t.shape[0], :t.shape[1]] = t
    cv2.putText(tile, jf.stem, (4, 14), 0, 0.45, (0, 0, 0), 1)
    cv2.putText(tile, label, (4, 27), 0, 0.4, (0, 0, 180), 1)
    tiles.append(tile)
while len(tiles) % 5:
    tiles.append(np.full_like(tiles[0], 255))
cv2.imwrite(str(HERE / "meta_sheet.png"), np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]))
for r in rows:
    print(r)
