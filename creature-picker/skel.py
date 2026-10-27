"""뼈대 선 가지치기 (skan, BSD-3).

skeletonize 뼈대에는 윤곽선의 작은 울퉁불퉁마다 잔가지가 붙는다. 끝 → 갈림길 가지 중 짧은 것을
없어질 때까지 되풀이해 쳐 내고, 남은 가지의 끝을 「튀어나온 곳」(다리 끝 · 날개 끝 · 머리 · 꼬리 후보)으로 쓴다.
점을 미리 놓아 줄 때만 쓰고, 움직임 자체는 사용자가 확인한 점으로 만든다.
"""
import cv2
import numpy as np
from scipy.sparse.csgraph import dijkstra
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

PRUNE = 0.06  # 이보다 짧은 끝 가지는 잔가지 (그림 긴 변 비율)


class Skel:
    def __init__(self, mask):
        self.mask = mask
        self.size = max(mask.shape)
        self.dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        img = skeletonize(mask > 0)
        self.sk = None
        if img.sum() < 10:
            return
        sk = Skeleton(img)
        for _ in range(10):  # 쳐 내면 갈림길이 사라져 새 끝 가지가 생기므로 되풀이
            df = summarize(sk, separator="_")
            short = df.index[(df["branch_type"] == 1) & (df["branch_distance"] < PRUNE * self.size)].to_list()
            if not short or len(short) >= len(df):
                break
            sk = sk.prune_paths(short)
        self.sk = sk

    def tips(self):
        """튀어나온 곳: [{"tip": (x, y), "base": (x, y), "len": px}] 긴 것부터."""
        if self.sk is None:
            return []
        sk = self.sk
        df = summarize(sk, separator="_")
        deg = np.asarray(sk.degrees)
        out = []
        for i, r in df.iterrows():
            if r["branch_type"] not in (0, 1):
                continue
            a, b = int(r["node_id_src"]), int(r["node_id_dst"])
            if deg[a] != 1:
                a, b = b, a
            ty, tx = sk.coordinates[a]
            by, bx = sk.coordinates[b]
            out.append({"tip": (float(tx), float(ty)), "base": (float(bx), float(by)), "len": float(r["branch_distance"])})
        return sorted(out, key=lambda o: -o["len"])

    def diameter(self):
        """가지친 뼈대에서 가장 먼 두 끝 사이 길 (x, y) 배열. 뱀 · 지렁이 몸 축 후보."""
        if self.sk is None:
            return None
        sk = self.sk
        deg = np.asarray(sk.degrees)
        ends = np.nonzero(deg == 1)[0]
        if len(ends) < 2:
            return None
        d, pred = dijkstra(sk.graph, indices=ends, return_predecessors=True)
        sub = d[:, ends]
        sub[~np.isfinite(sub)] = -1
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        path, cur = [], ends[j]
        while cur >= 0 and cur != ends[i]:
            path.append(cur)
            cur = pred[i, cur]
        path.append(ends[i])
        yx = sk.coordinates[np.array(path)]
        return yx[:, ::-1].astype(np.float32)
