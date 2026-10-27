"""2D ARAP (Igarashi & Igarashi 2009, 두 단계). Meta arap.py 와 같은 방법을 float64 · 희소 분해로 다시 쓴 것.

Meta 판은 float32 와 det()==0 검사 때문에 메시가 커지면 기준점을 그대로 둬도 모양이 어긋나고(수십 px),
느려진다. 기준점은 가장 가까운 꼭짓점에 건다.
"""
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class ARAP2D:
    def __init__(self, pins_xy, triangles, vertices, w=1000.0):
        v = np.asarray(vertices, np.float64)
        self.v = v
        n = len(v)
        self.w = w
        pins0 = np.asarray(pins_xy, np.float64)
        self.pin_vid = np.array([int(np.argmin(((v - p) ** 2).sum(1))) for p in pins0])
        self.pin_off = pins0 - v[self.pin_vid]  # 기준점과 꼭짓점의 처음 차이

        nbr = defaultdict(set)
        edges = set()
        for a, b, c in triangles:
            for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
                edges.add((min(x, y), max(x, y)))
                nbr[(min(x, y), max(x, y))].add(z)
        self.edges = sorted(edges)
        m = len(self.edges)

        rows1, cols1, vals1 = [], [], []
        grows, gcols, gvals = [], [], []
        self.e = np.zeros((m, 2))
        for k, (i, j) in enumerate(self.edges):
            ids = [i, j] + sorted(nbr[(i, j)])
            G = np.zeros((2 * len(ids), 4))
            for r, vid in enumerate(ids):
                x, y = v[vid]
                G[2 * r] = (x, y, 1, 0)
                G[2 * r + 1] = (y, -x, 0, 1)
            Gs = np.linalg.pinv(G)[:2]  # (c, s) = Gs @ [x0, y0, x1, y1, ...]
            ex, ey = v[j] - v[i]
            self.e[k] = (ex, ey)
            E = np.array([[ex, ey], [ey, -ex]])
            # 모서리 오차 = (v'_j - v'_i) - T_k e_k,  T_k e_k = E @ (c, s) = E @ Gs @ v'
            H = -E @ Gs
            H[0, 0] -= 1; H[1, 1] -= 1   # -v'_i
            H[0, 2] += 1; H[1, 3] += 1   # +v'_j
            for r in range(2):
                for cidx, vid in enumerate(ids):
                    for d in range(2):
                        val = H[r, 2 * cidx + d]
                        if val != 0:
                            rows1.append(2 * k + r); cols1.append(2 * vid + d); vals1.append(val)
            for r in range(2):
                for cidx, vid in enumerate(ids):
                    for d in range(2):
                        grows.append(2 * k + r); gcols.append(2 * vid + d); gvals.append(Gs[r, 2 * cidx + d])
        P = len(self.pin_vid)
        for q, vid in enumerate(self.pin_vid):
            rows1 += [2 * m + 2 * q, 2 * m + 2 * q + 1]; cols1 += [2 * vid, 2 * vid + 1]; vals1 += [w, w]
        A1 = sp.csr_matrix((vals1, (rows1, cols1)), shape=(2 * m + 2 * P, 2 * n))
        self.G = sp.csr_matrix((gvals, (grows, gcols)), shape=(2 * m, 2 * n))
        self.A1t = A1.T.tocsr()
        self.lu1 = splu((self.A1t @ A1).tocsc() + 1e-9 * sp.identity(2 * n, format="csc"))

        r2, c2, v2 = [], [], []
        for k, (i, j) in enumerate(self.edges):
            r2 += [k, k]; c2 += [i, j]; v2 += [-1.0, 1.0]
        for q, vid in enumerate(self.pin_vid):
            r2.append(m + q); c2.append(vid); v2.append(w)
        A2 = sp.csr_matrix((v2, (r2, c2)), shape=(m + P, n))
        self.A2t = A2.T.tocsr()
        self.lu2 = splu((self.A2t @ A2).tocsc() + 1e-9 * sp.identity(n, format="csc"))
        self.m = m

    def solve(self, pins_xy):
        p = np.asarray(pins_xy, np.float64) - self.pin_off
        b1 = np.concatenate([np.zeros(2 * self.m), self.w * p.reshape(-1)])
        v1 = self.lu1.solve(self.A1t @ b1)
        cs = (self.G @ v1).reshape(-1, 2)
        cs /= np.linalg.norm(cs, axis=1, keepdims=True) + 1e-12
        c, s = cs[:, 0], cs[:, 1]
        ex, ey = self.e[:, 0], self.e[:, 1]
        tx = c * ex + s * ey
        ty = -s * ex + c * ey
        bx = np.concatenate([tx, self.w * p[:, 0]])
        by = np.concatenate([ty, self.w * p[:, 1]])
        x = self.lu2.solve(self.A2t @ bx)
        y = self.lu2.solve(self.A2t @ by)
        return np.stack([x, y], 1)
