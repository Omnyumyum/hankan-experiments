"""Meta 사람 그림 찾기 모델(AnimatedDrawings `drawn_humanoid_detector`, MIT)의 상자를 mmdet 없이 torch 만으로 낸다.

    import find_box as fb
    det = fb.load()                      # 가중치 불러오기 (한 번)
    boxes = det(img_bgr)                 # [(x0, y0, x1, y1, 점수), ...] 점수 큰 순, 원본 좌표

모델: mmdet 2.x Mask R-CNN (ResNet-50 caffe + FPN), 클래스 하나(humanoid). 상자만 쓰므로 마스크 머리는 돌리지 않는다.
가중치: github.com/facebookresearch/AnimatedDrawings releases v0.0.1 `drawn_humanoid_detector.mar` 안 `latest.pth`(335MB, 옵티마이저 포함).
  `python find_box.py slim 그.pth 나갈.pt` 로 모델 값만 남기면 약 170MB.
mmdet 의 추론과 같은 계산을 따라 썼다: BGR 그대로 평균만 빼기, 짧은 변 800 · 긴 변 1333 에 맞춰 키우기(줄이기), 32 배수로 덧대기,
RPN(층마다 앞 1000개 · 층마다 NMS 0.7 · 1000개) → RoIAlign(aligned) → 상자 머리 → 점수 0.05 · NMS 0.5 · 100개.
AnimatedDrawings 는 사진 긴 변을 1000 으로 줄여 넣고 점수 0.5 이상 중 가장 높은 상자를 쓴다(`MIN_SCORE`, `detect_top`).
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms, nms, roi_align

WEIGHTS = Path(__file__).resolve().parent / "eval_real" / "data" / "models" / "drawn_humanoid_detector.pt"
MEAN = (103.53, 116.28, 123.675)  # BGR
MIN_SCORE = 0.5  # AnimatedDrawings torchserve 처리기의 threshold
PRE_SIDE = 1000  # image_to_annotations: 긴 변이 이보다 크면 줄여서 보냄


def _bn(sd, p, x):
    w, b, m, v = sd[p + ".weight"], sd[p + ".bias"], sd[p + ".running_mean"], sd[p + ".running_var"]
    s = w / torch.sqrt(v + 1e-5)
    return x * s[None, :, None, None] + (b - m * s)[None, :, None, None]


class Detector:
    def __init__(self, sd):
        self.sd = {k: v.float() for k, v in sd.items()}

    # ---- 뼈대: ResNet-50 caffe (줄이기를 1x1 conv 에서) ----
    def _block(self, p, x, stride, down):
        sd = self.sd
        y = F.relu(_bn(sd, p + ".bn1", F.conv2d(x, sd[p + ".conv1.weight"], stride=stride)))
        y = F.relu(_bn(sd, p + ".bn2", F.conv2d(y, sd[p + ".conv2.weight"], padding=1)))
        y = _bn(sd, p + ".bn3", F.conv2d(y, sd[p + ".conv3.weight"]))
        if down:
            x = _bn(sd, p + ".downsample.1", F.conv2d(x, sd[p + ".downsample.0.weight"], stride=stride))
        return F.relu(y + x)

    def _backbone(self, x):
        sd = self.sd
        x = F.relu(_bn(sd, "backbone.bn1", F.conv2d(x, sd["backbone.conv1.weight"], stride=2, padding=3)))
        x = F.max_pool2d(x, 3, 2, 1)
        outs = []
        for li, n in enumerate([3, 4, 6, 3]):
            for bi in range(n):
                x = self._block(f"backbone.layer{li + 1}.{bi}", x, (1 if li == 0 else 2) if bi == 0 else 1, bi == 0)
            outs.append(x)
        return outs

    def _fpn(self, c):
        sd = self.sd
        lat = [F.conv2d(c[i], sd[f"neck.lateral_convs.{i}.conv.weight"], sd[f"neck.lateral_convs.{i}.conv.bias"]) for i in range(4)]
        for i in range(3, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[2:], mode="nearest")
        outs = [F.conv2d(lat[i], sd[f"neck.fpn_convs.{i}.conv.weight"], sd[f"neck.fpn_convs.{i}.conv.bias"], padding=1) for i in range(4)]
        outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return outs

    # ---- RPN ----
    @staticmethod
    def _anchors(stride, h, w):
        ratios = torch.tensor([0.5, 1.0, 2.0])
        hr = torch.sqrt(ratios)
        wr = 1 / hr
        ws = stride * wr * 8
        hs = stride * hr * 8
        base = torch.stack([-0.5 * ws, -0.5 * hs, 0.5 * ws, 0.5 * hs], 1)
        sx = torch.arange(w, dtype=torch.float32) * stride
        sy = torch.arange(h, dtype=torch.float32) * stride
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        sh = torch.stack([xx.reshape(-1), yy.reshape(-1), xx.reshape(-1), yy.reshape(-1)], 1)
        return (sh[:, None, :] + base[None]).reshape(-1, 4)

    @staticmethod
    def _decode(rois, d, stds, shape):
        d = d * torch.tensor(stds)
        mr = abs(math.log(16 / 1000))
        dxy, dwh = d[:, :2], d[:, 2:].clamp(-mr, mr)
        pxy = (rois[:, :2] + rois[:, 2:]) * 0.5
        pwh = rois[:, 2:] - rois[:, :2]
        gxy = pxy + pwh * dxy
        gwh = pwh * dwh.exp()
        b = torch.cat([gxy - gwh * 0.5, gxy + gwh * 0.5], 1)
        h, w = shape
        b[:, 0::2] = b[:, 0::2].clamp(0, w)
        b[:, 1::2] = b[:, 1::2].clamp(0, h)
        return b

    def _rpn(self, feats, shape):
        sd = self.sd
        boxes, scores, lvl = [], [], []
        for i, f in enumerate(feats):
            t = F.relu(F.conv2d(f, sd["rpn_head.rpn_conv.weight"], sd["rpn_head.rpn_conv.bias"], padding=1))
            cls = F.conv2d(t, sd["rpn_head.rpn_cls.weight"], sd["rpn_head.rpn_cls.bias"])[0]
            reg = F.conv2d(t, sd["rpn_head.rpn_reg.weight"], sd["rpn_head.rpn_reg.bias"])[0]
            s = cls.permute(1, 2, 0).reshape(-1).sigmoid()
            r = reg.permute(1, 2, 0).reshape(-1, 4)
            a = self._anchors(4 * 2 ** i, f.shape[2], f.shape[3])
            if len(s) > 1000:
                s, k = s.topk(1000)
                r, a = r[k], a[k]
            boxes.append(self._decode(a, r, (1.0, 1.0, 1.0, 1.0), shape))
            scores.append(s)
            lvl.append(torch.full((len(s),), i))
        b, s, l = torch.cat(boxes), torch.cat(scores), torch.cat(lvl)
        ok = ((b[:, 2] - b[:, 0]) > 0) & ((b[:, 3] - b[:, 1]) > 0)
        b, s, l = b[ok], s[ok], l[ok]
        keep = batched_nms(b, s, l, 0.7)[:1000]
        return b[keep]

    def _roi(self, feats, rois):
        sd = self.sd
        scale = torch.sqrt((rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1]))
        lv = torch.floor(torch.log2(scale / 56 + 1e-6)).clamp(0, 3).long()
        out = torch.zeros(len(rois), 256, 7, 7)
        for i in range(4):
            k = torch.nonzero(lv == i).flatten()
            if len(k):
                r = torch.cat([torch.zeros(len(k), 1), rois[k]], 1)
                out[k] = roi_align(feats[i], r, (7, 7), 1 / (4 * 2 ** i), 0, aligned=True)
        x = out.flatten(1)
        x = F.relu(F.linear(x, sd["roi_head.bbox_head.shared_fcs.0.weight"], sd["roi_head.bbox_head.shared_fcs.0.bias"]))
        x = F.relu(F.linear(x, sd["roi_head.bbox_head.shared_fcs.1.weight"], sd["roi_head.bbox_head.shared_fcs.1.bias"]))
        cls = F.linear(x, sd["roi_head.bbox_head.fc_cls.weight"], sd["roi_head.bbox_head.fc_cls.bias"])
        reg = F.linear(x, sd["roi_head.bbox_head.fc_reg.weight"], sd["roi_head.bbox_head.fc_reg.bias"])
        return cls.softmax(1)[:, 0], reg

    @torch.no_grad()
    def __call__(self, img):
        """img: BGR uint8. 반환 [(x0, y0, x1, y1, 점수)] 점수 큰 순 (점수 0.05 이상 · 원본 좌표)."""
        h0, w0 = img.shape[:2]
        s = min(1333 / max(h0, w0), 800 / min(h0, w0))
        nw, nh = int(w0 * s + 0.5), int(h0 * s + 0.5)
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32) - np.array(MEAN, np.float32)
        ph, pw = -(-nh // 32) * 32, -(-nw // 32) * 32
        x = torch.zeros(1, 3, ph, pw)
        x[0, :, :nh, :nw] = torch.from_numpy(im).permute(2, 0, 1)
        feats = self._fpn(self._backbone(x))
        rois = self._rpn(feats, (nh, nw))
        sc, reg = self._roi(feats, rois)
        b = self._decode(rois, reg, (0.1, 0.1, 0.2, 0.2), (nh, nw))
        ok = sc > 0.05
        b, sc = b[ok], sc[ok]
        keep = nms(b, sc, 0.5)[:100]
        b, sc = b[keep], sc[keep]
        b = b / torch.tensor([nw / w0, nh / h0, nw / w0, nh / h0])
        return [(*map(float, bb), float(ss)) for bb, ss in zip(b, sc)]


def load(path=WEIGHTS):
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return Detector(ck.get("state_dict", ck))


def detect_top(det, img):
    """AnimatedDrawings 와 같이: 긴 변 1000 으로 줄여 넣고, 점수 MIN_SCORE 이상 중 가장 높은 상자(원본 좌표) · 점수 · 전체 목록."""
    h, w = img.shape[:2]
    s = 1.0
    if max(h, w) > PRE_SIDE:
        s = PRE_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)))
    boxes = [(x0 / s, y0 / s, x1 / s, y1 / s, sc) for x0, y0, x1, y1, sc in det(img)]
    good = [b for b in boxes if b[4] >= MIN_SCORE]
    return (good[0][:4], good[0][4], boxes) if good else (None, boxes[0][4] if boxes else 0.0, boxes)


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "slim":
    ck = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
    torch.save({k: v for k, v in ck["state_dict"].items() if not k.startswith("roi_head.mask_head")}, sys.argv[3])
