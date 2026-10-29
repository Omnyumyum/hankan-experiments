"""Meta AnimatedDrawings 의 「사람 찾기」(Mask R-CNN) · 「관절」(ResNet-50 SimpleBaseline) 공개 가중치를 mmdet · mmpose 없이 돌린다.

    from meta_pose import MetaPose
    mp = MetaPose("eval_real/data/hr/models")   # drawn_humanoid_detector/ · drawn_humanoid_pose_estimator/ (.mar 를 푼 곳)
    r = mp.run(img_bgr)   # dict: det(찾은 상자 점수들) · box · kp(17×3, 원본 좌표 x y 점수) · mask(잘라 낸 그림 마스크)

가중치: github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_{detector,pose_estimator}.mar (MIT 저장소의 공개 배포).
mmdet 2.x · mmpose 0.x 를 python 3.13 에 깔 수 없어서 같은 계산을 torch · torchvision 으로 다시 짰다(config.py 의 값 그대로):
  찾기  — caffe 식 ResNet-50 + FPN + RPN(앵커 32~512, 비 0.5 · 1 · 2) + RoIAlign(aligned) + 두 층 fc, 짧은 변 800 · 긴 변 1333 로 키움
  관절  — 찾은 상자로 자른 그림 전체를 1.25배 여유 · 192×256 으로, 좌우 뒤집어 한 번 더 돌려 평균(flip_test · shift_heatmap), 점수 = 열지도 최대값
  흐름  — AnimatedDrawings image_to_annotations.py 와 같음: 긴 변 1000 넘으면 줄임 → 점수 0.5 이상 상자 중 가장 높은 것 → 그 상자로 자름 → 관절 → 마스크
마스크 넓이 신호를 위해 AnimatedDrawings 의 segment(적응 문턱 · 닫기 · 가장자리 채우기 · 가장 큰 덩어리)도 같은 식으로 했다.
"""
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision.ops import batched_nms, roi_align

KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
            "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank"]
USED = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # AnimatedDrawings 뼈대가 쓰는 13 점 (눈 · 귀는 안 씀)
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


class CaffeBottleneck(Bottleneck):
    """mmdet style='caffe': 줄이기(stride)를 첫 1×1 에서 한다."""

    def __init__(self, inplanes, planes, stride=1, downsample=None, *args, **kw):
        super().__init__(inplanes, planes, stride, downsample, *args, **kw)
        self.conv1.stride = (stride, stride)
        self.conv2.stride = (1, 1)


def resnet50(caffe):
    m = ResNet(CaffeBottleneck if caffe else Bottleneck, [3, 4, 6, 3])
    m.fc = nn.Identity()
    return m


def backbone_feats(m, x):
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    c2 = m.layer1(x)
    c3 = m.layer2(c2)
    c4 = m.layer3(c3)
    c5 = m.layer4(c4)
    return [c2, c3, c4, c5]


class Detector(nn.Module):
    STRIDES = [4, 8, 16, 32, 64]
    STD = [0.1, 0.1, 0.2, 0.2]
    MAX_RATIO = abs(math.log(16 / 1000))

    def __init__(self):
        super().__init__()
        self.backbone = resnet50(caffe=True)
        self.lat = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in (256, 512, 1024, 2048)])
        self.fpn = nn.ModuleList([nn.Conv2d(256, 256, 3, padding=1) for _ in range(4)])
        self.rpn_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.rpn_cls = nn.Conv2d(256, 3, 1)
        self.rpn_reg = nn.Conv2d(256, 12, 1)
        self.fc6 = nn.Linear(256 * 49, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc_cls = nn.Linear(1024, 2)
        self.fc_reg = nn.Linear(1024, 4)

    def load(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        out = {}
        for k, v in sd.items():
            if k.startswith("backbone."):
                out[k] = v
            elif k.startswith("neck.lateral_convs."):
                i, rest = k.split(".")[2], k.split(".")[-1]
                out[f"lat.{i}.{rest}"] = v
            elif k.startswith("neck.fpn_convs."):
                i, rest = k.split(".")[2], k.split(".")[-1]
                out[f"fpn.{i}.{rest}"] = v
            elif k.startswith("rpn_head."):
                out[k[len("rpn_head."):]] = v
            elif k.startswith("roi_head.bbox_head."):
                k2 = k[len("roi_head.bbox_head."):].replace("shared_fcs.0", "fc6").replace("shared_fcs.1", "fc7")
                out[k2] = v
        missing, unexpected = self.load_state_dict(out, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        return self

    @staticmethod
    def anchors(stride, h, w):
        r = torch.tensor([0.5, 1.0, 2.0])
        hr = r.sqrt()
        wr = 1 / hr
        ws = stride * wr * 8
        hs = stride * hr * 8
        base = torch.stack([-0.5 * ws, -0.5 * hs, 0.5 * ws, 0.5 * hs], 1)  # 가운데 0 (mmdet center_offset=0)
        sx = torch.arange(w) * stride
        sy = torch.arange(h) * stride
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        sh = torch.stack([xx.reshape(-1), yy.reshape(-1), xx.reshape(-1), yy.reshape(-1)], 1).float()
        return (base[None] + sh[:, None]).reshape(-1, 4)

    @classmethod
    def decode(cls, boxes, d, std, shape):
        d = d * torch.tensor(std)
        dx, dy, dw, dh = d.unbind(1)
        dw = dw.clamp(-cls.MAX_RATIO, cls.MAX_RATIO)
        dh = dh.clamp(-cls.MAX_RATIO, cls.MAX_RATIO)
        px, py = (boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5
        pw, ph = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        gw, gh = pw * dw.exp(), ph * dh.exp()
        gx, gy = px + pw * dx, py + ph * dy
        h, w = shape
        return torch.stack([(gx - gw * 0.5).clamp(0, w), (gy - gh * 0.5).clamp(0, h),
                            (gx + gw * 0.5).clamp(0, w), (gy + gh * 0.5).clamp(0, h)], 1)

    @torch.no_grad()
    def forward(self, img):
        """img: BGR uint8. 돌려줌: [(x0, y0, x1, y1, 점수)] 원본 좌표, 점수 높은 순 (점수 0.05 이상, 최대 100)."""
        h0, w0 = img.shape[:2]
        s = min(1333 / max(h0, w0), 800 / min(h0, w0))
        nh, nw = int(h0 * s + 0.5), int(w0 * s + 0.5)
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        im -= np.array([103.53, 116.28, 123.675], np.float32)
        ph, pw = int(math.ceil(nh / 32) * 32), int(math.ceil(nw / 32) * 32)
        pad = np.zeros((ph, pw, 3), np.float32)
        pad[:nh, :nw] = im
        x = torch.from_numpy(pad).permute(2, 0, 1)[None]
        cs = backbone_feats(self.backbone, x)
        lat = [l(c) for l, c in zip(self.lat, cs)]
        for i in range(3, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[2:], mode="nearest")
        ps = [f(l) for f, l in zip(self.fpn, lat)]
        ps.append(F.max_pool2d(ps[-1], 1, stride=2))
        # RPN
        props, scores, lvls = [], [], []
        for li, (p, st) in enumerate(zip(ps, self.STRIDES)):
            t = F.relu(self.rpn_conv(p))
            sc = self.rpn_cls(t)[0].permute(1, 2, 0).reshape(-1).sigmoid()
            dl = self.rpn_reg(t)[0].permute(1, 2, 0).reshape(-1, 4)
            an = self.anchors(st, p.shape[2], p.shape[3])
            k = min(1000, sc.numel())
            sc, idx = sc.topk(k)
            bx = self.decode(an[idx], dl[idx], [1, 1, 1, 1], (nh, nw))
            ok = ((bx[:, 2] - bx[:, 0]) > 0) & ((bx[:, 3] - bx[:, 1]) > 0)
            props.append(bx[ok])
            scores.append(sc[ok])
            lvls.append(torch.full((int(ok.sum()),), li))
        props, scores, lvls = torch.cat(props), torch.cat(scores), torch.cat(lvls)
        keep = batched_nms(props, scores, lvls, 0.7)[:1000]
        rois = props[keep]
        # RoI 머리
        sz = ((rois[:, 2] - rois[:, 0]) * (rois[:, 3] - rois[:, 1])).sqrt()
        lv = torch.floor(torch.log2(sz / 56 + 1e-6)).clamp(0, 3).long()
        feats = torch.zeros(len(rois), 256, 7, 7)
        for li in range(4):
            m = lv == li
            if m.any():
                b = torch.cat([torch.zeros(int(m.sum()), 1), rois[m]], 1)
                feats[m] = roi_align(ps[li], b, 7, 1 / self.STRIDES[li], 0, aligned=True)
        t = F.relu(self.fc7(F.relu(self.fc6(feats.flatten(1)))))
        sc = F.softmax(self.fc_cls(t), 1)[:, 0]  # mmdet: 0 = humanoid, 1 = 배경
        bx = self.decode(rois, self.fc_reg(t), self.STD, (nh, nw)) / s
        ok = sc > 0.05
        bx, sc = bx[ok], sc[ok]
        keep = batched_nms(bx, sc, torch.zeros(len(sc), dtype=torch.long), 0.5)[:100]
        return [(*bx[i].tolist(), float(sc[i])) for i in keep]


class Pose(nn.Module):
    W, H = 192, 256

    def __init__(self):
        super().__init__()
        self.backbone = resnet50(caffe=False)
        layers = []
        cin = 2048
        for _ in range(3):
            layers += [nn.ConvTranspose2d(cin, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
            cin = 256
        self.deconv_layers = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(256, 17, 1)

    def load(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        out = {(k[len("keypoint_head."):] if k.startswith("keypoint_head.") else k): v for k, v in sd.items()}
        missing, unexpected = self.load_state_dict(out, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        return self

    def heat(self, x):
        return self.final_layer(self.deconv_layers(self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(
            self.backbone.layer1(self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))))))))

    @torch.no_grad()
    def forward(self, img):
        """img: 사람 상자로 자른 BGR. 돌려줌: 17×3 (x, y, 점수), img 좌표."""
        h, w = img.shape[:2]
        cx, cy = w * 0.5, h * 0.5
        bw, bh = float(w), float(h)
        ar = self.W / self.H
        if bw > ar * bh:
            bh = bw / ar
        else:
            bw = bh * ar
        sw, sh = bw * 1.25, bh * 1.25  # 원본에서 잘라 볼 크기 (mmpose padding 1.25)
        k = self.W / sw
        M = np.array([[k, 0, self.W * 0.5 - k * cx], [0, k, self.H * 0.5 - k * cy]], np.float32)
        inp = cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None]
        hm = self.heat(torch.cat([x, x.flip(3)]))
        a, b = hm[0], hm[1].flip(2)
        perm = list(range(17))
        for i, j in FLIP_PAIRS:
            perm[i], perm[j] = j, i
        b = b[perm]
        b[:, :, 1:] = b[:, :, :-1].clone()  # shift_heatmap
        hm = ((a + b) * 0.5).numpy()
        n, hh, ww = hm.shape
        flat = hm.reshape(n, -1)
        idx = flat.argmax(1)
        val = flat.max(1)
        px, py = (idx % ww).astype(np.float32), (idx // ww).astype(np.float32)
        for j in range(n):
            ix, iy = int(px[j]), int(py[j])
            if 1 < ix < ww - 1 and 1 < iy < hh - 1:
                px[j] += 0.25 * np.sign(hm[j, iy, ix + 1] - hm[j, iy, ix - 1])
                py[j] += 0.25 * np.sign(hm[j, iy + 1, ix] - hm[j, iy - 1, ix])
        # 열지도(48×64) → 잘라 본 칸(sw×sh) → 원본
        X = px * sw / ww + cx - sw * 0.5
        Y = py * sh / hh + cy - sh * 0.5
        return np.stack([X, Y, val], 1)


def segment(img):
    """AnimatedDrawings image_to_annotations.segment 와 같은 식의 그림 마스크 (상자로 자른 그림)."""
    g = np.min(img, axis=2)
    g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 115, 8)
    g = cv2.bitwise_not(g)
    k = np.ones((3, 3), np.uint8)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k, iterations=2)
    g = cv2.morphologyEx(g, cv2.MORPH_DILATE, k, iterations=2)
    h, w = g.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    mask[1:-1, 1:-1] = g
    ff = np.full(g.shape, 255, np.uint8)
    for x in range(0, w - 1, 10):
        cv2.floodFill(ff, mask, (x, 0), 0)
        cv2.floodFill(ff, mask, (x, h - 1), 0)
    for y in range(0, h - 1, 10):
        cv2.floodFill(ff, mask, (0, y), 0)
        cv2.floodFill(ff, mask, (w - 1, y), 0)
    ff[0, :] = ff[-1, :] = 0
    ff[:, 0] = ff[:, -1] = 0
    m = ff > 0
    n, lab = cv2.connectedComponents(m.astype(np.uint8))
    if n <= 1:
        return m
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    return ndimage.binary_fill_holes(lab == areas.argmax())


class MetaPose:
    def __init__(self, models):
        models = Path(models)
        torch.set_grad_enabled(False)
        self.det = Detector().load(models / "drawn_humanoid_detector" / "latest.pth").eval()
        self.pose = Pose().load(models / "drawn_humanoid_pose_estimator" / "best_AP_epoch_72.pth").eval()

    def run(self, img):
        if max(img.shape[:2]) > 1000:
            s = 1000 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s)
        dets = self.det(img)
        out = dict(det=[d[4] for d in dets[:5]], box=None, kp=None, mask=None, shape=img.shape[:2])
        found = [d for d in dets if d[4] >= 0.5]  # TorchServe 쪽 mmdet_handler 의 threshold
        if not found:
            return out
        x0, y0, x1, y1 = [int(round(v)) for v in found[0][:4]]
        x0, y0 = max(0, x0), max(0, y0)
        crop = img[y0:y1, x0:x1]
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            return out
        kp = self.pose(crop)
        out.update(box=(x0, y0, x1, y1), kp=kp, mask=segment(crop), n_found=len(found))
        return out
