"""점 프롬프트 분할 모델(SAM 계열)을 같은 모양으로 부르는 얇은 감싸개. 시험용 — 서비스 코드(rig_types.py)는 건드리지 않는다.

    m = load("mobile_sam")          # 모델 이름은 MODELS
    m.set_image(bgr)                # 그림 한 장 → 이미지 인코더 (느린 부분, 한 번만)
    masks, scores = m.predict(pts, box=None)   # pts: [(x, y), ...] 그림 좌표, 모두 「안쪽」 점. box: (x0, y0, x1, y1)
masks: (3, H, W) bool — 작은 것 → 큰 것 후보 3개(SAM 의 multimask), scores: 모델이 스스로 매긴 IoU 예상 3개.

코드는 공개 저장소를 그대로 git clone 해서 쓰고(설치하지 않음), 가중치는 공식 주소에서 받는다.
    eval_real/data/repos/{segment-anything, MobileSAM, sam2, EfficientSAM}
    eval_real/data/models/{sam_vit_b.pth, mobile_sam.pt, sam2.1_hiera_tiny.pt, sam2.1_hiera_small.pt}  (EfficientSAM 은 저장소 안 weights/)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
REPOS = DATA / "repos"
WEIGHTS = DATA / "models"
MODELS = ["sam_b", "mobile_sam", "sam2_t", "sam2_s", "effsam_t"]


class _SamV1:
    """segment-anything / MobileSAM (같은 SamPredictor)."""

    def __init__(self, name):
        import torch  # noqa: F401
        if name == "sam_b":
            sys.path.insert(0, str(REPOS / "segment-anything"))
            from segment_anything import SamPredictor, sam_model_registry
            net = sam_model_registry["vit_b"](checkpoint=str(WEIGHTS / "sam_vit_b.pth"))
        else:
            sys.path.insert(0, str(REPOS / "MobileSAM"))
            from mobile_sam import SamPredictor, sam_model_registry
            net = sam_model_registry["vit_t"](checkpoint=str(WEIGHTS / "mobile_sam.pt"))
        net.eval()
        self.p = SamPredictor(net)

    def set_image(self, bgr):
        import torch
        with torch.inference_mode():
            self.p.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def predict(self, pts, box=None):
        import torch
        with torch.inference_mode():
            m, s, _ = self.p.predict(point_coords=np.array(pts, np.float32), point_labels=np.ones(len(pts), np.int32),
                                     box=None if box is None else np.array(box, np.float32), multimask_output=True)
        return _order(m, s)


class _Sam2:
    def __init__(self, name):
        import torch  # noqa: F401
        sys.path.insert(0, str(REPOS / "sam2"))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        cfg, ck = {"sam2_t": ("sam2.1_hiera_t", "sam2.1_hiera_tiny.pt"), "sam2_s": ("sam2.1_hiera_s", "sam2.1_hiera_small.pt")}[name]
        net = build_sam2(f"configs/sam2.1/{cfg}.yaml", str(WEIGHTS / ck), device="cpu")
        self.p = SAM2ImagePredictor(net)

    def set_image(self, bgr):
        import torch
        with torch.inference_mode():
            self.p.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def predict(self, pts, box=None):
        import torch
        with torch.inference_mode():
            m, s, _ = self.p.predict(point_coords=np.array(pts, np.float32), point_labels=np.ones(len(pts), np.int32),
                                     box=None if box is None else np.array(box, np.float32), multimask_output=True)
        return _order(m > 0.5 if m.dtype != bool else m, s)


class _EffSam:
    """EfficientSAM-Ti. 상자는 점 두 개(왼쪽 위 = 2, 오른쪽 아래 = 3)로 넣는다."""

    def __init__(self, name):
        import torch
        sys.path.insert(0, str(REPOS / "EfficientSAM"))
        from efficient_sam.efficient_sam import build_efficient_sam
        self.net = build_efficient_sam(192, 3, checkpoint=str(REPOS / "EfficientSAM" / "weights" / "efficient_sam_vitt.pt")).eval()
        self.torch = torch

    def set_image(self, bgr):
        t = self.torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.hw = rgb.shape[:2]
        x = t.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255
        with t.inference_mode():
            self.emb = self.net.get_image_embeddings(x)

    def predict(self, pts, box=None):
        t = self.torch
        P = [list(map(float, p)) for p in pts]
        L = [1] * len(P)
        if box is not None:
            P += [[box[0], box[1]], [box[2], box[3]]]
            L += [2, 3]
        h, w = self.hw
        with t.inference_mode():
            lg, iou = self.net.predict_masks(self.emb, t.tensor([[P]]), t.tensor([[L]], dtype=t.float32), True, h, w, h, w)
        return _order(lg[0, 0].numpy() >= 0, iou[0, 0].numpy())


def _order(masks, scores):
    """후보 3개를 넓이 작은 것 → 큰 것 순으로."""
    masks = np.asarray(masks).astype(bool)
    o = np.argsort([m.sum() for m in masks], kind="stable")
    return masks[o], np.asarray(scores, np.float32)[o]


def load(name):
    if name in ("sam_b", "mobile_sam"):
        return _SamV1(name)
    if name in ("sam2_t", "sam2_s"):
        return _Sam2(name)
    if name == "effsam_t":
        return _EffSam(name)
    raise ValueError(name)
