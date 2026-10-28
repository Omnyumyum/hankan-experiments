"""기존 검사를 「Drawing 이 color_mask 를 쓴다면」으로 돌려 본다. rig_types.py 는 고치지 않고 실행할 때만 바꿔 끼운다.

    (creature-picker 폴더에서) ../venv/bin/python eval_real/color_mask_hook.py [--gate] check_wings.py [검사 인자 …]
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import color_mask as cm  # noqa: E402
import rig_types as rt  # noqa: E402

gate = "--gate" in sys.argv
if gate:
    sys.argv.remove("--gate")


def __init__(self, img):
    """rig_types.Drawing.__init__ 와 같고, 사진 정리 + 마스크 두 줄만 color_mask.crop_and_mask 로 바꿨다."""
    self.notes = []
    self.crop, self.mask, self.alpha = cm.crop_and_mask(img, self.notes, gate=gate)
    g = rt.cv2.cvtColor(self.crop, rt.cv2.COLOR_BGR2GRAY)
    self.ink = rt.cv2.adaptiveThreshold(g, 255, rt.cv2.ADAPTIVE_THRESH_GAUSSIAN_C, rt.cv2.THRESH_BINARY_INV, 41, 9)
    self.mask_q, self.alpha_q = rt.q.open_gaps(self.crop, self.mask, self.alpha, self.ink)
    self.skel = rt.Skel(self.mask)


rt.Drawing.__init__ = __init__
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(str(Path(script).resolve()), run_name="__main__")
