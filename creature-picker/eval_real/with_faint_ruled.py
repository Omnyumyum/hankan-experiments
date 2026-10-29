"""기존 검사 스크립트를 「사진 정리 = faint_ruled.clean_photo」로 바꿔 끼워 돌린다 (rig_types.py 는 고치지 않는다).

    ../venv/bin/python eval_real/with_faint_ruled.py check_wings.py
    ../venv/bin/python eval_real/with_faint_ruled.py eval_real/guess_eval.py --qd 폴더 --split all --out fr_new
합칠 때 Drawing.__init__ 의 clean_photo 한 줄을 faint_ruled.clean_photo 로 바꾸면 이 결과와 같아진다.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import faint_ruled as fr  # noqa: E402
import rig_types as rt  # noqa: E402

rt.clean_photo = fr.clean_photo  # Drawing 이 모듈 안에서 이름으로 찾아 부르므로 이것만 바꾸면 된다
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(str(Path(script).resolve()), run_name="__main__")
