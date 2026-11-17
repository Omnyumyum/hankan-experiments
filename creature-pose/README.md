# creature-pose

사람이 아닌 그림(네 발 동물부터)의 관절을 찾는 모델을 직접 학습해 보는 실험.
관절을 표시한 동물 낙서 데이터가 없어서, 서비스의 규칙(drawtale-ai `runtime/ai_server/quadruped.py`)이 찾은
관절을 정답 삼아 학습할 수 있는지부터 잰다.

## 기록

| 날짜 | 무엇 | 보고서 | 결론 |
|---|---|---|---|
| 2026-11-17 | 규칙 정답 품질 재기 (네 발, Quick, Draw! 17종 x 그림 꼴 3) | [REPORT_QUAD_PSEUDO.md](REPORT_QUAD_PSEUDO.md) | 학습해 볼 만함 **아님**: 거른 뒤 정밀도 48.8% (기준 90%). 머리 쪽 · 꼬리 · 뼈대 통째 틀림이 주원인 |

## 준비

```bash
cd creature-pose
uv venv -p 3.11 venv            # numpy 2 는 규칙 코드의 ndarray.ptp() 가 없어서 안 된다
uv pip install -p venv/bin/python "numpy<2" scipy "scikit-image<0.23" opencv-python-headless==4.10.0.84 requests pandas scikit-learn
git clone https://github.com/seominji58/drawtale-ai.git work/drawtale-ai && git -C work/drawtale-ai checkout 4c2147f
# Quick, Draw! 앞부분 (종류마다 처음 6MB, recognized 3,000장 넘게 들어 있음)
mkdir -p data/qd && for c in horse cow sheep pig dog cat elephant giraffe zebra lion tiger camel rhinoceros crocodile mouse raccoon squirrel; do
  curl -s -r 0-5999999 -o data/qd/$c.part https://storage.googleapis.com/quickdraw_dataset/full/simplified/$c.ndjson; done
```

## 파일

| 파일 | 무엇 |
|---|---|
| `qd.py` | Quick, Draw! 읽기 · 그림 꼴 세 가지(선만 · 굵은 선 · 채움)로 그리기 · 흔들기 좌표 변환 |
| `rule.py` | 서비스 규칙을 서버 없이 부르는 감싸기 (탐지 서버 = 「상자 없음」, `cut_out` → `find_joints`) |
| `check_fast.py` | 빠른 종이색 재기가 서비스 그대로와 같은 관절을 내는지 확인 |
| `features.py` · `pick_rule.py` | 몸 전체 옆모습 자동 고르기 (모양 숫자와 기준) |
| `sheet_body.py` | 몸 옆모습 눈 매기기용 표본 판 |
| `run_base.py` | 17종 x 3,000장 x 꼴 3 에 규칙 돌리기 |
| `run_jitter.py` | 흔들기(뒤집기 · ±5도 · 0.8/1.2배 · 선 굵기) 뒤 관절이 머무는지 |
| `make_grade_pick.py` · `sheet_grade.py` · `overlay.py` | 눈 매기기 500장 뽑기 · 뼈대 겹친 모아 보기 |
| `grades.py` · `auto_filter.py` | 매긴 결과 읽기 · 자동 거르기 기준 |
| `summarize.py` | 숫자 모으기 → `results/*.csv` |
| `figures.py` | 커밋용 모아 보기 그림 `figures/*.jpg` |
| `results/` | 눈 매기기 CSV, 숫자 CSV |
