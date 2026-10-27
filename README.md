# hankan-experiments (비공개)

drawtale 비휴머노이드 그림 움직이기 실험. 팀 저장소와 별개로 개인 실험 기록 (2026-10-27 비공개 저장소로 올림).

| 폴더 | 무엇 |
|---|---|
| `animal-monster/` | 모양 뼈대 + ARAP 로 동물 · 괴물 그림 움직이기, 네발 걷기(`quadruped.py`), 2D 기성 해법 조사 보고서(`reports/`) |
| `creature-picker/` | 종류 고르기 시험판 (네 발 · 긴 몸 · 날개 · 몸통만), 날개 칸 고르기, 경우의 수 · 점 흔들기 자동 검사 |

각 폴더 `README.md` 부터. `creature-picker` 는 `../animal-monster` 를 불러 쓴다.

```bash
cd creature-picker
python -m venv venv && venv/Scripts/python -m pip install -r requirements.txt   # Linux: venv/bin/python
venv/Scripts/python check_wings.py
```
