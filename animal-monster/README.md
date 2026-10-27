# 동물 · 괴물 그림 움직이기 실험 (2026-10-12)

사람 모양이 아닌 그림(동물, 알아볼 수 없는 괴물)을 움직일 수 있는지 세 방법으로 견줘 본 실험.
서버 코드는 바꾸지 않았다. 결과 요약은 `drawtale-frontend/docs/worklog.md` 2026-10-12 항목.

| 파일 | 무엇 |
|---|---|
| `gen.py` | 시험 그림 11장을 만든다 (동물 5, 괴물 6). `drawings/` 의 `real_*` 2장은 Meta 저장소 예제(MIT) |
| `overlay_meta.py` | 팀원 AI 서버 `/internal/v1/analyze` 결과(`results/meta_json/`)를 그림에 겹친다 → `results/meta_sheet.png` |
| `animal_pose.py` | 동물 관절 모델 ViTPose++-B (APT-36K, AP-10K 17점, rtmlib · onnxruntime CPU) → `results/animal_b_sheet.png` |
| `shape_rig.py` | **모양에서 뼈대를 뽑아 움직이기** (관절 모델 없음). 아무 그림 파일 · 폴더나 넣을 수 있다 → `shape/` (모아 보기 `shape_sheet.png`). 정리본은 `results/shape_sheet.png`, `results/gif/` |
| `quadruped.py` | **네발 동물(옆모습) 전용 걷기** (2026-10-20). 마스크 · 메시 · ARAP 은 `shape_rig` 것, 다리 찾기 · 걸음만 따로. 네발이 아니면 `shape_rig` 로 넘긴다 → `quadruped/` (모아 보기 `sheet.png`) |
| `gen_quadruped.py` | 네발 경우의 수 시험 그림 11장 → `drawings_quad/` |
| `reports/2D 비휴머노이드 자동 리깅 조사.md` | (2026-10-26) 2D 비휴머노이드 자동 리깅 상용 · 오픈소스 · 연구 · 생성 영상 조사. 근거 노트는 `research_notes/`. 결론: 조건을 다 갖춘 기성 해법 없음 → 이 폴더 방식 유지. 이어서 만든 종류 고르기 시험판은 `../creature-picker/` |
| `arap2d.py` | 2D ARAP 변형. Meta `arap.py` 와 같은 방법을 float64 · 희소 분해로 다시 쓴 것 |

## 돌리는 법

```bash
py -3.10 -m venv venv
venv/Scripts/python -m pip install numpy opencv-python scikit-image scipy imageio   # shape_rig 만 돌릴 때
venv/Scripts/python -m pip install rtmlib onnxruntime pyyaml                          # animal_pose 도 돌릴 때
venv/Scripts/python gen.py
venv/Scripts/python shape_rig.py                              # drawings/ 전체 → shape/
venv/Scripts/python shape_rig.py monster_blob worm            # drawings/ 에서 이름으로
venv/Scripts/python shape_rig.py 내그림.jpg 그림폴더/ --out 내결과   # 아무 그림 파일 · 폴더
venv/Scripts/python quadruped.py 내그림.jpg 그림폴더/ --out 내결과    # 네발 동물이면 이쪽 (아니면 알아서 shape_rig)
# 동물 모델: models/vitpose-b-apt36k.onnx 를 받아 둔다
#   https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/apt36k/vitpose-b-apt36k.onnx
venv/Scripts/python animal_pose.py b
```

### 내 그림으로 시험하기 (2026-10-20)

- 필요한 것은 이 폴더(`shape_rig.py` · `arap2d.py`)와 위 첫 줄 패키지뿐. AnimatedDrawings 저장소 · 이 PC 경로는 필요 없다
- 받는 것: png · jpg · jpeg · webp · bmp. 한글 경로, 투명 배경 PNG(흰 종이로 본다), 흑백, 휴대폰 사진(긴 변 1600px 로 줄인다)
- 결과: 그림마다 `이름.gif`(움직임) · `이름_rig.png`(잡은 몸 윤곽 · 팔다리 · 기준점), 모아 보기 `shape_sheet.png`
- 한 장이 안 돼도 나머지는 돌리고 끝에 실패 이유를 모아 보여 준다 (빈 종이 · 깨진 파일)
- 그림이 사진 가장자리에 닿으면(종이 가득 그렸거나 바짝 자른 사진) 결과 줄에 `edge!` 가 붙는다. 이때는 곧은 선(종이 가장자리)만 빼고 나머지를 그림으로 본다.
  전에는 닿은 그림을 통째로 지워 눈 점 하나를 움직이는 엉터리 결과가 나왔다
- 확인한 것: 기존 17장 결과는 고치기 전과 똑같다(GIF 바이트까지). 바짝 자른 강아지 · 문어, 종이 선 있는 사진, 그라데이션 그림자 jpg(3024×4032), 투명 PNG, 흑백, 한글 폴더 모두 움직임.
  **실제 아이 그림 사진은 아직 한 장도 안 돌려 봤다** — 색칠을 꽉 채운 그림, 연한 연필, 한 장에 여러 개, 줄 노트는 모른다

## quadruped 흐름 (2026-10-20)

그림마다 코드를 고치지 않으려고, 뼈대 선 가지 모양 대신 **옆모습 네발 동물의 공통 구조**(배 아래로 다리가 내려와 땅에 닿는다)에 기댄다.

1. 마스크는 `shape_rig.crop_and_mask`. 이어서 `open_gaps`: 틈 메우기가 메운 가까운 다리 사이 틈(종이 색 + 바깥 종이와 이어짐)을 되돌린다
2. `find_legs`: 아래에서 위로 한 줄씩 잘라 보다 몸통 폭 35% 넘는 덩어리가 나오면 몸통 밑, 그 위 몇 줄의 가장 넓은 구간 = 몸통 폭.
   그 아래 줄마다 다리를 센다 — 몸통 폭 아래에서 내려오고(코끼리 코 빠짐), 바닥까지 절반 넘게 내려오고(턱 · 귀 빠짐),
   눈 · 입이 있는 덩어리와 이어지지 않는다(숙인 머리 빠짐). 다리가 가장 많은 줄 중 가장 높은 줄 = 배. 다리마다 뿌리 · 무릎 · 발
3. 다리 2개 미만 · 7개 이상 · 너무 짧음(서 있는 그림) · 다리 3개 미만인데 몸이 긺(뱀) → `shape_rig` 로 넘김
4. `find_head_tail`: 몸통 가운데에서 몸 안으로 가장 먼 왼쪽 · 오른쪽 끝. 길은 몸 한가운데 능선을 따른다(최단 경로는 가장자리에 붙어 굵기를 못 잰다).
   머리 = 윤곽선과 떨어진 안쪽 선(눈 · 입)이 끝 가까이 있는 쪽(몸통 무늬는 가운데 가까이라 밀린다), 없으면 굵은 쪽.
   꼬리는 **있을 때만**: 끝에서 몸 폭 12% 들어가도 몸통 굵기의 35% 보다 가늘어야 꼬리 (둥근 엉덩이는 금방 굵어져서 꼬리 아님)
5. 다리 곡선 (`--gait`, 기본 `cmu`): CMU mocap 사람 걷기 2개(02_01 · 35_01)에서 뽑은 허벅지 · 무릎 곡선 50칸(`cmu_gait_curves.py`).
   사람 · 네발 짐승 모두 한 다리는 천천히 뒤로 밀고(약 60%) 빠르게 앞으로(약 40%), 무릎은 발 들 때만 잠깐 크게 접는다. 모양만 쓰고 크기는 상수로 맞춘다.
   예전 사인(`--gait sine`)은 앞뒤 대칭이고 주기 절반 동안 무릎이 접혀 긴 막대 다리가 엿가락처럼 휘었다. 비교 `results/gait_sine_vs_cmu.gif` — 긴 다리(말)는 확실히 나아지고 짧은 다리(강아지)는 거의 같다.
   데이터: CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), BVH 는 github.com/una-dinosauria/cmu-mocap
   `--bvh 파일` 이면 그 동작에서 곡선을 바로 뽑아 쓴다 (뛰기 → 속보 박자, 더 크게 · 빨리). 사람 그림과 같은 동작 파일을 함께 쓰는 구조는 `../cmu-motion/README.md`
6. 걸음: 뒷다리 · 앞다리로 나눠 1/4 박자씩(뒤 0, 앞 0.25, 반대 뒤 0.5, 반대 앞 0.75). 앞으로 나가는 다리는 무릎 접어 발 들기,
   몸통은 한 걸음에 두 번 오르내림, 머리 끄덕, 꼬리 흔들기. 각도는 파일 맨 위 상수

시험 (`drawings/` 17장 + `drawings_quad/` 11장 + 카톡 그림 2장 + 까다로운 입력 8장):
네발 그림은 다리 · 머리 방향 · 꼬리 유무가 모두 맞음 — 꼬리 없음(왼쪽 · 오른쪽 보기), 닥스훈트, 말, 다리 둘만 보임, 풀 뜯기, 코끼리 코, 늘어진 귀,
머리 큰 캐릭터, 윤곽선만, 색만, 기린 점무늬, 공룡, 강아지. 뱀 · 지렁이 · 물고기 · 덩어리 · 문어 · 서 있는 고양이는 `shape_rig` 로 넘어감.
모르는 것: 실제 아이 그림 사진(연필 · 크레파스), 앞모습 네발, 다리가 몸 앞뒤로 크게 겹친 그림, 앉거나 누운 자세. 붙여 그린 다리 한 쌍은 다리 하나로 걷는다

## shape_rig 흐름

1. 테두리에 닿는 긴 선(종이 가장자리 · 그림자)을 뺀 선으로 그림 영역을 자른다. 닿는 선이 안 닿는 선보다 많으면 그림이 닿은 것으로 보고 곧은 선만 뺀다
2. 마스크: 적응 임계값 → 틈 메우기(안이 채워질 때까지 5→17px 로 키움) → 바깥 flood fill. 몸에서 떨어진 조각도 다리를 놓아 붙인다
3. `skeletonize` 로 뼈대 선 → 몸 중심(가장 두꺼운 곳)에서 최단 경로 → 끝점마다 굵기가 확 커지는 곳을 뿌리로
4. 짧거나 길이에 비해 굵은 가지는 뺀다. 한 팔다리에서 갈라진 가지는 가장 긴 것 하나만
5. 팔다리마다 기준점 3개(뿌리 · 가운데 · 끝). 뿌리에서 아래로 향하면 다리(번갈아 까딱), 굵고 짧으면 머리처럼 살짝, 나머지는 흔들기. 팔다리가 없으면 말랑 · 통통
6. 격자 메시(6px) + ARAP → 삼각형마다 원본을 펴 붙여 GIF

### 긴 몸 갈래 (뱀 · 지렁이, 2026-10-19)

1~5 보다 먼저 `find_body_axis` 가 본다. 뼈대에서 가장 긴 길(지름)이 굵기 중앙값의 6배 이상(`elong`),
굵기가 고르고(95% / 중앙값 `w_ratio` ≤ 1.5), 그 길 둘레가 그림의 90% 이상을 덮으면(`cover`) 긴 몸이다.
- 축을 따라 기준점 7~14개(`make_body_pins`) → `pose_body`: 마디마다 방향을 사인 물결로 돌리고 물결은 머리 → 꼬리. 마디 길이 유지, 무게중심 제자리
- 머리는 끝 둘레의 「몸만 있으면 없을 선」(눈 · 입 · 혀)이 많은 끝. 테두리 몸은 가장자리 선, 한 줄 몸은 축 위 선을 빼고 센다. 비슷하면 굵은 끝
- 그림 `snake_s` · `snake_line` · `worm` 은 긴 몸, `snake_coil`(똬리)은 겹친 안쪽이 채워져 덩어리가 되어 팔다리 갈래로 간다(꼬리만 까딱)
- 기존 13장은 모두 팔다리 갈래 그대로 (가장 가까운 것: 고양이 `w_ratio` 1.75, 강아지 1.88). 고치기 전 결과 `results/snake_before_sheet.png`

## 알게 된 함정

- Meta `arap.py` 는 메시가 크면 기준점을 그대로 둬도 모양이 수십 px 어긋난다 (float32, `det()==0` 검사). 그래서 `arap2d.py`
- Delaunay 메시는 가는 선에서 꼭짓점 하나로만 붙은 경첩이 생기고, ARAP 이 그 부분을 마음대로 돌린다 → 격자 메시로 바꿈
- `python3` 은 이 PC 에서 Microsoft Store 바로가기라 멈춘다. `py` 나 venv 파이썬을 쓴다
