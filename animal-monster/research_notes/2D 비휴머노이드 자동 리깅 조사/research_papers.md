# 2D 그림(아이 그림·낙서·만화) 비휴머노이드 캐릭터 자동 리깅 · 키포인트 · 부위 분할 연구 조사 (2023–2026, 2026-10-26 기준)

> 범위: 논문/데이터셋별 사실 + 코드·가중치 공개 여부 + 라이선스 + 우리 요구사항(사진 1장, 완전 자동, 2D 그림체 유지, Python 서버 CPU 우선, 상업 이용, 저비용, 수십 초) 적합도.
> 라이선스는 가능하면 GitHub API(`api.github.com/repos/...`의 `license.spdx_id`, 2026-10-26 조회) 또는 LICENSE 파일로 확인했다. 확인 못 한 항목은 「미확인」으로 표시.
> 중요한 법적 원칙(일반론): GitHub 저장소에 LICENSE 가 없으면(API `None`) 기본적으로 저작권자가 모든 권리를 가진다 → 상업 이용 불가로 취급해야 한다.

## 1. Meta 의 AnimatedDrawings 이후 데이터셋·후속 연구 (ChildlikeSHAPES, Amateur Drawings, Smith 외 후속)

### Takeaway
ChildlikeSHAPES 는 Amateur Drawings 에서 뽑은 1.6만 장 아이 그림에 25개 부위(몸·얼굴) 픽셀 분할 라벨을 단 **데이터셋(CC-BY 4.0, 상업 가능)** 이고, 이를 학습한 SAM 기반 분할 모델(CharSegNet)의 **가중치 공개는 확인되지 않았다**. 사람 그림 중심이지만 논문은 "만화 동물로 적은 재학습으로 확장 가능"하다고 보고하므로, 우리가 직접 SAM/SAM2(Apache)에 이 데이터+동물 그림 소량으로 부위 분할기를 학습시키는 재료로는 쓸 만하다.

### Cited Findings
- **ChildlikeSHAPES 논문**: "ChildlikeSHAPES: Semantic Hierarchical Region Parsing for Animating Figure Drawings", Astitva Srivastava, Harrison Jesse Smith, Thu Nguyen-Phuoc, Yuting Ye (모두 Meta), arXiv:2504.08022, 2025-04-10 (학회 게재 여부는 미확인) — [arXiv](https://arxiv.org/html/2504.08022v1)
- 데이터: 16,075장 (학습 14,075 / 테스트 2,000), 출처는 Amateur Drawings Dataset(온라인 데모에 익명 제출된 아이 그림). "25 semantic part categories, organized broadly into body and facial features"(전체 목록은 부록 A.1) — [arXiv](https://arxiv.org/html/2504.08022v1)
- 모델: SAM 위에 3단계 계층(거친 분할 → 세밀 분할 → 얼굴 특징) 분할 모델. 응용으로 자동 얼굴 애니메이션, 리라이팅, 기존 AnimatedDrawings 개선 제시 — [arXiv](https://arxiv.org/html/2504.08022v1); [검색 요약](https://arxiv.org/pdf/2504.08022)
- 비휴머노이드/도메인 밖: "can effectively process figures from cave paintings to street signs, and can be adapted to new domains like cartoon animals with minimal retraining", Cartoon Dog 데이터셋에서 SOTA 보고 — [arXiv](https://arxiv.org/html/2504.08022v1)
- 공개 상태: 논문 v1 은 "upon acceptance" 공개 예정이라 썼으나, AnimatedDrawings README 에는 "ChildlikeSHAPES" 절이 있고 릴리스 페이지의 `datachunk_*` 조각을 합쳐 7z 로 푸는 방식(비밀번호 README 에 기재)으로 배포, 라이선스는 "ChildlikeSHAPES dataset is released under CC-BY 4.0" — [AnimatedDrawings README(로컬 clone, 커밋 b859684, 2025-09-03)](https://github.com/facebookresearch/AnimatedDrawings#childlikeshapes)
- 릴리스 v0.0.1(2025-02-17, @hjessmith)에 자산 144개. 조회 범위에서는 `ad_orig_img_fs_aa~aj`(각 1.95GB) 확인, **분할 모델 가중치는 보이지 않음** — [Releases](https://github.com/facebookresearch/AnimatedDrawings/releases)
- 저장소는 2025-09-03 보관(archived, 읽기 전용) — [Releases 페이지 요약](https://github.com/facebookresearch/AnimatedDrawings/releases)
- **Amateur Drawings Dataset**: 178,000장 이상, "user-accepted character bounding boxes, segmentation masks, and joint location annotations", 공개 데모를 통해 수집 — [arXiv 2303.12741](https://arxiv.org/abs/2303.12741)
- 라이선스: "Animated Drawings code, model weights, and Amateur Drawings dataset is released under the MIT license" — [README](https://github.com/facebookresearch/AnimatedDrawings#license)
- 원 논문: Smith, Zheng, Li, Jain, Hodgins, "A Method for Animating Children's Drawings of the Human Figure", ACM TOG 42(3), 2023 — [DOI](https://doi.org/10.1145/3592788)
- **후속 1**: "Animating Childlike Drawings with 2.5D Character Rigs", Harrison Jesse Smith, Nicky He, Yuting Ye, arXiv:2502.17866 (2025). 2.5D 캐릭터 모델과 주석(annotation) 연구·사용자 연구 포함. 초록 기준 비휴머노이드 언급 없음, 코드 공개 언급 없음 — [arXiv](https://arxiv.org/abs/2502.17866)
- 서베이(2025)는 AnimatedDrawings 를 "bipedal motion, reducing its generalizability to non-human objects or characters"로 평가 — [Sketch Animation: State-of-the-art Report, arXiv:2510.10218](https://arxiv.org/html/2510.10218)

### Inferences
- ChildlikeSHAPES 데이터는 CC-BY 4.0 이라 출처 표기만 하면 상업 서비스용 모델 학습에 쓸 수 있다(데이터 원천인 Amateur Drawings 는 MIT). 다만 25개 범주가 "body and facial features"로 사람 기준이라 꼬리·날개·여러 다리 라벨은 없을 가능성이 크다(범주 목록 직접 확인 필요).
- Amateur Drawings 의 주석은 "character" 박스·마스크·관절이며 관절은 사람 골격 기준(우리가 쓰는 15관절 모델의 학습원). 데모 사용자가 동물 그림을 올렸을 수는 있으나 비휴머노이드 라벨은 없다고 보는 것이 안전하다. 반대로 **박스+마스크(전경 분리)는 동물 그림에도 통하는 학습 재료**라, 우리 쪽 전경 마스크 단계에는 이미 쓰이고 있다.
- "적은 재학습으로 만화 동물에 적응"이라는 보고는 우리 계획(모양 뼈대 + 부위 이름 붙이기)에 대해 "SAM 계열 분할기에 동물 아이 그림 수백 장 라벨" 같은 경로가 현실적임을 시사한다(추론).

### Gaps
- ChildlikeSHAPES 25개 범주 정확한 목록: arXiv PDF 가 10MB 초과로 가져오지 못함. 로컬에서 데이터 압축을 풀어 확인 필요.
- CharSegNet 코드·가중치 공개 여부: 찾지 못함(미공개로 추정, 미확인).
- 2.5D Character Rigs 의 코드 공개, 자동화 정도: 초록 외 확인 못 함.
- Amateur Drawings 안에 동물 그림이 몇 % 섞였는지: 출처 없음.

## 2. 범주 무관 자세 추정(CAPE) — 그림에 통하나, 라이선스, 지원 이미지 없이(zero-shot) 쓸 수 있나

### Takeaway
POMNet·CapeFormer·PoseAnything·CapeX 는 코드가 Apache-2.0 이라 상업 이용 가능하지만, 모두 **"지원 이미지 + 그 위의 키포인트"(few-shot) 또는 텍스트 키포인트 설명**이 필요하고 사진(MP-100)으로 학습돼 아이 낙서 성능은 검증된 적이 없다. 텍스트만으로 되는 X-Pose 는 데이터(UniKPT)가 비상업 전용이라 제외 유지. 우리 용도로는 "동물 종류별 기준 그림 1장 + 키포인트"를 우리가 미리 만들어 두는 1-shot 방식이 가능은 하나, ViTPose++ 결과처럼 도메인 차이로 크게 기대하기 어렵다.

### Cited Findings
- **PoseAnything / "A Graph-Based Approach for Category-Agnostic Pose Estimation"** (Hirschorn & Avidan, ECCV 2024, arXiv:2311.17891): 키포인트를 그래프로 다뤄 대칭 깨기·구조 보존, MP-100 1-shot SOTA — [arXiv](https://arxiv.org/abs/2311.17891v2); [ECCV](https://eccv2024.ecva.net/virtual/2024/poster/1706)
- PoseAnything 저장소: "released under the Apache 2.0 license"(API: Apache-2.0), 1-shot/5-shot Tiny·Small 가중치를 Google Drive 로 공개(가중치 별도 라이선스 표기 없음), 입력은 **support image + query image**, MMPose 0.29 / mmcv-full 1.6.2 기반, HF Spaces 데모 있음 — [GitHub](https://github.com/orhir/PoseAnything)
- **POMNet** (Pose-for-Everything, ECCV 2022): 저장소 Apache-2.0 — [GitHub API 조회](https://github.com/luminxu/Pose-for-Everything)
- **CapeFormer** (CVPR 2023): 저장소 Apache-2.0 — [GitHub](https://github.com/flyinglynx/CapeFormer)
- **CapeX** ("Category-Agnostic Pose Estimation from Textual Point Explanation", ICLR 2025, arXiv:2406.00384): 지원 이미지 대신 키포인트의 텍스트 설명(그래프)으로 질의. 코드·데이터 공개, 저장소 Apache-2.0 — [arXiv](https://arxiv.org/pdf/2406.00384); [ICLR](https://iclr.cc/virtual/2025/poster/28113); [GitHub](https://github.com/matanr/capex)
- **X-Pose (UniPose)**: 텍스트 프롬프트 브랜치로 지원 이미지 없이 임의 범주 키포인트 추론 가능. 그러나 "UniKPT ... is only for non-commercial research purposes", 저장소 LICENSE 는 GitHub 상 `NOASSERTION`(커스텀) — [GitHub](https://github.com/IDEA-Research/X-Pose)
- 스케치 관련 최근 연구는 **사람** 한정: Sketch2PoseNet(arXiv:2510.26196)은 확산모델로 합성한 SKEP-120K(스케치–3D 사람 자세 12만 쌍)로 학습, 만화·아이 그림·막대 인간에서 더 정확하다고 보고 — [arXiv](https://arxiv.org/html/2510.26196v1)
- 부위 수준 포인팅: POP(POinting at Parts)는 MLLM 의 인스턴스 수준 포인팅 한계를 지적하고 few-shot·학습 없는 부위 포인팅을 제안 — [검색 결과 요약(출처 arXiv:2506.21317 추정)](https://arxiv.org/abs/2506.21317v1) (논문 본문 미확인)

### Inferences
- CAPE 계열은 "상업 OK 코드 + 지원 이미지 1장" 구조이므로, 우리가 "네발짐승 기준 그림", "새 기준 그림" 등을 직접 라벨해 두면 완전 자동 파이프라인 안에서 쓸 수는 있다(사용자 개입 없음). 다만 아이 그림은 같은 범주 안에서도 모양 편차가 커서, 지원 그림과 질의 그림의 대응이 사진보다 훨씬 약할 것이다(추론, 미검증).
- 가중치는 Google Drive 배포·라이선스 별도 표기 없음 → 저장소 Apache-2.0 을 따른다고 보는 것이 일반적이나, 학습 데이터 MP-100 원천 데이터셋 일부(예: 비상업 데이터셋)의 영향은 법적으로 회색지대(추론).
- 의존성(mmcv-full 1.6.2 / mmpose 0.29)은 AnimatedDrawings 가 쓰는 OpenMMLab 구버전과 비슷해 서버 통합은 가능하나, CPU 속도는 미확인.

### Gaps
- MetaPoint, ESCAPE, Edge-aware CAPE, OpenKD(open-vocabulary keypoint) 등은 이번 조사에서 저장소·라이선스를 확인하지 못함.
- CAPE 모델을 아이 그림/만화 동물에 평가한 논문은 찾지 못함(없다고 단정은 못 함).
- POP 논문 본문(그림 도메인 평가 여부) 미확인.

## 3. 스케치·만화 부위 분할 (SAM/SAM2/SAM3, 오픈 어휘 부위 분할) — 아이 그림에서 "다리/머리/꼬리" 찾기

### Takeaway
SAM2 는 Apache-2.0, SAM3 는 Meta 자체 "SAM License"(상업 허용·로열티 없음, 군사·무기 용도 금지 등 조건)로 둘 다 상업 서비스에 쓸 수 있다. SAM3 는 짧은 텍스트 개념 프롬프트로 분할하지만 848M 파라미터·CUDA 12.6 GPU 필요라 CPU 서버엔 무겁다. 아이 그림 부위 분할에 직접 검증된 공개 모델은 ChildlikeSHAPES(사람 부위)뿐이며, "꼬리·날개"까지 자동으로 이름 붙여 주는 공개·상업 가능 모델은 찾지 못했다.

### Cited Findings
- **SAM 3**: 텍스트·시각 프롬프트로 "exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase", 848M 파라미터, CUDA 12.6+ GPU·Python 3.12+·PyTorch 2.7+, 논문 arXiv:2511.16719, 2026-03-27 SAM 3.1 Object Multiplex 공개 — [GitHub](https://github.com/facebookresearch/sam3)
- SAM 3 라이선스 "SAM License": "non-exclusive, worldwide, non-transferable and royalty-free limited license"(상업 이용 가능), ITAR·군사·핵·첩보·총기 용도 금지, 역공학 금지, 재배포 시 라이선스 사본 동봉, 연구 발표 시 사용 표기, 사용자 수 제한 없음 — [LICENSE](https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE)
- **SAM 2**: 저장소 Apache-2.0 (GitHub API) — [GitHub](https://github.com/facebookresearch/sam2)
- **VLPart** (Meta, 오픈 어휘 부위 분할, ICCV 2023): 저장소 MIT, 2023-09 이후 갱신 없음·archived — [GitHub](https://github.com/facebookresearch/VLPart)
- **ChildlikeSHAPES / CharSegNet**: SAM 기반으로 아이 그림 25개 부위 분할, 만화 동물(Cartoon Dog)로 적은 재학습 전이 — [arXiv](https://arxiv.org/html/2504.08022v1)
- 서베이: SMPL 등 사람 신체 모델 의존 방법은 "limited in handling non-human or arbitrary forms" — [arXiv:2510.10218](https://arxiv.org/html/2510.10218)

### Inferences
- 현실적 조합(추론): (1) 모양 뼈대로 돌출부(가지)를 찾고 → (2) 각 가지 끝/덩어리 crop 을 VLM 또는 SAM3 텍스트 프롬프트("tail", "leg", "head")로 이름 붙이기. SAM3 는 사진 개념으로 학습돼 아이 그림의 "leg" 같은 부위 개념 성능은 미지수라 시험 필요.
- SAM3 를 쓰려면 GPU 서버가 필요하고, SAM License 는 Apache 와 다르므로(재배포 시 사본 동봉, 금지 용도) 팀 문서의 라이선스 표에 따로 적어야 한다.
- VLPart 는 MIT 이지만 사진 데이터(PACO, PartImageNet 등) 학습·관리 중단 상태라 낙서에서의 성능은 기대가 낮다(추론, 미검증).

### Gaps
- PartGLEE: `FoundationVision/PartGLEE` 경로가 404 → 실제 저장소·라이선스 미확인.
- "Segment Any Part", PartSLIP(3D 점군, `Colin97/PartSLIP` 404), SketchSeg 류 스케치 의미 분할의 라이선스·그림 성능 미확인.
- SAM3 를 아이 그림 부위에 평가한 자료 없음. CPU 추론 속도 자료 없음.

## 4. VLM(GPT/Gemini/Claude/Qwen-VL, Molmo, Florence-2)으로 그림 위 부위 가리키기 — 논문·벤치마크

### Takeaway
그림(아이 그림·만화)에서 VLM 포인팅 정확도를 평가한 논문·벤치마크는 찾지 못했다. 오픈 가중치 쪽에서는 Molmo-7B-D 가 Apache-2.0(단, "research and educational use" 권고 문구, 기반 Qwen2 는 Qwen 라이선스)으로 좌표 포인팅을 지원하지만 16GB급 GPU 가 필요하다. 결국 우리 그림 표본으로 직접 소규모 시험이 필요하다.

### Cited Findings
- **Molmo-7B-D-0924**: 가중치 "Apache 2.0", "intended for research and educational use" 안내, LLM 은 Qwen2-7B(Qwen License, 조건부 상업 허용), 비전은 CLIP ViT-L/14-336(MIT), PixMo(약 100만 이미지–텍스트) 학습 — [Hugging Face](https://huggingface.co/allenai/Molmo-7B-D-0924)
- Molmo 저장소 Apache-2.0 (GitHub API) — [GitHub](https://github.com/allenai/molmo)
- "Talking Points" (arXiv:2510.14583): VLM 이 객체·영역 수준 그라운딩에 머물러 픽셀 정밀 키포인트 이해가 부족하다고 지적, Point Descriptor + Point Localizer 제안 — [arXiv](https://arxiv.org/html/2510.14583v1)
- POP: 기존 MLLM 은 "instance-level pointing"에 머물러 부위 수준 포인팅이 약함 — [검색 요약](https://arxiv.org/abs/2506.21317v1) (본문 미확인)
- 위 검색에서 아이 그림 포인팅 벤치마크는 나오지 않음 — [검색 결과 근거: Talking Points 등](https://arxiv.org/pdf/2510.14583)

### Inferences
- 연구계도 "VLM 의 부위 수준·픽셀 정밀 포인팅은 약하다"고 보고하므로, VLM 에 좌표를 직접 찍게 하기보다 **우리가 모양 뼈대로 후보 점(가지 끝, 덩어리 중심)을 만들고 VLM 은 "이 번호 점이 머리/꼬리/다리인가" 분류만** 시키는 방식(set-of-marks 식)이 더 안정적일 것이다(추론; 팀의 기존 `experiments/joint-labeling/` 방향과 일치).
- 유료 API(GPT/Gemini/Claude)는 호출당 비용이 들어, 팀 규칙상 모델·횟수·금액을 먼저 승인받아야 한다.

### Gaps
- Florence-2 라이선스(MIT 로 알려져 있으나 이번에 확인 안 함), Qwen2.5-VL/Qwen3-VL 포인팅의 그림 성능, Molmo 후속(Molmo2 등) 라이선스: 미확인.
- 그림 도메인 포인팅 벤치마크: 찾지 못함.

## 5. 자동 2D 리깅·그림 애니메이션 연구 (비휴머노이드 지원·자동화·공개·라이선스·속도)

### Takeaway
"사진 1장 → 완전 자동 → 2D 그림체 유지 → 비휴머노이드 → 상업 라이선스"를 모두 만족하는 공개 연구는 **찾지 못했다**. 3D 리깅(UniRig·Make-It-Animatable, MIT)은 입력이 3D 메시이고, 영상 확산(Puppet-Master, Animate-X 등)은 무겁고 그림체가 변하며 일부는 라이선스가 없거나 비상업이다. 그림 전용 연구(DrawingSpinUp, From-Rigging-to-Waving, SketchAnim)는 사람 위주이거나 라이선스 없음/비상업이다. 우리 "모양 뼈대 + ARAP" 방식이 여전히 요구사항에 가장 맞는 축이다.

### Cited Findings
| 연구 | 입력 / 비휴머노이드 | 자동? | 코드·가중치 / 라이선스 | 계산 | 출처 |
|---|---|---|---|---|---|
| AnimatedDrawings (TOG 2023) | 사진 1장 / 사람만(이족) | 자동 | 공개 / MIT | CPU 가능 | [README](https://github.com/facebookresearch/AnimatedDrawings#license), [서베이](https://arxiv.org/html/2510.10218) |
| DrawingSpinUp (SIGGRAPH Asia 2024) | 그림 1장 → 3D 복원 → Mixamo 리깅 / 비휴머노이드 언급 없음 | Mixamo(외부 서비스) 사용 | 코드 공개, **LICENSE 없음**(API None) | RTX 4090(2080Ti 가능), 스타일화 학습 단계 포함 | [GitHub](https://github.com/LordLiang/DrawingSpinUp), [README](https://raw.githubusercontent.com/LordLiang/DrawingSpinUp/main/README.md) |
| From Rigging to Waving (TOG 2025) | 손그림 캐릭터 + 골격 애니 + 영상 확산(머리카락·치마 2차 동작) / 비휴머노이드 언급 없음 | 미확인 | 코드 공개, 웹페이지 CC BY-NC-SA 4.0(코드 라이선스 미확인) | 미확인 | [프로젝트](https://lordliang.github.io/From-Rigging-to-Waving/) |
| SketchAnim (CGF 2024, Rai·Gupta·Sharma) | 스케치 + 구동 영상 / 이족·**네발·무생물** 시연 | 수작업 대응 필요 여부 미확인 | GitHub 공개, 웹 CC BY-NC-SA 4.0(코드 라이선스 미확인) | "Real-time" | [프로젝트](https://graphics-research-group.github.io/SketchAnim/), [CGF](https://diglib.eg.org/items/244a6a84-d133-484a-83e4-85418e623419/full), [서베이](https://arxiv.org/html/2510.10218) |
| Monster Mash (SIGGRAPH Asia 2020) | 사용자가 그린 스트로크를 3D 로 부풀려 실시간 애니 / 임의 형태 | **사용자 조작 필요** | Apache-2.0 | 브라우저 실시간 | [GitHub](https://github.com/google/monster-mash), [서베이](https://arxiv.org/html/2510.10218) |
| Puppet-Master (Oxford VGG, arXiv:2408.04631) | 이미지 1장 + 드래그 궤적 / 물체·동물·사람 | 드래그 입력 필요 | 코드 공개, **LICENSE 없음**(API None); SVD 기반, 체크포인트 6.7GB + SAM ViT-H 2.56GB | 대형 GPU | [프로젝트](https://vgg-puppetmaster.github.io/), [GitHub](https://github.com/RuiningLi/puppet-master), [HF Space](https://huggingface.co/spaces/rayli/Puppet-Master/tree/main/ckpts) |
| DragAPart (2024) | 이미지 + 드래그(부위 단위 동작) | 드래그 필요 | **LICENSE 없음**(API None) | GPU | [GitHub](https://github.com/RuiningLi/DragAPart) |
| Animate-X (ICLR 2025) | 참조 이미지 + 춤 영상(DWPose) / "anthropomorphic" 포함, 다리 없는 생물 언급 없음 | 자동(동작 영상 필요) | Apache-2.0, SD2.1 기반 구성요소, 32프레임 768×512 | GPU | [GitHub](https://github.com/antgroup/animate-x) |
| AnimateZoo (arXiv:2404.04946) | 동물 영상 간 교차 종 애니메이션(영상 확산) | 자동 | 코드 공개 확인 못 함 | GPU | [arXiv](https://www.arxiv.org/abs/2404.04946) |
| UniRig (SIGGRAPH 2025) | **3D 메시** → 골격+스키닝, 비표준 토폴로지 | 자동 | MIT | GPU | [GitHub](https://github.com/VAST-AI-Research/UniRig) |
| Make-It-Animatable (CVPR 2025) | **3D 메시**(v2 는 메시만) / 사실상 휴머노이드(3DBiCar 로 머리 비율만 확장) | 자동 | MIT, HF 가중치 | GPU | [GitHub](https://github.com/jasongzy/Make-It-Animatable) |
| RigAnything (Adobe, TOG 2025, arXiv:2502.09615) | **3D 자산** → 루트부터 자기회귀로 관절 "성장" / 사람·네발·해양·곤충 등 | 자동 | 코드 공개 여부 미확인 | 미확인 | [arXiv](https://arxiv.org/html/2502.09615v1), [Adobe](https://research.adobe.com/publication/riganything-template-free-autoregressive-rigging-for-diverse-3d-assets) |
| AniDiffusion "How to Train Your Dragon" (arXiv:2503.15586) | 3–5장 예시 프레임으로 리그 추론, 다양한 토폴로지 | 캐릭터별 미세조정 | (팀 기확인: 라이선스 없음) | GPU | [arXiv](https://arxiv.org/abs/2503.15586) |

- 서베이(Sketch Animation STAR, 2025)가 꼽은 비휴머노이드 가능 그림 애니: MonsterMash, SketchAnim("biped, quadruped, and inanimate"), DrawingSpinUp(형상 일관 3D 변형). 텍스트→영상 계열(BreathingSketches, SketchAnimator, AnimateSketches, FlexClip)은 주로 벡터 스케치 대상 — [arXiv:2510.10218](https://arxiv.org/html/2510.10218)
- 서베이의 열린 과제: 계산 복잡도·확장성·사용 편의, 사람 모델(SMPL) 의존 방법의 비휴머노이드 한계 — [arXiv:2510.10218](https://arxiv.org/html/2510.10218)

### Inferences
- 3D 리깅 연구(UniRig·RigAnything)를 쓰려면 "그림 → 3D 메시"(DrawingSpinUp 식 복원 또는 image-to-3D)가 먼저 필요하고, 이는 GPU·수십 초~분 단위·그림체 변형 위험을 늘린다. 우리가 원하는 "2D 그대로" 와 거리가 있다(추론).
- 다만 **UniRig(MIT)** 는 우리 2D 그림을 두께 있는 메시로 부풀린 뒤(Monster Mash 식) 골격만 얻어 2D 로 다시 투영하는 "골격 추정기"로 쓸 여지가 있다. 2D 실루엣을 부풀린 메시에서 품질이 나올지는 미검증(추론).
- 영상 확산 계열은 결과가 "그림이 아닌 그림처럼 생긴 영상"이 되고(그림체 보존 약함), GPU 비용이 커서 요구사항 (c)(d)(f)(g)에 맞지 않는다. Puppet-Master·DragAPart·DrawingSpinUp 는 LICENSE 부재로 상업 사용 불가 취급.
- SketchAnim 은 네발·무생물까지 시연하고 실시간이라 방법론(영상에서 동작 추출 → 스케치에 전이)은 참고 가치가 크나, 웹 라이선스가 비상업이고 코드 라이선스 미확인이라 구현 아이디어 참고용.

### Gaps
- Sketch2Anim, Toon3D, Generative Rendering, MoReS, Anymate, SV4D, "Learning to rig 2D characters" 류: 이번 조사에서 확인 못 함.
- AnimateAnyone / UniAnimate / MimicMotion / ToonCrafter 라이선스: GitHub API 속도 제한으로 미확인. 이들이 비휴머노이드를 다룬다는 근거도 찾지 못함.
- RigAnything 코드 공개 여부, From-Rigging-to-Waving·SketchAnim 코드의 실제 LICENSE 파일: 미확인.
- 각 방법의 그림 1장당 처리 시간 수치: 대부분 문서에 없음.

## 6. 임의 골격용 동작 소스 (AnyTop, UniMate, Truebones Zoo, 절차적 걸음)

### Takeaway
AnyTop(코드 MIT, 사전학습 모델 제공)과 UniMate(SIGGRAPH Asia 2026, CC BY 4.0, 공개 예정)는 "처음 보는 골격"에도 동작을 생성하지만, 둘 다 학습 데이터가 **Truebones Zoo(유료 판매 상품)** 이며 AnyTop 은 "licensing clarification" 때문에 가공 데이터 배포를 보류했다. 생성된 동작을 상업 서비스에 쓸 때 Truebones 조건 확인이 선행돼야 하므로, 당장은 절차적 걸음·CMU mocap 재사용(팀 실험 진행 중)이 안전하다.

### Cited Findings
- **AnyTop** ("Character Animation Diffusion with Any Topology", arXiv:2502.17327, SIGGRAPH 2025): 골격 구조만 입력받아 동작 생성, 위상 정보를 attention 에 넣은 트랜스포머 디노이저, Truebones Zoo 로 학습 — [arXiv](https://arxiv.org/html/2502.17327v2)
- 학습 때 못 본 골격(고양이, 코모도왕도마뱀 등) zero-shot 추론 시연 — [프로젝트](https://anytop2025.github.io/Anytop-page/)
- 저장소 MIT(GitHub API: `Anytop2025/Anytop`, 2026-04 갱신), 사전학습 모델 다운로드 스크립트 제공, 이족·네발·비행·지네+뱀 별 모델과 통합 모델 존재 — [GitHub](https://github.com/Anytop2025/Anytop); [README](https://raw.githubusercontent.com/Anytop2025/Anytop/main/README.md)
- Truebones Zoo 는 "official Truebones website"(Gumroad 판매 페이지)에서 내려받아 직접 가공해야 하며, 가공 데이터는 "temporarily withheld due to licensing clarification" — [README](https://raw.githubusercontent.com/Anytop2025/Anytop/main/README.md); [Truebones Gumroad](https://truebones.gumroad.com/l/skZMC)
- **UniMate** ("One Unified Model to Animate Diverse Skeletons", Mou 외, SIGGRAPH Asia 2026, arXiv:2609.05415, 2026-09-04): 리깅된 3D 자산(정지 골격) + 텍스트 → 동작, 골격별 미세조정 없음, 이족·네발·조류·해양·곤충·뱀 등 미관측 7종에 일반화. UniML3D(Truebones + Mixamo + Objaverse-XL, 13,006 시퀀스). 학습 H100×8 하루, 추론 50 FPS. 저장소 `Friedrich-M/UniMate`, 라이선스 CC BY 4.0(논문 표기), 학습·데이터는 게재 후 공개 예정 — [arXiv](https://arxiv.org/html/2609.05415)

### Inferences
- UniMate 는 "텍스트로 동작 요청(예: '뱀이 기어간다')"이 되므로 동화 문장과 연결하기 좋아 보이나, (1) 아직 전부 공개되지 않았고 (2) Mixamo·Truebones 섞인 데이터라 생성물의 상업 이용 조건이 불분명하다(추론). 출력은 3D 관절 회전이라 우리 2D 골격으로 투영·리타깃 단계가 필요하다.
- AnyTop 은 골격 위상별 모델이 따로 있어 "처음 보는 괴물 골격"에는 통합 모델 품질이 관건이며 GPU 가 필요할 가능성이 높다(추론, 속도 수치 없음). 오프라인으로 대표 골격(네발·새·뱀)별 동작 몇 개를 미리 생성해 라이브러리로 저장하는 방식이면 서비스 시점 GPU 가 필요 없다(추론) — 단 Truebones 조건 확인 후.

### Gaps
- Truebones Zoo 의 실제 이용 약관(상업 게임·서비스에 동작 데이터/파생 생성물 사용 가능 여부): Gumroad 페이지 본문 미확인.
- AnyTop·UniMate 추론 속도(CPU 가능 여부), UniMate 저장소 실제 LICENSE 파일: 미확인.
- 절차적 걸음(procedural gait) 관련 2023–2026 그림 대상 논문: 이번 조사에서 찾지 못함.
