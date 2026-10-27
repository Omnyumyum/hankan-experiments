# 생성형 대안: 그림 한 장을 image-to-video / image-to-animation 모델로 움직이기 (2026-10 기준)

조사일: 2026-10-26. 가격은 모두 USD. 공식 문서를 우선했고, 집계 사이트(aggregator, 재판매처)에서만 확인된 수치는 "(집계 사이트, 미검증)"으로 표시했다.
우리 요구사항 약칭: (a) 사진 1장 (b) 완전 자동 (c) 아이 그림 그대로 보이기 (d) Python 서버 호출 (e) 상업 라이선스 (f) 저비용 (g) 수십 초 (h) 아동 안전.

## 1. 폐쇄형 API (Kling, Veo/Gemini, Runway, Pika, Luma, MiniMax Hailuo, Seedance, Sora 2, Wan API, Vidu) — 가격 · 속도 · 길이 · 정책

### Takeaway
2026-10 현재 초당 가격은 약 $0.015(MiniMax H3 Max Turbo, fal)부터 $0.40(Veo 3.1 표준)까지로 넓게 퍼져 있고, 720p 기준 주력 모델 대부분이 $0.05~0.15/초 구간이다. 하지만 **가장 큰 걸림돌은 가격이 아니라 약관이다**. Gemini API(Veo/Omni)는 18세 미만이 쓰거나 쓸 가능성이 큰 서비스에 쓰는 것을 금지하고, Sora 2 API는 2026-09-24에 종료됐다. Kling·Runway는 무료/일반 약관에서 업로드를 학습에 쓸 수 있다고 적어 두었다. 투명(alpha) 출력을 주는 폐쇄형 API는 찾지 못했다.

### Cited Findings
**가격 · 사양 (모델별)**
- **Kling**: fal 기준 "Kling Video" $0.07/초, "Kling Video v3 Image to Video [Pro]" $0.14/초 — [fal pricing](https://fal.ai/pricing). 집계 사이트들은 Kling 3.0 I2V $0.075/초(2026-03 스냅샷), Kling API 전체 $0.08~0.17/초(2026-05), 최소 과금 3초 / 최대 15초 모델 있음이라고 적었다 (집계 사이트, 미검증) — [costbench](https://costbench.com/software/ai-media-apis/kling-api/), [evolink](https://evolink.ai/blog/kling-3-api-pricing-integration-guide)
- Kling 4.0 API가 2026-10-15에 나왔고 $0.1134/초(Pro $0.1701), 클립 3~30초라는 보도가 있다 (집계 사이트, 미검증) — [aireiter](https://aireiter.com/blog/kling-3-api-pricing-guide-2026)
- Kling 공식 API는 선불 리소스 팩으로 판다: Starter 1,000 units ≈ $140(90일). 옛 V1.5/V2.1 Standard I2V 5초 = 2.0 units ≈ $0.28 (집계 사이트, 미검증) — [atlascloud](https://www.atlascloud.ai/blog/guides/kling-ai-api-pricing)
- **Google Veo 3.1 (Gemini API)**: 표준 $0.40/초(720p·1080p), $0.60/초(4K). Fast $0.10/$0.12/$0.30. Lite $0.05(720p)/$0.08(1080p). 무료 티어 없음. I2V도 같은 가격 (2차 출처) — [benchlm](https://benchlm.ai/media-pricing/veo), [the-decoder](https://the-decoder.com/google-veo-3-adds-1080p-916-video-and-drops-prices-by-half/)
- **주의**: Gemini API 공식 동영상 문서는 "Veo 3.1 Preview"가 **2026-10-22에 지원 종료·종료(shut down)**됐다고 하고, 지금은 Gemini Omni Flash를 영상 생성 모델로 안내한다. 정식(GA) Veo 3.1 모델이 남아 있는지는 문서에 없다 — [Gemini API video docs](https://ai.google.dev/gemini-api/docs/video)
- **Gemini Omni Flash** (`gemini-omni-1.1-flash`): 참조 이미지로 영상을 만들고(첫·끝 프레임 보간, 피사체 참조), 클립은 3~10초, 이어 붙여 최대 40초까지 된다. 360p/720p(기본)/1080p·4K(업스케일). **EEA·스위스·영국에서는 미성년자가 담긴 이미지를 올리거나 편집할 수 없다** — [Gemini Omni docs](https://ai.google.dev/gemini-api/docs/omni). 2026-07-01 출시, 1.1은 2026-08-27 preview — [eesel](https://www.eesel.ai/blog/gemini-omni-flash-pricing)
- Omni 1.1 Flash 가격: 360p ≈ $0.034/초, 720p ≈ $0.101/초, 1080p ≈ $0.152/초, 4K ≈ $0.304/초 (2차 출처, 2026-08-29) — [eesel](https://www.eesel.ai/blog/gemini-omni-1-1-flash-pricing)
- **Runway API**: 1 credit = $0.01. gen4.5 12 credits/초($0.12), gen4_turbo 5 credits/초($0.05), act_two 5 credits/초, veo3.1 오디오 없음 20 credits/초($0.20), veo3.1_fast 오디오 없음 10 credits/초($0.10) — [Runway API pricing](https://docs.dev.runwayml.com/guides/pricing/). Runway를 거치면 Veo 3.1을 아직 쓸 수 있다는 뜻이다.
- **Pika 2.2** (fal): 5초 720p $0.20, 1080p $0.45 (2026-07 확인, 집계 사이트) — [perkstack](https://perkstack.co/blog/cheapest-pika-2-2-api), [fal pika](https://fal.ai/models/fal-ai/pika/v2.2/text-to-video)
- **Luma**: Dream Machine/Ray2는 지원 종료(deprecated), 현재는 Ray3.2. Ray3.2 I2V 720p 5초 $0.30, 1080p 5초 $1.20 (집계 사이트, 미검증) — [wavespeed](https://wavespeed.ai/blog/cost-and-billing/luma-ai-pricing/), [costbench](https://www.costbench.com/software/ai-media-apis/luma-dream-machine-api/)
- **MiniMax Hailuo**: fal에서 H3 Max I2V $0.03/초, **H3 Max Turbo I2V $0.015/초**, H3 Max Reference-to-Video $0.05/초 — [fal pricing](https://fal.ai/pricing). platform.minimax.io의 MiniMax-H3 768P는 $0.08/초라는 인용이 있다 (미검증) — [segmind](https://www.segmind.com/models/minimax-h3-image-to-video/pricing). Hailuo 02 Standard 768p 6초 $0.27 — [fal llms.txt](https://fal.ai/models/fal-ai/minimax/hailuo-02/standard/image-to-video/llms.txt)
- **ByteDance Seedance**: fal Seedance 2 I2V $0.014/1,000 tokens — [fal pricing](https://fal.ai/pricing). BytePlus ModelArk 2.0 720p $7.00/백만 tokens, Fast 720p $5.60, 2.5 $10.70 (2차 출처) — [saascrmreview](https://saascrmreview.com/seedance-pricing/), [BytePlus docs](https://docs.byteplus.com/docs/ModelArk/1544106). Seedance 2.0은 4~15초, 최대 720p, 2026-04-16 기준 100개 넘는 나라의 기업 고객에게 열렸고 **미국은 빠졌다. 한국 포함 여부는 기사에 없다** — [the-decoder](https://the-decoder.com/bytedance-rolls-out-seedance-2-0-to-100-countries-but-keeps-the-us-off-the-list/)
- **OpenAI Sora 2 API**: $0.10/초(720p). 그러나 **2026-09-24에 종료돼 더는 쓸 수 없고, "1:1로 대체할 API가 없다"** — [OpenAI Sora 2 model page](https://developers.openai.com/api/docs/models/sora-2)
- **Alibaba Wan API**: Wan 2.6은 2025-12 Alibaba Cloud Bailian의 상용 API로만 나왔고 가중치는 공개되지 않았다 — [wan27.org](https://www.wan27.org/blog/wan-2-6-open-source-guide) (비공식 사이트). Atlas Cloud에서 Wan 2.6 T2V/I2V $0.07/초 (집계 사이트) — [atlascloud](https://www.atlascloud.ai/blog/wan-2-6-api-guide). fal에서 Wan 2.2 A14B는 $0.10/초라는 인용이 있다 — [atlascloud](https://www.atlascloud.ai/blog/cheapest-ai-video-generation-api-2026)
- **Vidu**: 1 credit = $0.005. Q3-pro I2V 1080P 30 cr/초($0.15), 720P 25 cr/초($0.125), 540P 10 cr/초($0.05). Q3-turbo 1080P 14 cr/초($0.07). **사용량이 적은 시간대(off-peak)에는 절반 값** (2차 출처). 공식 문서에는 Q4 Preview I2V도 보인다 — [cutout.pro](https://www.cutout.pro/learn/?p=2422), [Vidu platform docs](https://platform.vidu.com/docs/pricing)

**속도 (latency)**
- 2026 API 평균 응답: Veo 3.1 약 15~25초, Kling 3.0 약 15~30초, Wan 2.7 약 30~60초, Seedance 2.0 약 45초 이상 (집계 사이트, 측정 조건 불명, 미검증) — [atlascloud](https://www.atlascloud.ai/es/blog/case-studies/Top-5-ai-video-apis-compared-speed-latency-and-cost-per-second-2026)
- 리더보드 측정: Seedance 2.0 720p 35초, Veo 3 Fast 39.8초 (2026-04-06 갱신) — [magichour leaderboard](https://magichour.ai/model-leaderboard/image-to-video)

**업로드 학습 · 보관 · 미성년자 정책**
- Gemini API 추가 약관(2026-03-23 시행): "API를 쓰려면 18세 이상이어야 한다", "**18세 미만이 대상이거나 접속할 가능성이 큰** 웹사이트·앱·서비스의 일부로 쓰지 않는다". 유료는 프롬프트·응답을 제품 개선에 쓰지 않고, 정책 위반 탐지용으로 제한된 기간만 기록한다. 무료는 개선·개발에 쓴다 — [Gemini API Additional Terms](https://ai.google.dev/gemini-api/terms)
- Kling 약관(2026-04-21 갱신): 13세(또는 현지 최소 연령) 미만 사용 금지, 미성년자는 보호자를 통해서만 쓸 수 있다. 업로드 콘텐츠를 "홍보, 제품/기능 업그레이드, 신제품 연구"에 쓸 권리를 둔다. 문서에는 API/기업 사용자를 다르게 다룬다는 말이 없다 — [Kling user policy](https://kling.ai/docs/user-policy)
- Runway 사용 정책은 "Characters & Game Worlds" 항목에서 "18세 미만의 얼굴·목소리를 바탕으로 한 캐릭터"와 "18세 미만 사용자를 끌어들이려는 캐릭터나 게임(**아동 대상 서사 요소나 시각 스타일 포함**)"을 금지한다 — [Runway usage policy](https://runway.com/safety/usage-policy). 2차 분석에 따르면 Runway는 Enterprise가 아니면 입력·출력을 학습에 쓸 라이선스를 갖는다 — [terms.law](https://terms.law/ai-output-rights/runway/) (미검증)
- BytePlus Seedance: 실제 사람 얼굴이 담긴 이미지로는 I2V를 못 한다(hard block). 콘텐츠 사전 필터가 "아동·청소년의 사생활 침해, 심신 건강 위해" 내용을 거른다 — [BytePlus Content Pre-filter](https://docs.byteplus.com/ko/docs/ModelArk/Content_Pre-filter), [the-decoder](https://the-decoder.com/bytedance-rolls-out-seedance-2-0-to-100-countries-but-keeps-the-us-off-the-list/)

### Inferences
- 우리 서비스는 사용자가 아이이므로, **Gemini API(Veo·Omni)를 직접 쓰는 것은 약관상 막힌 것으로 봐야 한다**. Vertex AI(Google Cloud) 약관은 다를 수 있지만 이번 조사에서 확인하지 못했다. Runway를 거쳐 Veo를 부르는 길도 Runway의 "아동 대상 시각 스타일" 조항과 부딪칠 수 있다. 이 조항이 Characters 제품에만 해당하는지 일반 영상 생성에도 해당하는지는 해석이 필요하다.
- 가격 관점의 후보는 MiniMax H3 Max Turbo($0.015/초), Vidu Q3 540p($0.05/초, off-peak 절반), Runway gen4_turbo($0.05/초), Kling($0.07~0.14/초)이다. 다만 Kling·Runway는 업로드 학습 조항 때문에 **기업 계약(학습 제외)이 필요**하다.
- 응답 15~60초는 요구사항 (g) "수십 초"와 맞는다. 하지만 재생성(retry)까지 치면 한 장면에 1~2분 걸릴 수 있다.
- 폐쇄형 모델이 시판 중이어도 수명이 짧다. Sora 2는 출시 약 1년 만에 종료됐고 Veo 3.1 Preview도 종료됐다. 공급자 하나에 묶이면 위험하므로 공급자 추상화 계층(예: fal 같은 경유 서비스)을 두는 편이 낫다.

### Gaps
- 폐쇄형 API 가운데 **투명(alpha) 배경 출력**을 주는 곳은 찾지 못했다. 확인한 문서 어디에도 그런 옵션이 없었다.
- loop(끝 프레임 = 첫 프레임) 지원: Omni의 첫·끝 프레임 보간으로 같은 이미지를 양끝에 넣는 방식은 가능해 보이지만, 공식 loop 옵션은 확인하지 못했다. Kling·Vidu·Hailuo의 첫·끝 프레임 기능 역시 이번 조사에서 1차 출처를 확보하지 못했다.
- 한국 리전에서 쓸 수 있는지(Gemini API 한국 지원 여부, Seedance ModelArk 한국 포함 여부)는 1차 출처로 확인하지 못했다.
- Kling·MiniMax·Vidu 공식 페이지의 2026-10 현재 단가는 공식 페이지를 직접 읽지 못해 대부분 2차 출처에 기댔다.
- 각 API가 "흰 종이 위 크레용 선"을 얼마나 지키는지 비교한 정량 평가는 찾지 못했다.

## 2. 오픈 모델 — 라이선스 · GPU · 속도 · 비휴머노이드 대응

### Takeaway
상업적으로 쓸 수 있고 한국에서도 쓸 수 있는 오픈 I2V 기반은 **Wan 2.2(Apache 2.0)**와 **LTX-Video/LTX-2(매출 1천만 달러 미만 무료)**가 중심이다. HunyuanVideo-I2V는 **라이선스가 한국을 적용 지역에서 명시적으로 빼서** 쓸 수 없다. 포즈 기반 모델(AnimateAnyone 계열)은 사람 뼈대를 입력으로 받아 네발·뱀에는 맞지 않는다. LivePortrait는 얼굴 표정 전용이다(사람·고양이·개). 속도는 24GB 소비자 GPU에서 5초 720p 한 편에 수 분이 걸려 "수십 초"에 못 미친다. H100급 GPU와 증류(distilled) 모델이 있어야 수십 초 안에 든다.

### Cited Findings
- **Wan 2.2**: 모든 모델이 Apache 2.0. T2V-A14B, I2V-A14B(480P/720P, MoE), TI2V-5B(720P, 24fps), S2V-14B, Animate-14B. A14B는 GPU 하나로 돌리려면 최소 80GB VRAM(offload로 줄일 수 있음), TI2V-5B는 24GB(RTX 4090급). TI2V-5B는 "소비자 GPU 한 장에서 5초 720P를 9분 안에" 만든다 — [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- Wan 2.5·2.6은 가중치를 공개하지 않았다(API 전용). Wan 2.7(2026-03)은 일부만 Apache 2.0으로 열고 고급 기능은 API 전용이라는 비공식 정리가 있다 (미검증) — [wan27.org](https://wan27.org/blog/wan-3-open-source), [wan27.org 2.6](https://www.wan27.org/blog/wan-2-6-open-source-guide)
- **Wan-Alpha**: RGB와 alpha 채널을 함께 학습해 투명 배경 영상을 만든다. v1.0·v2.0 모두 Wan2.1-14B-T2V 기반 가중치와 추론 코드를 공개했다. HF에 I2V 변형 저장소(htdong/Wan-Alpha-I2V)도 있다 — [Wan-Alpha HF](https://huggingface.co/htdong/Wan-Alpha/blob/main/README.md), [Wan-Alpha-I2V](https://huggingface.co/htdong/Wan-Alpha-I2V), [논문 arXiv 2509.24979](https://arxiv.org/pdf/2509.24979)
- **HunyuanVideo-I2V**: Tencent Hunyuan Community License. 적용 지역은 "**유럽연합, 영국, 대한민국을 제외한** 전 세계". 출시일 기준 MAU 1억 명을 넘으면 Tencent에 별도 라이선스를 받아야 한다. 출력물에 대한 권리는 주장하지 않는다 — [HunyuanVideo-I2V LICENSE (HF)](https://huggingface.co/tencent/HunyuanVideo-I2V/blob/main/LICENSE)
- **LTX-Video**: README는 2025-03-05 업데이트에서 OpenRail-M으로 상업적 사용을 허용했다고 적었다. 최신 체크포인트는 13B distilled(v0.9.8)로, "H100에서 HD 영상 10초, 저해상도 미리보기 3초"를 주장한다. 2B distilled는 15배 빠르다. 기본 1216×704 30fps, 최대 60초. I2V를 지원한다 — [LTX-Video GitHub](https://github.com/Lightricks/LTX-Video)
- **LTX-2**(2025-10-23 발표, 오디오+비디오): LTX-2 Community License로, 연매출(모회사·계열사 합산) 1천만 달러 미만이면 상업 사용이 무료, 그 이상이면 유료 라이선스가 필요하다 — [LTX license](https://ltx.io/model/license), [scancode ltx-2-cla-2026](https://scancode-licensedb.aboutcode.org/ltx-2-cla-2026.html), [st-hakky 정리](https://book.st-hakky.com/en/data-science/ltx-2-5-commercial-license-guide)
- **Stable Video Diffusion**: Stability AI Community License로, 연매출 100만 달러 미만이면 상업 사용 무료(등록 필요). 그 이상은 Enterprise 계약 — [Stability membership](https://stability.ai/membership), [fast.io 리뷰](https://fast.io/resources/stability-ai-review-2026/). SVD에 바로 이 라이선스가 걸리는지는 2차 출처라 불명확하다.
- **ToonCrafter**: 코드는 Apache-2.0이지만 "상업 제품이 아닌 오픈소스 연구 탐색"이라는 면책 문구가 있다. 입력은 **카툰 키프레임 2장(시작·끝)**과 선택적 스케치 가이드. 512×320, 최대 16프레임, A100에서 약 24GB VRAM, 약 24초(50 DDIM steps) — [ToonCrafter GitHub](https://github.com/Doubiiu/ToonCrafter)
- **LivePortrait**: 사람·고양이·개의 **얼굴(초상) 표정**을 움직인다. Animals 모드는 X-Pose(MultiScaleDeformableAttention) 빌드가 필요하고 Linux/Windows NVIDIA에서만 테스트됐다 — [LivePortrait GitHub](https://github.com/KlingTeam/LivePortrait). 의존하는 InsightFace는 코드는 MIT지만 **학습 데이터와 학습된 모델은 비상업 연구용**이다. ComfyUI-LivePortraitKJ는 대안으로 MediaPipe(MIT/Apache)와 face-alignment 검출기를 넣었다 — [ComfyUI-LivePortraitKJ](https://github.com/kijai/ComfyUI-LivePortraitKJ), [insightface 미러 README](https://github.com/zhuyingSeu/insightface/blob/master/README.md)
- **Animate-X**(ICLR 2025): LDM 기반 "universal" 캐릭터 애니메이션. 의인화(anthropomorphic) 캐릭터용 Pose Indicator를 쓰고, A2Bench(의인화 캐릭터 500개 + 댄스 영상)를 만들었다. 기존 방법은 사람 그림에만 맞고 의인화 캐릭터에는 일반화가 잘 안 된다고 지적한다. 구동 입력은 여전히 사람 댄스 영상이다 — [Animate-X arXiv](https://arxiv.org/html/2410.10306v2)
- **Puppet-Master**: 이미지 1장과 몇 개 점의 "drag" 궤적을 주면 물체 **부위 단위 움직임** 영상을 만든다. 사전학습 I2V 모델을 Objaverse-Animation-HQ(합성 3D 애니메이션)로 미세조정했다. 코드와 체크포인트를 2024-08 공개 — [Puppet-Master arXiv](https://arxiv.org/abs/2408.04631v2), [GitHub](https://github.com/RuiningLi/puppet-master)

### Inferences
- 한국 서비스라서 HunyuanVideo 계열(I2V·Custom 등)은 **처음부터 후보에서 뺀다**.
- AnimateAnyone · UniAnimate · MimicMotion · Animate-X는 사람 포즈 시퀀스로 구동한다. 네발·뱀·벌레·괴물·사물에는 **구동 신호 자체가 없다**. 비휴머노이드에 쓸 만한 오픈 방식은 텍스트 프롬프트 I2V(Wan/LTX)나 궤적/drag 제어(Puppet-Master, DragAnything, MotionCtrl 계열)다.
- LivePortrait는 몸 동작(걷기·점프)을 만들지 못하고 얼굴만 움직인다. 아이 그림의 동물 얼굴은 X-Pose 고양이·개 키포인트 검출이 실패할 가능성이 크다(추정). 라이선스 면에서도 InsightFace를 빼야 한다.
- ToonCrafter는 키프레임 2장이 필요하므로 "사진 1장"만으로는 쓸 수 없다. 리깅으로 만든 시작·끝 포즈를 사이에서 보간하는 **하이브리드 보조 도구**로나 생각해 볼 만하고, 연구용 면책 문구도 걸린다.
- 자체 호스팅 비용(추정): TI2V-5B가 24GB GPU에서 5초에 약 9분이면, 시간당 $0.5~1 GPU를 써도 클립당 $0.1 안팎이다. 대신 대기시간이 요구사항 (g)를 크게 넘는다. LTX distilled + H100이 속도 면에서 유일하게 "수십 초"에 가깝다.

### Gaps
- CogVideoX, MimicMotion, UniAnimate, AnimateAnyone(공식 구현은 미공개 여부), Follow-Your-Click, DragAnything, MotionCtrl, Motion-I2V, Puppet-Master의 정확한 라이선스는 이번 조사에서 LICENSE 파일을 직접 확인하지 못했다.
- X-Pose(LivePortrait 동물 모드 의존성)의 상업 라이선스는 확인하지 못했다.
- Wan 2.2 I2V-A14B를 H100에서 돌릴 때의 클립당 시간, LTX-2 I2V가 아이 그림을 얼마나 지키는지는 공식 수치를 찾지 못했다.

## 3. 아이 그림에서 알려진 실패 유형과 평가 · 사례

### Takeaway
학술·실무 출처 모두 생성형 영상 모델은 **아이 그림 스타일을 3D/실사 쪽으로 끌고 가는 편향(3D bias, photorealism drift)**이 있다고 말한다. Meta 연구진은 아이 그림이 학습 데이터에 극히 드물어 "잘 되리라 기대하는 것은 무리"라고 보고 리깅(2.5D) 방식을 택했다. 아이 그림 + I2V를 정량 평가한 공개 자료는 찾지 못했다.

### Cited Findings
- Meta(Smith, He, Ye, 2025) "Animating Childlike Drawings with 2.5D Character Rigs": "생성형 AI 모델은 이미지 한 장에서 2D 애니메이션을 자동으로 만드는 매력적이고 실현 가능한 선택지가 됐지만, **움직임을 정밀하게 제어하거나 캐릭터 정체성을 지키는 것은 아직 풀리지 않은 과제**다." "초상 사진이나 전문 작품과 달리 아이 그림은 학습 데이터에 극히 드물다. 이런 모델이 아이 그림을 잘 움직이리라 기대하는 것은 무리다." Figure 10에서 멀티뷰 확산 모델의 "3D bias"가 아이 그림 스타일을 지키지 못함을 보였다 — [arXiv 2502.17866](https://arxiv.org/html/2502.17866v1)
- 실무 가이드: 스타일 드리프트(선 굵기·명암 밀도·채도의 프레임 간 변화)는 캐릭터 드리프트와 별개의 문제다. **평평한 셀 셰이딩 입력은 깨끗하게 움직이지만, 명암·질감이 많은 입력은 2~3초 무렵부터 실사 쪽으로 부드러워진다**. 대응책은 실사 금지 negative 블록, 고정 스타일 앵커, 스타일 참조 프레임 묶음 — [invideo FAQ](https://invideo.io/faq/how-do-you-stop-ai-from-drifting-into-photorealism-when/)
- 사례·앱: Higgsfield "Sketch-to-Video"는 Sora 2 기반으로 아이 손그림을 1080p 카툰으로 만든다고 홍보했다(Sora 2 API는 2026-09 종료) — [higgsfield](https://higgsfield.ai/blog/Sketch-to-Video-Cartoons). DrawMingo, Revive Drawn, Anijam 등이 "원래 선과 스타일 유지"를 내세운다(마케팅 문구, 검증 없음) — [futuretools DrawMingo](https://futuretools.io/tools/drawmingo), [neiroset Revive Drawn](https://ai.neiroset.com/en/tools/video/revive-drawn), [anijam](https://www.anijam.ai/tools/sketch-to-video-ai/)
- 부모용 블로그도 생성 결과가 "삐뚤한 다리, 정확한 보라색, 짝짝이 웃음"을 잃으면 다시 돌리라고 권한다. 생성형이 아이 그림 특유의 디테일을 자주 잃는다는 뜻이다 — [magicshot](https://magicshot.ai/blog/your-kids-drawing-brought-to-life)
- 2025-10 "Sketch Animation: State-of-the-art Report"(Rai, Sharma, 2026-02 개정)가 키프레임 보간·물리·데이터 기반·딥러닝 방식을 정리했다(본문은 읽지 못함) — [arXiv 2510.10218](https://arxiv.org/abs/2510.10218)

### Inferences
- 아이 그림에서 예상되는 실패 유형: (1) 크레용 질감이 매끈해지거나 3D 장난감처럼 변함 (2) 비대칭·과장 비율이 "정상" 해부학으로 바로잡힘 (3) 다리·팔이 생기거나 늘어남(특히 다리 없는 괴물·뱀) (4) 흰 종이 배경이 방·풀밭 같은 장면으로 바뀜 (5) 선 떨림(flicker). 4·5초 짧은 클립일수록 드리프트가 덜하다는 실무 관찰은 우리 3~5초 요구와 맞는다.
- 요구사항 (c) "내 그림이라고 알아보기"가 핵심이므로, 생성형은 결과 검사(첫 프레임과의 색·윤곽 유사도) + 재시도 + 실패 시 리깅 대체 경로가 꼭 필요하다(추정).

### Gaps
- 아이 그림 데이터셋(예: Amateur Drawings)으로 여러 I2V 모델을 비교 평가한 논문·블로그는 찾지 못했다.
- 비휴머노이드(뱀·벌레·괴물) 아이 그림에 I2V를 돌렸을 때의 성공률 보고는 찾지 못했다. 직접 소규모 시험이 필요하다(유료 API는 비용 승인 후).

## 4. 실무에서 보이는 하이브리드 아이디어

### Takeaway
생성형 모델을 "최종 렌더러"가 아니라 **움직임 공급원**이나 **배경 분리된 소재 생성기**로 쓰는 방법이 우리 요구(아이 그림 그대로 보이기)와 더 잘 맞는다. 투명 출력은 오픈 쪽 Wan-Alpha가 유일한 직접 해법이었고, 폐쇄형은 흰 배경으로 생성한 뒤 매팅(matting)해야 한다.

### Cited Findings
- 투명 배경 영상 생성: Wan-Alpha가 RGBA를 함께 만들어 "피사체가 이미 잘린 채로" 나온다. ComfyUI 워크플로가 있다 — [ComfyUI docs Wan-Alpha](https://docs.comfy.org/tutorials/video/wan/wan-alpha), [Wan-Alpha HF](https://huggingface.co/htdong/Wan-Alpha/blob/main/README.md)
- 궤적(drag) 제어로 부위 움직임: Puppet-Master — [arXiv](https://arxiv.org/abs/2408.04631v2)
- 키프레임 사이 생성 보간: ToonCrafter(시작·끝 2장 + 스케치 가이드) — [GitHub](https://github.com/Doubiiu/ToonCrafter)
- 스타일 유지는 리깅/2.5D 쪽이 낫다는 연구 판단 — [arXiv 2502.17866](https://arxiv.org/html/2502.17866v1)

### Inferences
- **안 A: 동작 라이브러리 미리 만들기.** 캐릭터마다 걷기·점프·손 흔들기 등 4~6개 동작을 한 번만 생성(흰 배경, 3~5초, loop용 첫=끝 프레임)하고, 매팅해 여러 이야기·쪽에서 다시 쓴다. 이야기마다 새로 생성하지 않아 비용과 대기시간이 준다.
- **안 B: 생성 → 마스크 → 원본 질감 다시 입히기.** 생성 영상에서 부위 움직임(optical flow, 윤곽)만 뽑아 원본 그림 조각에 변형으로 적용한다. 픽셀이 원본에서 오므로 크레용 질감이 지켜진다. 구현 난도가 높고, 공개된 실제 사례는 찾지 못했다.
- **안 C: 비휴머노이드에만 생성형.** 사람 모양은 지금의 AnimatedDrawings, 네발 등은 기존 모양 뼈대 실험, 분류가 안 되는 괴물·사물만 I2V로 보낸다.
- **안 D: 스프라이트 시트.** 생성 영상에서 8~12프레임을 골라 스프라이트로 저장하고 프런트에서 재생한다(용량·loop 처리가 쉬움).
- **안 E: 뼈대 영상으로 제어.** 우리 쪽 리깅이 만든 거친 움직임 영상을 video-to-video/참조 영상 입력(예: Kling·Seedance의 reference/video 입력, Wan Animate)으로 넣어 다듬는다. 다만 그만큼 스타일 드리프트 위험이 다시 생긴다.

### Gaps
- 안 B·E를 실제로 서비스에 쓴 공개 사례는 찾지 못했다.
- 폐쇄형 API 가운데 첫·끝 프레임 같은 loop 생성 품질 비교 자료는 없었다.

## 5. 이야기 1편 비용 대략 계산 (클립 4~8개 × 3~5초)

### Takeaway
이야기 1편은 생성 시간 12~40초 분량이다. 위 단가와 재시도 1.5배를 적용하면 **최저가 모델 약 $0.3~1, 중간 모델($0.05~0.10/초) 약 $1~6, 고급 모델($0.40/초) $7~24** 수준이다. 캐릭터당 동작 라이브러리를 한 번만 만들어 재사용하면 캐릭터당 약 $0.4~4로 고정된다.

### Cited Findings (계산에 쓴 단가)
- MiniMax H3 Max Turbo I2V $0.015/초, H3 Max I2V $0.03/초 — [fal pricing](https://fal.ai/pricing)
- Vidu Q3-pro 540P $0.05/초(off-peak 절반) — [cutout.pro](https://www.cutout.pro/learn/?p=2422)
- Runway gen4_turbo $0.05/초, gen4.5 $0.12/초 — [Runway API pricing](https://docs.dev.runwayml.com/guides/pricing/)
- Kling $0.07~0.14/초 — [fal pricing](https://fal.ai/pricing)
- Omni Flash 720p ≈ $0.10/초 — [eesel](https://www.eesel.ai/blog/gemini-omni-1-1-flash-pricing)
- Veo 3.1 표준 $0.40/초 — [benchlm](https://benchlm.ai/media-pricing/veo)
- Kling 계열은 최소 과금 3초 — [costbench](https://costbench.com/software/ai-media-apis/kling-api/) (미검증). Seedance 2.0 최소 4초 — [the-decoder](https://the-decoder.com/bytedance-rolls-out-seedance-2-0-to-100-countries-but-keeps-the-us-off-the-list/)

### Inferences (계산)
| 단가 | 12초(4×3초) | 40초(8×5초) | 재시도 ×1.5 적용 범위 |
|---|---|---|---|
| $0.015/초 (H3 Max Turbo) | $0.18 | $0.60 | $0.27~0.90 |
| $0.05/초 (Vidu 540p, gen4_turbo) | $0.60 | $2.00 | $0.90~3.00 |
| $0.10/초 (Omni 720p, Kling 하단) | $1.20 | $4.00 | $1.80~6.00 |
| $0.14/초 (Kling v3 Pro) | $1.68 | $5.60 | $2.52~8.40 |
| $0.40/초 (Veo 3.1 표준) | $4.80 | $16.00 | $7.20~24.00 |
- 최소 과금 시간(4~5초) 때문에 3초 클립이 5초 값으로 매겨질 수 있다. 그러면 12초 → 20초 분량으로 과금된다.
- 동작 라이브러리 방식: 캐릭터당 6동작 × 4초 = 24초 → $0.015/초면 $0.36, $0.10/초면 $2.4, 재시도를 넣으면 $0.5~3.6. 이야기가 늘어도 추가 비용이 없다.
- 비교: 리깅(AnimatedDrawings)은 자체 서버 CPU/GPU 비용만 들고 호출당 과금이 없다.
- 원화 환산은 환율에 따라 달라서 생략했다(예: 1달러 ≈ 1,400원이면 $1 ≈ 1,400원, 환율은 미검증).

### Gaps
- 실패율(재시도 배수)은 아이 그림 실측이 없어 1.5배로 가정했다.
- 공식 단가 대부분이 2차 출처라 실제 계약가·대량 할인은 다를 수 있다.

## 6. 아동 안전 · 개인정보 (COPPA, 한국 개인정보보호법, 공급자 약관)

### Takeaway
아이 그림을 제3자 API로 보내는 일은 **법보다 공급자 약관이 먼저 막는다**. Gemini API는 18세 미만 대상 서비스를 금지하고, Runway는 아동 대상 시각 스타일을 금지하는 조항을 두었다. Kling은 업로드를 연구·개선에 쓸 권리를 갖는다. 법적으로는 한국 개인정보보호법상 만 14세 미만 정보 처리에 법정대리인 동의가 필요하고, 국외 이전은 따로 알리고 동의를 받아야 한다. 미국 COPPA 개정(2026-04-22 준수 기한)은 AI 학습용 제3자 제공에 별도 부모 동의를 요구한다.

### Cited Findings
- Gemini API: 18세 이상만 사용, 18세 미만이 대상이거나 접속할 가능성이 큰 서비스에서 쓰지 못함. 유료는 학습에 쓰지 않음 — [Gemini API Additional Terms](https://ai.google.dev/gemini-api/terms)
- Gemini Omni: EEA·스위스·영국에서 미성년자가 담긴 이미지는 올리기·편집 불가 — [Gemini Omni docs](https://ai.google.dev/gemini-api/docs/omni)
- Runway: "아동 대상 서사 요소나 시각 스타일을 포함해 18세 미만 사용자를 끌어들이려는 캐릭터나 게임" 금지(Characters & Game Worlds 항목) — [Runway usage policy](https://runway.com/safety/usage-policy)
- Kling: 13세 미만 사용 금지, 미성년자는 보호자를 통해서만. 업로드를 홍보·업그레이드·연구에 쓸 권리 — [Kling user policy](https://kling.ai/docs/user-policy)
- BytePlus: 실제 얼굴 I2V 차단, 아동 사생활 침해 콘텐츠 사전 필터 — [BytePlus Content Pre-filter](https://docs.byteplus.com/ko/docs/ModelArk/Content_Pre-filter)
- 한국 개인정보보호법 제22조의2: 만 14세 미만 아동의 개인정보 처리에는 법정대리인 동의가 필요하다. 국외 수집·이전은 제3자 제공·처리 위탁과 따로 구분해 적어야 한다. 아동 대상 해외사업자도 법정대리인 동의 없이 아동 정보를 수집하지 않도록 해야 한다 — [국민연금공단 개인정보 처리방침 신구대조(2025-02)](https://m.nps.or.kr/html/download/policy/20250210_oldnew.pdf), [법무법인 세종(shinkim) 뉴스레터](https://shinkim.com/kor/media/newsletter/pdf/2452) (2차 출처, 조문은 [국가법령정보센터](https://www.law.go.kr/법령/개인정보보호법)에서 확인 필요)
- 미국 COPPA 개정: 2025-04-22 확정, 2025-06-23 시행, **2026-04-22까지 준수**. 제3자와 그 목적을 고지해야 하고, AI 학습 같은 서비스에 필수가 아닌 제3자 제공에는 **별도의 검증 가능한 부모 동의**가 필요하다 — [Loeb & Loeb](https://www.loeb.com/en/insights/publications/2025/05/childrens-online-privacy-in-2025-the-amended-coppa-rule), [Data Protection Report](https://www.dataprotectionreport.com/2025/06/ftcs-coppa-rule-changes-include-ai-training-consent-requirement/)

### Inferences
- 그림 자체(얼굴 사진이 아닌 선 그림)는 단독으로는 개인정보가 아닐 수 있다. 하지만 계정·이름·사진 메타데이터(EXIF 위치 등)와 묶이면 개인정보가 될 수 있다. 보내기 전에 **EXIF 제거, 그림 영역만 잘라내기, 이름 글씨 가리기**가 필요하다(추정, 법률 검토 필요).
- 공급자를 고를 때 가격보다 먼저 볼 것: (1) 아동 대상 서비스 허용 여부 (2) 업로드 학습 제외(zero data retention / 기업 계약) (3) 보관 기간 (4) 처리 국가. 이번 조사 범위에서는 이 넷을 모두 공식 문서로 통과한 폐쇄형 API를 찾지 못했다.
- 그래서 **아이 데이터가 밖으로 나가지 않는 자체 호스팅 오픈 모델(Wan 2.2 Apache 2.0, LTX 계열)**이 아동 안전 면에서는 가장 깔끔하다. 대가는 GPU 비용과 대기시간이다.
- 프롬프트를 이야기 문장에서 자동으로 만든다면, 생성 결과에 폭력·공포 장면이 나오지 않는지 사후 검사도 필요하다(기존 safety-filter 실험과 연결).

### Gaps
- Kling·MiniMax·Vidu·BytePlus의 **기업/API 계약에서 학습 제외와 보관 기간**이 어떻게 되는지는 공개 문서에서 확인하지 못했다.
- Vertex AI(Google Cloud)의 Veo가 Gemini API처럼 18세 미만 조항을 두는지 확인하지 못했다.
- 한국 개인정보보호법 제28조의8(국외 이전) 원문과 개인정보보호위원회 아동 가이드라인 원문은 직접 읽지 못했다. 법률 자문이 필요하다.
