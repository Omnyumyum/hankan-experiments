# 상용 2D 서비스 · 앱 — 비휴머노이드 그림 자동 리깅 · 애니메이션 조사 (2026-10-26 기준)

조사 범위: 아이가 종이에 그린 그림 사진 1장 → 사람 손 없이 자동으로 움직이기. 2D 그림 느낌 유지, 서버/웹에서 API·SDK 호출, 상업 이용 가능, 싸고 수십 초 안. 비휴머노이드(네발 동물, 뱀·지렁이, 새, 곤충, 정체불명 괴물, 별·구름 같은 물체).
이미 확인해서 깊이 안 본 것: Tripo, Anything World, UniRig, AniDiffusion, Spine, Cascadeur.

분류 약속:
- **[리깅형]** 그림에 뼈대/메시를 붙여 그림 픽셀을 변형 → 그림 느낌이 그대로 남음 (우리가 원하는 방식)
- **[생성 영상형]** 이미지→영상 생성 모델(Kling 등)로 새로 그려낸 영상 → 선·색이 바뀌거나 흔들릴 수 있음, 2D 그림 "변형"이 아님
- **[3D형]** 그림을 3D 모델로 바꿔 움직임 → 요구 (c) 위반
- **[수작업 도구]** 자동 리깅 없음, 사람이 뼈를 놓아야 함 → 요구 (b) 위반

---

## 1. 아이·아마추어 2D 그림을 자동으로 움직이는 상용 제품은 무엇이고, 비휴머노이드를 지원하는가?

### Takeaway
"그림 사진 1장 → 자동 → 2D 그림 그대로 변형 → 비휴머노이드도 됨"을 모두 만족하는 상용 제품은 찾지 못했다. 자동 + 비휴머노이드를 내세우는 소비자 앱(DrawToLife, Drawmingo, KidsAI, Drawings Alive, Animate my Drawing 등)은 사실상 전부 **생성 영상형**이고, 리깅형 자동 제품은 Meta Animated Drawings(사람 형태만) 하나뿐이며, 리깅형 전문 도구(Cartoon Animator, Live2D, Moho, Rive, Toon Boom, Adobe Character Animator)는 사람이 뼈·레이어를 준비해야 한다.

### Cited Findings

**A. 리깅형 · 자동 (사람 형태 한정)**
- Meta Animated Drawings: 2021년 공개된 아이 "사람 그림" 자동 애니메이션 방법, 부모가 그림을 올리면 춤추고·뛰고·점프하는 캐릭터가 됨 — [About Meta (2021-12)](https://about.fb.com/news/2021/12/using-ai-to-animate-childrens-drawings/)
- 2023-04-14 오픈소스 공개, 브라우저 데모(sketch.metademolab.com) 무료 체험 — [GIGAZINE](https://gigazine.net/gsc_news/en/20230414-meta-animated-drawings); [InfoQ](https://www.infoq.com/news/2023/04/meta-animated-drawings/)
- 2026-10 기준 리뷰 비교표에서 Animated Drawings 는 "Skeleton rigging / Figures only(사람 형태만) / 무료 / Web" 으로 분류 — [draw-to-life.com 블로그 (2026-08-08, 10-13 갱신)](https://draw-to-life.com/blog/best-apps-bring-kids-drawings-to-life/) (경쟁사 DrawToLife 가 쓴 글이라 자사 1위로 둔 편향 있음)
- 데모 사이트는 지금도 제목 "Animated Drawings | Meta FAIR" 로 응답하나 본문 확인은 못 함 — [sketch.metademolab.com](https://sketch.metademolab.com/) (운영 여부 미확인)
- Animated Drawings 를 감싼 자체 호스팅 앱 "Drawing Animator (Drawing Dance)": Starlette JSON API + SQLite + 한 번에 하나씩 렌더 큐 + 순수 JS 프런트 — [GitHub sampurchase/drawing-animator](https://github.com/sampurchase/drawing-animator) (오픈소스, 상용 서비스 아님, 역시 사람 형태 전제)

**B. 생성 영상형 · 자동 · 비휴머노이드 "됨" (아이 그림 전용 소비자 앱)**
- DrawToLife: 그림 1장 → AI 그림 6가지 화풍, 9개 언어 낭독 동화, 말하는 애니메이션 영상. 표에서 "Generative AI video / 비휴머노이드 ✓ / 연 $4.99–$79.99 / iOS·Android / API 언급 없음" — [draw-to-life.com 블로그](https://draw-to-life.com/blog/best-apps-bring-kids-drawings-to-life/); [draw-to-life.com/animate-drawings](https://draw-to-life.com/animate-drawings/)
- Drawings Alive: 그림 → 화풍 변환, 6초 애니메이션 영상, 3D 모델(.glb, .usdz) AR. 짧은 설명 문장을 함께 넣어 AI 를 안내 — [dataconomy](https://dataconomy.com/tools/drawings-alive/); [G2 AI marketplace](https://ai.g2.com/marketplace/tools/drawings-alive); 표에서 "Generative + 3D/AR, 연 $59–$199" — [draw-to-life.com 블로그](https://draw-to-life.com/blog/best-apps-bring-kids-drawings-to-life/)
- Drawmingo: "AI-powered animations", 갤러리에 "Dancing Mice", "Animal Parade" 등 동물 예시. 영상 1개 10크레딧, 2~5분 소요, 768p MP4. 방식(리깅/생성) 공개 안 함, API·상업 라이선스 언급 없음 — [drawmingo.com](https://drawmingo.com/) (768p MP4·수 분 소요라는 특성상 생성 영상형으로 추정 — 미확인)
- KidsAI: 아이 그림을 6초·10초 영상으로. "gentle, faithful motion", "The lines stay your child's; only the rendering is added by AI". 애니메이션 5~22크레딧, 유료 월 $3.99~. 모델명·API 공개 안 함 — [kidsai.art](https://kidsai.art/)
- Animate my Drawing: AI Cartoon (iPad, 개발자 Jeremy Hoffmann): 사진 찍고 화풍 선택(Original="preserves the original hand-drawn charm", Realistic, 3D Cartoon 등). 크레딧 $0.99/4개 ~ $7.99/50개 — [App Store](https://apps.apple.com/us/app/animate-my-drawing-ai-cartoon/id6760906020)
- Pixley: 2~8세용, 그림·사진 → 애니메이션 캐릭터가 개인화 영상 이야기·동화·학습 게임 주인공. 캐릭터 생성 "about a minute". 기술·가격·B2B 공개 없음 — [pixleyai.com](https://www.pixleyai.com/)
- Doodle Dreams: 애니메이션에 최적화된 "테마별 색칠 도안 팩"을 골라 색칠 → 애니메이션, 월 $3.75~ — [G2 AI marketplace](https://ai.g2.com/marketplace/tools/doodle-dreams) (자유 그림이 아니라 도안 기반)
- AniMagic: Animate Drawings: 여러 춤 스타일, 카메라·갤러리에서 직접 그린 캐릭터 올리기 — [App Store](https://apps.apple.com/us/app/animagic-animate-drawings/id6448135645) (방식·비휴머노이드 지원 미확인. 춤 위주라 Animated Drawings 계열 사람형일 가능성 — 추정)
- Kidify: 아이 첫 그림을 말하는 애니메이션 캐릭터로 — [App Store](https://apps.apple.com/app/id1446324384) (방식 미확인)
- SKETCH 앱: 표에서 "Generative 3D / 괴물만 / 크레딧 $0.99–$89.99" — [draw-to-life.com 블로그](https://draw-to-life.com/blog/best-apps-bring-kids-drawings-to-life/) (원 출처 앱 페이지 미확인)

**C. 도안(템플릿) 방식 · 설치형/AR — 자유 그림 아님**
- teamLab Sketch Aquarium (2013~): "Users color in a fish on paper provided" — 미리 인쇄된 물고기 도안에 색칠 → 스캔 → 가상 수족관에서 헤엄. 전시 설치 작품 — [teamLab](https://teamlab.art/w/aquarium/milanosalone2014); [The New Children's Museum](https://thinkplaycreate.org/blog/introducing-sketch-aquarium/)
- Quiver (QuiverVision): 색칠 팩을 받아 인쇄 → 색칠 → 앱으로 스캔하면 3D AR 로 움직임. 350+ AR 색칠 도안, 55+ 교육 팩 — [quivervision.com](https://quivervision.com/); [SoftwareAdvice](https://www.softwareadvice.com.au/software/563166/Quiver-3D-Coloring-App) (도안 = 미리 만든 3D 모델에 색만 입힘 → [3D형]+템플릿)
- Crayola Color Alive: 단종 여부를 확인할 출처를 찾지 못함 (Gaps)

**D. 리깅형 전문 도구 — 자동 아님 (수작업 도구)**
- Reallusion Cartoon Animator 5: "aligning bones along an image shape" 식 수작업 2D Bone Editor, 핀으로 영역 고정. 네발 동물용 "Quadruped Motion Library"(개·고양이·말 등) 와 사용자 정의 템플릿 — [Reallusion 2D Character Creation](https://www.reallusion.com/cartoon-animator/2D-character-creation.html)
  - 네발 동물 PSD 캐릭터 템플릿: 레이어 이미지를 자기 그림으로 바꾸면 리그가 자동 설정 — [검색 요약, Reallusion 페이지 기반](https://www.reallusion.com/cartoon-animator/2D-character-creation.html) → 단, 부위별로 나뉜 PSD 레이어를 사람이 만들어야 함
  - "AI 뼈 자동 바인딩(사람·물체·동물 인식 후 뼈 추가)"은 사용자 기능 요청 게시판에 올라온 **요청**일 뿐 구현된 기능 아님 — [Reallusion Feedback Tracker](https://www.reallusion.com/FeedBackTracker/Issue/AI-Integration-Ideas-and-requests)
  - Python/Lua 스크립팅은 사용자 요청 상태 — [Feedback Tracker: Extensions Plugins support](https://cdn-www.reallusion.com/FeedBackTracker/Issue/Extentions-Plugins-support)
- Live2D Cubism 5.1: AI "자동 디포머 생성"은 **휴머노이드 모델 전신용**, PSD 가져온 직후 디포머 구성 생성. "자동 흔들림 생성"은 장신구·머리카락용 — [Live2D 5.1 New Features](https://docs.live2d.com/en/cubism-editor-manual/new-function5-1/); [AUTOMATON (2024-07-30)](https://automaton-media.com/articles/newsjp/20240730-303793/)
  - PRO 버전 월 ¥2,280~, AI 기능은 PRO — [AUTOMATON](https://automaton-media.com/articles/newsjp/20240730-303793/); [Live2D 구매 비교](https://www.live2d.com/en/comparison/)
- Adobe Character Animator: Illustrator/Photoshop 레이어 아트를 웹캠·마이크로 실시간 조종하는 퍼펫 도구, 태그 기반 리깅(부위를 사람이 선택해 태그) — [Adobe Express page](https://express.adobe.com/page/B3M67/); [Toolfarm](https://www.toolfarm.com/?p=17984)
- Adobe Express "Animate characters (Animate from Audio)": 음성에 맞춰 립싱크하는 기능인데 **제공 캐릭터 중 선택만 가능, 자기 캐릭터 업로드 불가**가 가장 많이 요청된 기능 — [Adobe Community](https://community.adobe.com/questions-329/animate-a-character-using-adobe-express-1221026); [Adobe HelpX](https://helpx.adobe.com/express/using/how-to-animate-from-audio.html). 오디오 2분·1GB 제한 — [Adobe Community](https://community.adobe.com/t5/adobe-express-discussions/voice-to-animation-audio-limit/td-p/14673575)
  - 2025년 "character animation upload" 관련 버그 신고 글이 있어 업로드 기능이 시험 중일 가능성 — [Adobe Community bug report](https://community.adobe.com/bug-reports-328/internal-server-error-adobe-express-character-animation-upload-1469903) (본문 접속 실패, 내용 미확인)
- Moho Smart Bones: 뼈를 골라 액션을 만들어 두면 변형이 따라오는 방식, 수작업 셋업 — [Moho features](https://moho.lostmarble.com/pages/features)
- Rive: 정점·베지어 핸들을 뼈에 묶고 가중치를 직접 지정 — [Rive bone tips](https://rive.app/docs/editor/manipulating-shapes/bone-tips.md)
- Toon Boom Harmony 25: AI 기능은 이미지 확장·해상도 업·마스크 영역 제거 등 "pipeline quickening tasks". 단일 그림 자동 리깅 언급 없음 — [Toon Boom docs](https://docs.toonboom.com/help/harmony-25/premium/ai/ai-features.html). 별도 AI 도구 묶음 Ember(대본 분해·업스케일 등) — [검색 요약, Kidscreen](https://kidscreen.com/?p=223313)

**E. 3D형 · 기타**
- Krikey AI: 동물·생물·물체 등 비휴머노이드는 자사 **3D** Cartoon Character store 라이브러리로 제공, Illustrator 2D 아트를 "3D talking avatar"로 — [Krikey learn](https://www.krikey.ai/learn/how-to-create-animated-characters); [Krikey Illustrator integration](https://www.krikey.ai/integrations/turn-illustrator-2d-art-into-3d-talking-avatar)
- Viggle: 이미지 1장 + 동작(영상/템플릿) → 해부 구조를 분석해 그림을 동작에 맞게 휘게 함 — [Solan Sync](https://solansync.beehiiv.com/p/business-idea-animate-characters-free-latest-aitools). API 예시는 사람 캐릭터뿐, 동물·비휴머노이드 지원 명시 없음. 3D 캐릭터는 Gaussian splat(.vsplat), 동작은 GLB/FBX(Mixamo·MetaHuman 호환) — [Viggle developers](https://viggle.ai/developers)
- Canva: 표에서 "Design templates + motion" — [draw-to-life.com 블로그](https://draw-to-life.com/blog/best-apps-bring-kids-drawings-to-life/) (그림 부위를 움직이는 게 아니라 통째 모션 효과로 추정)
- FlipaClip: 수작업 프레임 애니메이션 앱 — [draw-to-life.com 블로그](https://draw-to-life.com/blog/best-apps-bring-kids-drawings-to-life/); 한국 App Store 에 "AI 애니메이션 메이커" 문구 — [App Store KR](https://apps.apple.com/kr/app/flipaclip-%EC%95%A0%EB%8B%88%EB%A9%94%EC%9D%B4%EC%85%98-%EB%A7%8C%EB%93%A4%EA%B8%B0-%EC%95%84%EB%B0%94%ED%83%80-ai-%EA%B7%B8%EB%A6%BC/id6746458207)

**F. 한국 · 중국 · 일본**
- 한국어 검색에서 나온 앱은 대부분 "사진 → 애니 화풍 변환" 앱(Anime AI 그림, AI 애니메이션 아트 생성기 등)이고, 아이 그림 비휴머노이드 자동 리깅 국내 서비스는 찾지 못함 — [App Store 검색 결과 예: AI 애니메이션 아트 생성](https://apps.apple.com/kr/app/ai-%EC%95%A0%EB%8B%88%EB%A9%94%EC%9D%B4%EC%85%98-%EC%95%84%ED%8A%B8-%EC%83%9D/id6624297688)
- 중국어 검색("儿童画 动起来")도 결국 Drawmingo, Doodle Dreams, Drawings Alive 등 영어권 서비스만 나옴 — [G2 Doodle Dreams](https://ai.g2.com/marketplace/tools/doodle-dreams); [futuretools Drawmingo](https://www.futuretools.io/tools/drawmingo)

### Inferences
- 2024~2026 "아이 그림 살리기" 앱 시장은 이미지→영상 생성 모델 위에 얹은 래퍼가 주류가 되었다. "비휴머노이드 됨"이라는 주장은 생성 모델이 뭐든 움직이게 그려 주기 때문이지, 부위를 인식해 리깅하는 것이 아니다.
- 리깅형으로 그림 픽셀을 그대로 보존하면서 자동인 상용 제품은 사람형(Animated Drawings 계열) 외에는 공백이다. 즉 비휴머노이드 리깅은 우리가 직접 만들어야 할 가능성이 높다(experiments/animal-monster 의 모양 뼈대 방식과 일치).
- teamLab·Quiver 의 "도안에 색칠" 방식은 비휴머노이드를 안정적으로 움직이는 가장 오래된 상용 해법이다. 동물 종류별 도안(미리 만든 리그)을 주고 아이가 그 위에 그리게 하면 자동·2D 유지가 가능하지만, "아이가 자유롭게 그린 그림"이라는 서비스 취지와 충돌한다(대안 UX 로만 참고).

### Gaps
- Crayola Color Alive, Disney Research 색칠 AR, DoodleLens 의 현재 운영 상태 — 신뢰할 출처를 못 찾음
- Animaker, Vyond, Plask, DeepMotion, CapCut·TikTok "그림 움직이기" 효과 — 이번 조사에서 1차 출처 확인 못 함. (일반 지식으로는 Animaker·Vyond 는 자체 템플릿 캐릭터, Plask·DeepMotion 은 3D 사람 mocap 이지만 **미확인**)
- Adobe After Effects / Photoshop Puppet Warp "자동 핀" 기능 존재 여부 — 출처 확인 못 함 (일반적으로 핀은 사람이 찍음, **미확인**)
- DrawToLife·Drawmingo·KidsAI·Pixley 가 어떤 영상 모델을 쓰는지 — 모두 비공개
- sketch.metademolab.com 데모가 현재 동작하는지 — 본문을 못 읽음

---

## 2. 각 후보의 API/SDK, 비용, 상업 임베드 라이선스, 속도

### Takeaway
서버에서 바로 호출할 수 있는 공개 API 가 확인된 것은 **Viggle API**(크레딧 1개 $0.01, 사람 중심)와 **범용 이미지→영상 API**(Kling 등, 5초 약 $0.56)뿐이며 둘 다 리깅형 2D 가 아니다. 아이 그림 전용 앱들은 API 를 공개하지 않는다. 리깅형 도구 중 SDK 가 있는 Live2D 는 "사용자가 모델을 들여오거나 만드는 앱" = Expandable Application 으로 분류되어 소규모 사업자도 별도 심사·계약이 필요하다.

### Cited Findings

**Viggle API**
- 자가 가입(self-serve) API, JST-2 기반: 재사용 캐릭터 자산, 영상 렌더, 3D 변환 — [Viggle developers](https://viggle.ai/developers)
- 신규 100 무료 크레딧, 1크레딧 = $0.01. Viggle-Animate $0.01/구동 영상 1초, Video Remix $0.01/출력 1초, 3D 캐릭터 $0.25/개, H3 $0.01/생성 1초(4K $0.05) — [Viggle developers](https://viggle.ai/developers)
- 비동기 작업, 폴링으로 완료 확인. 지연 시간 명시 없음 — [Viggle developers](https://viggle.ai/developers)
- 상업 이용: 입력에 대한 권리가 있으면 출력 상업 이용 가능, API 입출력을 학습에 쓰지 않음 — [Viggle developers](https://viggle.ai/developers)
- 소비자 Pro 플랜 월 $9.99 — [techjockey](https://www.techjockey.com/detail/viggle-ai)

**범용 이미지→영상 API (생성 영상형 기준점)**
- fal.ai 의 Kling 3.0 Pro: 오디오 끔 $0.112/초 → 5초 약 $0.56. Kling 2.5 Turbo Pro: 첫 5초 $0.35, 이후 $0.07/초 — [apiframe 블로그](https://apiframe.ai/blog/kling-3-0-api-providers); [fal.ai image-to-video](https://fal.ai/explore/image-to-video-apis) (2차 출처 요약, 가격은 수시 변동)
- 아이 그림 이미지 애니메이션 예제를 AIML API 문서가 제공(5초 영상 생성) — [AIML API docs](https://docs.aimlapi.com/use-cases/animate-images-a-childrens-encyclopedia)

**Live2D Cubism SDK**
- SDK 는 무료로 받아 개발 가능, 출시 시 SDK 출시 라이선스(Publication License Agreement) 계약·지불 필요. 개인·소규모 사업자는 면제(단, Expandable Application 제외) — [Live2D SDK license](https://www.live2d.com/en/?p=586)
- 소규모 기준 연 매출 1,000만 엔 미만 — [Live2D 검색 요약](https://www.live2d.com/en/?p=607)
- Expandable Application = "SDK 를 쓰는 서비스 중 확장성이 큰 것", 예: VTuber 트래킹 소프트웨어, **최종 사용자가 Live2D 모델을 직접 들여오거나 만드는 도구** → 모든 출판사(소규모 포함)가 사전 심사 + 특별 계약 — [Live2D SDK license](https://www.live2d.com/en/?p=586)
- 요금(검색 요약): 대기업 초기 ¥300,000 + 플랫폼당 연 ¥1,200,000 + 매출 5%, 중기업 초기 ¥50,000 + 연 ¥240,000 + 5% — [Live2D 검색 요약](https://live2d.com/en/?p=589) (원문 표 직접 확인 못 함)
- 자동 디포머 생성은 휴머노이드 전신, PSD 가져오기 기반 — [Live2D 5.1 New Features](https://docs.live2d.com/en/cubism-editor-manual/new-function5-1/)

**Cartoon Animator 5**
- 데스크톱 저작 도구. 외부 Python 스크립트·명령줄 일괄 처리 지원은 사용자 요청 단계로 보임 — [Reallusion Feedback Tracker](https://cdn-www.reallusion.com/FeedBackTracker/Issue/Extentions-Plugins-support); 해당 페이지에 SDK/API/서버 자동화 언급 없음 — [Reallusion 2D Character Creation](https://www.reallusion.com/cartoon-animator/2D-character-creation.html)

**Adobe Express Animate characters**
- 웹사이트에 "Animate from Audio" 임베드 문의 글이 있음 — [Adobe Community](https://community.adobe.com/t5/adobe-express-developers-discussions/embed-animate-from-audio-in-website/m-p/14026800/highlight/true) (답변 내용 미확인). 자기 캐릭터 업로드가 안 되므로 우리 용도엔 무관

**아이 그림 전용 소비자 앱 (API 없음)**
- DrawToLife, Drawings Alive, SKETCH, FlipaClip, Canva, DoodleTale 모두 API 언급 없음 — [draw-to-life.com 블로그](https://draw-to-life.com/blog/best-apps-bring-kids-drawings-to-life/)
- Drawmingo: API·상업 라이선스 언급 없음, 2~5분 — [drawmingo.com](https://drawmingo.com/)
- KidsAI: API·상업 라이선스 언급 없음 — [kidsai.art](https://kidsai.art/)
- Pixley: B2B/API 정보 없음, 캐릭터 생성 약 1분 — [pixleyai.com](https://www.pixleyai.com/)

**Meta Animated Drawings (참고, 현재 사용 중)**
- Python, macOS·Ubuntu 시험, 오픈소스 — [About Meta](https://about.fb.com/news/2021/12/using-ai-to-animate-childrens-drawings/); [InfoQ](https://www.infoq.com/news/2023/04/meta-animated-drawings/)

### Inferences
- 요구 (d) API + (e) 상업 + (f) 저가를 만족하는 외부 서비스는 생성 영상형(Kling·Viggle 등)밖에 없다. 비용은 5초당 $0.05(Viggle-Animate, 구동 영상 5초 기준 단순 계산) ~ $0.56(Kling 3.0 Pro) 수준으로, 캐릭터마다 장면 수만큼 곱해진다. 장면 4개 × 5초면 Kling 기준 약 $2.2/편(단순 계산, 추정).
- 생성 영상형은 요구 (c) "아이 그림 그대로"를 보장하지 못한다. KidsAI 처럼 "선은 그대로"라고 홍보해도 픽셀 보존이 아닌 모델의 재그리기이므로, 느린 학습자 아이에게 "내 그림이 변했다"는 혼란을 줄 위험이 있다(추정, 직접 시험 안 함).
- Live2D 는 (1) 휴머노이드 전용 자동화, (2) 부위별 PSD 필요, (3) 우리 구조가 Expandable Application 으로 분류될 공산이 커서 소규모 면제를 못 받음 → 후보에서 제외가 타당.
- 처리 시간: Drawmingo 2~5분, Pixley 약 1분으로, 요구 (g) 수십 초를 넘기는 경우가 많다. Viggle·Kling 지연은 공개 수치가 없다.

### Gaps
- Viggle API 가 동물·괴물 그림을 받아 주는지, 처리 시간 — 문서에 없음. 직접 시험 필요(유료, 허락 필요)
- Live2D 요금표 원문, 우리 서비스(서버에서 모델 자동 생성 후 사용자에게 재생만)가 Expandable Application 에 정확히 해당하는지 — Live2D 에 문의해야 확인 가능
- Kling·Viggle 의 한국 아동 개인정보·아동 이미지 업로드 관련 약관 — 이번 범위에서 확인 못 함

---

## 3. 비휴머노이드에서 부위를 자동 인식하는가, 아니면 그림 전체 효과(흔들기·튀기기)만인가?

### Takeaway
확인한 상용 제품 중 **비휴머노이드 그림의 부위(다리·꼬리·날개)를 자동 인식해 2D 리그를 만드는 것은 하나도 없다.** 생성 영상형은 부위 인식 없이 영상 전체를 새로 그리고, 도안형(teamLab·Quiver)은 미리 만든 리그에 아이 색만 입히며, 리깅 도구는 사람이 뼈를 놓거나 템플릿 레이어를 맞춰야 한다.

### Cited Findings
- Cartoon Animator: 네발 동물은 PSD 템플릿 레이어 교체 또는 수작업 뼈 정렬. AI 자동 뼈 인식은 사용자 요청 상태 — [Reallusion 2D Character Creation](https://www.reallusion.com/cartoon-animator/2D-character-creation.html); [Feedback Tracker](https://www.reallusion.com/FeedBackTracker/Issue/AI-Integration-Ideas-and-requests)
- Live2D 자동 디포머 생성: 휴머노이드 전신 대상 — [Live2D 5.1 New Features](https://docs.live2d.com/en/cubism-editor-manual/new-function5-1/)
- Viggle: 그림의 해부 구조를 분석해 동작에 맞춰 휘게 함(사람 동작 영상/템플릿이 구동원) — [Solan Sync](https://solansync.beehiiv.com/p/business-idea-animate-characters-free-latest-aitools); API 예시는 사람형만 — [Viggle developers](https://viggle.ai/developers)
- teamLab: 제공된 물고기 도안에 색칠 → 스캔 → 헤엄 — [teamLab](https://teamlab.art/w/aquarium/milanosalone2014)
- Quiver: 인쇄된 도안 색칠 → 스캔 → 3D AR — [quivervision.com](https://quivervision.com/)
- Doodle Dreams: "애니메이션에 최적화된" 테마 도안 팩 — [G2](https://ai.g2.com/marketplace/tools/doodle-dreams)
- Drawings Alive: 사용자가 짧은 설명을 넣어 AI 생성을 안내(=부위 인식이 아니라 프롬프트 기반 생성) — [G2 AI marketplace](https://ai.g2.com/marketplace/tools/drawings-alive)
- Krikey: 비휴머노이드는 자사 3D 캐릭터 라이브러리로 대체 — [Krikey learn](https://www.krikey.ai/learn/how-to-create-animated-characters)
- Animated Drawings: 사람 형태만 — [draw-to-life.com 블로그](https://draw-to-life.com/blog/best-apps-bring-kids-drawings-to-life/)

### Inferences
- "그림 전체 효과(흔들기·숨쉬기·통통 튀기)"를 별도 상품으로 내세운 상용 API 는 찾지 못했지만, 이는 기술적으로 단순(메시 변형·사인파)해서 상품화되지 않은 것으로 보인다. 별·구름·정체불명 괴물처럼 부위가 없는 그림은 이 방식이 오히려 적합하므로 자체 구현이 현실적이다(추정).
- 결국 상용 시장은 (가) 사람형 자동 리깅, (나) 생성 영상, (다) 도안형 세 갈래로 나뉘며, "자유 그림 + 비휴머노이드 + 픽셀 보존 + 자동"은 비어 있다. 이 공백이 우리 실험(모양 뼈대 + 이미지 모델 부위 이름)의 차별점이 될 수 있다.

### Gaps
- 생성 영상형 서비스들이 내부에서 부위 인식(포즈 추정 등)을 섞는지 — 모두 비공개
- Adobe 의 실험 기능(Firefly 영상 등)에 2D 퍼펫 자동 리깅이 들어왔는지 — 출처 확인 못 함

---

## 4. 아이 그림 애니메이션 전용 B2B API 를 파는 회사가 있는가?

### Takeaway
**찾지 못했다.** 아이 그림 전용 서비스(DrawToLife, Drawmingo, KidsAI, Pixley, Drawings Alive, Doodle Dreams)는 모두 소비자 앱/웹이고 API·파트너 프로그램을 공개하지 않는다. B2B 로 쓸 수 있는 것은 범용 생성 영상 API(Kling 등)와 Viggle API, 그리고 오픈소스 Animated Drawings 를 직접 호스팅하는 방법뿐이다.

### Cited Findings
- 리뷰 비교표의 7개 앱 모두 API 언급 없음 — [draw-to-life.com 블로그](https://draw-to-life.com/blog/best-apps-bring-kids-drawings-to-life/)
- 검색 요약에서도 "대부분 오픈소스이거나 소비자 클라우드 플랫폼이지 전통적 B2B SDK 는 아니다" — [검색 결과: drawing-animator](https://github.com/sampurchase/drawing-animator); [kidsai.art](https://kidsai.art/)
- 범용 API 에서 아이 그림 애니메이션 사용 예시 문서 — [AIML API docs](https://docs.aimlapi.com/use-cases/animate-images-a-childrens-encyclopedia)
- Viggle 자가 가입 API, 출력 상업 이용 가능 — [Viggle developers](https://viggle.ai/developers)
- Kidify, KidsAI 는 아동 안전(광고 없음, 학습 미사용) 강조 — [kidsai.art](https://kidsai.art/); Drawmingo 는 COPPA 준수·부모 통제 강조 — [drawmingo.com](https://drawmingo.com/)

### Inferences
- 아이 그림 전용 회사들은 범용 영상 모델 API 를 감싼 형태로 보이므로(추정), B2B 로 그들을 거치면 원가 위에 마진이 붙을 뿐 품질상 이점이 적다. 생성 영상형을 쓴다면 Kling/fal.ai 등 원 API 를 직접 쓰는 편이 싸다.
- 우리 요구 (c) 를 지키려면 외부 B2B 대신 자체 리깅(Animated Drawings + 비휴머노이드 확장) 노선을 유지하고, 생성 영상형은 "보조 연출(배경 움직임 등)" 정도로만 검토하는 것이 맞아 보인다(추정, 팀 논의 필요 — pt_docs 우선).

### Gaps
- 한국 에듀테크(키즈 앱) 회사 중 그림 애니메이션 B2B 를 제공하는 곳 — 한국어 검색으로 찾지 못함. 국내 업계 문의가 필요
- DrawToLife·Pixley 등에 직접 문의 시 B2B/화이트라벨이 가능한지 — 공개 정보 없음
