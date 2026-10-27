# API 계약 대조 — 한칸이야기 (2026-10-27)

**세 줄 요약**
1. 화면이 쓰는 API(그림 올리기 · 관절 · 이야기 · 순서 맞추기)의 **필드 이름 · 타입 · 필수 여부 · 상태 값 · 오류 코드 매핑은 프론트 · 백엔드 · 설계 문서가 서로 맞는다.** 백엔드 테스트 95개도 모두 통과했다.
2. **시연을 막을 수 있는 것은 배포 주소다.** 프론트는 서버 주소를 `/api/v1` 로 박아 두었다. 그런데 배포 문서는 `VITE_API_BASE` 라는 설정을 쓰라고 한다. 이 설정은 프론트 코드에 없다. 배포한 화면(Static Web Apps)으로 시연하면 서버에 닿지 못한다. 로컬 시연이면 해당 없다.
3. 나머지는 **문서가 코드보다 늦은 곳**이다. 시간 제한 값(60 · 120 · 300초), 렌더 시간(8 · 35~40 · 47.5초), `max_frames`, 빠진 오류 코드가 문서마다 다르다. AI 서버가 다시 켜지면 이미 분석한 그림으로는 이야기를 못 만드는 점도 시연 전에 알아 두어야 한다.

---

## 확인한 판

| 저장소 | 브랜치 | 커밋 |
|---|---|---|
| drawtale-frontend | feat/backend-contract | `43dc34e` (2026-10-19) |
| drawtale-backend | dev | `c565f09` (2026-10-20) |
| drawtale-pt_docs | main | `ac8f5ef` (2026-10-27) |

- 읽기만 했다. 팀 저장소에 push · 이슈 · PR · 댓글은 하지 않았다. `.env` 는 열지 않았다 (클론에 없다).
- 「설계」 = pt_docs, 「백 계약」 = 백엔드 `docs/api-contract.md`, 「프론트 계약」 = 프론트 `docs/api-contract.md`.
- **확실한 정도** — 높음: 코드를 읽고 직접 돌려 확인함 · 중간: 코드를 읽었지만 돌려 보지는 않음 · 추측: 그럴 수 있다는 판단이고 재현하지 않음.

---

## 1. 시연을 막음

| # | 무엇이 | 어디서 | 근거 (어긋나는 곳) | 누가 고칠지 | 확실한 정도 |
|---|---|---|---|---|---|
| A1 | **배포한 프론트가 서버 주소를 모른다.** 서버 주소가 `const BASE = "/api/v1"` 로 박혀 있다. 로그인도 `fetch(\`/api/v1${path}\`)` 를 쓴다. `VITE_API_BASE` 같은 설정은 코드 어디에도 없다 (`import.meta.env` 는 `VITE_ENGINE` · 카카오 · 구글 client id 뿐). 로컬에서는 vite 프록시(`vite.config.ts` `/api → 127.0.0.1:8000`) 덕분에 된다 | 프론트 `src/api/index.ts:15`, `src/api/auth.ts:17` | 백엔드 `deploy/README.md:17` 「프론트 배포 \| Azure Static Web Apps. `VITE_API_BASE` 를 위 백엔드 주소로」. 설계 `11-server-architecture.md:79` 는 프론트를 Static Web Apps 에, 백엔드를 VM(Caddy `api.<도메인>`)에 따로 둔다. 같은 출처가 아니므로 상대 경로 `/api/v1` 은 Static Web Apps 자신에게 간다 | 프론트 (주소 설정 추가). 백엔드 배포 문서도 이름을 맞춘다 | 코드에 설정이 없다는 것: **높음**. Static Web Apps 에서 `/api` 가 실제로 어떻게 답하는지(404 등): **추측** — 배포해 보지 않았다 |
| A2 | **HTTPS 화면이 HTTP 서버를 부르면 브라우저가 막는다.** Caddy 설정에 「도메인이 없으면 `:80` 으로, 시연용」이라고 적혀 있다. Static Web Apps 는 HTTPS 로만 열린다 | 백엔드 `deploy/Caddyfile:1-3` | 설계 `11-server-architecture.md` 는 Caddy 가 HTTPS 인증서를 받는다고 적는다. `:80` 로 두면 그 전제가 깨진다 (mixed content) | 백엔드 (시연에도 도메인 + HTTPS) | **중간** — 브라우저의 일반 규칙이다. 이 배포로 재현하지는 않았다 |

> A1 · A2 는 **배포 주소로 시연할 때만** 막는다. 시연 PC 에서 `npm run dev`(또는 `vite preview`) + 로컬 백엔드 · AI 로 한다면 해당 없다. **어느 쪽으로 시연할지가 정해져 있지 않으면 그것부터 정한다.**

---

## 2. 고치면 좋음

| # | 무엇이 | 어디서 | 근거 (어긋나는 곳) | 누가 고칠지 | 확실한 정도 |
|---|---|---|---|---|---|
| B1 | **AI 서버가 다시 켜지면 이미 분석한 그림의 이야기가 실패한다.** 렌더가 `SESSION_NOT_FOUND` 를 받으면 `_ai_error` 가 `AI_ERROR` 를 던지고 이야기 Job 이 통째로 `failed` 가 된다. 화면은 `ENGINE_ERROR` →「잠깐 문제가 생겼어요 · 처음으로」를 띄운다. 문장 · 음성은 이미 만들어 놓은 상태다 | 백엔드 `app/services/jobs.py:351-359`, `:383-385`, `app/services/ai_client.py:153-154` | 설계 `04-flowchart.md:321` 은 「AI 세션 없음 → 원본 그림으로 계속」, `:323` 은 「렌더의 그 밖 실패 → 멈춘다」. 코드에서 「세션 없음」은 **DB 에 `ai_request_id` 가 비었을 때뿐**이다. AI 쪽에서 세션을 잃은 경우(`SESSION_NOT_FOUND`)는 「그 밖 실패」로 간다. 설계 `05` S-08 (`05-screen-spec.md:466`)은 「부분 실패를 전체 실패로 만들지 않는다」. 둘 중 어느 쪽을 따를지 설계가 정해야 한다 | 설계 문서 (`SESSION_NOT_FOUND` 를 어느 줄로 볼지) → 백엔드 | 코드 흐름: **높음**. AI 서버가 다시 켜질 때 세션이 사라지는지는 백 계약 3-3 표(「AI 서버 재시작 등」, `docs/api-contract.md:522`)에 근거함 — AI 코드는 보지 않았다 |
| B2 | **시간 제한 값이 네 군데에서 다르다.** 코드 기본값은 60초, `.env.example` 은 300초, 배포는 120초다. 프론트 주석과 프론트 계약은 「백엔드 AI 타임아웃(300초)과 같다」고 적는다 | 백엔드 `app/core/config.py:15` (`ai_timeout_seconds = 60`), `.env.example:12` (300), `deploy/README.md:61` (120) · 프론트 `src/api/index.ts:20-22`, `docs/api-contract.md:126` | 설계 `01` NFR-01 (`01-requirements.md:75`), `04-flowchart.md:416` 은 **프론트 대기 한도**(분석 30초 · 이야기 300초)만 정한다. AI 렌더 한도는 정하지 않았다. 배포 실측 렌더가 47.5초(`deploy/README.md:60`)라서 `.env` 없이 띄우면(60초) 아슬아슬하다 | 백엔드 (코드 기본값을 배포값에 맞춤) · 프론트 (주석 · 프론트 계약 고침) | **높음** (값은 코드와 문서에서 읽음) |
| B3 | **렌더 시간이 문서마다 다르다** — 8초 / 35~40초 / 47.5초 | 백 계약 3-3 「소요 시간 약 35~40초」 (`docs/api-contract.md:541`), `deploy/Caddyfile:11`, `deploy/README.md:60` (47.5초) | 설계 `03-feature-spec.md:918` · `05-screen-spec.md:465` 는 「렌더 8초」(동작 파일을 짧게 한 뒤). 위 B2 의 한도를 정하려면 **시연 VM 에서 다시 잰 값 하나**가 필요하다 | 백엔드 (VM 에서 재고 문서 하나로) | **높음** (숫자가 다르다는 것). 어느 값이 지금 맞는지는 **모름** |
| B4 | **`max_frames` 를 AI 서버가 받지 않는데 백 계약은 받는다고 적는다.** 백엔드 코드 주석은 「조용히 버려진다 (2026-10-20 확인)」. 그래서 S-05 미리보기도 동작 전체를 그린다 | 백엔드 `app/schemas/ai.py:82-85`, `app/services/jobs.py:125-132` | 백 계약 3-3 표 `max_frames` 줄 (`docs/api-contract.md:504`), 2-3 (`:206-207` 「`max_frames=150`」) | 백엔드 (계약서 고침) | **높음** (백엔드 주석 기준. AI 코드는 보지 않음) |
| B5 | **미리보기 두 개가 엇갈려 옛 관절 영상이 남을 수 있다.** 분석 Job 이 끝나면 같은 작업 안에서 미리보기를 렌더한다. 그동안 어른이 S-06 에서 관절을 고치면 `run_preview_job` 이 하나 더 돈다. 분석 쪽 렌더가 **나중에** 끝나면 고치기 전 관절로 만든 영상으로 덮어쓴다 | 백엔드 `app/services/jobs.py:93-97`(분석 뒤 렌더), `:173-186`, `app/api/v1/characters.py:168-172` | 백 계약 2-3 (`docs/api-contract.md:209`) 「관절을 보정하면 `rendering` 으로 돌아가 새로 만들고 옛 파일은 지운다」 | 백엔드 | **추측** — 재현하지 않았다. 렌더가 오래 걸릴수록(B3) 일어나기 쉽다 |
| B6 | **E-01 「반복해서 걸러질 때」가 프론트에 없다.** 오류 화면은 코드별 이동만 한다. 연속 실패 횟수를 세지 않는다 | 프론트 `src/screens/E01Error.tsx:19-28` | 설계 `05-screen-spec.md:689-698`: 2회 연속 → S-03D, 3회 → 어른, 지원 수준 1은 1회에서 바로 S-03D (`06` 4-3) | 프론트 | **높음** (`src` 에서 횟수를 세는 코드를 찾지 못함) |
| B7 | **음수 관절 좌표는 계약과 다른 코드로 온다.** `Joint.x/y` 에 `ge=0` 이 걸려 있어서 음수는 `422 VALIDATION_ERROR` 다. 그림보다 큰 값만 `400 JOINT_OUT_OF_IMAGE` 다. AI 가 준 관절에 음수가 있으면 분석 전체가 `AI_INVALID_RESPONSE` 로 실패한다 | 백엔드 `app/schemas/common.py:34-35`, `app/api/v1/characters.py:156-157`, `app/services/ai_client.py:83-88` | 백 계약 1-4 (`docs/api-contract.md:102`) · 2-4 (`:228`) 「관절이 이미지 밖이면 `400 JOINT_OUT_OF_IMAGE`」 | 백엔드 (문서 또는 검사 하나로) | 422/400 동작: **높음** (직접 돌려 봄). 프론트는 끈 관절을 0~1 로 자르므로(`S06Joints.tsx:55-56`) 화면에서는 잘 안 생긴다. AI 가 음수를 주는지는 **추측** |
| B8 | **백 계약 1-4 오류 표에 없는 코드가 있다** — `AI_INVALID_RESPONSE`(502, Job error), `NOT_FOUND` · `HTTP_ERROR`(없는 경로 등) | 백엔드 `app/services/ai_client.py:81,88,131,138`, `app/core/errors.py:39` | 백 계약 1-4 표 (`docs/api-contract.md:97-120`). 「모든 에러는 같은 형식」은 맞다. 다만 코드 목록이 모자라다. 프론트는 이 셋을 `ENGINE_ERROR` 로 보내므로 설계 `05` E-01 「그 밖의 모든 코드」와 맞다 | 백엔드 (문서에 추가) | **높음** (`/api/v1/nope` → `{"code":"NOT_FOUND"}` 확인) |
| B9 | **S-05-07 「누워 있던 친구가 일어났어요!」를 실을 자리가 계약에 없다.** 서버가 그림을 돌려 분석했는지 알리는 필드가 백엔드 응답에도, 프론트 모양(`ServerCharacter`)에도 없다 | 백엔드 `app/schemas/character.py:29-46` · 프론트 `src/api/index.ts:53-63` | 설계 `05-screen-spec.md:313` (S-05-07), `06-input-guideline.md:344-383` (3-4 B 「서버가 돌려 보고 잘 나오는 방향을 쓴다」) | 설계 문서 (계획인지 지금 할 일인지) → 백엔드 · AI · 프론트 | 필드가 없다는 것: **높음**. 이것이 시연 범위인지는 **모름** |

---

## 3. 참고

| # | 무엇이 | 어디서 | 근거 | 누가 | 확실한 정도 |
|---|---|---|---|---|---|
| C1 | 프론트 계약이 스스로 어긋난다. 1절은 「`/api/v1/auth/*` 를 부르지만 **백엔드에 없다**」고 적는다. 4절과 백 계약 2-7 · 2-8 에는 있다 | 프론트 `docs/api-contract.md:36` | 백 계약 2-7 (`docs/api-contract.md:329-371`), 백엔드 `app/api/v1/auth.py` | 프론트 (문서) | 높음 |
| C2 | 프론트 계약 「제목이 없어서 `「{장소}에 간 날」`을 프론트가 만든다」. 지금은 긴 이야기면 `book.title` 을 먼저 쓴다 | 프론트 `docs/api-contract.md:119` ↔ `src/api/index.ts:159` | 백 계약 2-6 `book.title` | 프론트 (문서) | 높음 |
| C3 | 백 계약 2-5 의 ⚠ 「선택지를 서버가 내려줄지 · 코드 값으로 할지 합의 필요」가 아직 열려 있다. 실제로는 설계가 카드 24장을 정했다. 프론트가 라벨(50자 이하 문자열)을 보낸다 | 백 계약 `docs/api-contract.md:271-272` | 설계 `05-screen-spec.md:443-448`, 프론트 `src/api/index.ts:239-242` | 백엔드 (⚠ 닫기) | 높음 |
| C4 | 렌더 실패 · 시간 초과의 어른용 메시지가 「**그림 분석** 중 오류 / 시간 초과」다 (이야기 렌더에서도). 로그도 `AI analyze failed` 로 찍힌다. 아이 화면에는 안 나오지만 운영 기록을 볼 때 헷갈린다 | 백엔드 `app/services/ai_client.py:65-67`, `:115` | 백 계약 1-4 「`message`: 사용자에게 그대로 보여줘도 되는 문장」 | 백엔드 | 높음 |
| C5 | 이야기 Job 이 300초를 넘길 수도 있다. 긴 그림책 대본(OpenAI 30초 × 다시 쓰기) + 대화 음성(60초) + 렌더(120초)를 더하면 프론트 한도 300초에 가깝다. 넘으면 서버는 계속 만드는데 화면은 `ENGINE_TIMEOUT` 이다 | 백엔드 `app/core/config.py:31,40`, `deploy/README.md:61` · 프론트 `src/api/index.ts:22` | 설계 NFR-01 「이야기 300초」 | 백엔드 (재기) | **추측** — 최악의 경우를 더한 값이다. 재지 않았다 |
| C6 | S-08 에서 `ENGINE_TIMEOUT` 이 나면 E-01 「다시 하기」가 **S-03(그림 올리기)** 으로 보낸다. 아이는 그림부터 다시 해야 한다. 프론트는 설계대로 했다 | 프론트 `src/screens/E01Error.tsx:27` | 설계 `05-screen-spec.md:685` 가 `ENGINE_TIMEOUT → S-03` 하나만 정한다. 이야기 단계의 시간 초과를 S-07 로 보낼지는 설계가 정할 일 | 설계 문서 | 높음 (설계와 일치함을 확인) |
| C7 | E-01 보조 문구가 설계와 다르다. `NO_CHARACTER`: 프론트 「사람을 한 명만 크게 그려 볼까요?」 ↔ 설계 「지금은 두 발로 선 친구만 움직일 수 있어요」 | 프론트 `src/lib.ts:44` | 설계 `05-screen-spec.md:681` | 프론트 | 높음 |
| C8 | 사진 경로: 10MB 검사를 줄이기 **전의** 원본 크기로 한다. 줄이면 들어갈 사진도 막힌다. 다만 설계 S-03-09 · R-09 가 그렇게 정했다 | 프론트 `src/screens/S03Upload.tsx:32` | 설계 `05-screen-spec.md:255`, `06-input-guideline.md:291` | 설계 문서 (필요하면) | 높음. 시연 기본 경로(화면에 그리기, `photoUpload: false`)와는 관계없다 |
| C9 | 사진 경로: 2MB 미만 · 1600px 이하 JPEG 는 다시 그리지 않고 그대로 보낸다. 사진의 방향 메모(EXIF)가 픽셀에 반영되지 않은 채 간다 | 프론트 `src/lib.ts:32` | 설계 `06-input-guideline.md:355-364` (3-4 A) 가 「지금 프론트에도 AI 서버에도 없다」고 이미 적었다 | 프론트 | 높음 (코드). 실제로 누운 결과가 나오는지는 **추측** |
| C10 | 사용자가 옮기지 않은 관절은 AI 값을 그대로 다시 보낸다. AI 가 그림 너비보다 조금 큰 값을 주면 S-06 저장이 `JOINT_OUT_OF_IMAGE` 로 실패한다 (화면에는 저장 실패 안내) | 프론트 `src/api/index.ts:227-231` · 백엔드 `characters.py:156-157` | 백 계약 1-2 「AI 도 응답은 원본 이미지 좌표로」 | 백엔드 또는 AI (값 자르기) | **추측** |
| C11 | 개발 모드(`npm run dev`, StrictMode)에서는 분석 · 이야기 요청이 두 번 간다. 렌더도 두 번 돈다. 팀이 이미 아는 일이다 | 프론트 `docs/team-status.md:162` | — | 시연 운영 | 높음 (팀 문서) |
| C12 | 시연 중 **새로고침 · 탭 닫기**를 하면 `pagehide` 가 원본을 지운다(「원본 그림 보관」이 꺼져 있을 때). 그 뒤 같은 그림으로 만든 이야기는 MP4 없이 캔버스로 움직인다. 계약대로의 동작이다 | 프론트 `docs/api-contract.md:84-89` · 백 계약 2-10 | 백 계약 `docs/api-contract.md:399` | 시연 운영 (S-13 「원본 그림 보관」을 켜 두거나 새로고침하지 않기) | 높음 (직접 돌려 봄: 지운 뒤 `image_url` · `animation_url` 이 `null`) |

---

## 4. 맞는 것 (확인함)

| 항목 | 프론트 `src/api/index.ts` | 백엔드 코드 | 설계 |
|---|---|---|---|
| 그림 올리기 | `image` + `keep_original` 문자열 (`:184-186`) | `File` · `Form(bool)` (`characters.py:73-82`) | — |
| 크기 · 형식 | 10MB · `image/*` 를 올리기 전에 막고, PNG/JPEG 가 아니면 JPEG 로 바꿈 | `max_upload_mb=10`, PNG/JPEG 만 (`characters.py:27,83-94`) | R-08 · R-09 |
| Job 상태 | `pending · running · succeeded · failed` | `JobStatus` (`schemas/common.py:53-57`) | `04` 10장 |
| polling | 1.5초 (`:17`) | — | 1.5초 (`04-flowchart.md:188`, `05` 7장) |
| 대기 한도 | 분석 30초 · 이야기 300초 (`:19,22`, S-04 타이머 `S04Analyzing.tsx:27`) | — | NFR-01 |
| 잠깐 끊김 | `TypeError` · `HTTP_502/503/504` 면 다시 물음 (`:101-103`) | — | `05` 7장 「연결 끊김」 |
| 관절 | 15개 전부, 원본 px, 소수 1자리 (`:227-231`) | 15개 · 이름 검사 · 정렬 (`common.py:75-80`) | — |
| 이야기 요청 | `place/problem/action/result` · `voice_mode` · `protagonist_voice` · `protagonist_name`(10자) · `length` · `spoken` | `StoryCreateRequest` (`schemas/story.py:41-79`) 와 이름 · 값 · 길이가 같다 | — |
| 이야기 응답 | `book.title/pages/order_pages`, `text`, `audio_url`, `animation_url` (`.mp4(\?|$)` 로 영상 판단 — SAS 주소의 `?` 도 받음, `S09Story.tsx:57`) | `StoryResponse` (`schemas/story.py:115-134`) | — |
| 순서 맞추기 기록 | `attempts · completed · card_count · level` | `ActivityCreateRequest` 범위 1~100 · 2~4 · 1~3 | — |
| ERROR_MAP | `src/api/index.ts:33-43` | 백엔드가 낼 수 있는 코드 전부 — 표에 없는 것은 `ENGINE_ERROR` | `05-screen-spec.md:704-714` 표와 **한 줄도 다르지 않다** |
| 미리보기 | `preview_status: rendering` 이면 1.5초마다, 90초까지 (`CharacterPreview.tsx:8-9`) | 분석 Job 이 `succeeded` 가 된 뒤 렌더 (`jobs.py:93-97`) | `05` S-05-06 |

---

## 5. 백엔드 테스트

- 실행: Python 3.11.17 가상환경에 `requirements-dev.txt` 를 설치하고 `pytest -q` (백엔드 dev `c565f09`).
- 결과: **95 passed (4.4초)**. `ruff check app tests` 도 통과.
- 키가 필요 없다. `tests/conftest.py` 가 SQLite · 로컬 저장소 · 목 AI 로 바꾸고 OpenAI · TTS · ElevenLabs 키를 비운다. 그래서 **실제 OpenAI · ElevenLabs · AI 서버 · Azure Blob · PostgreSQL 은 시험하지 않았다.** 키를 쓰는 부분은 테스트 안에서 가짜로 바꿔 끼운다.
- 따로 확인한 것 (목 AI, 같은 conftest): 음수 관절 → 422 `VALIDATION_ERROR`, 너비와 같은 x → 200, 너비보다 큰 x → 400 `JOINT_OUT_OF_IMAGE`, 이름 11자 → 422, 긴 이야기 → 8쪽 · `order_pages [1,2,4,6]`, 목 AI 의 `animation_url` 은 PNG 주소, 원본 지운 뒤 `image_url` · `animation_url` 은 `null`, 없는 경로 → `NOT_FOUND`.

---

## 6. 시연 전에 할 일 (순서대로)

1. **시연을 배포 주소로 할지 로컬로 할지 정한다.** 배포라면 A1(프론트 서버 주소 설정)과 A2(HTTPS)를 먼저 고친다.
2. 시연 VM(또는 PC)에서 **렌더 시간을 한 번 재고**, `AI_TIMEOUT_SECONDS` 를 그 값의 2배 이상으로 `.env` 에 적는다 (B2 · B3).
3. 시연 직전에는 **AI 서버를 다시 켜지 않는다.** 다시 켰다면 그림부터 새로 올린다 (B1).
4. 시연은 배포 빌드로 하고, 새로고침하지 않는다 (C11 · C12).
