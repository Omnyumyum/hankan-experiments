# 2D 비휴머노이드 그림 자동 리깅 · 메시 변형 오픈소스 구성요소 조사 (2026-10-26)

> 조사 범위: 사진 1장 → 마스크 → 메시 → 뼈대 → 변형 파이프라인 중 **이미 만든 것(마스크, scipy Delaunay 메시, skeletonize 뼈대, ARAP2D, 네발 다리 찾기, 몸통 꿈틀)** 을 바꾸거나 낫게 할 오픈소스.
> 저장소 별(stars) · 마지막 push 날짜는 2026-10-26 GitHub API 조회값. 라이선스는 가능한 한 실제 LICENSE 파일을 열어 확인했다. 확인 못 한 것은 "(미검증)"으로 표시.
> 우리 코드 확인: `shape_rig.py` 는 `cv2`, `scipy.spatial.Delaunay`, `skimage.morphology.skeletonize`, 자체 `arap2d.ARAP2D` 만 쓴다 (로컬 파일 import 확인). 즉 지금은 Triangle 같은 라이선스 위험 의존성이 없다.

## 1. 2D 변형 방법과 구현체 (ARAP 변형, BBW, LBS, MLS, Puppet warp, cage/harmonic 좌표)

### Takeaway
Python 에서 가장 쓸 만한 묶음은 **libigl Python 바인딩(pip `libigl`, 코어 MPL-2.0)** 으로, `arap`, `bbw`(Bounded Biharmonic Weights), `harmonic`, `biharmonic_coordinates`, `lbs_matrix`, `mvc`, `kelvinlets` 가 모두 들어 있다. 단, 같은 휠에 비상업 Triangle(`restricted`)과 GPL/AGPL(`copyleft` cgal·tetgen) 모듈이 기본 빌드되므로 그 모듈은 쓰지 말아야 한다. 뼈대 없는 모양(별·구름·덩어리)에는 MLS(Jarvis73, MIT, numpy) 가 가볍게 붙일 만하다. JS 쪽에는 쓸 만한 ARAP/puppet warp 라이브러리가 사실상 없다.

### Cited Findings
- **libigl (C++)**: 저장소 루트에 `LICENSE.MPL2` 와 `LICENSE.GPL` 두 파일이 있다 (GitHub API 는 GPL-3.0 으로 표시하지만 실제로는 코어 MPL-2.0 + copyleft 하위 모듈 GPL 의 이중 구조). 별 5,114, 마지막 push 2026-09-21 — [libigl/libigl](https://github.com/libigl/libigl), [LICENSE.MPL2](https://github.com/libigl/libigl/blob/main/LICENSE.MPL2)
- **libigl Python 바인딩**: `pyproject.toml` 분류자 `License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)`, 버전 2.6.4.dev1, nanobind 기반. 별 371, push 2026-09-04 — [libigl-python-bindings](https://github.com/libigl/libigl-python-bindings), [pyproject.toml](https://github.com/libigl/libigl-python-bindings/blob/main/pyproject.toml)
- 바인딩 `src/` 에 `arap.cpp`, `bbw.cpp`, `biharmonic_coordinates.cpp`, `harmonic.cpp`, `lbs_matrix.cpp`, `mvc.cpp`(mean value coordinates), `kelvinlets.cpp`, `boundary_conditions.cpp`, `active_set.cpp`, `min_quad_with_fixed.cpp`, `lscm.cpp`, `slim.cpp` 존재 — [src 트리](https://github.com/libigl/libigl-python-bindings/tree/main/src)
- `bbw` 바인딩은 `igl::BBWData` 의 `partition_unity`, active-set 파라미터(`max_iter`, `inactive_threshold` 등)를 노출하고 내장 active-set QP 로 푼다 (MOSEK 불필요로 보임) — [src/bbw.cpp](https://github.com/libigl/libigl-python-bindings/blob/main/src/bbw.cpp)
- `arap` 바인딩은 에너지 종류 `ARAP_ENERGY_TYPE_SPOKES`, `SPOKES_AND_RIMS`, `ELEMENTS` 를 노출 (2D 삼각형 메시는 ELEMENTS/SPOKES_AND_RIMS 가 해당). 삼각형별 강성(rigidity) 가중치 인자는 찾지 못함 — [src/arap.cpp](https://github.com/libigl/libigl-python-bindings/blob/main/src/arap.cpp)
- 바인딩 `CMakeLists.txt` 기본값: `LIBIGL_COPYLEFT_CGAL ON`, `LIBIGL_COPYLEFT_TETGEN ON`, `LIBIGL_RESTRICTED_TRIANGLE ON` → `igl.copyleft.*`, `igl.triangle.*` (`triangulate`, `cdt`, `refine`, `scaf`) 가 같이 빌드됨 — [CMakeLists.txt](https://github.com/libigl/libigl-python-bindings/blob/main/CMakeLists.txt), [src/triangle/](https://github.com/libigl/libigl-python-bindings/tree/main/src/triangle)
- 공식 문서: 설치 `python -m pip install libigl`, `igl.copyleft` 는 "GPL-licensed functionality (booleans, meshing)" — [libigl python docs](https://libigl.github.io/libigl-python-bindings/)
- **gptoolbox** (Alec Jacobson, MATLAB, BBW/ARAP 원 저자 코드 다수): Apache-2.0, 별 682, push 2026-10-23. MATLAB 이라 서버에 바로 못 씀, 참고 구현용 — [alecjacobson/gptoolbox](https://github.com/alecjacobson/gptoolbox)
- **MLS 이미지 변형** (Schaefer 2006, affine/similarity/rigid 3종, 닫힌 해라 실시간 가능) — [논문 PDF](https://people.engr.tamu.edu/schaefer/research/mls.pdf)
  - Jarvis73/Moving-Least-Squares: "Numpy & PyTorch implementation of three algorithms", **MIT**, 별 365, 마지막 push 2023-12 (정체) — [GitHub](https://github.com/Jarvis73/Moving-Least-Squares)
  - mpizenberg/rust_mls: Rust 구현, **MPL-2.0**, 별 12, 2022-04 — [GitHub](https://github.com/mpizenberg/rust_mls)
  - 그 밖의 C++/MATLAB 구현은 라이선스 없음 또는 GPL (예: zhangzhensong/movingleastsquare GPL-2.0, CUDA_MLS GPL-3.0) — [GitHub 검색 결과: zhangzhensong](https://github.com/zhangzhensong/movingleastsquare), [CUDA_MLS](https://github.com/EvangelosKatsavrias/CUDA_MLS)
- **Puppet warp 계열**: mikecokina/puppet-warp — "Python library ... inspired by Photoshop's Puppet Warp", GitHub 메타데이터는 GPL-3.0, 별 61, push 2026-01. 단 main/master 루트에서 LICENSE 파일을 찾지 못했다 (라이선스 표기 불일치, 어느 쪽이든 GPL 이면 상업 서비스 통합에 부적합) — [GitHub](https://github.com/mikecokina/puppet-warp)
- **Igarashi 2005 ARAP** (2단계 닫힌 해: 회전 → 스케일 보정, 선형계 2개) 원 논문 · 구현 해설 — [프로젝트 페이지](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/research/rigid), [JGT 2009 구현 논문](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/takeo_jgt09_arapFlattening.pdf)
- **JS 구현**: GitHub 검색("arap deformation language:javascript", "arap language:typescript") 및 npm 검색("arap", "puppet warp")에서 쓸 만한 ARAP 라이브러리가 나오지 않음. 근접한 것: `laplacian-deformation` (npm 3.0.0, MIT, 2018, 정체) — [mikolalysenko/laplacian-deformation](https://github.com/mikolalysenko/laplacian-deformation); `@flighthq/skeleton2d` (npm 0.4.0, 2026-08, "2D skeleton: bone hierarchy ... inverse-bind, and weighted" 스키닝, MIT) — [flighthq/flight](https://github.com/flighthq/flight)
- **BBW 독립 구현들**(azer89/BBW, BBlashko/Real-TimeDeformations 등)은 모두 라이선스 표기 없음, 2015~2019 정체 — [azer89/BBW](https://github.com/azer89/BBW), [BBlashko](https://github.com/BBlashko/Real-TimeDeformations)

### Inferences
- **가장 큰 개선 후보 = "BBW 가중치 + LBS"를 ARAP 와 병행**: 서버(Python)에서 libigl `bbw` 로 뼈마디(우리 skeleton 노드)별 가중치를 한 번 계산하면, 매 프레임은 `V' = Σ w_j T_j V` 행렬곱뿐이라 **브라우저 GPU(Pixi custom shader / three.js SkinnedMesh)에서 돌릴 수 있다**. 지금처럼 서버에서 프레임마다 ARAP 를 풀어 영상/좌표를 만드는 구조보다 서버 비용과 대기 시간이 줄어든다. 다만 LBS 는 관절 굽힘에서 "candy-wrapper"/부피 감소가 생길 수 있어, 아이 그림처럼 두꺼운 다리·꼬리에서는 ARAP 보다 덜 자연스러울 수 있다 → 실험 필요.
- libigl 휠을 쓰면 코드에서 `igl.triangle`, `igl.copyleft` 를 import 하지 않는 것으로 충분한지는 법률 판단 영역. 서버에서만 실행(배포 안 함)이면 GPL 의무는 대체로 생기지 않지만, Triangle 의 "commercial products" 금지 문구는 서버 사용도 걸릴 수 있다. 가장 안전한 길은 **libigl 을 소스에서 `LIBIGL_RESTRICTED_TRIANGLE=OFF`, `LIBIGL_COPYLEFT_*=OFF` 로 직접 빌드**하거나, BBW/ARAP 만 우리 코드로 옮기는 것.
- ARAP 자체는 우리 재구현으로 충분하다. libigl ARAP 로 바꿔도 얻는 것은 속도(C++) 정도이고, 강성 가중치(머리는 단단하게, 꼬리는 말랑하게)는 libigl 에도 인자로 없으므로 **우리 ARAP2D 에 삼각형별 가중치를 넣는 쪽이 간단**하다.
- MLS(rigid) 는 메시·뼈대가 없어도 윤곽 위 몇 개 제어점만으로 변형되므로, 별·구름·덩어리 괴물의 "숨쉬기/흔들기"에 ARAP 대신 쓸 수 있다. Jarvis73 구현은 MIT 이고 numpy 라 바로 붙일 수 있으나, 픽셀 단위 역변환이라 해상도가 크면 느릴 수 있다 (속도 미측정).

### Gaps
- libigl PyPI 휠에 `igl.triangle` 바이너리가 실제로 들어 있는지는 휠을 풀어 보지 않아 미확인 (CMake 기본값 ON 으로 추정).
- "ARAP with rigidity"(삼각형별 강성) 의 공개 Python/JS 구현은 찾지 못함.
- Skinning cages / Green coordinates 의 유지되는 Python·JS 구현은 찾지 못함 (libigl 의 `mvc`, `harmonic` 정도가 대체).
- JS/WebGL ARAP 데모 저장소(예: "arap.js")는 검색으로 확인되지 않음 — 존재 여부 미검증.

## 2. 실루엣에서 2D 뼈대/리그 자동 추출 (medial axis 가지치기, DSE, skan, RigMesh, Pinocchio, Monster Mash, Sketch2Pose)

### Takeaway
skeletonize 결과를 **그래프로 바꾸고 짧은 잔가지를 쳐내는 부분**은 skan(BSD-3) 또는 DSE 가지치기(MIT)로 바로 개선할 수 있다. RigMesh·Pinocchio·RigNet·UniRig 등 "자동 리깅" 연구 코드는 대부분 3D 용이거나 라이선스가 맞지 않아, 2D 비휴머노이드 뼈대 추출은 여전히 "medial axis + 가지치기 + 규칙" 이 실용적인 최선이다.

### Cited Findings
- **skan**: "pure Python library to analyse skeleton images", BSD-3-Clause, 별 151, push 2026-06 — [jni/skan](https://github.com/jni/skan), [문서](https://skeleton-analysis.org/stable/)
  - `skan.csr.Skeleton` 에 `path_lengths()`, `prune_paths(indices)` 가 있고, `summarize()` 결과에 `branch_type` 열(끝점-끝점 / 분기-끝점 / 분기-분기 / 고리 구분)이 있다 — [csr.py 소스](https://github.com/jni/skan/blob/main/src/skan/csr.py)
- **DSE (Discrete Skeleton Evolution)**: 끝 가지마다 "원래 모양에 기여하는 픽셀 수"로 가중치를 매겨 반복적으로 잔가지를 제거하고 줄기를 보존 — [Wikipedia](https://en.wikipedia.org/wiki/Discrete_skeleton_evolution)
  - Python 구현 originlake/DSE-skeleton-pruning: **MIT** (2020, s.zhong) — [GitHub](https://github.com/originlake/DSE-skeleton-pruning), [LICENSE](https://github.com/originlake/DSE-skeleton-pruning/blob/master/LICENSE)
  - TopoStats 도 skeleton 가지치기 모듈을 갖고 있음 (라이선스 미확인) — [topostats.tracing.pruning](https://topostats.readthedocs.io/en/stable/_modules/topostats/tracing/pruning.html)
- **sknw** (skeleton → networkx 그래프): BSD-3-Clause — [Image-Py/sknw](https://github.com/Image-Py/sknw)
- **centerline** (다각형 중심선): MIT — [fitodic/centerline](https://github.com/fitodic/centerline) (Voronoi 기반이라는 설명은 미검증)
- **kimimaro** (TEASAR 뼈대화): **GPL-3.0** → 부적합 — [seung-lab/kimimaro](https://github.com/seung-lab/kimimaro)
- **scikit-image** (`skeletonize`, `medial_axis`, `find_contours`): 별 6,624, push 2026-10-23, 라이선스 BSD 계열(GitHub 는 NOASSERTION 표시, 미검증) — [scikit-image](https://github.com/scikit-image/scikit-image)
- **Monster Mash** (Google): "sketch a character, inflate it into 3D, and promptly animate it", **Apache-2.0 (src 폴더)**, 단 "third-party libraries may have more restrictive licenses" 이고 빌드 의존성에 **Triangle** 과 SDL 2.0 이 있음. Emscripten 으로 WebAssembly 빌드, 데모 monstermash.zone. 별 1,330, push 2026-03 — [google/monster-mash](https://github.com/google/monster-mash), [LICENSE](https://github.com/google/monster-mash/blob/main/LICENSE)
- **Sketch2Pose** (kbrodt/sketch2pose): LICENSE 파일 없음(=기본 저작권, 사용 불가로 간주), 마지막 push 2022-05 — [GitHub](https://github.com/kbrodt/sketch2pose)
- **AnimatedDrawings 내부**: 메시는 `skimage.measure.find_contours` → `approximate_polygon(tolerance=0.25)` 외곽 + 40×40 격자 내부점 → `scipy.spatial.Delaunay` → 중심이 윤곽 밖인 삼각형 제거 (`animated_drawing.py` `_generate_mesh`, 533~578행). 뼈대는 사람 관절 추정 모델 출력이며 저장소는 "pose estimation model assumes a human-like skeleton" 라고 명시 — 로컬 `AnimatedDrawings/animated_drawings/model/animated_drawing.py`, [README](https://github.com/facebookresearch/AnimatedDrawings)
- **Make-It-Animatable** (CVPR 2025, 3D 메시 자동 리깅): MIT (2025) — [jasongzy/Make-It-Animatable](https://github.com/jasongzy/Make-It-Animatable)
- 상용 쪽 참고: Cascadeur 2025.3 이 네발 짐승 Quick Rigging(Alpha) 추가 — 오픈소스 아님 — [CG Channel](https://www.cgchannel.com/2025/11/nekki-releases-cascadeur-2025-3/)

### Inferences
- 우리 `skeletonize` 뼈대에 **skan `Skeleton` + `summarize` 로 가지 길이 · 종류를 얻고, "분기-끝점" 이면서 짧은 가지를 `prune_paths` 로 제거**하는 단계를 넣으면 손떨림 잔가지 문제를 줄일 수 있다. DSE(MIT) 는 길이가 아닌 "모양 기여도" 기준이라 아이 그림의 굵기 변화에 더 강할 가능성이 있다. 둘 다 순수 Python 이라 수 초 이내 처리 예상 (미측정).
- Monster Mash 는 "그림 → 부풀린 3D → 애니메이션" 이라 결과가 아이 그림의 2D 느낌을 유지한다는 요구(c)와 일부 어긋나고, Triangle 의존성 때문에 그대로 상업 사용은 위험하다. 코드 참고(WASM 빌드 구조, 실시간 변형 UI) 용도로만 가치가 있다.
- AnimatedDrawings 의 메시 생성은 우리 방식과 사실상 같다(scipy Delaunay + 중심 필터). 여기서 더 얻을 것은 없다.

### Gaps
- RigMesh(2012), Pinocchio 2D 변형의 공개 코드 · 라이선스는 이번에 확인하지 못함.
- "모양만 보고 다리/꼬리/머리를 구분"하는 2D 전용 오픈소스(비휴머노이드 관절 이름 붙이기)는 찾지 못함.

## 3. 웹 런타임 (Pixi.js Mesh/MeshRope, Three.js 스키닝, Rive, DragonBones, Spine, Inochi2D, Paper.js)

### Takeaway
브라우저에서 아이 그림 텍스처를 메시로 움직이려면 **PixiJS v8 (MIT) `MeshSimple`/`Mesh`(정점 매 프레임 갱신) + 뱀·지렁이는 `MeshRope`** 가 가장 직접적이다. Spine 은 런타임도 Spine 에디터 라이선스가 사용자마다 필요해 부적합, Rive 는 런타임은 MIT 지만 메시·뼈를 에디터에서 만들어야 해 자동 파이프라인과 맞지 않는다. DragonBones 런타임(MIT)은 데이터 JSON 을 서버가 직접 만들 수 있다면 후보지만 유지보수가 약하다.

### Cited Findings
- **PixiJS**: MIT, 별 48,468, push 2026-10-22 — [pixijs/pixijs](https://github.com/pixijs/pixijs)
  - v8 메시 종류: `Mesh`(사용자 geometry/shader), `MeshSimple`(정점·UV·인덱스 배열, `autoUpdate = true` 면 `mesh.vertices` 매 프레임 수정 가능), `MeshRope`("Bends a texture along a series of control points, often used for trails, snakes"), `MeshPlane`(격자 왜곡), `PerspectiveMesh`. 문서에 skinning/bones 언급 없음 — [Pixi v8 Mesh 가이드](https://pixijs.com/8.x/guides/components/scene-objects/mesh)
- **pixi-spine / spine-v8**: `pixijs/spine-v8` 는 archived(2024-11), `pixijs-userland/spine` 은 루트 LICENSE 파일 없음 — [pixijs/spine-v8](https://github.com/pixijs/spine-v8), [pixijs-userland/spine](https://github.com/pixijs-userland/spine)
- **Spine Runtimes License** (2025-04-05판): "Each user of the Products must obtain their own Spine Editor license" → 상업 서비스에 쓰려면 라이선스 구매 필요 — [spine-runtimes LICENSE](https://github.com/EsotericSoftware/spine-runtimes/blob/master/LICENSE)
- **Rive**: `rive-runtime` · `rive-wasm` 모두 MIT, 각각 별 1,239 / 1,015, push 2026-10-24 — [rive-runtime](https://github.com/rive-app/rive-runtime), [rive-wasm](https://github.com/rive-app/rive-wasm)
  - 뼈 · 메시 문서는 에디터 작업만 다루고, Auto-Weights("nearest bone" 기반 자동 가중치)도 **에디터 기능**. 런타임에서 메시/뼈를 만드는 API 언급 없음 — [Rive Bones 문서](https://rive.app/docs/editor/manipulating-shapes/bones)
  - Rive Scripting 은 Luau 기반, "Scripts run in the Rive editor and export with your .riv file" — [Rive 블로그](https://rive.app/blog/why-scripting-runs-on-luau), [rive-scripting skill 설명](https://skills.sh/stevysmith/rive-skills/rive-scripting)
- **DragonBonesJS**: MIT, 별 857, push 2026-01-23, Egret · PixiJS · Phaser · Hilo · Cocos Creator · Three.js 폴더 지원. 메시/FFD · 에디터 오픈소스 여부는 README 에서 확인 안 됨 — [DragonBones/DragonBonesJS](https://github.com/DragonBones/DragonBonesJS)
- **three.js**: MIT, 별 116,929, push 2026-10-25 — [mrdoob/three.js](https://github.com/mrdoob/three.js); `SkinnedMesh`/`Skeleton`/`Bone` 으로 GPU 스키닝 — [three.js SkinnedMesh 문서](https://threejs.org/docs/#api/en/objects/SkinnedMesh)
- **Inochi2D** (오픈소스 Live2D 대체): BSD-2-Clause, 별 1,827, push 2026-10-03 — [Inochi2D/inochi2d](https://github.com/Inochi2D/inochi2d). Rust 포트 **inox2d**: BSD-2 계열, "experimental", OpenGL + **WASM(WebGL)** 동작, Deforms(메시 정점 오프셋) · Physics 지원, Mesh groups 미지원 — [Inochi2D/inox2d](https://github.com/Inochi2D/inox2d)
- **Paper.js**: MIT (LICENSE.txt), 별 15,104 — [paperjs/paper.js](https://github.com/paperjs/paper.js). 벡터 경로 라이브러리로 래스터 텍스처 메시 변형 기능은 아님(미검증).
- **Creature** (Kestrel Moon) 런타임 예: Creature_Cocos2DX Apache-2.0 이지만 에디터는 상용 — [kestrelm/Creature_Cocos2DX](https://github.com/kestrelm/Creature_Cocos2DX)

### Inferences
- **추천 구조**: 서버(Python)가 `{vertices, uvs, triangles, bones, weights}` 또는 `{프레임별 정점 좌표}` JSON 을 만들고, 프론트는 Pixi v8 `MeshSimple` 로 아이 그림 텍스처를 그 메시에 입혀 재생. 프레임별 정점 좌표 방식은 지금 ARAP 결과를 그대로 쓸 수 있고(정점 수백 개 × 프레임 수십 개 = 수십~수백 KB), BBW+LBS 방식은 뼈 변환만 보내면 되어 더 작다.
- 뱀·지렁이 "꿈틀" 은 Pixi `MeshRope` 로 몸통 축(우리 skeleton 중심선)을 제어점으로 주면 서버 계산 없이 브라우저에서 바로 만들 수 있다. 다만 MeshRope 는 텍스처를 띠 모양으로 펴서 쓰므로, 굽은 그림은 먼저 축을 따라 펴서(straighten) 텍스처를 만들어야 한다 → 우리 쪽 전처리 필요.
- Spine 포맷을 "생성"해서 쓰는 방식은 런타임 라이선스 때문에 배제. DragonBones JSON 생성은 기술적으로 가능하겠지만 Pixi v8 호환 여부 · 유지보수 상태가 불확실하므로 자체 JSON + Pixi Mesh 가 더 단순하다.
- 모바일 성능: Pixi/three.js 모두 정점 수백~수천 개 메시 1~3개는 저가 기기에서도 부담이 적은 규모로 판단 (실측 자료는 못 찾음).

### Gaps
- Pixi v8 · three.js 메시 변형의 저가 모바일 기기 실측 벤치마크는 찾지 못함.
- DragonBonesJS 가 Pixi v8 을 지원하는지, 런타임이 weighted mesh 를 지원하는지 README 로 확인 못 함.
- Rive Scripting 이 런타임에 메시 정점을 바꿀 수 있는지 공식 문서로 확인 못 함.

## 4. 마스크에서 메시 생성 (Triangle, meshpy, CGAL, earcut, poly2tri, cdt2d, CDT, delaunator) 과 라이선스

### Takeaway
지금 쓰는 **scipy Delaunay + 중심 필터(= AnimatedDrawings 방식)** 는 라이선스상 깨끗하다. 품질(경계 구속, 가는 삼각형 방지)을 올리고 싶으면 **artem-ogre/CDT + PythonCDT (MPL-2.0, 2026-10 활발)** 가 가장 맞다. Shewchuk Triangle 과 이를 감싼 `triangle`, `meshpy`, `igl.triangle`, Monster Mash 의 Triangle 의존성은 상업 서비스에서 피해야 한다.

### Cited Findings
- **Triangle (Shewchuk)**: "Although Triangle is freely available, it is copyrighted by the author and may not be sold or included in commercial products without a license." — [Triangle 홈페이지](https://www.cs.cmu.edu/~quake/triangle.html)
- **meshpy**: LICENSE 파일 안에 Triangle 조항("part of a commercial system is permissible ONLY BY DIRECT ARRANGEMENT")과 TetGen **AGPL v3**(또는 유료 상용) 조항이 같이 들어 있음. 별 591, push 2026-10-19 — [inducer/meshpy LICENSE](https://github.com/inducer/meshpy/blob/main/LICENSE)
- **drufat/triangle** (pip `triangle`): 래퍼 LGPL-3.0, 별 265, push 2025-01. 내부에 Shewchuk Triangle C 코드를 포함하는 것으로 추정(미검증) — [drufat/triangle](https://github.com/drufat/triangle)
- **CGAL**: GitHub 라이선스 NOASSERTION. 일반적으로 GPL/LGPL + 상용 이중 라이선스로 알려져 있음(이번에 LICENSE 미확인) — [CGAL/cgal](https://github.com/CGAL/cgal)
- **artem-ogre/CDT** (C++ constrained Delaunay, header-only): **MPL-2.0**, 별 1,456, push 2026-10-25. Python 바인딩 **PythonCDT**: MPL-2.0, 별 68, push 2026-10-13 — [CDT](https://github.com/artem-ogre/CDT), [PythonCDT](https://github.com/artem-ogre/PythonCDT)
- **mapbox/earcut** (JS 다각형 삼각분할): ISC, 별 2,623, push 2026-10-23 — [mapbox/earcut](https://github.com/mapbox/earcut). Python 바인딩 mapbox_earcut_python: 바인딩 ISC, earcut 본체 라이선스는 include 폴더 참조 — [LICENSE.md](https://github.com/skogler/mapbox_earcut_python/blob/master/LICENSE.md)
- **mapbox/delaunator** (JS Delaunay): ISC, 별 2,636, push 2026-06 — [mapbox/delaunator](https://github.com/mapbox/delaunator)
- **poly2tri.js** (JS CDT): BSD-3 (LICENSE.txt), 마지막 push 2018-10 (정체) — [r3mi/poly2tri.js](https://github.com/r3mi/poly2tri.js)
- **cdt2d** (JS CDT): MIT, 마지막 push 2019-07 (정체) — [mikolalysenko/cdt2d](https://github.com/mikolalysenko/cdt2d)
- **shapely** (점-내부 판정, 다각형 연산): BSD-3, push 2026-10 — [shapely/shapely](https://github.com/shapely/shapely)
- **Open3D**: 라이선스 NOASSERTION(MIT 로 알려짐, 미검증), 2D 메시 생성기는 아님 — [isl-org/Open3D](https://github.com/isl-org/Open3D)
- AnimatedDrawings 와 우리 `shape_rig.py` 모두 `scipy.spatial.Delaunay` 사용 — 로컬 `AnimatedDrawings/animated_drawings/model/animated_drawing.py:21,567`, `experiments/animal-monster/shape_rig.py:20`

### Inferences
- scipy Delaunay + 중심 필터의 약점은 (1) 오목한 곳에서 경계 간선이 보장되지 않아 다리 사이가 "붙는" 삼각형이 생길 수 있음, (2) 격자 내부점이라 삼각형 크기 조절이 거침. 다리 사이가 붙어 같이 움직이는 문제를 본 적이 있다면 **PythonCDT 로 외곽선을 구속 간선으로 넣는 것**이 직접적인 해결책이다.
- 브라우저에서 메시를 만들 필요가 생기면 earcut(ISC)은 내부점 없이 외곽선만 삼각분할하므로(미검증) 변형용으로는 가늘고 긴 삼각형이 많아 부적합할 수 있고, delaunator + 내부점 + 중심 필터(우리 서버 방식의 JS 판) 가 낫다.
- 메시 생성은 서버에 두는 것이 단순하다(한 번만 계산).

### Gaps
- PythonCDT 의 pip 휠 제공 여부 · Windows/Linux 설치 난이도는 확인 못 함.
- earcut 이 Steiner 점을 넣지 않는다는 점은 README 로 직접 확인하지 않음.

## 5. 팔다리 없는 모양의 2차 움직임 · 물리 (soft-body, PBD, shape matching, squash & stretch, 숨쉬기)

### Takeaway
덩어리 · 별 · 구름에는 **shape matching (Müller 2005)** 이나 스프링-질량을 메시 정점에 직접 거는 방식이 가장 잘 맞고, 구현이 짧아 자체 구현이 현실적이다. Matter.js · p2.js · Box2D 는 모두 MIT 이지만 강체 엔진이라 텍스처 메시를 말랑하게 만드는 데는 직접 맞지 않는다.

### Cited Findings
- **Meshless Deformations Based on Shape Matching** (Müller, Heidelberger, Teschner, Gross, SIGGRAPH 2005): 에너지 대신 "현재 위치 → 목표 위치(휴지 상태와의 최적 강체/선형 맞춤)" 로 당겨 무조건 안정, 연결 정보 불필요 — [논문 PDF](https://cg.informatik.uni-freiburg.de/publications/2005_SIGGRAPH_deformationShapeMatching.pdf), [SOFA ShapeMatching 플러그인 설명](https://www.sofa-framework.org/applications/plugins/shape-matching/)
- **Matter.js**: MIT, 별 18,492, push 2026-09-30 — [liabru/matter-js](https://github.com/liabru/matter-js)
- **p2.js**: MIT — [schteppe/p2.js](https://github.com/schteppe/p2.js) / **Box2D**: MIT (2022, Erin Catto) — [erincatto/box2d](https://github.com/erincatto/box2d)
- **inox2d**: 런타임에 Physics(Inochi2D 의 진자/스프링식 2차 움직임) 지원 표기 — [Inochi2D/inox2d](https://github.com/Inochi2D/inox2d)
- Ten Minute Physics(Matthias Müller) 예제 저장소 matthias-research/pages 는 main/master 루트에 LICENSE 파일 없음 → 코드 복사 불가로 간주 — [matthias-research/pages](https://github.com/matthias-research/pages)
- GitHub 검색 "soft body 2d javascript" 결과는 사실상 없음 (별 0 저장소 1개) — [GitHub 검색](https://github.com/search?q=soft+body+2d+javascript&type=repositories)

### Inferences
- 우리 ARAP2D 에 이미 "핸들 위치 → 메시" 가 있으므로, 2차 움직임은 **핸들(또는 몇 개 제어점)에 감쇠 스프링을 걸고 ARAP/MLS 로 메시를 따라오게** 하는 것이 가장 적은 추가 코드다 (예: 몸통 중심이 튀면 머리 · 귀 핸들이 늦게 따라옴).
- 순수 절차 움직임(숨쉬기 = 세로 스케일 sin, 흔들기 = 바닥 고정 회전, squash & stretch = 부피 보존 sx·sy=1)은 브라우저에서 메시 정점 변환만으로 충분하며 라이브러리가 필요 없다.
- shape matching 은 정점 수백 개 규모면 JS 로 매 프레임 계산 가능 (2×2 극분해만 필요) — 직접 구현 권장. 공개 2D JS 구현 중 라이선스가 확실한 것은 찾지 못했다.

### Gaps
- 라이선스가 확실한 2D shape-matching / PBD JS 구현을 찾지 못함.
- 아이 그림 텍스처에 soft-body 를 건 사례 · 데모 영상은 찾지 못함.

## 6. "아무 낙서나 자동으로 움직이기" 를 이미 하는 GitHub 프로젝트

### Takeaway
2026-10 기준으로도 **AnimatedDrawings(MIT, 2025-09 보관 처리)** 가 사실상 유일한 성숙한 오픈소스이고, 그마저 사람형 뼈대 전제다. 최근 연구(DrawingSpinUp, AniClipart)는 라이선스 파일이 없거나 3D · GPU 무거운 방식이라 우리 요구(상업, 수십 초, 2D 유지)에 바로 쓰기 어렵다. 즉 비휴머노이드 자동화는 우리가 만든 쪽이 앞서 있고, 위 구성요소로 "부분 교체" 하는 전략이 맞다.

### Cited Findings
- **facebookresearch/AnimatedDrawings**: MIT (코드 · 모델 가중치 · Amateur Drawings 데이터셋), ChildlikeSHAPES 데이터셋 CC-BY 4.0. "archived by the owner on Sep 3, 2025. It is now read-only." 네발 예제는 있으나 "primarily designed with human-like characters in mind". 별 12,808 — [GitHub](https://github.com/facebookresearch/AnimatedDrawings)
- **DrawingSpinUp** (SIGGRAPH Asia 2024, 그림 → 3D 캐릭터): LICENSE 파일 없음, RTX 4090(2080 Ti 도 가능) 사용, 리깅은 **Mixamo**(외부 서비스) 로 수동 단계 — [LordLiang/DrawingSpinUp](https://github.com/LordLiang/DrawingSpinUp)
- **AniClipart** (클립아트 애니메이션): LICENSE 파일 없음. 전처리에 keypoint 검출 + 뼈대 복잡도 파라미터(`epsilon`, `max_iter`, `factor`) — [kingnobro/AniClipart](https://github.com/kingnobro/AniClipart)
- GitHub 검색 "animate children drawing" 상위: AnimatedDrawings 와 그 포크, 학생 프로젝트(snuhcs-course/swpp-2023-project-team-9), `Saber5656/inkstory`("children's scanned drawings into animated picture books", 별 0), `PrinceMridul/Once-Upon-A-Doodle`(별 0) — 모두 라이선스 표기 없음 — [GitHub 검색](https://github.com/search?q=animate+children+drawing&type=repositories)
- GitHub 검색 "doodle animation rig" 결과 0건 — [GitHub 검색](https://github.com/search?q=doodle+animation+rig&type=repositories)

### Inferences
- 우리 파이프라인을 통째로 바꿀 오픈소스는 없다. 개선 우선순위(효과 대비 비용 순):
  1. **skan/DSE 로 뼈대 가지치기** (BSD-3/MIT, 순수 Python, 바로 붙음).
  2. **PythonCDT 로 경계 구속 메시** (MPL-2.0) — 다리 사이 붙음 문제가 있을 때.
  3. **Pixi v8 MeshSimple/MeshRope 로 브라우저 재생** — 서버는 메시+정점 프레임(또는 뼈+가중치) JSON 만 보내기. 서버 비용 · 대기 시간 감소.
  4. **libigl `bbw` + LBS 실험** — 브라우저 GPU 스키닝으로 가는 길. libigl 휠의 Triangle/copyleft 모듈 처리 방침 결정 필요.
  5. **MLS (Jarvis73, MIT)** — 별 · 구름 · 덩어리처럼 뼈대가 의미 없는 모양용.
  6. **shape matching / 스프링 2차 움직임 자체 구현** — 젤리 느낌.
- 피해야 할 것: Triangle 계열(`triangle`, `meshpy`, `igl.triangle`, Monster Mash 빌드), Spine 런타임, GPL(kimimaro, puppet-warp), 라이선스 없는 연구 코드(DrawingSpinUp, AniClipart, Sketch2Pose, Ten Minute Physics).

### Gaps
- 2025~2026 년 발표된 "2D 비휴머노이드 낙서 자동 애니메이션" 논문 중 상업 허용 코드가 있는 것은 찾지 못함 (검색 범위가 GitHub · 일반 웹에 한정, arXiv 전수 조사는 안 함).
- 저장소 별 · 날짜 일부(puppet-warp, DSE, sknw 등)는 GitHub API 요청 한도 초과로 메타데이터를 다시 확인하지 못했고, LICENSE 파일만 확인함.
