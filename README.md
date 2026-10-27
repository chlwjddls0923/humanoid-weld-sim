# humanoid-weld-sim — 휴머노이드 조선 소조립 정형 용접 시뮬레이션 (MuJoCo)

한국생산제조학회(KSMTE) 2026 추계학술대회 특별세션 **P06** 발표용 코드와 결과입니다.
휴머노이드가 조선 소조립 T형 필렛 용접을 할 때, 휴머노이드만 겪는 오차(서는 위치 오차, 몸통 흔들림)가 있어도
용접 허용 오차를 지킬 수 있는지, 지키려면 어떤 보정이 필요한지를 Unitree G1 모델로 확인합니다.

- 결과 요약: [`results/RESULTS.md`](results/RESULTS.md) (본 결과) → [`results_validation/VALIDATION.md`](results_validation/VALIDATION.md) (가정 민감도) → [`results_validation/FINAL.md`](results_validation/FINAL.md) (논문용 최종 숫자)
- 초록 보완 실험: `results_validation/abstract_supplement/SUPPLEMENT.md` (용접선 변형, 토치 각도, 센서급)

## 재현 방법

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                       # mujoco, numpy, matplotlib, pillow, imageio-ffmpeg
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git   # G1 모델 (아래 "모델 출처")
python build_scene.py                                 # mujoco_menagerie/unitree_g1/weld_scene.xml 생성

python run_study.py                                   # 1) 본 실험 → results/            (10코어 약 2분)
python run_study.py --rot-ratio 0 --out results_rot0  #    회전 흔들림을 끈 비교
python run_validation.py                              # 2) 검증 스윕 → results_validation/ (약 15분)
python validate_top.py                                #    TOP Table IV 대조 → model_check.txt
python run_final.py                                   # 3) 지연×주파수, 잡음 크기 → FINAL 표 (약 8분)
python run_supplement.py                              # 4) 초록 보완 실험 → abstract_supplement/
python make_video.py                                  #    데모 영상 results/demo.mp4
```

빠른 확인: `python run_study.py --trials 30 --fine-trials 30` (약 15초). 모든 스크립트는 seed 2026이 기본이며, 결과 폴더가 이미 있으면 건너뜁니다.
장면을 직접 보려면 `mjpython -m mujoco.viewer --mjcf=mujoco_menagerie/unitree_g1/weld_scene.xml`.
화면 없는 Linux 서버에서 영상을 만들 때는 `MUJOCO_GL=egl python make_video.py`.

## 파일

| 파일 | 역할 |
|---|---|
| `build_scene.py` | G1 모델에 용접 토치를 붙이고 작업대·T형 필렛 부재 장면을 만든다 (`--seam-length`, `--part-yaw`로 변형 장면) |
| `sim_core.py` | 역기구학(IK), 터치 센싱 보정(2점 + 판 끝면), 몸통 흔들림(병진 + 회전), 4개 조건(R·H0·H1·H2) 시뮬레이션 |
| `run_study.py` | 몬테카를로 실행 → CSV·그래프·`params.txt`. 옵션: `--rot-ratio`, `--two-point-touch`, `--conds`, `--comp-latency`(ms), `--sway-freq`, `--comp-noise-scale`, `--scene` |
| `run_validation.py` / `run_final.py` / `run_supplement.py` | 검증·최종·초록 보완 실험을 순서대로 돌리고 표를 만든다 |
| `validate_top.py` | TOP 논문 Table IV의 몸통 오차를 G1 골반에 넣었을 때의 토치 변위 |
| `make_video.py` | 3장면 데모 영상 |
| `results/` | v2 본 결과 (300회 × 조건, seed 2026) |
| `results_rot0/` | 같은 코드에서 회전 흔들림만 끈 비교 |
| `results_v1_translation_only/` | v1 결과 보존용 (병진만, 2점 터치, 백색 추정 잡음) |
| `results_validation/` | 검증·최종·초록 보완 실험 결과와 정리 문서 |

## 조건

| 조건 | 서는 위치 오차 | 몸통 흔들림 | 출발 전 터치 보정 | 실시간 몸통 보상 |
|---|---|---|---|---|
| R 고정형 로봇 | 없음 | 없음 | O | X |
| H0 휴머노이드 · 보정 없음 | O | O | X | X |
| H1 휴머노이드 · 터치 보정 (현재 기술) | O | O | O | X |
| H2 휴머노이드 · 터치 + 실시간 보상 (개발 과제) | O | O | O | O |

부재 위치 오차는 모든 조건에 들어갑니다.

## 변수 (`sim_core.Params`)

| 변수 | 값 | 근거 |
|---|---|---|
| 몸통 흔들림 (병진 최대 변위) | 0, 0.5, 1, 2, 3, 5 mm (세분 실험 0~3 mm 11단계) | 실기 휴머노이드가 서서 작업할 때 몸통(base) x축 오차 1.654 mm(Chen·Xu et al. 2025, Table IV, 세트 ℳ) 주변을 덮도록 |
| 몸통 흔들림 (회전) | 병진 피크 1 mm당 0.0089 rad, roll:pitch:yaw = 0.0091:0.0114:0.0045 | 같은 표의 몸통 회전/병진 비율(\|rot\| 0.0153 rad / \|trans\| 1.72 mm). `--rot-ratio 0`이면 병진만 |
| 흔들림 주파수 | 0.3~2 Hz 사인 3개 합, 축별 무작위 위상 | 가정 (논문에 주파수 정보 없음) |
| 서는 위치 오차 | 0~50 mm, 방향 ±5° | 가정 |
| 부재 위치 오차 | 0~10 mm, ±3° | 가정 |
| 터치 측정 오차 | ±0.25 mm | ABB SmarTac 사양 |
| 터치 점 | ① 용접선 시작·끝에서 20 mm 안쪽 2곳, 점마다 세로판 면 + 바닥판 면 탐색 ② 판 끝면 1회 탐색 → 용접선 방향 위치 | FANUC/ABB 시작점 탐색 방식 |
| 터치 탐색 시작 거리 | 믿는 면에서 80 mm | 서는 위치 50 mm + 부재 10 mm 오차보다 크게 |
| 실시간 보상(H2) | 몸통 자세 추정 지연 20 ms, 잡음 0.1 mm · 0.3 mrad (1σ), 2 Hz 1차 저역통과 | 가정 |
| 용접 속도 | 6 mm/s, 용접선 220 mm (양 끝 10 mm 제외) | 가정 |
| 허용 오차 | 0.5 / 1.0 / 1.5 mm | 단일 출처 없음 → 세 기준 모두 보고 |

## 측정

용접 중 0.05 s마다 토치 끝과 **실제** 용접선 사이의 거리를 계산하고, 한 번의 용접에서 가장 큰 값(최대 이탈)이
허용 오차 이하이면 성공으로 셉니다. IK가 수렴하지 않거나 관절 한계에 걸리거나 터치 탐색이 판 안에서 시작하면 실패로 셉니다.
용접 시작점의 용접선 방향 오차(`along_start_mm`), 토치 축 각도 오차(`ang_max_deg`, 작업각·진행각 성분), 부호 있는 가로 오프셋도 기록합니다.

## 단순화 (논문에 한계로 명시)

- 다리로 균형 잡는 동작은 시뮬레이션하지 않고, 그 결과인 몸통 흔들림(병진 + 회전)을 골반에 주입한다.
- 회전/병진 비율은 실기 측정 한 점(TOP 세트 ℳ)에서 가져온 값이다. 같은 논문의 큰 동작 세트 𝒯에서는 비율이 훨씬 작으므로(0.0007 rad/mm) 큰 흔들림에서 회전을 비례시킨 것은 외삽이다.
- 아크·열·비드 같은 용접 물리는 다루지 않고 토치 경로 정확도만 평가한다.
- 팔은 역기구학으로 목표 자세에 정확히 도달한다고 가정한다 (순수 기구학 시뮬레이션, 모터 추종 오차 미포함).
- 터치 탐색은 직선과 평면의 교점으로 계산하며 판의 유한한 크기와 토치 끝 반지름은 고려하지 않는다.

## 주요 결과 (허용 오차 1.0 mm, 조건별 300회)

| 몸통 흔들림 | R 고정형 | H0 보정 없음 | H1 터치 보정(현재) | H2 + 실시간 보상 |
|---|---|---|---|---|
| 0 mm | 100% | 0% | 100% | 97.0% |
| 0.5 mm | — | 0% | 0% | 90.7% |
| 1 mm | — | 0% | 0% | 67.0% |
| 2 mm | — | 0% | 0% | 6.0% |
| 3 mm | — | 0% | 0% | 0% |
| 5 mm | — | 0% | 0% | 0% |

- 현재 기술(H1)이 성공률 95%를 유지하는 몸통 흔들림 한계: 허용 1.0 mm 기준 0.1~0.6 mm(회전/병진 비율 0~0.0089 rad/mm에 따라). 실기 수준 1.5 mm에서는 가정을 바꿔도 0%.
- 실기 수준 흔들림에서 허용 1.5 mm를 지키려면 실시간 보상의 추정 지연이 15 ms 이하(0.3~2 Hz; 저주파 30 ms, 고주파 12 ms). 허용 1.0 mm는 지연 10 ms 이하와 추정 잡음 0.05 mm / 0.15 mrad 이하가 함께 필요.
- 자세한 수치와 신뢰구간은 `results_validation/FINAL.md`.

## 인용

- Z. Chen, H. Xu, H. Zhang, L. Zhang, H. Li, D. Wang, J. Yu, Y. Yang, Z. Zhou, R. Xiong, "TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation with Humanoid Robots", arXiv:2508.00355, 2025 (Zhejiang Univ.).
  사용한 값: Table IV, 실기 모션캡처, 작은 동작 세트 ℳ, 20개 동작 평균 — 몸통(base) 오차 x/y/z 1.654/0.461/0.123 mm, roll/pitch/yaw 0.0091/0.0114/0.0045 rad; 손끝(EE) x 오차 5.068 mm.
  주의: 논문에 "1.7 mm", "5 mm"라는 표현은 없다. 로봇은 G1이 아니라 1.65 m·60 kg·41자유도 기종이다. 큰 동작 세트 𝒯에서는 몸통 x 오차가 29.2 mm다. 흔들림 주파수는 보고되지 않았다.
- ABB SmarTac 사양서 — 터치 센싱 정확도 ±0.25 mm.
- FANUC 터치 센싱 T형 필렛 설정 — 시작·끝 2점 탐색.
- HD현대삼호 협동로봇 용접 사례(Rainbow Robotics) — 터치 센싱 + 아크 센싱 보정.
- KOBELCO, Robotic Welding System for Shipbuilding (2018) — 3D CAD에서 용접선 자동 추출.

## 모델 출처와 라이선스

- Unitree G1 모델: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) `unitree_g1` (BSD-3-Clause, 원 모델은 Unitree Robotics 제공). 이 저장소에는 포함하지 않으며 `build_scene.py`가 클론한 폴더 안에 토치가 붙은 `g1_weld.xml`과 장면 파일을 생성합니다.
- 이 저장소의 코드와 결과: MIT License (`LICENSE`).

## 이 결과를 만든 환경

macOS(Apple Silicon, Python 3.14, `pip install mujoco` → MuJoCo 3.17.0, numpy 2.5)에서 10코어로 계산했습니다.
같은 seed(2026)면 같은 플랫폼에서는 수치가 완전히 같고, 다른 플랫폼에서는 부동소수점 차이로 소수점 아래가 미세하게 다를 수 있습니다.
전체 실행량(v2 + 검증 + 최종)은 98,800회 용접, 약 26분입니다.
