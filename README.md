# 휴머노이드 조선 소조립 정형 용접 시뮬레이션 (MuJoCo)

PPT 13·16쪽의 "휴머노이드 소조립 정형 용접"이 휴머노이드만 겪는 오차(서는 위치 오차, 몸통 흔들림)가 있어도
용접 허용 오차를 지킬 수 있는지, 지키려면 어떤 보정이 필요한지 확인하는 시뮬레이션입니다.

## 파일

| 파일 | 역할 |
|---|---|
| `build_scene.py` | Unitree G1 모델(mujoco_menagerie)에 용접 토치를 붙이고 작업대·T형 필렛 부재 장면을 만든다 |
| `sim_core.py` | 역기구학(IK), 터치 센싱 보정, 몸통 흔들림(병진 + 회전), 4개 조건(R·H0·H1·H2) 시뮬레이션 |
| `run_study.py` | 몬테카를로 반복 실행 → `results/`에 CSV·그래프·사용한 변수(`params.txt`) 저장 |
| `make_video.py` | 3장면 데모 영상(`results/demo.mp4`) |
| `results/` | 아래 "실행 환경"에서 실제로 돌린 v2 결과 (요약은 `results/RESULTS.md`) |
| `results_rot0/` | 같은 코드에서 회전 흔들림만 끈 비교 실험 (`--rot-ratio 0`) |
| `results_v1_translation_only/` | v1 결과 보존용 (병진 흔들림만, 2점 터치, 백색 추정 잡음) |

## 맥(Apple Silicon)에서 실행

```bash
python3 -m venv ~/weld-sim && source ~/weld-sim/bin/activate
pip install mujoco numpy matplotlib pillow imageio-ffmpeg
cd weld_demo
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
python build_scene.py                 # weld_scene.xml 생성
python run_study.py --trials 30 --fine-trials 30   # 빠른 확인 (약 15초)
python run_study.py                   # 본 실험 (기본 300회, 10코어 약 2분)
python run_study.py --rot-ratio 0 --out results_rot0   # 회전 흔들림을 끈 비교 실험
python make_video.py                  # 데모 영상 (약 45초)
```

장면을 직접 돌려 보려면: `mjpython -m mujoco.viewer --mjcf=mujoco_menagerie/unitree_g1/weld_scene.xml`

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
| 터치 점 | ① 용접선 시작·끝에서 20 mm 안쪽 2곳, 점마다 세로판 면 + 바닥판 면 탐색 → 용접선의 위치·방향 ② 판 끝면 1회 탐색 → 용접선 방향 위치 | FANUC/ABB 시작점 탐색 방식 |
| 터치 탐색 시작 거리 | 믿는 면에서 80 mm | 서는 위치 50 mm + 부재 10 mm 오차보다 크게 |
| 실시간 보상(H2) | 몸통 자세 추정 지연 20 ms, 잡음 0.1 mm · 0.3 mrad (1σ), 2 Hz 1차 저역통과 | 가정 |
| 용접 속도 | 6 mm/s, 용접선 220 mm (양 끝 10 mm 제외) | 가정 |
| 허용 오차 | 0.5 / 1.0 / 1.5 mm | 단일 출처 없음 → 세 기준 모두 보고 |

## 측정

용접 중 0.05 s마다 토치 끝과 **실제** 용접선 사이의 거리를 계산하고, 한 번의 용접에서 가장 큰 값(최대 이탈)이
허용 오차 이하이면 성공으로 셉니다. IK가 수렴하지 않거나 관절 한계에 걸리거나 터치 탐색이 판 안에서 시작하면 실패로 셉니다.
용접 시작점이 용접선 방향으로 얼마나 밀렸는지(`along_start_mm`)도 기록하지만, 허용값 출처가 없어 성공 판정에는 넣지 않습니다.

## 단순화 (논문에 한계로 명시)

- 다리로 균형 잡는 동작은 시뮬레이션하지 않고, 그 결과인 몸통 흔들림(병진 + 회전)을 골반에 주입한다.
- 회전/병진 비율은 실기 측정 한 점(TOP 세트 ℳ)에서 가져온 값이다. 같은 논문의 큰 동작 세트 𝒯에서는 비율이 훨씬 작으므로(0.0007 rad/mm) 큰 흔들림에서 회전을 비례시킨 것은 외삽이다.
- 아크·열·비드 같은 용접 물리는 다루지 않고 토치 경로 정확도만 평가한다.
- 팔은 역기구학으로 목표 자세에 정확히 도달한다고 가정한다 (순수 기구학 시뮬레이션, 모터 추종 오차 미포함).
- 터치 탐색은 직선과 평면의 교점으로 계산하며 판의 유한한 크기와 토치 끝 반지름은 고려하지 않는다.

## v2에서 바뀐 점 (v1 결과는 `results_v1_translation_only/`)

1. 몸통 흔들림에 회전(roll·pitch·yaw)을 추가했다. v1은 병진만 넣어 토치 오차가 흔들림과 1:1이었는데, 골반–토치 거리 348 mm 때문에 실제로는 회전이 더 크게 작용한다.
2. 판 끝면 터치를 추가해 용접선 방향 위치도 보정한다. v1은 이 방향을 도면값으로 두어 용접 시작점이 평균 20 mm, 최대 약 55 mm 밀렸고, 성공 판정(용접선까지의 수직 거리)에 잡히지 않았다.
3. 터치 탐색 시작 거리를 40 → 80 mm로 늘렸다. v1은 5%의 시행에서 탐색이 판 안에서 시작했다.
4. H2 추정 잡음을 20 Hz 백색잡음에서 2 Hz 저역통과 유색잡음으로 바꾸고 회전 잡음을 추가했다.
5. 본 실험 흔들림 단계에 0.5 mm를 넣고, 세분 실험을 0~3 mm 11단계로 바꿨다.
6. TOP 논문 인용을 원문 Table IV 값으로 정정했다(아래 "인용").

## 결과 (요약은 `results/RESULTS.md`)

허용 오차 1.0 mm, 조건별 300회:

| 몸통 흔들림 | R 고정형 | H0 보정 없음 | H1 터치 보정(현재) | H2 + 실시간 보상 |
|---|---|---|---|---|
| 0 mm | 100% | 0% | 100% | 97.0% |
| 0.5 mm | — | 0% | 0% | 90.7% |
| 1 mm | — | 0% | 0% | 67.0% |
| 2 mm | — | 0% | 0% | 6.0% |
| 3 mm | — | 0% | 0% | 0% |
| 5 mm | — | 0% | 0% | 0% |

현재 기술(H1)로 성공률 95% 이상을 유지하는 몸통 흔들림(병진 피크) 한계: 허용 1.0 mm → 0.1 mm, 허용 1.5 mm → 0.2 mm.
회전을 빼면(`results_rot0/`) 각각 0.5 mm, 0.75 mm.

## 인용

- Z. Chen, H. Xu, H. Zhang, L. Zhang, H. Li, D. Wang, J. Yu, Y. Yang, Z. Zhou, R. Xiong, "TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation with Humanoid Robots", arXiv:2508.00355, 2025 (Zhejiang Univ.).
  사용한 값: Table IV, 실기 모션캡처, 작은 동작 세트 ℳ, 20개 동작 평균 — 몸통(base) 오차 x/y/z 1.654/0.461/0.123 mm, roll/pitch/yaw 0.0091/0.0114/0.0045 rad; 손끝(EE) x 오차 5.068 mm.
  주의: 논문에 "1.7 mm", "5 mm"라는 표현은 없다. 로봇은 G1이 아니라 1.65 m·60 kg·41자유도 기종이다. 큰 동작 세트 𝒯에서는 몸통 x 오차가 29.2 mm다. 흔들림 주파수는 보고되지 않았다.
- ABB SmarTac 사양서 — 터치 센싱 정확도 ±0.25 mm.
- FANUC 터치 센싱 T형 필렛 설정 — 시작·끝 2점 탐색.
- HD현대삼호 협동로봇 용접 사례(Rainbow Robotics) — 터치 센싱 + 아크 센싱 보정.
- KOBELCO, Robotic Welding System for Shipbuilding (2018) — 3D CAD에서 용접선 자동 추출.

## 검증 실험

v2 결과의 가정 민감도와 통계 신뢰도를 `results_validation/`에서 확인했다. 정리는 `results_validation/VALIDATION.md`에 있다.

```bash
python run_validation.py     # H2 지연 스윕, 회전 비율, 흔들림 주파수, seed 2027 재실행 + 표 (10코어 약 15분)
python validate_top.py       # TOP Table IV 몸통 오차 → G1 토치 변위 (model_check.txt)
```

요약: H1(현재 기술)이 실기 수준 흔들림에서 실패한다는 결론은 모든 가정에서 유지되지만, 한계값은 회전 비율에 따라 0.1~0.6 mm(허용 1.0 mm)다.
H2는 흔들림 1.5 mm·허용 1.5 mm에서 추정 지연 15 ms 이하가 필요하다. 이 값은 흔들림 주파수와 회전 비율 가정에 좌우된다.

## 이 결과를 만든 환경

macOS(Apple Silicon, Python 3.14, `pip install mujoco` → MuJoCo 3.17.0, numpy 2.5)에서 10코어로 계산했습니다.
본 실험 7,900회 용접에 약 2분, 영상 약 45초. 같은 seed(2026)면 같은 플랫폼에서는 수치가 완전히 같고,
다른 플랫폼에서는 부동소수점 차이로 소수점 아래가 미세하게 다를 수 있습니다.
화면 없는 Linux 서버(KW-VIP)에서 영상을 만들 때는 `MUJOCO_GL=egl python make_video.py`.
