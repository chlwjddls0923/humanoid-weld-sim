# 휴머노이드 조선 소조립 정형 용접 시뮬레이션 (MuJoCo)

PPT 13·16쪽의 "휴머노이드 소조립 정형 용접"이 휴머노이드만 겪는 오차(서는 위치 오차, 몸통 흔들림)가 있어도
용접 허용 오차를 지킬 수 있는지, 지키려면 어떤 보정이 필요한지 확인하는 시뮬레이션입니다.

## 파일

| 파일 | 역할 |
|---|---|
| `build_scene.py` | Unitree G1 모델(mujoco_menagerie)에 용접 토치를 붙이고 작업대·T형 필렛 부재 장면을 만든다 |
| `sim_core.py` | 역기구학(IK), 터치 센싱 보정, 몸통 흔들림, 4개 조건(R·H0·H1·H2) 시뮬레이션 |
| `run_study.py` | 몬테카를로 반복 실행 → `results/`에 CSV와 그래프 저장 |
| `make_video.py` | 3장면 데모 영상(`results/demo.mp4`) |
| `results/` | 이 폴더의 결과는 아래 "실행 환경"에서 실제로 돌린 값 |

## 맥(Apple Silicon)에서 실행

```bash
python3 -m venv ~/weld-sim && source ~/weld-sim/bin/activate
pip install mujoco numpy matplotlib pillow imageio-ffmpeg
cd weld_demo
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
python build_scene.py                 # weld_scene.xml 생성
python run_study.py --trials 30 --fine-trials 30   # 빠른 확인 (수 분)
python run_study.py                   # 본 실험 (기본 300회)
python make_video.py                  # 데모 영상
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
| 몸통 흔들림(최대 변위) | 0, 1, 2, 3, 5 mm | 서서 작업하는 휴머노이드의 몸통 오차 약 1.7 mm (Chen et al., arXiv:2508.00355, 2025) |
| 서는 위치 오차 | 0~50 mm, 방향 ±5° | 가정 |
| 부재 위치 오차 | 0~10 mm, ±3° | 가정 |
| 터치 측정 오차 | ±0.25 mm | ABB SmarTac 사양 |
| 터치 점 | 용접선 시작·끝에서 20 mm 안쪽 2곳, 점마다 세로판 면 + 바닥판 면 탐색 | FANUC 시작/끝 탐색 방식 |
| 실시간 보상(H2) | 몸통 위치 추정 지연 20 ms, 잡음 0.1 mm | 가정 |
| 용접 속도 | 6 mm/s, 용접선 220 mm (양 끝 10 mm 제외) | 가정 |
| 허용 오차 | 0.5 / 1.0 / 1.5 mm | 단일 출처 없음 → 세 기준 모두 보고 |

## 측정

용접 중 0.05 s마다 토치 끝과 **실제** 용접선 사이의 거리를 계산하고, 한 번의 용접에서 가장 큰 값(최대 이탈)이
허용 오차 이하이면 성공으로 셉니다. IK가 수렴하지 않거나 관절 한계에 걸리면 실패로 셉니다.

## 단순화 (논문에 한계로 명시)

- 다리로 균형 잡는 동작은 시뮬레이션하지 않고, 그 결과인 몸통 흔들림을 골반에 주입한다 (병진만, 회전 흔들림 미포함).
- 아크·열·비드 같은 용접 물리는 다루지 않고 토치 경로 정확도만 평가한다.
- 팔은 역기구학으로 목표 자세에 정확히 도달한다고 가정한다 (모터 추종 오차 미포함).
- 회전 흔들림과 모터 추종 오차가 빠져 있으므로 실제 휴머노이드는 이 결과보다 더 불리할 수 있다.

## 결과 (요약은 `results/RESULTS.md`)

허용 오차 1.0 mm, 조건별 300회:

| 몸통 흔들림 | R 고정형 | H0 보정 없음 | H1 터치 보정(현재) | H2 + 실시간 보상 |
|---|---|---|---|---|
| 0 mm | 100% | 0% | 100% | 100% |
| 1 mm | — | 0% | 29.0% | 100% |
| 2 mm | — | 0% | 0% | 99.7% |
| 3 mm | — | 0% | 0% | 87.7% |
| 5 mm | — | 0% | 0% | 38.3% |

현재 기술(H1)로 성공률 95% 이상을 유지하는 몸통 흔들림 한계: 허용 1.0 mm → 0.5 mm, 허용 1.5 mm → 1.0 mm.

## 이 결과를 만든 환경

결과는 Linux 작업 환경(CPU 2코어)에서 MuJoCo 3.17.0 공식 C 라이브러리로 계산했습니다. 그 환경에서는
`pip install mujoco`가 막혀 있어, 같은 라이브러리를 ctypes로 호출하는 최소 래퍼를 썼습니다(코드는 동일).
맥에서 `pip install mujoco`로 설치하면 이 코드가 그대로 실행되며, 같은 seed(2026)면 소수점 아래 미세한 차이 외에는 같은 수치가 나와야 합니다(플랫폼별 부동소수점 차이).
화면 없는 Linux 서버에서 영상을 만들 때는 `MUJOCO_GL=egl python make_video.py`.
