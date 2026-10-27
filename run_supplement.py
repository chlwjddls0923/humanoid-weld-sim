"""Abstract-supplement experiments (v2 defaults, seed 2026) -> results_validation/abstract_supplement/

  seam/    1. seam variants: length 150 / 220 (baseline) / 300 mm, part yaw +30 / -30 deg
           R, H0, H1, H2 at 1.5 mm sway, plus H1 at 0.3 mm
  angle/   2. torch-axis angle error on the v2 main conditions (R, H0, H1, H2 x 0/0.5/1/1.5/2 mm)
  sensor/  3. H2 at 1.5 mm sway for a table of sensor classes (latency, noise scale) - edit SENSORS
  offset/  4. signed transverse offsets (web side +, base side -), from seam/baseline runs
  axial/   5. torch-axis (CTWD) component: angle set rerun + sway z-ratio 0.25 / 1.0 for H2 @ 1.5 mm
  stance_margin/  6. station planning for the yaw +30 part: reach-margin grid, then R/H1/H2 at the chosen stance

Usage:  python run_supplement.py [--stages seam,angle,sensor,axial,stance_margin,report] [--jobs N]
Folders that already hold a summary.csv are skipped. Existing result folders are never written.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

from sim_core import Params, Pose, Trial, WeldModel, run_trial, sway_coefs

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.join(HERE, "mujoco_menagerie", "unitree_g1")
OUT = os.path.join(HERE, "results_validation", "abstract_supplement")
SEED = 2026
TARGET = 95.0
TOP_SWAY = 1.5

# experiment 1: (label, build_scene args, scene file)
VARIANTS = [
    ("base_L220_yaw0", [], "weld_scene.xml"),
    ("L150", ["--seam-length", "0.15", "--name", "weld_scene_L150"], "weld_scene_L150.xml"),
    ("L300", ["--seam-length", "0.30", "--name", "weld_scene_L300"], "weld_scene_L300.xml"),
    ("yaw+30", ["--part-yaw", "30", "--name", "weld_scene_yaw30"], "weld_scene_yaw30.xml"),
    ("yaw-30", ["--part-yaw", "-30", "--name", "weld_scene_yawm30"], "weld_scene_yawm30.xml"),
]
# experiment 3: provisional sensor classes -> (latency ms, noise scale; 1.0 = 0.1 mm / 0.3 mrad)
SENSORS = [
    ("A 레이저 비전 심트래커급", 30, 1.0),
    ("B 모션캡처급", 5, 1.5),
    ("C 레이저 트래커급", 2, 0.2),
    ("D 고속 토치 센서 가정", 5, 0.5),
    ("E 기준 (v2 기본값)", 20, 1.0),
]
ANGLE_SWAYS = "0,0.5,1,1.5,2"
ANGLE_TOL_DEG = 5.0      # assumption, no source
# experiment 5: auxiliary component-wise criterion (transverse <= 1.0 mm AND axial <= 2.0 mm; CTWD tolerance is an assumption)
Z_RATIOS = [0.25, 1.0]
# experiment 6: station shifts (m) along the seam towards its far end, and the stance-error budget corners
STANCE_SHIFTS = [0.0, 0.05, 0.10, 0.15]
STANCE_BUDGET = (0.050, 5.0)     # radius 50 mm (disk, as sampled), +-5 deg


def run_study(out, *args, jobs):
    assert os.path.abspath(out).startswith(OUT + os.sep), out
    if os.path.exists(os.path.join(out, "summary.csv")):
        print("   (exists, skipped)", os.path.relpath(out, OUT), flush=True)
        return 0.0
    cmd = [sys.executable, os.path.join(HERE, "run_study.py"), "--out", out, "--jobs", str(jobs),
           "--fine", "", "--seed", str(SEED), *args]
    print(">>", " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.time() - t0


def log_time(name, seconds):
    if seconds == 0.0:
        return
    p = os.path.join(OUT, "timing.json")
    t = json.load(open(p)) if os.path.exists(p) else {}
    t[name] = round(seconds, 1)
    json.dump(t, open(p, "w"), indent=1, ensure_ascii=False)


def load(path):
    return [{k: (v if k == "cond" else float(v)) for k, v in r.items()} for r in csv.DictReader(open(path))]


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def row(rows, cond, sway):
    for r in rows:
        if r["cond"] == cond and abs(r["sway_mm"] - sway) < 1e-9:
            return r
    return None


def pct_ci(r, tol):
    k, n = round(r[f"success_{tol}mm_pct"] / 100 * r["n"]), int(r["n"])
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f} [{lo:.1f}–{hi:.1f}]"


# ------------------------------------------------------------------ stages
def stage_seam(jobs):
    for label, bargs, scene in VARIANTS:
        if bargs and not os.path.exists(os.path.join(G1, scene)):
            subprocess.run([sys.executable, os.path.join(HERE, "build_scene.py"), *bargs], check=True, stdout=subprocess.DEVNULL)
        sc = os.path.join(G1, scene)
        log_time(f"seam/{label}/sway1.5", run_study(os.path.join(OUT, "seam", label, "sway1.5"), "--scene", sc,
                 "--sways", str(TOP_SWAY), jobs=jobs))
        log_time(f"seam/{label}/H1_sway0.3", run_study(os.path.join(OUT, "seam", label, "H1_sway0.3"), "--scene", sc,
                 "--conds", "H1", "--sways", "0.3", jobs=jobs))


def stage_angle(jobs):
    log_time("angle", run_study(os.path.join(OUT, "angle"), "--sways", ANGLE_SWAYS, jobs=jobs))


def stage_sensor(jobs):
    for label, lat, scale in SENSORS:
        name = f"{label.split()[0]}_lat{lat:g}ms_noise{scale:g}x"
        log_time(f"sensor/{name}", run_study(os.path.join(OUT, "sensor", name), "--conds", "H2", "--sways", str(TOP_SWAY),
                 "--comp-latency", str(lat), "--comp-noise-scale", str(scale), jobs=jobs))


def stage_axial(jobs):
    log_time("axial/main", run_study(os.path.join(OUT, "axial", "main"), "--sways", ANGLE_SWAYS, jobs=jobs))
    for z in Z_RATIOS:
        log_time(f"axial/zratio_{z:g}", run_study(os.path.join(OUT, "axial", f"zratio_{z:g}"), "--conds", "H2",
                 "--sways", str(TOP_SWAY), "--sway-z-ratio", str(z), jobs=jobs))


def plan_stance(scene):
    """Reach-margin grid: shift the station along the seam towards its far end and require IK
    convergence (H1, no sway, no part error) at the 8 worst cases of the stance-error budget:
    4 points on the 50 mm circle (+-x, +-y) x yaw +-5 deg. (The square corners (+-50, +-50) lie
    at radius 71 mm, outside the sampled disk, so they are not used.)"""
    wm0 = WeldModel(scene, Params())
    far_sign = 1.0 if np.linalg.norm(wm0.seam1 - wm0.nominal.pos) > np.linalg.norm(wm0.seam0 - wm0.nominal.pos) else -1.0
    d = far_sign * wm0.u_n[:2]
    e, yaw = STANCE_BUDGET
    grid = []
    for sh in STANCE_SHIFTS:
        xy = tuple(float(v) for v in np.array(Params().stance_xy) + sh * d)
        wm = WeldModel(scene, Params(stance_xy=xy))
        ok = 0
        for dx, dy in ((e, 0.0), (-e, 0.0), (0.0, e), (0.0, -e)):
            for dyaw in (-yaw, yaw):
                rng = np.random.default_rng(0)
                st = Pose(wm.nominal.pos + np.array([dx, dy, 0.0]), wm.nominal.yaw + np.deg2rad(dyaw))
                tr = Trial(st, np.zeros(3), 0.0, sway_coefs(rng, wm.P), sway_coefs(rng, wm.P))
                ok += int(run_trial(wm, tr, "H1", 0.0, rng)["feasible"])
        # pelvis -> torch-tip distance at seam start / middle / end (lever arm for rotational sway)
        L = np.linalg.norm(wm.seam1 - wm.seam0)
        q, lever = wm.q_home.copy(), []
        for sp in (wm.P.seam_margin, L / 2, L - wm.P.seam_margin):
            q, _ = wm.ik(wm.nominal, wm.seam0 + wm.u_n * sp, wm.a_draw, q)
            lever.append(round(float(np.linalg.norm(wm.fk(wm.nominal, q)[0] - wm.nominal.pos) * 1000)))
        grid.append(dict(shift_mm=sh * 1000, stance_xy=xy, corners_ok=ok, lever_mm=lever))
    passing = [g for g in grid if g["corners_ok"] == 8]
    chosen = passing[0] if passing else grid[-1]
    plan = dict(direction_xy=[float(v) for v in d], grid=grid, chosen=chosen, all_corners_pass=bool(passing))
    os.makedirs(os.path.join(OUT, "stance_margin"), exist_ok=True)
    json.dump(plan, open(os.path.join(OUT, "stance_margin", "plan.json"), "w"), indent=1)
    return plan


def stage_stance_margin(jobs):
    scene = os.path.join(G1, "weld_scene_yaw30.xml")
    if not os.path.exists(scene):
        subprocess.run([sys.executable, os.path.join(HERE, "build_scene.py"), "--part-yaw", "30", "--name", "weld_scene_yaw30"],
                       check=True, stdout=subprocess.DEVNULL)
    pj = os.path.join(OUT, "stance_margin", "plan.json")
    plan = json.load(open(pj)) if os.path.exists(pj) else plan_stance(scene)
    xy = ",".join(f"{v:.4f}" for v in plan["chosen"]["stance_xy"])
    base = os.path.join(OUT, "stance_margin", "yaw+30_planned")
    log_time("stance_margin/sway1.5", run_study(os.path.join(base, "sway1.5"), "--scene", scene, "--stance-xy", xy,
             "--conds", "R,H1,H2", "--sways", str(TOP_SWAY), jobs=jobs))
    log_time("stance_margin/H1_sway0.3", run_study(os.path.join(base, "H1_sway0.3"), "--scene", scene, "--stance-xy", xy,
             "--conds", "H1", "--sways", "0.3", jobs=jobs))


# ------------------------------------------------------------------ report
def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ["Noto Sans CJK KR", "AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic"]:
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def report():
    md, feas = [], []
    plt = setup_plt()

    # ---- 1. seam variants
    md.append("## 실험 1. 용접선 조건 일반화\n")
    md.append("| 변형 | 조건 | 흔들림 | n | feasible % | 최대 이탈 평균 mm | 성공률 1.0 mm [95% CI] | 성공률 1.5 mm [95% CI] | 시작점 오차 p95 mm |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    seam = {}
    for label, _, _ in VARIANTS:
        a = load(os.path.join(OUT, "seam", label, "sway1.5", "summary.csv"))
        b = load(os.path.join(OUT, "seam", label, "H1_sway0.3", "summary.csv"))
        seam[label] = a + b
        feas.append((f"seam/{label}", a + b))
        for r in [row(a, "R", 0.0), row(b, "H1", 0.3), row(a, "H0", TOP_SWAY), row(a, "H1", TOP_SWAY), row(a, "H2", TOP_SWAY)]:
            md.append(f"| {label} | {r['cond']} | {r['sway_mm']:g} | {int(r['n'])} | {r['feasible_pct']:.1f} | {r['max_mean_mm']:.2f} | "
                      f"{pct_ci(r, 1.0)} | {pct_ci(r, 1.5)} | {r['along_start_abs_p95_mm']:.2f} |")
    base = seam["base_L220_yaw0"]
    md.append("\n기준(220 mm, 0°) 대비 성공률 차이 (%p):\n")
    md.append("| 변형 | R 1.0/1.5 | H1@0.3 1.0/1.5 | H0@1.5 1.0/1.5 | H1@1.5 1.0/1.5 | H2@1.5 1.0/1.5 | 최대 \\|차이\\| |")
    md.append("|---|---|---|---|---|---|---|")
    worst_all = {}
    for label, _, _ in VARIANTS[1:]:
        cells, worst = [], 0.0
        for cond, sw in [("R", 0.0), ("H1", 0.3), ("H0", TOP_SWAY), ("H1", TOP_SWAY), ("H2", TOP_SWAY)]:
            r0, r1 = row(base, cond, sw), row(seam[label], cond, sw)
            d = [r1[f"success_{t}mm_pct"] - r0[f"success_{t}mm_pct"] for t in (1.0, 1.5)]
            worst = max(worst, max(abs(x) for x in d))
            cells.append(f"{d[0]:+.1f} / {d[1]:+.1f}")
        worst_all[label] = worst
        md.append(f"| {label} | " + " | ".join(cells) + f" | {worst:.1f} |")
    fails = [(label, r["cond"], r["sway_mm"], r["feasible_pct"]) for label, rows in seam.items() for r in rows if r["feasible_pct"] < 100]
    md.append("\nIK 수렴·터치 탐색 실패: " + ("없음 (모든 변형·조건 feasible 100%)" if not fails else
              ", ".join(f"{l} {c}@{s:g} mm feasible {f:.1f}%" for l, c, s, f in fails)))
    within = [l for l, w in worst_all.items() if w <= 5.0]
    md.append(f"\n**한 줄 결론:** 기준 대비 성공률 차이가 ±5%p 안인 변형: {', '.join(within) if within else '없음'}; "
              f"벗어난 변형: {', '.join(f'{l} (최대 {w:.1f}%p)' for l, w in worst_all.items() if w > 5.0) or '없음'}.")
    # figure 1
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    conds = [("H1", 0.3, "H1 · 흔들림 0.3 mm"), ("H1", TOP_SWAY, "H1 · 흔들림 1.5 mm"), ("H2", TOP_SWAY, "H2 · 흔들림 1.5 mm")]
    x = np.arange(len(VARIANTS))
    w = 0.26
    for k, (cond, sw, name) in enumerate(conds):
        ys = [row(seam[l], cond, sw)["success_1.0mm_pct"] for l, _, _ in VARIANTS]
        ys15 = [row(seam[l], cond, sw)["success_1.5mm_pct"] for l, _, _ in VARIANTS]
        ax.bar(x + (k - 1) * w, ys, w, label=f"{name} (허용 1.0)", color=["#2563eb", "#93c5fd", "#16a34a"][k])
        ax.plot(x + (k - 1) * w, ys15, "k_", ms=9, mew=1.5, label="허용 1.5 mm" if k == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in VARIANTS], fontsize=8)
    ax.set_ylabel("용접 성공률 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("용접선 길이·부재 각도에 따른 성공률 (막대: 허용 1.0 mm, 가로선: 1.5 mm)", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seam", "fig_seam_variants.png"), dpi=200)
    plt.close(fig)

    # ---- 2. torch angle
    ang = load(os.path.join(OUT, "angle", "summary.csv"))
    feas.append(("angle", ang))
    md.append(f"\n## 실험 2. 토치 각도 오차 (기준 ±{ANGLE_TOL_DEG:g}°, 가정)\n")
    md.append("| 조건 | 흔들림 mm | 최대 각도 오차 평균 ° | 최대 각도 오차 p95 ° | 작업각 성분 최대 평균 ° | 진행각 성분 최대 평균 ° | 각도 ≤5° 비율 % | 위치 성공률 1.0 mm % |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in ang:
        md.append(f"| {r['cond']} | {r['sway_mm']:g} | {r['ang_max_mean_deg']:.2f} | {r['ang_max_p95_deg']:.2f} | "
                  f"{r['work_max_mean_deg']:.2f} | {r['travel_max_mean_deg']:.2f} | {r['ang_le5deg_pct']:.1f} | {r['success_1.0mm_pct']:.1f} |")
    h0 = [r for r in ang if r["cond"] == "H0"]
    hx = [r for r in ang if r["cond"] in ("H1", "H2")]
    md.append(f"\n**한 줄 결론:** 터치 보정이 있는 H1·H2의 토치 각도 오차는 최대 {max(r['ang_max_p95_deg'] for r in hx):.2f}°(p95)로 "
              f"±{ANGLE_TOL_DEG:g}° 기준에 모두 들어오고, 보정 없는 H0만 {min(r['ang_max_mean_deg'] for r in h0):.1f}~{max(r['ang_max_p95_deg'] for r in h0):.1f}°로 서는 방향 오차(±5°)와 부재 오차(±3°)가 그대로 남는다.")

    # ---- 3. sensor classes
    md.append("\n## 실험 3. 센서급별 실시간 보상 (H2, 흔들림 1.5 mm)\n")
    md.append("| 조합 | 지연 ms | 잡음 배율 (병진 / 회전) | n | feasible % | 최대 이탈 평균 mm | 성공률 1.0 mm [95% CI] | 성공률 1.5 mm [95% CI] |")
    md.append("|---|---|---|---|---|---|---|---|")
    sens = []
    for label, lat, scale in SENSORS:
        name = f"{label.split()[0]}_lat{lat:g}ms_noise{scale:g}x"
        r = row(load(os.path.join(OUT, "sensor", name, "summary.csv")), "H2", TOP_SWAY)
        sens.append((label, lat, scale, r))
        feas.append((f"sensor/{name}", [r]))
        md.append(f"| {label} | {lat} | {scale:g}× ({0.1 * scale:.3f} mm / {0.3 * scale:.3f} mrad) | {int(r['n'])} | {r['feasible_pct']:.1f} | "
                  f"{r['max_mean_mm']:.2f} | {pct_ci(r, 1.0)} | {pct_ci(r, 1.5)} |")
    ok = [f"{l} ({r['success_1.0mm_pct']:.1f}%)" for l, _, _, r in sens if r["success_1.0mm_pct"] >= TARGET]
    md.append(f"\n**한 줄 결론:** 허용 1.0 mm를 95% 이상 만족하는 조합: {', '.join(ok) if ok else '없음'}.")
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    x = np.arange(len(sens))
    ax.bar(x - 0.18, [r["success_1.0mm_pct"] for *_, r in sens], 0.36, label="허용 1.0 mm", color="#2563eb")
    ax.bar(x + 0.18, [r["success_1.5mm_pct"] for *_, r in sens], 0.36, label="허용 1.5 mm", color="#93c5fd")
    ax.axhline(TARGET, color="k", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l.split()[0]}\n{lat} ms / {sc:g}×" for l, lat, sc, _ in sens], fontsize=8)
    ax.set_ylabel("용접 성공률 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("센서급(지연 / 잡음)별 실시간 보상 성공률, 흔들림 1.5 mm", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sensor", "fig_sensor_classes.png"), dpi=200)
    plt.close(fig)

    # ---- 4. signed offsets (from baseline seam runs)
    md.append("\n## 실험 4. 부호 있는 가로 오프셋 분포 (web 쪽 +, base 쪽 −)\n")
    md.append("| 조건 | 흔들림 mm | 최대 +오프셋 중앙값 mm | 최대 +오프셋 p95 mm | 최대 −오프셋 중앙값 mm | 최대 −오프셋 p5 mm | 참고: 최대 이탈 평균 mm |")
    md.append("|---|---|---|---|---|---|---|")
    for cond, sw in [("R", 0.0), ("H1", 0.3), ("H1", TOP_SWAY), ("H2", TOP_SWAY), ("H0", TOP_SWAY)]:
        r = row(base, cond, sw)
        md.append(f"| {cond} | {sw:g} | {r['off_web_med_mm']:+.2f} | {r['off_web_p95_mm']:+.2f} | {r['off_base_med_mm']:+.2f} | {r['off_base_p05_mm']:+.2f} | {r['max_mean_mm']:.2f} |")
    r1, r2 = row(base, "H1", 0.3), row(base, "H2", TOP_SWAY)
    md.append(f"\n**한 줄 결론:** H1(0.3 mm)은 +{r1['off_web_p95_mm']:.2f} / {r1['off_base_p05_mm']:.2f} mm, H2(1.5 mm)는 +{r2['off_web_p95_mm']:.2f} / {r2['off_base_p05_mm']:.2f} mm(p95 / p5)로 "
              "양쪽이 거의 대칭이다. 흔들림이 무작위 위상이라 어느 한쪽으로 치우칠 이유가 없다.")

    # ---- 5. torch-axis (CTWD) component
    if os.path.exists(os.path.join(OUT, "axial", "main", "summary.csv")):
        ax = load(os.path.join(OUT, "axial", "main", "summary.csv"))
        feas.append(("axial/main", ax))
        md.append("\n## 실험 5. 토치 축 방향 오차 (CTWD 변화; + 멀어짐, − 파고듦)\n")
        md.append(f"수직 거리² − (가로² + 축 방향²) 최대 오차: {max(r['perp_check_max_mm2'] for r in ax):.2e} mm²\n")
        md.append("| 조건 | 흔들림 mm | 최대 \\|축\\| 평균 mm | p95 mm | 최대 + 중앙값 mm | 최대 − 중앙값 mm | ±1 mm 안 % | ±2 mm 안 % | 참고: 최대 \\|가로\\| 평균 mm |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in ax:
            tra = np.mean([abs(x) for x in (r["off_web_med_mm"], r["off_base_med_mm"])])
            md.append(f"| {r['cond']} | {r['sway_mm']:g} | {r['ax_abs_max_mean_mm']:.2f} | {r['ax_abs_max_p95_mm']:.2f} | {r['ax_max_med_mm']:+.2f} | "
                      f"{r['ax_min_med_mm']:+.2f} | {r['ax_within1_pct']:.1f} | {r['ax_within2_pct']:.1f} | {r['max_mean_mm']:.2f} (수직 거리) |")
        md.append("\n보조 기준(가로 ≤ 1.0 mm 그리고 축 방향 ≤ 2.0 mm; CTWD 허용값은 가정)과 1차 지표 비교:\n")
        md.append("| 조건 | 흔들림 mm | 1차: 수직 거리 ≤ 1.0 mm % | 1차: ≤ 1.5 mm % | 보조: 가로 ≤ 1.0 & 축 ≤ 2.0 % |")
        md.append("|---|---|---|---|---|")
        for r in ax:
            md.append(f"| {r['cond']} | {r['sway_mm']:g} | {r['success_1.0mm_pct']:.1f} | {r['success_1.5mm_pct']:.1f} | {r['success_comp_t1.0_a2.0_pct']:.1f} |")
        md.append("\n흔들림 z 비율 민감도 (H2, 흔들림 1.5 mm):\n")
        md.append("| z 비율 | 최대 \\|축\\| 평균 mm | 1차 ≤ 1.0 / 1.5 mm % | 보조 기준 % |")
        md.append("|---|---|---|---|")
        for z in [0.5] + Z_RATIOS:
            r = row(ax, "H2", TOP_SWAY) if z == 0.5 else row(load(os.path.join(OUT, "axial", f"zratio_{z:g}", "summary.csv")), "H2", TOP_SWAY)
            if z != 0.5:
                feas.append((f"axial/zratio_{z:g}", [r]))
            md.append(f"| {z:g}{' (기본)' if z == 0.5 else ''} | {r['ax_abs_max_mean_mm']:.2f} | {r['success_1.0mm_pct']:.1f} / {r['success_1.5mm_pct']:.1f} | {r['success_comp_t1.0_a2.0_pct']:.1f} |")

    # ---- 6. station planning for the yaw +30 part
    pj = os.path.join(OUT, "stance_margin", "plan.json")
    if os.path.exists(pj) and os.path.exists(os.path.join(OUT, "stance_margin", "yaw+30_planned", "sway1.5", "summary.csv")):
        plan = json.load(open(pj))
        md.append("\n## 실험 6. yaw +30° 부재에 대한 서는 위치 계획 (도달 여유)\n")
        md.append("| 이동량 mm | stance (x, y) m | 예산 최악 8개 중 IK 수렴 | 골반–토치 거리 시작/중간/끝 mm |")
        md.append("|---|---|---|---|")
        for g in plan["grid"]:
            md.append(f"| {g['shift_mm']:g} | ({g['stance_xy'][0]:.3f}, {g['stance_xy'][1]:.3f}) | {g['corners_ok']}/8 | {' / '.join(str(v) for v in g['lever_mm'])} |")
        c = plan["chosen"]
        md.append(f"\n고른 값: {c['shift_mm']:g} mm (용접선 방향 {np.round(plan['direction_xy'], 3).tolist()} 쪽), stance ({c['stance_xy'][0]:.3f}, {c['stance_xy'][1]:.3f})"
                  + ("" if plan["all_corners_pass"] else " — 어느 이동량도 8/8을 만족하지 못해 가장 큰 이동량을 사용"))
        pl = load(os.path.join(OUT, "stance_margin", "yaw+30_planned", "sway1.5", "summary.csv")) + \
             load(os.path.join(OUT, "stance_margin", "yaw+30_planned", "H1_sway0.3", "summary.csv"))
        feas.append(("stance_margin/yaw+30_planned", pl))
        un = seam["yaw+30"]
        md.append("\n| 조건 | 흔들림 | 기준 0°: feasible / 최대 이탈 / 성공 1.0 / 1.5 | +30° 계획 전 | +30° 계획 후 |")
        md.append("|---|---|---|---|---|")
        worst = 0.0
        for cond, sw in [("R", 0.0), ("H1", 0.3), ("H1", TOP_SWAY), ("H2", TOP_SWAY)]:
            cells = []
            for rows in (base, un, pl):
                r = row(rows, cond, sw)
                cells.append(f"{r['feasible_pct']:.1f}% / {r['max_mean_mm']:.2f} mm / {r['success_1.0mm_pct']:.1f} / {r['success_1.5mm_pct']:.1f}")
            r0, r2 = row(base, cond, sw), row(pl, cond, sw)
            worst = max(worst, max(abs(r2[f"success_{t}mm_pct"] - r0[f"success_{t}mm_pct"]) for t in (1.0, 1.5)))
            md.append(f"| {cond} | {sw:g} | " + " | ".join(cells) + " |")
        md.append(f"\n계획 후 기준 대비 성공률 최대 |차이|: {worst:.1f}%p")

    # ---- totals
    t = json.load(open(os.path.join(OUT, "timing.json")))
    n_total = sum(int(r["n"]) for _, rows in feas for r in rows)
    md.append("\n## 실행량\n")
    md.append("| 실행 | 시간 (s) |\n|---|---|")
    for k, v in t.items():
        md.append(f"| {k} | {v} |")
    md.append(f"| 합계 | {sum(t.values()):.0f} |")
    md.append(f"\n용접 수 합계: {n_total:,}회. feasible 최소: {min(r['feasible_pct'] for _, rows in feas for r in rows):.1f}%.")
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="seam,angle,sensor,axial,stance_margin,report")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    st = a.stages.split(",")
    if "seam" in st:
        stage_seam(a.jobs)
    if "angle" in st:
        stage_angle(a.jobs)
    if "sensor" in st:
        stage_sensor(a.jobs)
    if "axial" in st:
        stage_axial(a.jobs)
    if "stance_margin" in st:
        stage_stance_margin(a.jobs)
    if "report" in st:
        report()


if __name__ == "__main__":
    main()
