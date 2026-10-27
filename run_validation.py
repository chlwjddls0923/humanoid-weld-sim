"""Validation runs for the v2 study (results/). Everything goes to results_validation/.

  1. latency/    H2 estimator latency sweep (5/10/20 ms, plus --extra-latencies)
  2. rot_ratio/  sensitivity to the base rotation/translation ratio
  3. frequency/  sensitivity to the sway frequency band
  4. stats/      Wilson 95% CIs for results/*.csv and a seed-2027 rerun
  report         tables (tables.md) and the latency figure from the CSVs above

Usage:
  python run_validation.py                          # all stages + report
  python run_validation.py --stages latency --latencies 2,1   # add latency levels
  python run_validation.py --stages report          # rebuild tables/figures only
The existing results/, results_rot0/, results_v1_translation_only/ are only read, never written.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_validation")
SEED = 2026
TOLS = (0.5, 1.0, 1.5)
TARGET = 95.0
TOP_SWAY = 1.5                      # mm, sway level comparable to TOP set M base x error (1.654 mm)

LAT_SWAYS = "0,0.5,1,1.5,2,3"
ROT_RATIOS = [("0", 0.0), ("0.0007", 0.0007), ("0.003", 0.003), ("0.0089", None)]   # None = v2 default
ROT_SWAYS = "0,0.5,1,1.5,2,3,5"
FINE_0_1 = ",".join(f"{v:g}" for v in np.round(np.arange(0, 1.0001, 0.1), 1))
FREQ_BANDS = [("low_0.3-0.8Hz", "0.3,0.8"), ("base_0.3-2.0Hz", "0.3,2.0"), ("high_1.0-2.0Hz", "1.0,2.0")]
FREQ_SWAYS = "0.5,1,2"


# ------------------------------------------------------------------ running
def run_study(out, *args, jobs):
    """Run run_study.py into `out` (must be under results_validation/). Returns wall time (s)."""
    assert os.path.abspath(out).startswith(OUT + os.sep), out
    cmd = [sys.executable, os.path.join(HERE, "run_study.py"), "--out", out, "--jobs", str(jobs), *args]
    print(">>", " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return time.time() - t0


def log_time(name, seconds):
    p = os.path.join(OUT, "timing.json")
    t = json.load(open(p)) if os.path.exists(p) else {}
    t[name] = round(seconds, 1)
    json.dump(t, open(p, "w"), indent=1, ensure_ascii=False)


def stage_latency(jobs, latencies):
    for lat in latencies:
        name = f"lat_{lat:g}ms"
        log_time(f"latency/{name}", run_study(os.path.join(OUT, "latency", name), "--conds", "H2",
                 "--sways", LAT_SWAYS, "--fine", "", "--comp-latency", str(lat), "--seed", str(SEED), jobs=jobs))


def stage_rot(jobs):
    for label, ratio in ROT_RATIOS:
        extra = [] if ratio is None else ["--rot-ratio", str(ratio)]
        name = f"rot_{label}"
        log_time(f"rot_ratio/{name}", run_study(os.path.join(OUT, "rot_ratio", name), "--conds", "H1,H2",
                 "--sways", ROT_SWAYS, "--fine", FINE_0_1, "--fine-trials", "200", "--seed", str(SEED), *extra, jobs=jobs))


def stage_freq(jobs):
    for name, band in FREQ_BANDS:
        log_time(f"frequency/{name}", run_study(os.path.join(OUT, "frequency", name), "--conds", "H1,H2",
                 "--sways", FREQ_SWAYS, "--fine", FINE_0_1, "--fine-trials", "200", "--sway-freq", band,
                 "--seed", str(SEED), jobs=jobs))


def stage_seed(jobs):
    log_time("stats/seed2027", run_study(os.path.join(OUT, "stats", "seed2027"), "--seed", "2027", jobs=jobs))


# ------------------------------------------------------------------ analysis
def load(path):
    return [{k: (v if k == "cond" else float(v)) for k, v in r.items()} for r in csv.DictReader(open(path))]


def wilson(k, n, z=1.959964):
    """Wilson score interval for k successes out of n -> (lo, hi) in %."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def with_ci(rows):
    out = []
    for r in rows:
        r = dict(r)
        n = int(r["n"])
        for tol in TOLS:
            key = f"success_{tol}mm_pct"
            lo, hi = wilson(round(r[key] / 100 * n), n)
            r[f"{key}_ci95_lo"], r[f"{key}_ci95_hi"] = lo, hi
        out.append(r)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def limit(rows, cond, tol, target=TARGET):
    """Largest sway level such that every level up to it reaches `target` % success.
    Returns (limit, next level that fails) — limit None means even 0 mm fails."""
    sel = sorted((r for r in rows if r["cond"] == cond), key=lambda r: r["sway_mm"])
    lim = None
    for r in sel:
        if r[f"success_{tol}mm_pct"] >= target:
            lim = r["sway_mm"]
        else:
            return lim, r["sway_mm"]
    return lim, None


def fmt_lim(lim_next):
    lim, nxt = lim_next
    if lim is None:
        return f"없음 (0 mm에서도 <{TARGET:g}%)"
    if nxt is None:
        return f"≥ {lim:g} mm (실험 범위 끝까지)"
    return f"{lim:g} mm (다음 단계 {nxt:g} mm에서 미달)"


def rate(rows, cond, sway, tol):
    for r in rows:
        if r["cond"] == cond and abs(r["sway_mm"] - sway) < 1e-9:
            return r[f"success_{tol}mm_pct"]
    return float("nan")


def ci_txt(rows, cond, sway, tol):
    for r in rows:
        if r["cond"] == cond and abs(r["sway_mm"] - sway) < 1e-9:
            k, n = round(r[f"success_{tol}mm_pct"] / 100 * r["n"]), int(r["n"])
            lo, hi = wilson(k, n)
            return f"{100 * k / n:.1f}% [{lo:.1f}–{hi:.1f}]"
    return "—"


def setup_fonts():
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
    md = []
    feas = []                                         # (folder, feasible % min) over every summary

    def track(folder, rows):
        feas.append((folder, min(r["feasible_pct"] for r in rows), sum(int(r["n"]) for r in rows)))

    # ---- 1. latency
    lat_dirs = sorted((d for d in os.listdir(os.path.join(OUT, "latency")) if d.startswith("lat_")),
                      key=lambda d: float(d[4:-2]))
    lat = {float(d[4:-2]): load(os.path.join(OUT, "latency", d, "summary.csv")) for d in lat_dirs}
    for d in lat_dirs:
        track(f"latency/{d}", lat[float(d[4:-2])])
    md.append("## 1. H2 추정 지연\n")
    md.append("| 지연 | " + " | ".join(f"흔들림 {s:g} mm" for s in map(float, LAT_SWAYS.split(","))) + " |")
    md.append("|---|" + "---|" * len(LAT_SWAYS.split(",")))
    for tol in (1.0, 1.5):
        for L, rows in lat.items():
            md.append(f"| {L:g} ms (허용 {tol}) | " + " | ".join(f"{rate(rows, 'H2', s, tol):.1f}"
                      for s in map(float, LAT_SWAYS.split(","))) + " |")
    md.append("\n성공률 95% 이상을 유지하는 흔들림 한계 (단계 0/0.5/1/1.5/2/3 mm):\n")
    md.append("| 지연 | 허용 1.0 mm | 허용 1.5 mm | 흔들림 1.5 mm 성공률 (허용 1.0) [95% CI] | (허용 1.5) [95% CI] |")
    md.append("|---|---|---|---|---|")
    for L, rows in lat.items():
        md.append(f"| {L:g} ms | {fmt_lim(limit(rows, 'H2', 1.0))} | {fmt_lim(limit(rows, 'H2', 1.5))} | "
                  f"{ci_txt(rows, 'H2', TOP_SWAY, 1.0)} | {ci_txt(rows, 'H2', TOP_SWAY, 1.5)} |")
    for tol in (1.0, 1.5):
        ok = [L for L, rows in lat.items() if rate(rows, "H2", TOP_SWAY, tol) >= TARGET]
        md.append(f"\n흔들림 {TOP_SWAY} mm에서 허용 {tol} mm 성공률 ≥ {TARGET:g}%를 만족하는 지연: "
                  + (f"{', '.join(f'{L:g}' for L in sorted(ok))} ms (최대 {max(ok):g} ms)" if ok else "없음"))
    plt = setup_fonts()
    for tol in (1.0, 1.5):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(lat)))
        for (L, rows), c in zip(lat.items(), cols):
            sel = sorted((r for r in rows if r["cond"] == "H2"), key=lambda r: r["sway_mm"])
            ax.plot([r["sway_mm"] for r in sel], [r[f"success_{tol}mm_pct"] for r in sel], marker="o", ms=4,
                    color=c, label=f"지연 {L:g} ms")
        ax.axhline(TARGET, color="k", lw=0.8, ls=":")
        ax.axvline(1.654, color="#dc2626", lw=0.8, ls="--")
        ax.text(1.69, 57, "TOP 실측\n몸통 x 1.654 mm", color="#dc2626", fontsize=7)
        ax.set_xlabel("몸통 흔들림 최대 변위 (mm)")
        ax.set_ylabel("용접 성공률 (%)")
        ax.set_title(f"실시간 보상(H2)의 추정 지연별 성공률 (허용 {tol} mm)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "latency", f"fig_latency_tol{tol}.png"), dpi=200)
        plt.close(fig)

    # ---- 2. rotation ratio
    md.append("\n## 2. 회전 비율\n")
    md.append("| 회전 비율 (rad/mm) | H1 한계 허용 1.0 | H1 한계 허용 1.5 | H1 @1.5 mm (허용 1.0 / 1.5) | H2 @1.5 mm (허용 1.0 / 1.5) | H2 한계 허용 1.0 | H2 한계 허용 1.5 |")
    md.append("|---|---|---|---|---|---|---|")
    for label, _ in ROT_RATIOS:
        d = os.path.join(OUT, "rot_ratio", f"rot_{label}")
        main, thr = load(os.path.join(d, "summary.csv")), load(os.path.join(d, "threshold.csv"))
        track(f"rot_ratio/rot_{label}", main)
        track(f"rot_ratio/rot_{label} (threshold)", thr)
        md.append(f"| {label} | {fmt_lim(limit(thr, 'H1', 1.0))} | {fmt_lim(limit(thr, 'H1', 1.5))} | "
                  f"{rate(main, 'H1', TOP_SWAY, 1.0):.1f} / {rate(main, 'H1', TOP_SWAY, 1.5):.1f} % | "
                  f"{rate(main, 'H2', TOP_SWAY, 1.0):.1f} / {rate(main, 'H2', TOP_SWAY, 1.5):.1f} % | "
                  f"{fmt_lim(limit(main, 'H2', 1.0))} | {fmt_lim(limit(main, 'H2', 1.5))} |")

    # ---- 3. frequency
    md.append("\n## 3. 흔들림 주파수\n")
    md.append("| 주파수 대역 | H1 한계 허용 1.0 | H1 한계 허용 1.5 | H2 허용 1.0: 0.5 / 1 / 2 mm | H2 허용 1.5: 0.5 / 1 / 2 mm | H1 최대 이탈 평균 @1 mm |")
    md.append("|---|---|---|---|---|---|")
    for name, band in FREQ_BANDS:
        d = os.path.join(OUT, "frequency", name)
        main, thr = load(os.path.join(d, "summary.csv")), load(os.path.join(d, "threshold.csv"))
        track(f"frequency/{name}", main)
        track(f"frequency/{name} (threshold)", thr)
        h1max = next(r["max_mean_mm"] for r in main if r["cond"] == "H1" and r["sway_mm"] == 1.0)
        md.append(f"| {band.replace(',', '–')} Hz | {fmt_lim(limit(thr, 'H1', 1.0))} | {fmt_lim(limit(thr, 'H1', 1.5))} | "
                  + " / ".join(f"{rate(main, 'H2', s, 1.0):.1f}" for s in (0.5, 1, 2)) + " % | "
                  + " / ".join(f"{rate(main, 'H2', s, 1.5):.1f}" for s in (0.5, 1, 2)) + f" % | {h1max:.2f} mm |")

    # ---- 4. stats: Wilson CIs + seed 2027
    sdir = os.path.join(OUT, "stats")
    v2_main, v2_thr = load(os.path.join(HERE, "results", "summary.csv")), load(os.path.join(HERE, "results", "threshold.csv"))
    write_csv(os.path.join(sdir, "summary_v2_ci95.csv"), with_ci(v2_main))
    write_csv(os.path.join(sdir, "threshold_v2_ci95.csv"), with_ci(v2_thr))
    s27_main = load(os.path.join(sdir, "seed2027", "summary.csv"))
    s27_thr = load(os.path.join(sdir, "seed2027", "threshold.csv"))
    write_csv(os.path.join(sdir, "summary_seed2027_ci95.csv"), with_ci(s27_main))
    write_csv(os.path.join(sdir, "threshold_seed2027_ci95.csv"), with_ci(s27_thr))
    track("stats/seed2027", s27_main)
    track("stats/seed2027 (threshold)", s27_thr)
    md.append("\n## 4. seed 2026 vs 2027\n")
    md.append("| 항목 | seed 2026 (v2) | seed 2027 |")
    md.append("|---|---|---|")
    for tol in TOLS:
        md.append(f"| H1 한계 허용 {tol} mm | {fmt_lim(limit(v2_thr, 'H1', tol))} | {fmt_lim(limit(s27_thr, 'H1', tol))} |")
    for tol in (1.0, 1.5):
        for s in (0.0, 0.5, 1.0, 2.0):
            md.append(f"| H2 @{s:g} mm 허용 {tol} | {ci_txt(v2_main, 'H2', s, tol)} | {ci_txt(s27_main, 'H2', s, tol)} |")
    md.append("\nv2 한계 실험 Wilson 95% CI (허용 1.0 mm):\n")
    md.append("| 흔들림 | " + " | ".join(f"{r['sway_mm']:g}" for r in v2_thr) + " |")
    md.append("|---|" + "---|" * len(v2_thr))
    md.append("| 성공률 [CI] | " + " | ".join(ci_txt(v2_thr, "H1", r["sway_mm"], 1.0) for r in v2_thr) + " |")

    # ---- timing / feasibility
    md.append("\n## 소요 시간 / feasible\n")
    t = json.load(open(os.path.join(OUT, "timing.json")))
    md.append("| 실행 | 시간 (s) |")
    md.append("|---|---|")
    for k, v in t.items():
        md.append(f"| {k} | {v} |")
    md.append(f"| 합계 | {sum(t.values()):.0f} |")
    md.append("\n| 폴더 | 용접 수 | feasible 최소 % |")
    md.append("|---|---|---|")
    for f, m, n in feas:
        md.append(f"| {f} | {n} | {m:.1f} |")
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="latency,rot,freq,seed,report")
    ap.add_argument("--latencies", default="5,10,20", help="H2 latencies in ms for the latency stage")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    stages = a.stages.split(",")
    if "latency" in stages:
        stage_latency(a.jobs, [float(x) for x in a.latencies.split(",")])
    if "rot" in stages:
        stage_rot(a.jobs)
    if "freq" in stages:
        stage_freq(a.jobs)
    if "seed" in stages:
        stage_seed(a.jobs)
    if "report" in stages:
        report()


if __name__ == "__main__":
    main()
