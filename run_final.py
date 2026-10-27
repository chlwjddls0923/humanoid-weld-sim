"""Last two validation experiments for the v2 study (H2 only, seed 2026).

  latency_freq/  estimator latency x sway frequency band -> max latency that keeps 95 % success
                 at 1.5 mm sway (1.5 mm and 1.0 mm tolerance); boundaries are refined once
  noise/         estimator noise scale (translation and rotation together) at 10 / 15 ms latency

Usage:
  python run_final.py                 # both experiments (with refinement) + tables + figure
  python run_final.py --stages report # rebuild tables/figure from existing CSVs
Outputs: results_validation/latency_freq/, results_validation/noise/, results_validation/final_tables.md
Existing result folders are never written.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

from sim_core import Params

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_validation")
SEED = 2026
TARGET = 95.0
TOP_SWAY = 1.5

BANDS = [("low_0.3-0.8Hz", "0.3,0.8"), ("base_0.3-2.0Hz", "0.3,2.0"), ("high_1.0-2.0Hz", "1.0,2.0")]
LATENCIES = [5, 10, 15, 20, 30]
LF_SWAYS = "1,1.5,2"
NOISE_LAT = [10, 15]
NOISE_SCALES = [0, 0.25, 0.5, 1, 2]
NOISE_SWAYS = "0,1.5"


def run_study(out, *args, jobs):
    assert os.path.abspath(out).startswith(OUT + os.sep), out
    if os.path.exists(os.path.join(out, "summary.csv")):
        print("   (exists, skipped)", os.path.relpath(out, OUT), flush=True)
        return 0.0
    cmd = [sys.executable, os.path.join(HERE, "run_study.py"), "--out", out, "--jobs", str(jobs),
           "--conds", "H2", "--fine", "", "--seed", str(SEED), *args]
    print(">>", " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.time() - t0


def log_time(name, seconds):
    if seconds == 0.0:
        return
    p = os.path.join(OUT, "timing_final.json")
    t = json.load(open(p)) if os.path.exists(p) else {}
    t[name] = round(seconds, 1)
    json.dump(t, open(p, "w"), indent=1, ensure_ascii=False)


def load(path):
    return [{k: (v if k == "cond" else float(v)) for k, v in r.items()} for r in csv.DictReader(open(path))]


def rate(rows, sway, tol):
    for r in rows:
        if abs(r["sway_mm"] - sway) < 1e-9:
            return r[f"success_{tol}mm_pct"]
    return float("nan")


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


# ------------------------------------------------------------------ experiment 1
def lf_dir(band, lat):
    return os.path.join(OUT, "latency_freq", band, f"lat_{lat:g}ms")


def lf_rows(band, lat):
    return load(os.path.join(lf_dir(band, lat), "summary.csv"))


def lf_run(band, freq, lat, jobs):
    log_time(f"latency_freq/{band}/lat_{lat:g}ms",
             run_study(lf_dir(band, lat), "--sways", LF_SWAYS, "--sway-freq", freq, "--comp-latency", str(lat), jobs=jobs))


def lf_latencies(band):
    d = os.path.join(OUT, "latency_freq", band)
    return sorted(float(x[4:-2]) for x in os.listdir(d) if x.startswith("lat_"))


def boundary(band, tol):
    """(largest passing latency, smallest failing latency) at TOP_SWAY; None when none pass / none fail."""
    ok = [L for L in lf_latencies(band) if rate(lf_rows(band, L), TOP_SWAY, tol) >= TARGET]
    bad = [L for L in lf_latencies(band) if rate(lf_rows(band, L), TOP_SWAY, tol) < TARGET]
    return (max(ok) if ok else None), (min(bad) if bad else None)


def stage_latency_freq(jobs):
    for band, freq in BANDS:
        for lat in LATENCIES:
            lf_run(band, freq, lat, jobs)
    # refine each boundary once with the midpoint (both tolerances)
    for band, freq in BANDS:
        for tol in (1.5, 1.0):
            lo, hi = boundary(band, tol)
            if lo is not None and hi is not None and hi - lo > 1.0:
                lf_run(band, freq, round((lo + hi) / 2), jobs)
            elif lo is not None and hi is None:          # passes even at the largest latency
                lf_run(band, freq, 2 * max(LATENCIES), jobs)
            elif lo is None and hi is not None and hi > 1.0:   # fails even at the smallest latency
                lf_run(band, freq, round(hi / 2), jobs)


# ------------------------------------------------------------------ experiment 2
def noise_dir(lat, scale):
    return os.path.join(OUT, "noise", f"lat_{lat:g}ms", f"scale_{scale:g}")


def stage_noise(jobs):
    for lat in NOISE_LAT:
        for sc in NOISE_SCALES:
            log_time(f"noise/lat_{lat:g}ms/scale_{sc:g}",
                     run_study(noise_dir(lat, sc), "--sways", NOISE_SWAYS, "--comp-latency", str(lat),
                               "--comp-noise-scale", str(sc), jobs=jobs))


# ------------------------------------------------------------------ report
def report():
    P = Params()
    md = ["## 지연 × 주파수 (H2, 흔들림 1.5 mm)\n"]
    all_lats = sorted({L for b, _ in BANDS for L in lf_latencies(b)})
    for tol in (1.5, 1.0):
        md.append(f"허용 {tol} mm 성공률 (%), 흔들림 1.5 mm:\n")
        md.append("| 대역 | " + " | ".join(f"{L:g} ms" for L in all_lats) + " |")
        md.append("|---|" + "---|" * len(all_lats))
        for band, freq in BANDS:
            cells = []
            for L in all_lats:
                if L in lf_latencies(band):
                    r = rate(lf_rows(band, L), TOP_SWAY, tol)
                    cells.append(f"**{r:.1f}**" if r >= TARGET else f"{r:.1f}")
                else:
                    cells.append("–")
            md.append(f"| {freq.replace(',', '–')} Hz | " + " | ".join(cells) + " |")
        md.append("")
    md.append("95% 이상을 만족하는 최대 지연 (돌린 단계 기준):\n")
    md.append("| 대역 | 허용 1.5 mm | 허용 1.0 mm | 허용 1.0, 흔들림 1 mm | 허용 1.5, 흔들림 2 mm |")
    md.append("|---|---|---|---|---|")
    for band, freq in BANDS:
        cells = []
        for tol in (1.5, 1.0):
            lo, hi = boundary(band, tol)
            if lo is None:
                cells.append(f"불가 ({min(lf_latencies(band)):g} ms에서도 {rate(lf_rows(band, min(lf_latencies(band))), TOP_SWAY, tol):.1f}%)")
            elif hi is None:
                cells.append(f"≥ {lo:g} ms (실험 범위 끝)")
            else:
                cells.append(f"{lo:g} ms ({hi:g} ms에서 미달)")
        lats = lf_latencies(band)
        s1 = [L for L in lats if rate(lf_rows(band, L), 1.0, 1.0) >= TARGET]
        s2 = [L for L in lats if rate(lf_rows(band, L), 2.0, 1.5) >= TARGET]
        cells.append(f"{max(s1):g} ms" if s1 else "불가")
        cells.append(f"{max(s2):g} ms" if s2 else "불가")
        md.append(f"| {freq.replace(',', '–')} Hz | " + " | ".join(cells) + " |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ["Noto Sans CJK KR", "AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic"]:
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for (band, freq), col in zip(BANDS, ["#2563eb", "#16a34a", "#dc2626"]):
        lats = lf_latencies(band)
        ys = [rate(lf_rows(band, L), TOP_SWAY, 1.5) for L in lats]
        ax.plot(lats, ys, marker="o", ms=4, color=col, label=f"{freq.replace(',', '–')} Hz")
    ax.axhline(TARGET, color="k", lw=0.8, ls=":")
    ax.set_xlabel("몸통 상태 추정 지연 (ms)")
    ax.set_ylabel("용접 성공률 (%)")
    ax.set_title("실시간 보상(H2): 지연 × 흔들림 주파수\n흔들림 1.5 mm, 허용 1.5 mm")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="흔들림 주파수 대역", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "latency_freq", "fig_latency_x_freq.png"), dpi=200)
    plt.close(fig)

    md.append("\n## 추정 잡음 크기 (H2, 0.3–2.0 Hz)\n")
    md.append("| 잡음 배율 | 병진 1σ | 회전 1σ (토치 환산) | 지연 10 ms: 흔들림 0 / 1.5 mm, 허용 1.0 | 허용 1.5 @1.5 mm | 지연 15 ms: 흔들림 0 / 1.5 mm, 허용 1.0 | 허용 1.5 @1.5 mm |")
    md.append("|---|---|---|---|---|---|---|")
    lever = 0.348
    for sc in NOISE_SCALES:
        cells = [f"{sc:g}×", f"{P.comp_noise * sc * 1e3:.3f} mm",
                 f"{P.comp_noise_rot * sc * 1e3:.3f} mrad ({P.comp_noise_rot * sc * lever * 1e3:.3f} mm)"]
        for lat in NOISE_LAT:
            rows = load(os.path.join(noise_dir(lat, sc), "summary.csv"))
            a, b = rate(rows, 0.0, 1.0), rate(rows, TOP_SWAY, 1.0)
            cells.append(f"{a:.1f} / {'**' if b >= TARGET else ''}{b:.1f}{'**' if b >= TARGET else ''} %")
            cells.append(f"{rate(rows, TOP_SWAY, 1.5):.1f} %")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    for lat in NOISE_LAT:
        ok = [sc for sc in NOISE_SCALES if rate(load(os.path.join(noise_dir(lat, sc), "summary.csv")), TOP_SWAY, 1.0) >= TARGET]
        if ok:
            m = max(ok)
            md.append(f"- 지연 {lat} ms: 흔들림 1.5 mm·허용 1.0 mm에서 95% 이상이 되는 잡음은 **{m:g}배 이하** "
                      f"(병진 ≤ {P.comp_noise * m * 1e3:.3f} mm, 회전 ≤ {P.comp_noise_rot * m * 1e3:.3f} mrad)")
        else:
            md.append(f"- 지연 {lat} ms: 잡음을 0으로 해도 95%에 못 미침 → 지연 자체가 한계")

    t = json.load(open(os.path.join(OUT, "timing_final.json")))
    md.append("\n## 소요 시간\n")
    md.append("| 실행 | 시간 (s) |\n|---|---|")
    for k, v in t.items():
        md.append(f"| {k} | {v} |")
    md.append(f"| 합계 | {sum(t.values()):.0f} |")
    open(os.path.join(OUT, "final_tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="latency_freq,noise,report")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    st = a.stages.split(",")
    if "latency_freq" in st:
        stage_latency_freq(a.jobs)
    if "noise" in st:
        stage_noise(a.jobs)
    if "report" in st:
        report()


if __name__ == "__main__":
    main()
