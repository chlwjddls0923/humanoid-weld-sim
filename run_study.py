"""Monte-Carlo study: can a humanoid keep fillet-weld torch accuracy on a shipyard
sub-assembly part under standing-position error and body sway?

Usage:
  python run_study.py                      # default: 300 trials, sway 0/0.5/1/2/3/5 mm
  python run_study.py --trials 50 --jobs 4 # quick run
  python run_study.py --rot-ratio 0        # translation-only sway (v1 behaviour)
  python run_study.py --two-point-touch    # no end-face search, seam position from the drawing (v1)
  python run_study.py --conds H2 --fine "" --comp-latency 5   # H2 only, 5 ms estimator latency
  python run_study.py --sway-freq 0.3,0.8  # low-frequency sway band
  python run_study.py --conds H2 --comp-noise-scale 0.5   # half the H2 estimator noise
Outputs (results/):
  trials.csv      one row per simulated weld
  summary.csv     success rate per condition / sway / tolerance
  threshold.csv   fine sway sweep for H1 (current technology)
  fig1_success_vs_sway.png, fig2_stance_error.png, fig3_threshold.png
"""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np

from sim_core import Params, WeldModel, sample_trial, run_trial

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "mujoco_menagerie", "unitree_g1", "weld_scene.xml")
_WM = None


def _init(params_kw, xml=XML):
    global _WM
    _WM = WeldModel(xml, Params(**params_kw))


def _job(args):
    cond, sway_mm, i, seed = args
    wm = _WM
    rng_trial = np.random.default_rng([seed, i])            # same errors for every condition
    trial = sample_trial(wm, rng_trial, humanoid=(cond != "R"))
    cid = {"R": 0, "H0": 1, "H1": 2, "H2": 3}[cond]
    rng_run = np.random.default_rng([seed, i, cid, int(round(sway_mm * 100))])
    r = run_trial(wm, trial, cond, sway_mm / 1000.0, rng_run)
    st = np.linalg.norm(trial.stance.pos[:2] - wm.nominal.pos[:2]) * 1000
    return dict(cond=cond, sway_mm=sway_mm, trial=i, max_mm=r["max"] * 1000, rms_mm=r["rms"] * 1000,
                feasible=int(r["feasible"]), ik_err_mm=r["ik_err"] * 1000, touch_ok=int(r["touch_ok"]),
                along_start_mm=r["along_start"] * 1000, along_end_mm=r["along_end"] * 1000,
                off_web_max_mm=r["off_web_max"] * 1000, off_base_max_mm=r["off_base_max"] * 1000,
                ang_max_deg=r["ang_max"], ang_mean_deg=r["ang_mean"], work_max_deg=r["work_max"],
                travel_max_deg=r["travel_max"], stance_err_mm=st,
                stance_yaw_deg=np.rad2deg(trial.stance.yaw - wm.nominal.yaw),
                part_err_mm=np.linalg.norm(trial.part_shift[:2]) * 1000,
                part_yaw_deg=np.rad2deg(trial.part_yaw))


def run_jobs(jobs, n_proc, params_kw, xml=XML):
    with Pool(n_proc, initializer=_init, initargs=(params_kw, xml)) as pool:
        return pool.map(_job, jobs, chunksize=4)


def summarize(rows, tols):
    out = []
    keys = sorted({(r["cond"], r["sway_mm"]) for r in rows}, key=lambda k: ("RH0H1H2".find(k[0]), k[1]))
    for cond, sw in keys:
        sel = [r for r in rows if r["cond"] == cond and r["sway_mm"] == sw]
        mx = np.array([r["max_mm"] for r in sel])
        fe = np.array([r["feasible"] for r in sel], bool)
        al = np.abs([r["along_start_mm"] for r in sel])       # weld start offset along the seam
        row = dict(cond=cond, sway_mm=sw, n=len(sel), feasible_pct=100 * fe.mean(),
                   max_mean_mm=mx.mean(), max_p95_mm=np.percentile(mx, 95), rms_mean_mm=np.mean([r["rms_mm"] for r in sel]),
                   along_start_abs_mean_mm=al.mean(), along_start_abs_p95_mm=np.percentile(al, 95))
        for tol in tols:
            row[f"success_{tol}mm_pct"] = 100 * np.mean((mx <= tol) & fe)
        ang = np.array([r["ang_max_deg"] for r in sel])
        ow, ob = np.array([r["off_web_max_mm"] for r in sel]), np.array([r["off_base_max_mm"] for r in sel])
        row.update(ang_max_mean_deg=ang.mean(), ang_max_p95_deg=np.percentile(ang, 95),
                   work_max_mean_deg=np.mean([r["work_max_deg"] for r in sel]),
                   travel_max_mean_deg=np.mean([r["travel_max_deg"] for r in sel]),
                   ang_le5deg_pct=100 * np.mean((ang <= 5.0) & fe),
                   off_web_med_mm=np.median(ow), off_web_p95_mm=np.percentile(ow, 95),
                   off_base_med_mm=np.median(ob), off_base_p05_mm=np.percentile(ob, 5))
        out.append(row)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def plots(summary, rows, thr, tols, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib import font_manager
    for name in ["Noto Sans CJK KR", "AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic"]:
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    style = {"R": ("고정형 로봇 (기준)", "#6b7280", "s"), "H0": ("휴머노이드 · 보정 없음", "#dc2626", "x"),
             "H1": ("휴머노이드 · 출발 전 터치 보정", "#2563eb", "o"),
             "H2": ("휴머노이드 · 터치 보정 + 실시간 몸통 보상", "#16a34a", "^")}
    r_rows = [s for s in summary if s["cond"] == "R"]

    fig, axes = plt.subplots(1, len(tols), figsize=(4.2 * len(tols), 3.6), sharey=True)
    for ax, tol in zip(np.atleast_1d(axes), tols):
        for c in ["H0", "H1", "H2"]:                  # only the conditions that were run (--conds)
            sel = sorted((s for s in summary if s["cond"] == c), key=lambda s: s["sway_mm"])
            if sel:
                ax.plot([s["sway_mm"] for s in sel], [s[f"success_{tol}mm_pct"] for s in sel],
                        marker=style[c][2], color=style[c][1], label=style[c][0])
        if r_rows:
            ax.axhline(r_rows[0][f"success_{tol}mm_pct"], color=style["R"][1], ls="--", lw=1, label=style["R"][0])
        ax.set_title(f"허용 오차 {tol} mm")
        ax.set_xlabel("몸통 흔들림 최대 변위 (mm)")
        ax.grid(alpha=0.3)
        ax.set_ylim(-3, 103)
    np.atleast_1d(axes)[0].set_ylabel("용접 성공률 (%)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(outdir, "fig1_success_vs_sway.png"), dpi=200)

    fig2_conds = [c for c in ["H0", "H1"] if any(r["cond"] == c and r["sway_mm"] == 0 for r in rows)]
    if fig2_conds:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for c in fig2_conds:
            sel = [r for r in rows if r["cond"] == c and r["sway_mm"] == 0]
            ax.scatter([r["stance_err_mm"] for r in sel], [r["max_mm"] for r in sel], s=8,
                       color=style[c][1], label=style[c][0], alpha=0.7)
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_yscale("log")
        # plain decimals (0.1, 1, 10) instead of 10^-1: Korean fonts have no U+2212 for mathtext
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel("서는 위치 오차 (mm)")
        ax.set_ylabel("용접선 최대 이탈 (mm, 로그)")
        ax.set_title("서는 위치 오차의 영향 (흔들림 0 mm)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig2_stance_error.png"), dpi=200)
    plt.close("all")

    if thr:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        xs = [t["sway_mm"] for t in thr]
        for tol, col in zip(tols, ["#93c5fd", "#2563eb", "#1e3a8a"]):
            ax.plot(xs, [t[f"success_{tol}mm_pct"] for t in thr], marker="o", ms=3, color=col, label=f"허용 {tol} mm")
        ax.axhline(95, color="k", lw=0.8, ls=":")
        ax.set_xlabel("몸통 흔들림 최대 변위 (mm)")
        ax.set_ylabel("성공률 (%)")
        ax.set_title("현재 기술(출발 전 터치 보정)의 흔들림 한계")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig3_threshold.png"), dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--sways", default="0,0.5,1,2,3,5")
    ap.add_argument("--tols", default="0.5,1.0,1.5")
    ap.add_argument("--fine", default="0,0.1,0.2,0.3,0.4,0.5,0.75,1,1.5,2,3",
                    help="sway levels (mm) for the H1 threshold sweep; empty to skip")
    ap.add_argument("--rot-ratio", type=float, default=None,
                    help="peak base rotation per mm of peak translation (rad/mm); default from TOP Table IV, 0 = translation only")
    ap.add_argument("--two-point-touch", action="store_true",
                    help="skip the end-face search; position along the seam taken from the drawing")
    ap.add_argument("--conds", default="R,H0,H1,H2", help="conditions for the main study, e.g. H1,H2")
    ap.add_argument("--comp-latency", type=float, default=None,
                    help="H2 body-state estimation latency in ms (default 20)")
    ap.add_argument("--sway-freq", default=None,
                    help="sway frequency band 'lo,hi' in Hz (default 0.3,2.0)")
    ap.add_argument("--comp-noise-scale", type=float, default=None,
                    help="scale H2 estimator noise (translation 0.1 mm and rotation 0.3 mrad together); 0 = noise-free")
    ap.add_argument("--fine-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--scene", default=XML, help="scene XML (default: weld_scene.xml built by build_scene.py)")
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from existing CSVs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sways = [float(s) for s in a.sways.split(",")]
    tols = [float(t) for t in a.tols.split(",")]
    params_kw = {}
    if a.rot_ratio is not None:
        params_kw["sway_rot_ratio"] = a.rot_ratio
    if a.two_point_touch:
        params_kw["touch_end"] = False
    if a.comp_latency is not None:
        params_kw["comp_latency"] = a.comp_latency / 1000.0
    if a.sway_freq:
        params_kw["sway_freq"] = tuple(float(f) for f in a.sway_freq.split(","))
    if a.comp_noise_scale is not None:
        base = Params()
        params_kw["comp_noise"] = base.comp_noise * a.comp_noise_scale
        params_kw["comp_noise_rot"] = base.comp_noise_rot * a.comp_noise_scale
    conds = [c.strip() for c in a.conds.split(",") if c.strip()]
    P = Params(**params_kw)
    print(f"conds={conds}, sway_freq={P.sway_freq} Hz, comp_latency={P.comp_latency * 1000:g} ms, "
          f"sway_rot_ratio={P.sway_rot_ratio:.4f} rad/mm, touch_end={P.touch_end}, "
          f"touch_approach={P.touch_approach * 1000:.0f} mm, comp noise {P.comp_noise * 1000:.2f} mm / "
          f"{P.comp_noise_rot * 1000:.2f} mrad low-passed at {P.comp_noise_cutoff:g} Hz")
    if a.plot_only:
        def load(name):
            p = os.path.join(a.out, name)
            if not os.path.exists(p):
                return []
            out = []
            for r in csv.DictReader(open(p)):
                out.append({k: (v if k == "cond" else float(v)) for k, v in r.items()})
            return out
        plots(load("summary.csv"), load("trials.csv"), load("threshold.csv"), tols, a.out)
        return

    with open(os.path.join(a.out, "params.txt"), "w") as f:
        f.write(f"scene = {os.path.relpath(a.scene, HERE)}\n")   # relative: no user paths in results
        for k, v in vars(P).items():
            f.write(f"{k} = {v}\n")
    jobs = [("R", 0.0, i, a.seed) for i in range(a.trials)] if "R" in conds else []
    for c in ["H0", "H1", "H2"]:
        if c in conds:
            jobs += [(c, s, i, a.seed) for s in sways for i in range(a.trials)]
    print(f"main study: {len(jobs)} welds")
    rows = run_jobs(jobs, a.jobs, params_kw, a.scene)
    write_csv(os.path.join(a.out, "trials.csv"), rows)
    summary = summarize(rows, tols)
    write_csv(os.path.join(a.out, "summary.csv"), summary)

    thr = []
    if a.fine:
        fine = [float(s) for s in a.fine.split(",")]
        fjobs = [("H1", s, i, a.seed + 1) for s in fine for i in range(a.fine_trials)]
        print(f"threshold sweep: {len(fjobs)} welds")
        frows = run_jobs(fjobs, a.jobs, params_kw, a.scene)
        thr = summarize(frows, tols)
        write_csv(os.path.join(a.out, "threshold.csv"), thr)

    plots(summary, rows, thr, tols, a.out)
    for s in summary:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})


if __name__ == "__main__":
    main()
