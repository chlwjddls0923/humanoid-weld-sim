"""Model sanity check against the TOP paper (Chen, Xu et al., arXiv:2508.00355, 2025, Table IV).

Puts the paper's measured base (pelvis) pose error on the G1 pelvis while the torch is held on
the seam, and compares the resulting torch-tip displacement with the paper's end-effector error.
The paper reports mean absolute errors per axis (signs unknown), so every sign combination of the
base error is tried and the range is reported.

Usage:  python validate_top.py   ->  results_validation/model_check.txt
"""
import itertools
import os

import numpy as np

from sim_core import Params, Pose, WeldModel

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "mujoco_menagerie", "unitree_g1", "weld_scene.xml")
OUT = os.path.join(HERE, "results_validation", "model_check.txt")

# Table IV, real-robot motion capture, mean over 20 motions. trans mm (x,y,z), rot rad (roll,pitch,yaw)
TOP = {
    "M (small, slow motions)": dict(base_t=[1.654, 0.461, 0.123], base_r=[0.0091, 0.0114, 0.0045],
                                    ee_t=[5.068, 2.192, 0.560], ee_r=[0.0213, 0.0138, 0.0048]),
    "T (large, fast motions)": dict(base_t=[29.232, 2.485, 1.169], base_r=[0.0163, 0.0119, 0.0032],
                                    ee_t=[33.735, 3.049, 1.491], ee_r=[0.0502, 0.0127, 0.0103]),
}


def tip_shift(wm, q, dt_mm, dr_rad):
    base = wm.nominal
    tip0, _ = wm.fk(base, q)
    moved = Pose(base.pos + np.asarray(dt_mm) / 1000.0, base.yaw, np.asarray(dr_rad, float))
    tip1, _ = wm.fk(moved, q)
    return (tip1 - tip0) * 1000.0


def main():
    wm = WeldModel(XML, Params())
    P = wm.P
    L = np.linalg.norm(wm.seam1 - wm.seam0)
    poses = {}
    q = wm.q_home.copy()
    for name, s in [("start", P.seam_margin), ("mid", L / 2), ("end", L - P.seam_margin)]:
        q, e = wm.ik(wm.nominal, wm.seam0 + wm.u_n * s, np.array(P.torch_dir, float), q)
        poses[name] = q.copy()
    tip_mid, _ = wm.fk(wm.nominal, poses["mid"])
    lever = np.linalg.norm(tip_mid - wm.nominal.pos) * 1000

    lines = ["TOP Table IV vs G1 weld model — torch-tip displacement from the measured base error",
             "Source: Chen, Xu et al., 'TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation",
             "        with Humanoid Robots', arXiv:2508.00355 (2025), Table IV (real robot, motion capture, 20 motions).",
             "Model : Unitree G1 (mujoco_menagerie) + torch, arm posed by IK on the seam at the nominal station.",
             f"        pelvis -> torch tip distance at seam middle: {lever:.0f} mm",
             "Note  : the paper's robot is not a G1 (1.65 m, 60 kg, 41 DoF). Its numbers are per-axis mean absolute",
             "        errors over many motions and its EE error also contains arm tracking error, so only the order of",
             "        magnitude is comparable. Signs are unknown -> all 2^6 sign combinations are evaluated.", ""]
    for set_name, v in TOP.items():
        bt, br, ee = np.array(v["base_t"]), np.array(v["base_r"]), np.array(v["ee_t"])
        lines += [f"=== set {set_name}",
                  f"input  base error   trans x/y/z {bt.tolist()} mm, rot roll/pitch/yaw {br.tolist()} rad",
                  f"paper  EE error     trans x/y/z {ee.tolist()} mm  (norm of means {np.linalg.norm(ee):.2f} mm)",
                  f"paper  base error   norm of means {np.linalg.norm(bt):.2f} mm",
                  ""]
        lines.append(f"{'seam point':<11}{'trans only':>12}{'rot only':>12}{'both, all +':>13}"
                     f"{'both, min':>11}{'both, median':>14}{'both, max':>11}   per-axis |d| (all +), world x/y/z")
        for pname, q in poses.items():
            t_only = np.linalg.norm(tip_shift(wm, q, bt, [0, 0, 0]))
            r_only = np.linalg.norm(tip_shift(wm, q, [0, 0, 0], br))
            both_pos = tip_shift(wm, q, bt, br)
            norms = [np.linalg.norm(tip_shift(wm, q, bt * np.array(s[:3]), br * np.array(s[3:])))
                     for s in itertools.product([1, -1], repeat=6)]
            lines.append(f"{pname:<11}{t_only:>10.2f}mm{r_only:>10.2f}mm{np.linalg.norm(both_pos):>11.2f}mm"
                         f"{min(norms):>9.2f}mm{np.median(norms):>12.2f}mm{max(norms):>9.2f}mm   "
                         + " / ".join(f"{abs(c):.2f}" for c in both_pos))
        lines.append("")
    lines += ["Reading: for set M the paper's base translation alone (≈1.7 mm) cannot explain its EE error (≈5.1 mm in x);",
              "adding the measured base rotation on the G1 lever arm gives a tip displacement of the same size as the",
              "paper's EE error. This supports injecting rotational sway with the set-M rotation/translation ratio.",
              "The match is in magnitude only: in the model the displacement is largest along world z (torch held",
              "low and forward, pitch/roll swing it vertically), while the paper's EE error is largest along its x,",
              "whose frame the paper does not define. Sign combinations change the set-M result from ~1.3 to ~7.2 mm.",
              "For set T the base translation dominates, consistent with the paper's footnote that the base accounts",
              "for most of the EE error."]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
