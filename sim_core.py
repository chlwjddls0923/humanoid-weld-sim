"""Core of the humanoid fillet-welding simulation.

Kinematic simulation in MuJoCo:
  * Unitree G1 (mujoco_menagerie) with a welding torch on the right wrist
  * T-joint fillet part on a table (scene built by build_scene.py)
  * The right arm follows the seam with damped-least-squares IK (position control)
  * Humanoid-specific errors are injected on the pelvis (floating base):
      - standing-position error (static, after walking to the station)
      - body sway (time-varying translation while standing)
  * Part placement error is applied to the part on the table
  * Touch sensing (2 points, web face + base plate) corrects the path before welding

Conditions
  R  : fixed robot   - no stance error, no sway, touch correction
  H0 : humanoid      - stance error + sway, no correction (drawing path as-is)
  H1 : humanoid      - stance error + sway, touch correction before welding (current tech)
  H2 : humanoid      - H1 + real-time compensation of measured body sway (future tech)

Balance control itself is NOT simulated; the sway is injected on the pelvis.
Welding physics (arc, heat, bead) is NOT simulated; only torch-path accuracy.
"""
from dataclasses import dataclass, field
import numpy as np
import mujoco

ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# ------------------------------------------------------------------ parameters
@dataclass
class Params:
    # station: where the humanoid is told to stand (pelvis, world frame)
    stance_xy: tuple = (0.09, -0.01)
    stance_yaw_deg: float = 0.0
    # welding
    weld_speed: float = 0.006            # m/s (≈ 360 mm/min)
    seam_margin: float = 0.010           # skip first/last 10 mm of the seam
    dt: float = 0.05                     # evaluation step (s)
    torch_dir: tuple = (1.0, 0.0, -1.0)  # torch pointing direction (45° into the corner)
    # touch sensing
    touch_inset: float = 0.020           # touch points 20 mm inside seam ends
    touch_noise: float = 0.00025         # ±0.25 mm (ABB SmarTac spec), uniform
    touch_times: tuple = (-6.0, -3.0)    # s before welding starts
    # error ranges (sampled uniformly)
    stance_err_max: float = 0.050        # 0–50 mm
    stance_yaw_max_deg: float = 5.0      # 0–5°
    part_err_max: float = 0.010          # 0–10 mm
    part_yaw_max_deg: float = 3.0        # 0–3°
    # sway model
    sway_freq: tuple = (0.3, 2.0)        # Hz band of sway components
    sway_z_ratio: float = 0.5            # vertical sway relative to horizontal
    # H2 real-time compensation (state-estimation of the body)
    comp_latency: float = 0.020          # 20 ms
    comp_noise: float = 0.0001           # 0.1 mm (1 sigma)
    # IK
    ik_tol: float = 1e-6                 # 0.001 mm
    ik_iters: int = 300
    ik_damping: float = 0.03
    ori_weight: float = 0.3


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Pose:
    """Planar rigid pose of the pelvis: position (3) + yaw."""
    pos: np.ndarray
    yaw: float

    def R(self):
        return rotz(self.yaw)

    def to_world(self, p_local):
        return self.R() @ p_local + self.pos

    def to_local(self, p_world):
        return self.R().T @ (p_world - self.pos)


# ------------------------------------------------------------------ model wrapper
class WeldModel:
    def __init__(self, xml_path, params: Params = None):
        self.P = params or Params()
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        m = self.m
        key = m.key("stand").id
        self.q_stand = m.key_qpos[key].copy()
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in ARM_JOINTS])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in ARM_JOINTS])
        self.qlo = np.array([m.jnt_range[m.joint(n).id][0] for n in ARM_JOINTS])
        self.qhi = np.array([m.jnt_range[m.joint(n).id][1] for n in ARM_JOINTS])
        self.site = m.site("torch_tip").id
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        # seam geometry from the scene (drawing position)
        self.d.qpos[:] = self.q_stand
        mujoco.mj_kinematics(m, self.d)
        self.seam0 = self.d.site_xpos[m.site("seam_start").id].copy()
        self.seam1 = self.d.site_xpos[m.site("seam_end").id].copy()
        self.seam_mid = 0.5 * (self.seam0 + self.seam1)
        z = self.q_stand[2]
        self.nominal = Pose(np.array([self.P.stance_xy[0], self.P.stance_xy[1], z]),
                            np.deg2rad(self.P.stance_yaw_deg))
        self.q_home = self.q_stand[self.qadr].copy()

    # ---- forward kinematics with a given pelvis pose
    def set_state(self, base: Pose, q_arm):
        d = self.d
        d.qpos[:] = self.q_stand
        d.qpos[0:3] = base.pos
        d.qpos[3:7] = yaw_quat(base.yaw)
        d.qpos[self.qadr] = q_arm

    def fk(self, base: Pose, q_arm):
        self.set_state(base, q_arm)
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3)[:, 0].copy()

    # ---- damped least-squares IK for torch tip position + torch axis
    def ik(self, base: Pose, p_t, a_t, q0):
        P, m, d = self.P, self.m, self.d
        q = q0.copy()
        a_t = a_t / np.linalg.norm(a_t)
        err = np.inf
        for _ in range(P.ik_iters):
            self.set_state(base, q)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.site]
            a = d.site_xmat[self.site].reshape(3, 3)[:, 0]
            e_p = p_t - p
            e_r = np.cross(a, a_t)
            err = max(np.linalg.norm(e_p), 0.05 * np.linalg.norm(e_r))
            if err < P.ik_tol:
                break
            mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp[:, self.dadr], P.ori_weight * self.jacr[:, self.dadr]])
            e = np.concatenate([e_p, P.ori_weight * e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + P.ik_damping ** 2 * np.eye(6), e)
            q = np.clip(q + dq, self.qlo, self.qhi)
        return q, err


# ------------------------------------------------------------------ one trial
@dataclass
class Trial:
    stance: Pose            # actual pelvis pose (before sway)
    part_shift: np.ndarray  # part translation (xy)
    part_yaw: float         # part rotation about seam midpoint
    sway_coef: np.ndarray   # (3 axes, k comps, [amp, freq, phase])


def sample_trial(wm: WeldModel, rng, humanoid=True):
    P = wm.P
    # draw the stance error even for the fixed robot so that every condition
    # sees the same part error and sway pattern for a given seed
    r, th = P.stance_err_max * rng.uniform(), rng.uniform(0, 2 * np.pi)
    dyaw = np.deg2rad(rng.uniform(-P.stance_yaw_max_deg, P.stance_yaw_max_deg))
    if humanoid:
        stance = Pose(wm.nominal.pos + np.array([r * np.cos(th), r * np.sin(th), 0.0]), wm.nominal.yaw + dyaw)
    else:
        stance = Pose(wm.nominal.pos.copy(), wm.nominal.yaw)
    r, th = P.part_err_max * rng.uniform(), rng.uniform(0, 2 * np.pi)
    shift = np.array([r * np.cos(th), r * np.sin(th), 0.0])
    pyaw = np.deg2rad(rng.uniform(-P.part_yaw_max_deg, P.part_yaw_max_deg))
    k = 3
    coef = np.stack([rng.uniform(0.5, 1.0, (3, k)),
                     rng.uniform(*P.sway_freq, (3, k)),
                     rng.uniform(0, 2 * np.pi, (3, k))], axis=-1)
    return Trial(stance, shift, pyaw, coef)


def sway_raw(trial: Trial, t, z_ratio):
    c = trial.sway_coef
    t = np.atleast_1d(t)[:, None, None]
    s = (c[..., 0] * np.sin(2 * np.pi * c[..., 1] * t + c[..., 2])).sum(-1)  # (T,3)
    s[:, 2] *= z_ratio
    return s


class SwayFn:
    """Body sway s(t) scaled so its peak 3-D displacement equals `amp`."""

    def __init__(self, trial, amp, t_lo, t_hi, z_ratio):
        tt = np.linspace(t_lo, t_hi, 4000)
        peak = np.linalg.norm(sway_raw(trial, tt, z_ratio), axis=1).max()
        self.k = 0.0 if amp == 0 else amp / peak
        self.trial, self.z = trial, z_ratio

    def __call__(self, t):
        return self.k * sway_raw(self.trial, t, self.z)


def part_transform(wm, trial):
    R = rotz(trial.part_yaw)
    c = wm.seam_mid
    return lambda p: R @ (p - c) + c + trial.part_shift, R


def actual_seam(wm, trial):
    T, R = part_transform(wm, trial)
    return T(wm.seam0), T(wm.seam1), R


def touch_measure(wm, trial, base_true: Pose, base_belief: Pose, y_nom_point, rng):
    """Touch-sense the seam at the nominal point `y_nom_point` (believed world frame).

    The robot searches toward the web face along its believed x axis and down onto the
    base plate. It converts contacts into the believed world frame using base_belief while
    its body is really at base_true. Returns the measured seam point (believed frame).
    """
    P = wm.P
    s0, s1, R = actual_seam(wm, trial)
    # actual planes in world: web face (normal = R @ x) through s0; base plate z = s0.z
    n_w = R @ np.array([1.0, 0.0, 0.0])
    # map believed-frame search line to world: world = base_true ∘ base_belief^-1 (believed)
    def b2w(p):
        return base_true.to_world(base_belief.to_local(p))
    def w2b(p):
        return base_belief.to_world(base_true.to_local(p))
    # x-search: line through (x, y_i, z_i + 0.03) along believed +x
    o = b2w(y_nom_point + np.array([-0.04, 0.0, 0.03]))
    u = base_true.R() @ base_belief.R().T @ np.array([1.0, 0.0, 0.0])
    lam = n_w @ (s0 - o) / (n_w @ u)
    hit_w = w2b(o + lam * u)
    # z-search: line through (x_i - 0.02, y_i, z + 0.05) along -z
    o2 = b2w(y_nom_point + np.array([-0.02, 0.0, 0.05]))
    hit_z = w2b(np.array([o2[0], o2[1], s0[2]]))
    meas = np.array([hit_w[0], y_nom_point[1], hit_z[2]])
    # sensor noise on the two searched coordinates
    meas[[0, 2]] += rng.uniform(-P.touch_noise, P.touch_noise, 2)
    return meas


def corrected_path(wm, m1, m2):
    """Rigid (yaw + translation) map of the drawing seam onto the measured line."""
    P = wm.P
    n0, n1 = wm.seam0, wm.seam1
    u_n = (n1 - n0) / np.linalg.norm(n1 - n0)
    n_t1 = n0 + P.touch_inset * u_n
    n_t2 = n1 - P.touch_inset * u_n
    d_n = n_t2 - n_t1
    d_m = m2 - m1
    yaw = np.arctan2(d_m[1], d_m[0]) - np.arctan2(d_n[1], d_n[0])
    R = rotz(yaw)
    def T(p):
        return R @ (p - n_t1) + m1
    return T, R


def run_trial(wm: WeldModel, trial: Trial, cond: str, sway_amp: float, rng, q_cache=None,
              dt=None, record=False):
    """Simulate one weld. Returns dict with max/rms deviation (m) and feasibility.

    record=True also returns the joint/pelvis trajectory for rendering.
    """
    P = wm.P
    L = np.linalg.norm(wm.seam1 - wm.seam0)
    T_weld = (L - 2 * P.seam_margin) / P.weld_speed
    ts = np.arange(0.0, T_weld + 1e-9, dt or P.dt)
    sway = SwayFn(trial, sway_amp if cond != "R" else 0.0, min(P.touch_times) - 1, T_weld + 1, P.sway_z_ratio)
    stance = trial.stance
    belief = wm.nominal                       # robot thinks it stands exactly at the station

    def base_at(t):
        s = sway(t)[0]
        return Pose(stance.pos + s, stance.yaw)

    def est_sway(t):                          # H2 state estimate (delayed + noisy)
        return sway(t - P.comp_latency)[0] + rng.normal(0, P.comp_noise, 3)

    # ---- path in the believed frame
    u_n = (wm.seam1 - wm.seam0) / np.linalg.norm(wm.seam1 - wm.seam0)
    a_draw = np.array(P.torch_dir, float)
    if cond == "H0":
        Tp, Rp = (lambda p: p), np.eye(3)
    else:
        pts = []
        for k, sign in enumerate([+1, -1]):
            base_n = wm.seam0 if sign > 0 else wm.seam1
            y_pt = base_n + sign * P.touch_inset * u_n
            tt = P.touch_times[k]
            b_true = base_at(tt)
            b_bel = belief if cond != "H2" else Pose(belief.pos + est_sway(tt), belief.yaw)
            pts.append(touch_measure(wm, trial, b_true, b_bel, y_pt, rng))
        Tp, Rp = corrected_path(wm, pts[0], pts[1])

    s0w, s1w, _ = actual_seam(wm, trial)
    u_w = (s1w - s0w) / np.linalg.norm(s1w - s0w)

    q = wm.q_home.copy() if q_cache is None else q_cache
    devs, worst_ik = [], 0.0
    rec = dict(q=[], base=[], tip=[])
    for t in ts:
        p_draw = wm.seam0 + u_n * (P.seam_margin + P.weld_speed * t)
        p_cmd, a_cmd = Tp(p_draw), Rp @ a_draw
        if cond == "H2":
            base_cmd = Pose(belief.pos + est_sway(t), belief.yaw)
        else:
            base_cmd = belief
        q, e = wm.ik(base_cmd, p_cmd, a_cmd, q)
        worst_ik = max(worst_ik, e)
        b = base_at(t)
        tip, _ = wm.fk(b, q)
        v = tip - s0w
        devs.append(np.linalg.norm(v - (v @ u_w) * u_w))   # distance to actual seam line
        if record:
            rec["q"].append(q.copy())
            rec["base"].append(b)
            rec["tip"].append(tip)
    devs = np.array(devs)
    out = dict(max=devs.max(), rms=np.sqrt((devs ** 2).mean()), ik_err=worst_ik,
               feasible=worst_ik < 1e-5, devs=devs, ts=ts)
    if record:
        out.update(rec, seam=(s0w, s1w), part_R=part_transform(wm, trial)[1])
    return out
