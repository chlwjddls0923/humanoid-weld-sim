"""Core of the humanoid fillet-welding simulation.

Kinematic simulation in MuJoCo:
  * Unitree G1 (mujoco_menagerie) with a welding torch on the right wrist
  * T-joint fillet part on a table (scene built by build_scene.py)
  * The right arm follows the seam with damped-least-squares IK (position control)
  * Humanoid-specific errors are injected on the pelvis (floating base):
      - standing-position error (static, after walking to the station)
      - body sway (time-varying translation AND rotation while standing)
  * Part placement error is applied to the part on the table
  * Touch sensing corrects the path before welding:
      2 points (web face + base plate) fix the seam line,
      1 search against the web-plate end fixes the position along the seam

Conditions
  R  : fixed robot   - no stance error, no sway, touch correction
  H0 : humanoid      - stance error + sway, no correction (drawing path as-is)
  H1 : humanoid      - stance error + sway, touch correction before welding (current tech)
  H2 : humanoid      - H1 + real-time compensation of the estimated body sway (future tech)

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

# Base pose error of a real humanoid during standing manipulation: motion capture, small-motion
# set M, mean over 20 motions (Chen, Xu et al., arXiv:2508.00355, 2025, Table IV).
# Translation x/y/z in mm, rotation roll/pitch/yaw in rad. Used to scale the rotational sway.
TOP_BASE_TRANS_MM = np.array([1.654, 0.461, 0.123])
TOP_BASE_ROT_RAD = np.array([0.0091, 0.0114, 0.0045])


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
    touch_approach: float = 0.080        # search starts 80 mm from the believed surface (> max stance + part error)
    touch_end: bool = True               # 3rd search against the web-plate end (position along the seam)
    touch_times: tuple = (-9.0, -6.0, -3.0)   # s before welding: point 1, point 2, plate end
    # error ranges (sampled uniformly)
    stance_err_max: float = 0.050        # 0–50 mm
    stance_yaw_max_deg: float = 5.0      # 0–5°
    part_err_max: float = 0.010          # 0–10 mm
    part_yaw_max_deg: float = 3.0        # 0–3°
    # sway model: translation peak = study variable A (mm); rotation peak = A * sway_rot_ratio
    sway_freq: tuple = (0.3, 2.0)        # Hz band of sway components (assumption)
    sway_z_ratio: float = 0.5            # vertical sway relative to horizontal
    # rad of peak base rotation per mm of peak base translation. TOP Table IV (set M):
    # |rot| 0.0153 rad / |trans| 1.72 mm = 0.0089 rad/mm, i.e. the base rotates about a pivot
    # ~110 mm below the pelvis (hip joints). 0 reproduces translation-only sway (v1).
    sway_rot_ratio: float = float(np.linalg.norm(TOP_BASE_ROT_RAD) / np.linalg.norm(TOP_BASE_TRANS_MM))
    sway_rot_axes: tuple = tuple(float(v) for v in TOP_BASE_ROT_RAD / np.linalg.norm(TOP_BASE_ROT_RAD))
    # H2 real-time compensation (state estimate of the body)
    comp_latency: float = 0.020          # 20 ms
    comp_noise: float = 0.0001           # 0.1 mm (1 sigma, stationary) on the estimated base position
    comp_noise_rot: float = 0.0003       # 0.3 mrad (1 sigma) on the estimated base rotation (≈0.1 mm at the torch)
    comp_noise_cutoff: float = 2.0       # Hz, first-order low-pass of the estimator noise
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


def euler_xyz(rpy):
    """Rotation matrix from (roll, pitch, yaw) applied about the body x, y, z axes."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.asarray(rpy, float), "xyz")
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


@dataclass
class Pose:
    """Rigid pose of the pelvis: position (3), station yaw, and a small sway rotation (roll, pitch, yaw)."""
    pos: np.ndarray
    yaw: float
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self._R = rotz(self.yaw) @ euler_xyz(self.tilt)
        self._quat = np.zeros(4)
        mujoco.mju_mat2Quat(self._quat, self._R.ravel())

    def R(self):
        return self._R

    def quat(self):
        return self._quat

    def to_world(self, p_local):
        return self._R @ p_local + self.pos

    def to_local(self, p_world):
        return self._R.T @ (p_world - self.pos)


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
        self.u_n = (self.seam1 - self.seam0) / np.linalg.norm(self.seam1 - self.seam0)
        # seam frame: n_web = horizontal normal of the web face pointing from the robot into the
        # web plate, u_n = seam direction, z up. The part may be yawed in the scene.
        self.n_web = np.cross(self.u_n, np.array([0.0, 0.0, 1.0]))
        self.R_seam = np.column_stack([self.n_web, self.u_n, np.array([0.0, 0.0, 1.0])])
        self.a_draw = self.R_seam @ np.array(self.P.torch_dir, float)   # torch axis in world (drawing)
        # both plates extend beyond the seam ends by the same amount; the end face is touched to
        # locate the seam along its own direction (plate long axis = local y = seam direction)
        g = m.geom("web_plate")
        self.web_half_t = float(g.size[0])
        end_pt = self.d.geom_xpos[g.id] - self.d.geom_xmat[g.id].reshape(3, 3) @ np.array([0.0, float(g.size[1]), 0.0])
        self.plate_overhang = float((self.seam0 - end_pt) @ self.u_n)
        z = self.q_stand[2]
        self.nominal = Pose(np.array([self.P.stance_xy[0], self.P.stance_xy[1], z]),
                            np.deg2rad(self.P.stance_yaw_deg))
        self.q_home = self.q_stand[self.qadr].copy()

    # ---- forward kinematics with a given pelvis pose
    def set_state(self, base: Pose, q_arm):
        d = self.d
        d.qpos[:] = self.q_stand
        d.qpos[0:3] = base.pos
        d.qpos[3:7] = base.quat()
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
    sway_coef: np.ndarray       # translation: (3 axes, k comps, [amp, freq, phase])
    sway_rot_coef: np.ndarray   # rotation:    (3 axes, k comps, [amp, freq, phase])


def sway_coefs(rng, P: Params, k=3):
    return np.stack([rng.uniform(0.5, 1.0, (3, k)),
                     rng.uniform(*P.sway_freq, (3, k)),
                     rng.uniform(0, 2 * np.pi, (3, k))], axis=-1)


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
    return Trial(stance, shift, pyaw, sway_coefs(rng, P), sway_coefs(rng, P))


def _sines(coef, t):
    t = np.atleast_1d(t)[:, None, None]
    return (coef[..., 0] * np.sin(2 * np.pi * coef[..., 1] * t + coef[..., 2])).sum(-1)  # (T,3)


class SwayFn:
    """Body sway: translation (m) and rotation (rad, roll/pitch/yaw) of the pelvis.

    Translation is scaled so its peak 3-D displacement equals `amp` (m); rotation so its peak
    rotation-vector norm equals amp[mm] * sway_rot_ratio, split over the axes by sway_rot_axes.
    """

    def __init__(self, trial, amp, t_lo, t_hi, P: Params):
        tt = np.linspace(t_lo, t_hi, 4000)
        self.w_t = np.array([1.0, 1.0, P.sway_z_ratio])
        self.w_r = np.asarray(P.sway_rot_axes, float)
        pk_t = np.linalg.norm(_sines(trial.sway_coef, tt) * self.w_t, axis=1).max()
        pk_r = np.linalg.norm(_sines(trial.sway_rot_coef, tt) * self.w_r, axis=1).max()
        self.k_t = 0.0 if amp == 0 else amp / pk_t
        self.k_r = 0.0 if amp == 0 else amp * 1e3 * P.sway_rot_ratio / pk_r
        self.trial = trial

    def __call__(self, t):
        """-> (T, 6): translation (m) and rotation (rad)."""
        return np.concatenate([self.k_t * _sines(self.trial.sway_coef, t) * self.w_t,
                               self.k_r * _sines(self.trial.sway_rot_coef, t) * self.w_r], axis=1)


class ColoredNoise:
    """Zero-mean Gaussian noise with a first-order low-pass spectrum (cutoff fc), on a time grid."""

    def __init__(self, rng, sigma, fc, t_lo, t_hi, dt):
        sigma = np.asarray(sigma, float)
        n = int(np.ceil((t_hi - t_lo) / dt)) + 2
        a = np.exp(-2 * np.pi * fc * dt)
        w = rng.normal(0.0, 1.0, (n, len(sigma)))
        x = np.empty_like(w)
        x[0] = w[0]
        for i in range(1, n):
            x[i] = a * x[i - 1] + np.sqrt(1 - a * a) * w[i]
        self.x = x * sigma
        self.t_lo, self.dt = t_lo, dt

    def __call__(self, t):
        i = int(round((t - self.t_lo) / self.dt))
        return self.x[min(max(i, 0), len(self.x) - 1)]


def part_transform(wm, trial):
    R = rotz(trial.part_yaw)
    c = wm.seam_mid
    return lambda p: R @ (p - c) + c + trial.part_shift, R


def actual_seam(wm, trial):
    T, R = part_transform(wm, trial)
    return T(wm.seam0), T(wm.seam1), R


def _search(b_true: Pose, b_bel: Pose, o_b, u_b, plane_p, plane_n):
    """One touch search. The robot moves along a line defined in its believed world frame
    (origin o_b, direction u_b) while its body is really at b_true. Returns the contact with the
    actual plane (point plane_p, normal plane_n) expressed in the believed frame, and the travel
    distance (negative = the search would have started inside the part)."""
    o_w = b_true.to_world(b_bel.to_local(o_b))
    u_w = b_true.R() @ b_bel.R().T @ u_b
    lam = plane_n @ (plane_p - o_w) / (plane_n @ u_w)
    return b_bel.to_world(b_true.to_local(o_w + lam * u_w)), lam


def touch_point(wm, trial, b_true: Pose, b_bel: Pose, y_pt, rng):
    """Web-face (x) and base-plate (z) search at the nominal seam point y_pt (believed frame).
    Returns the measured seam point in the believed frame; y is the commanded (nominal) y."""
    P = wm.P
    s0, _, R = actual_seam(wm, trial)
    Rs, n_web, ez = wm.R_seam, wm.n_web, np.array([0.0, 0.0, 1.0])
    hit_x, l1 = _search(b_true, b_bel, y_pt - P.touch_approach * n_web + 0.03 * ez, n_web, s0, R @ n_web)
    hit_z, l2 = _search(b_true, b_bel, y_pt - 0.02 * n_web + P.touch_approach * ez, -ez, s0, ez)
    # assemble in the seam frame: searched normal / along-seam (commanded) / searched height
    loc = np.array([(Rs.T @ hit_x)[0], (Rs.T @ y_pt)[1], (Rs.T @ hit_z)[2]])
    loc[[0, 2]] += rng.uniform(-P.touch_noise, P.touch_noise, 2)   # sensor noise on the searched axes
    return Rs @ loc, min(l1, l2) > 0


def touch_end(wm, trial, b_true: Pose, b_bel: Pose, m1, m2, rng):
    """Search along the measured seam line against the web-plate end face to locate the seam
    start along its own direction (start-point search). Returns the measured seam start on the
    measured line (believed frame)."""
    P = wm.P
    _, _, R = actual_seam(wm, trial)
    Tpart, _ = part_transform(wm, trial)
    end_w = Tpart(wm.seam0 - wm.plate_overhang * wm.u_n)       # a point on the actual end face
    u_m = (m2 - m1) / np.linalg.norm(m2 - m1)
    n_m = np.array([u_m[1], -u_m[0], 0.0])                     # measured web-face normal (into the web)
    o_b = (m1 - (P.touch_inset + wm.plate_overhang + P.touch_approach) * u_m
           + wm.web_half_t * n_m + np.array([0.0, 0.0, 0.03]))
    hit, lam = _search(b_true, b_bel, o_b, u_m, end_w, R @ wm.u_n)
    s_end = (hit - m1) @ u_m + rng.uniform(-P.touch_noise, P.touch_noise)
    return m1 + (s_end + wm.plate_overhang) * u_m, lam > 0


def corrected_path(wm, m1, m2, s0_meas=None):
    """Rigid (yaw + translation) map of the drawing seam onto the measured seam line. Without an
    end-face search the position along the seam is taken from the drawing (s0_meas=None)."""
    P = wm.P
    u_m = (m2 - m1) / np.linalg.norm(m2 - m1)
    yaw = np.arctan2(u_m[1], u_m[0]) - np.arctan2(wm.u_n[1], wm.u_n[0])
    R = rotz(yaw)
    if s0_meas is None:
        s0_meas = m1 - P.touch_inset * u_m
    def T(p):
        return R @ (p - wm.seam0) + s0_meas
    return T, R


def run_trial(wm: WeldModel, trial: Trial, cond: str, sway_amp: float, rng, q_cache=None,
              dt=None, record=False):
    """Simulate one weld. Returns dict with max/rms deviation (m), start/end offset along the
    seam (m) and feasibility.

    record=True also returns the joint/pelvis trajectory for rendering.
    """
    P = wm.P
    u_n = wm.u_n
    L = np.linalg.norm(wm.seam1 - wm.seam0)
    T_weld = (L - 2 * P.seam_margin) / P.weld_speed
    ts = np.arange(0.0, T_weld + 1e-9, dt or P.dt)
    t_lo = min(P.touch_times) - 1.0
    sway = SwayFn(trial, sway_amp if cond != "R" else 0.0, t_lo, T_weld + 1.0, P)
    noise = ColoredNoise(rng, [P.comp_noise] * 3 + [P.comp_noise_rot] * 3, P.comp_noise_cutoff,
                         t_lo - 1.0, T_weld + 1.0, P.dt)
    stance = trial.stance
    belief = wm.nominal                       # robot thinks it stands exactly at the station

    def base_at(t):                           # actual pelvis pose
        s = sway(t)[0]
        return Pose(stance.pos + s[:3], stance.yaw, s[3:])

    def est_base(t):                          # H2 state estimate: delayed + low-pass noise
        s = sway(t - P.comp_latency)[0] + noise(t)
        return Pose(belief.pos + s[:3], belief.yaw, s[3:])

    believed = est_base if cond == "H2" else (lambda t: belief)

    # ---- path in the believed frame
    a_draw = wm.a_draw
    touch_ok = True
    if cond == "H0":
        Tp, Rp = (lambda p: p), np.eye(3)
    else:
        pts = []
        for k, (end, sign) in enumerate([(wm.seam0, +1), (wm.seam1, -1)]):
            tt = P.touch_times[k]
            m_k, ok = touch_point(wm, trial, base_at(tt), believed(tt), end + sign * P.touch_inset * u_n, rng)
            pts.append(m_k)
            touch_ok &= ok
        s0_meas = None
        if P.touch_end:
            tt = P.touch_times[2]
            s0_meas, ok = touch_end(wm, trial, base_at(tt), believed(tt), pts[0], pts[1], rng)
            touch_ok &= ok
        Tp, Rp = corrected_path(wm, pts[0], pts[1], s0_meas)

    s0w, s1w, R_part = actual_seam(wm, trial)
    u_w = (s1w - s0w) / np.linalg.norm(s1w - s0w)
    # actual seam frame for the signed offsets and the torch angle components
    n_web_w = R_part @ wm.n_web
    ez = np.array([0.0, 0.0, 1.0])
    e_trans = (n_web_w + ez) / np.sqrt(2.0)       # transverse: + towards the web plate, - towards the base plate
    a_ref = R_part @ wm.a_draw                     # torch axis the drawing asks for, on the actual part
    a_ref = a_ref / np.linalg.norm(a_ref)
    e_axial = -a_ref                               # along the torch axis: + away from the work (CTWD up), - plunging

    q = wm.q_home.copy() if q_cache is None else q_cache
    devs, worst_ik = [], 0.0
    trans, axial, angs, works, travels = [], [], [], [], []
    tip_first = tip = None
    rec = dict(q=[], base=[], tip=[])
    for t in ts:
        p_draw = wm.seam0 + u_n * (P.seam_margin + P.weld_speed * t)
        p_cmd, a_cmd = Tp(p_draw), Rp @ a_draw
        q, e = wm.ik(believed(t), p_cmd, a_cmd, q)
        worst_ik = max(worst_ik, e)
        b = base_at(t)
        tip, ax = wm.fk(b, q)
        if tip_first is None:
            tip_first = tip
        v = tip - s0w
        v_perp = v - (v @ u_w) * u_w
        devs.append(np.linalg.norm(v_perp))                # distance to actual seam line
        trans.append(v_perp @ e_trans)
        axial.append(v_perp @ e_axial)
        # torch axis error vs. the drawing axis on the actual part: total, and split into
        # travel angle (tilt along the seam) and work angle (tilt in the plane normal to the seam)
        angs.append(np.degrees(np.arccos(np.clip(ax @ a_ref, -1.0, 1.0))))
        travels.append(np.degrees(np.arcsin(np.clip(ax @ u_w, -1.0, 1.0)) - np.arcsin(np.clip(a_ref @ u_w, -1.0, 1.0))))
        works.append(np.degrees(np.arctan2(ax @ ez, ax @ n_web_w) - np.arctan2(a_ref @ ez, a_ref @ n_web_w)))
        if record:
            rec["q"].append(q.copy())
            rec["base"].append(b)
            rec["tip"].append(tip)
    devs, trans, axial, angs = np.array(devs), np.array(trans), np.array(axial), np.array(angs)
    out = dict(max=devs.max(), rms=np.sqrt((devs ** 2).mean()), ik_err=worst_ik, touch_ok=touch_ok,
               feasible=(worst_ik < 1e-5) and touch_ok,
               off_web_max=trans.max(), off_base_max=trans.min(),       # signed transverse offsets (m)
               trans_abs_max=np.abs(trans).max(),
               ax_max=axial.max(), ax_min=axial.min(), ax_abs_max=np.abs(axial).max(),   # along torch axis (m)
               perp_check=np.abs(devs ** 2 - trans ** 2 - axial ** 2).max(),            # should be ~0
               ang_max=angs.max(), ang_mean=angs.mean(),                 # torch axis error (deg)
               work_max=np.abs(works).max(), travel_max=np.abs(travels).max(),
               along_start=(tip_first - s0w) @ u_w - P.seam_margin,      # 0 = weld starts where it should
               along_end=(tip - s0w) @ u_w - (L - P.seam_margin),
               devs=devs, ts=ts)
    if record:
        out.update(rec, seam=(s0w, s1w), part_R=part_transform(wm, trial)[1])
    return out
