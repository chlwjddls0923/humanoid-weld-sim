"""Render a 3-scene demo video of the humanoid fillet-welding simulation.

  ① 도면 그대로 (보정 없음)              -> H0
  ② 출발 전 터치 보정 (현재 기술)        -> H1
  ③ 터치 보정 + 실시간 몸통 보상 (개발 과제) -> H2

The same standing error, part error and body sway are used in all three scenes.
Usage:  python make_video.py [--sway 2.0] [--out results/demo.mp4]
Needs ffmpeg (or `pip install imageio-ffmpeg`).
On a headless Linux server set MUJOCO_GL=egl.
"""
import argparse
import os
import subprocess

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from sim_core import Params, Pose, Trial, WeldModel, run_trial, sway_coefs, yaw_quat

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "mujoco_menagerie", "unitree_g1", "weld_scene.xml")

SCENES = [("H0", "① 도면 그대로 — 보정 없음"),
          ("H1", "② 출발 전 터치 보정 — 현재 기술 수준"),
          ("H2", "③ 터치 보정 + 실시간 몸통 보상 — 개발 과제")]
W, H, BAR = 960, 540, 84
FPS, PLAYBACK = 30, 4.0          # video is 4x faster than real welding
TOL = 0.001


def find_font(size):
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
              "/System/Library/Fonts/AppleSDGothicNeo.ttc",
              "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
              "C:/Windows/Fonts/malgun.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def demo_trial(wm, rng):
    """A fixed, readable example: 30 mm / 2° standing error, 5 mm / 1° part error."""
    stance = Pose(wm.nominal.pos + np.array([-0.018, 0.024, 0.0]), wm.nominal.yaw + np.deg2rad(2.0))
    return Trial(stance, np.array([0.004, -0.003, 0.0]), np.deg2rad(1.0),
                 sway_coefs(rng, wm.P), sway_coefs(rng, wm.P))


def add_sphere(scn, pos, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), pos, np.eye(3).ravel(), np.array(rgba, np.float32))
    scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sway", type=float, default=2.0, help="body sway peak (mm)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "demo.mp4"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    wm = WeldModel(XML, Params())
    m = wm.m
    d = mujoco.MjData(m)
    trial = demo_trial(wm, np.random.default_rng(a.seed))
    part_mocap = m.body("part").mocapid[0]
    cam_wide, cam_close = m.camera("wide").id, m.camera("close").id
    renderer = mujoco.Renderer(m, H, W)
    f_big, f_small = find_font(34), find_font(24)
    proc = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{2 * W}x{H + BAR}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast", a.out],
                            stdin=subprocess.PIPE)
    stats = []
    for cond, title in SCENES:
        r = run_trial(wm, trial, cond, a.sway / 1000, np.random.default_rng(a.seed),
                      dt=PLAYBACK / FPS, record=True)
        stats.append((title, r["max"] * 1000, r["rms"] * 1000))
        # place the part where it really is
        c = wm.seam_mid
        Rp = r["part_R"]
        d.mocap_pos[part_mocap] = c - Rp @ c + trial.part_shift
        d.mocap_quat[part_mocap] = yaw_quat(trial.part_yaw)
        n = len(r["q"])
        for i in range(-int(1.2 * FPS), n + int(1.0 * FPS)):   # hold title before, result after
            j = min(max(i, 0), n - 1)
            b = r["base"][j]
            d.qpos[:] = wm.q_stand
            d.qpos[0:3], d.qpos[3:7] = b.pos, b.quat()
            d.qpos[wm.qadr] = r["q"][j]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)              # camera poses
            panels = []
            for cam in (cam_wide, cam_close):
                renderer.update_scene(d, camera=cam)
                if i >= 0:
                    for k in range(0, j + 1):
                        ok = r["devs"][k] <= TOL
                        add_sphere(renderer.scene, r["tip"][k], 0.0009, (0.1, 0.9, 0.3, 1) if ok else (1, 0.15, 0.1, 1))
                panels.append(renderer.render())
            frame = np.concatenate(panels, axis=1)
            img = Image.new("RGB", (2 * W, H + BAR), (17, 24, 39))
            img.paste(Image.fromarray(frame), (0, BAR))
            dr = ImageDraw.Draw(img)
            dr.text((24, 10), title, font=f_big, fill=(255, 255, 255))
            cur = r["devs"][j] * 1000 if i >= 0 else 0.0
            mx = r["devs"][: j + 1].max() * 1000 if i >= 0 else 0.0
            info = (f"서는 위치 오차 30 mm · 2°   부재 오차 5 mm · 1°   몸통 흔들림 {a.sway:g} mm(+회전)   |   "
                    f"현재 이탈 {cur:5.2f} mm   최대 {mx:5.2f} mm   (초록 ≤ 1 mm, 빨강 > 1 mm, 실제 속도의 4배)")
            dr.text((26, 52), info, font=f_small, fill=(203, 213, 225))
            proc.stdin.write(np.asarray(img).tobytes())
    # final summary card
    img = Image.new("RGB", (2 * W, H + BAR), (17, 24, 39))
    dr = ImageDraw.Draw(img)
    dr.text((60, 60), f"같은 조건(몸통 흔들림 {a.sway:g} mm)에서 용접선 최대 이탈", font=f_big, fill=(255, 255, 255))
    for k, (title, mx, rms) in enumerate(stats):
        col = (134, 239, 172) if mx <= 1.0 else (252, 165, 165)
        dr.text((80, 160 + 90 * k), f"{title}", font=f_big, fill=(226, 232, 240))
        dr.text((1300, 160 + 90 * k), f"최대 {mx:6.2f} mm", font=f_big, fill=col)
    dr.text((60, 500), "허용 오차 1 mm 기준 · 단일 예시이며 통계 결과는 summary.csv 참고", font=f_small, fill=(148, 163, 184))
    for _ in range(int(3 * FPS)):
        proc.stdin.write(np.asarray(img).tobytes())
    proc.stdin.close()
    proc.wait()
    for s in stats:
        print(f"{s[0]}: max {s[1]:.2f} mm, rms {s[2]:.2f} mm")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
