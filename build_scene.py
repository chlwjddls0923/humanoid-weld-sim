"""Build the MuJoCo scene for the humanoid fillet-welding demo.

Reads the Unitree G1 model from mujoco_menagerie, attaches a welding torch to
the right wrist, and adds a work table with a T-joint fillet part.
Writes weld_scene.xml next to g1.xml so the mesh paths keep working.

Usage:  python build_scene.py [path/to/mujoco_menagerie]
        python build_scene.py --seam-length 0.30 --part-yaw 30 --name weld_scene_L300_yaw30
        (variants are written next to weld_scene.xml; the default scene is unchanged)
"""
import argparse
import math
import os
import sys

# ---- part geometry (metres); seam = line where web plate meets base plate ----
TABLE_TOP = 0.69     # height of table top
SEAM_X = 0.36        # x of the web plate face that the robot welds on
SEAM_Y0 = -0.32      # seam start (y)
SEAM_Y1 = -0.10      # seam end (y)
BASE_T = 0.010       # base plate thickness
WEB_T = 0.010        # web plate thickness
WEB_H = 0.100        # web plate height

TORCH = """
                          <body name="torch" pos="0.075 0 0">
                            <geom name="torch_body" type="capsule" fromto="0 0 0 0.13 0 0" size="0.013"
                              rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0" group="2" density="0"/>
                            <geom name="torch_nozzle" type="capsule" fromto="0.13 0 0 0.165 0 0" size="0.008"
                              rgba="0.80 0.50 0.20 1" contype="0" conaffinity="0" group="2" density="0"/>
                            <site name="torch_tip" pos="0.175 0 0" size="0.003" rgba="1 0.2 0.1 1" group="2"/>
                          </body>"""


def cam(name, pos, target, fovy=45):
    """MJCF camera at `pos` looking at `target` (z up)."""
    import math
    f = [t - p for p, t in zip(pos, target)]
    n = math.sqrt(sum(v * v for v in f)); f = [v / n for v in f]
    x = [f[1], -f[0], 0.0]; n = math.sqrt(x[0] ** 2 + x[1] ** 2); x = [v / n for v in x]
    y = [x[1] * f[2] - x[2] * f[1], x[2] * f[0] - x[0] * f[2], x[0] * f[1] - x[1] * f[0]]
    v = " ".join(f"{c:.4f}" for c in x + y)
    p = " ".join(f"{c:.4f}" for c in pos)
    return f'<camera name="{name}" pos="{p}" xyaxes="{v}" fovy="{fovy}"/>'


def scene_xml(seam_length=None, part_yaw_deg=0.0):
    """Default: seam SEAM_Y0..SEAM_Y1 along +y. Variants keep the seam midpoint and rotate the
    part (plates + seam) about it by part_yaw_deg; the table is enlarged to stay underneath."""
    sy = 0.5 * (SEAM_Y0 + SEAM_Y1)
    half = 0.5 * (SEAM_Y1 - SEAM_Y0) if seam_length is None else 0.5 * seam_length
    y0, y1 = sy - half, sy + half
    sl = half + 0.04                                 # half length of plates
    seam_z = TABLE_TOP + BASE_T
    if seam_length is None and part_yaw_deg == 0.0:
        part_attr, cx, cy = 'pos="0 0 0"', 0.0, 0.0  # byte-identical default scene
        table = f'pos="{SEAM_X + 0.10:.4f} {sy:.4f} {TABLE_TOP / 2:.4f}" size="0.22 {sl + 0.06:.4f} {TABLE_TOP / 2:.4f}"'
    else:
        q = math.radians(part_yaw_deg) / 2
        part_attr, cx, cy = f'pos="{SEAM_X:.4f} {sy:.4f} 0" quat="{math.cos(q):.6f} 0 0 {math.sin(q):.6f}"', SEAM_X, sy
        table = f'pos="{SEAM_X:.4f} {sy:.4f} {TABLE_TOP / 2:.4f}" size="{sl + 0.12:.4f} {sl + 0.12:.4f} {TABLE_TOP / 2:.4f}"'
    return f"""<mujoco model="humanoid fillet welding demo">
  <include file="g1_weld.xml"/>
  <statistic center="0.25 -0.2 0.7" extent="1.2"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.25 0.25 0.25" specular="0.2 0.2 0.2"/>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.62 0.70" rgb2="0.12 0.14 0.17" width="512" height="3072"/>
    <texture type="2d" name="floor" builtin="checker" mark="edge" rgb1="0.33 0.34 0.36" rgb2="0.28 0.29 0.31"
      markrgb="0.5 0.5 0.5" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="4 4" reflectance="0.05"/>
    <material name="steel" rgba="0.55 0.58 0.62 1" specular="0.6" shininess="0.6"/>
    <material name="table" rgba="0.30 0.32 0.36 1"/>
  </asset>
  <worldbody>
    <light pos="0.5 -0.5 2.5" dir="-0.2 0.2 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    <!-- work table -->
    <geom name="table" type="box" {table} material="table"/>
    <!-- T-joint part: base plate + web plate (drawing position) -->
    <body name="part" {part_attr} mocap="true">
      <geom name="base_plate" type="box" material="steel"
        pos="{SEAM_X + 0.02 - cx:.4f} {sy - cy:.4f} {TABLE_TOP + BASE_T / 2:.4f}" size="0.09 {sl:.4f} {BASE_T / 2:.4f}"/>
      <geom name="web_plate" type="box" material="steel"
        pos="{SEAM_X + WEB_T / 2 - cx:.4f} {sy - cy:.4f} {seam_z + WEB_H / 2:.4f}" size="{WEB_T / 2:.4f} {sl:.4f} {WEB_H / 2:.4f}"/>
      <!-- seam (green line) -->
      <geom name="seam" type="capsule" fromto="{SEAM_X - cx:.4f} {y0 - cy:.4f} {seam_z:.4f} {SEAM_X - cx:.4f} {y1 - cy:.4f} {seam_z:.4f}"
        size="0.0012" rgba="0.1 0.85 0.3 1" contype="0" conaffinity="0"/>
      <site name="seam_start" pos="{SEAM_X - cx:.4f} {y0 - cy:.4f} {seam_z:.4f}" size="0.002"/>
      <site name="seam_end" pos="{SEAM_X - cx:.4f} {y1 - cy:.4f} {seam_z:.4f}" size="0.002"/>
    </body>
    {cam("wide", (0.75, -1.85, 1.30), (0.22, -0.16, 0.80), 40)}
    {cam("close", (0.24, -0.60, 0.92), (SEAM_X, (SEAM_Y0 + SEAM_Y1) / 2, TABLE_TOP + BASE_T), 38)}
  </worldbody>
</mujoco>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.join(os.path.dirname(__file__), "mujoco_menagerie"))
    ap.add_argument("--seam-length", type=float, default=None, help="seam length in m (default 0.22)")
    ap.add_argument("--part-yaw", type=float, default=0.0, help="part yaw about the seam midpoint, deg")
    ap.add_argument("--name", default="weld_scene", help="output file name without .xml")
    a = ap.parse_args()
    g1_dir = os.path.join(a.root, "unitree_g1")
    src = open(os.path.join(g1_dir, "g1.xml"), encoding="utf-8").read()
    anchor = 'class="visual" mesh="right_rubber_hand"/>'
    if anchor not in src:
        sys.exit("right_rubber_hand geom not found in g1.xml - menagerie version differs")
    src = src.replace(anchor, anchor + TORCH, 1)
    src = src.replace('<mujoco model="g1_29dof_rev_1_0">', '<mujoco model="g1_weld">', 1)
    open(os.path.join(g1_dir, "g1_weld.xml"), "w", encoding="utf-8").write(src)
    out = os.path.join(g1_dir, a.name + ".xml")
    open(out, "w", encoding="utf-8").write(scene_xml(a.seam_length, a.part_yaw))
    print("wrote", out)


if __name__ == "__main__":
    main()
