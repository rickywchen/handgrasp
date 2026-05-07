"""
Quick re-plot: loads existing trial JSON and regenerates heatmap images.
Run this instead of run_bottle_heatmap.py when you only changed the
visualisation (empirical_heatmap.py) and don't want to redo 260 trials.
"""
import sys
import numpy as np
import mujoco
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from empirical_grasp_sampler import get_object_info
from empirical_heatmap import EmpiricalHeatmap, ObjInfo

ROOT       = Path(__file__).resolve().parent.parent
SCENES_DIR = ROOT / "assets" / "scenes"
OUTPUT_DIR = ROOT / "outputs" / "empirical"

# FIX: JSON is saved by the sampler into assets/outputs/empirical/,
# but replot reads from outputs/empirical/ — align them to the same place.
# The sampler saves to: <xml_path parent>/../outputs/empirical/
# xml_path = assets/scenes/scene_bottle.xml → saves to assets/outputs/empirical/
# So we must look there:
JSON_PATH  = ROOT / "assets" / "outputs" / "empirical" / "empirical_trials_scene_bottle.json"

# Fallback: also check the root outputs dir (in case user moved it)
if not JSON_PATH.exists():
    JSON_PATH = OUTPUT_DIR / "empirical_trials_scene_bottle.json"

if not JSON_PATH.exists():
    print(f"ERROR: Trial JSON not found in either:")
    print(f"  {ROOT / 'assets' / 'outputs' / 'empirical' / 'empirical_trials_scene_bottle.json'}")
    print(f"  {OUTPUT_DIR / 'empirical_trials_scene_bottle.json'}")
    print("Run run_bottle_heatmap.py first!")
    sys.exit(1)

print(f"Found trial data at: {JSON_PATH}")

# Load geometry from scene (no simulation needed, just model load)
model = mujoco.MjModel.from_xml_path(str(SCENES_DIR / "scene_bottle.xml"))
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)
for _ in range(300):
    mujoco.mj_step(model, data)

raw = get_object_info(model, data, "cube")

# Bottle sections from scene_bottle.xml for tapered mesh rendering
# (z_world_bot, z_world_top, radius)
bottle_sections = [
    (0.420, 0.440, 0.030),   # base_section
    (0.440, 0.480, 0.030),   # body_lower
    (0.480, 0.520, 0.030),   # slippery_band
    (0.520, 0.550, 0.022),   # shoulder
    (0.550, 0.590, 0.015),   # neck
    (0.590, 0.600, 0.018),   # top_cap
]

obj = ObjInfo(
    body_name       = "bottle",        # display name in plot titles
    com             = raw.com,
    geom_centre     = raw.geom_centre,
    height          = raw.height,
    width           = raw.width,
    depth           = raw.depth,
    geom_type       = raw.geom_type,
    table_z         = raw.table_z,
    top_z           = raw.top_z,
    bottle_sections = bottle_sections,  # FIX: pass sections for tapered mesh
)

print(f"Loading trial data from: {JSON_PATH}")
heatmap = EmpiricalHeatmap.from_json(str(JSON_PATH), obj)
paths   = heatmap.save_all(str(OUTPUT_DIR), tag="bottle")

print(f"\nDone! Regenerated {len(paths)} files:")
for p in paths:
    print(f"  {p}")