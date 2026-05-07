"""
run_bottle_heatmap.py
Standalone script: run the empirical affordance trial loop on the bottle
scene and immediately generate the heatmap visualisation.

Usage:
    cd <project root>
    python scripts/run_bottle_heatmap.py

Outputs → outputs/empirical/
    empirical_trials_scene_bottle.json
    empirical_2d_bottle.png
    empirical_3d_bottle.png
    empirical_mesh_bottle.png
    empirical_report_bottle.txt
"""

import sys
import json
import numpy as np
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from empirical_grasp_sampler import run_trials, get_object_info, ObjInfo
from empirical_heatmap import EmpiricalHeatmap, TrialResult, ObjInfo as HeatmapObjInfo

ROOT       = Path(__file__).resolve().parent.parent
SCENES_DIR = ROOT / "assets" / "scenes"
OUTPUT_DIR = ROOT / "outputs" / "empirical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

XML_PATH = str(SCENES_DIR / "scene_bottle.xml")
OBJ_NAME = "cube"   # bottle body is named "cube" in the XML

# ── Step 1: Run physical trials ───────────────────────────────────────────────
print("=" * 60)
print("  BOTTLE AFFORDANCE TRIAL RUN")
print("=" * 60)
print(f"  Scene:   {XML_PATH}")
print(f"  Output:  {OUTPUT_DIR}")
print()

results = run_trials(
    xml_path       = XML_PATH,
    obj_name       = OBJ_NAME,
    n_top          = 20,
    n_side_heights = 10,
    n_side_angles  = 24,
    show_viewer    = True,
    verbose        = True,
    rng_seed       = 42,
)

# ── Step 2: Load geometry for heatmap ─────────────────────────────────────────
import mujoco
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)
for _ in range(300):
    mujoco.mj_step(model, data)

raw = get_object_info(model, data, OBJ_NAME)

# Convert ObjInfo from sampler to heatmap's ObjInfo
BOTTLE_SECTIONS = [
    (0.420, 0.440, 0.030),   # base
    (0.440, 0.480, 0.030),   # body_lower
    (0.480, 0.520, 0.030),   # slippery_band
    (0.520, 0.550, 0.022),   # shoulder
    (0.550, 0.590, 0.015),   # neck
    (0.590, 0.600, 0.018),   # cap
]

obj = HeatmapObjInfo(
    body_name       = "bottle",
    com             = raw.com,
    geom_centre     = raw.geom_centre,
    height          = raw.height,
    width           = raw.width,
    depth           = raw.depth,
    geom_type       = raw.geom_type,
    table_z         = raw.table_z,
    top_z           = raw.top_z,
    bottle_sections = BOTTLE_SECTIONS,
)

# ── Step 3: Build and save empirical heatmap ──────────────────────────────────
heatmap = EmpiricalHeatmap(results, obj)
paths   = heatmap.save_all(str(OUTPUT_DIR), tag="bottle")

print(f"\n  Saved {len(paths)} output files:")
for p in paths:
    print(f"    {p}")

print("\n  Done! Open the PNG files in outputs/empirical/ to see the affordance map.")
print("""
  What to expect in the bottle affordance map:
    GREEN  (high lift)  → grippy body zone (z=0.52–0.59) — best grasp height
    YELLOW (partial)    → transition zones — inconsistent
    RED    (fail/slip)  → slippery band (z=0.48–0.52, µ=0.15) — gripper slides
    RED    (fail/tip)   → base zone — CoM too low, arm geometry issues
    ASYMMETRY in XY     → real CoM is +Y offset → some angles succeed more
""")