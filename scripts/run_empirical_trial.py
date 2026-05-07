"""
Run Empirical Grasp Trials — v1
Franka Panda + MuJoCo 3.x

Main entry point for the empirical affordance mapping pipeline:

  1. Load scene (bottle or shifted-CoM cylinder)
  2. Sample N contact points across the object surface
  3. Execute each grasp physically in MuJoCo
  4. Record lift distance + success/failure at every contact point
  5. Build and save empirical affordance heatmap from real results

This produces a PHYSICS-GROUNDED affordance map — the colour at each
surface point reflects what ACTUALLY happened when the robot grasped there,
not a formula.

Usage:
    python scripts/run_empirical_trial.py
    python scripts/run_empirical_trial.py --scene bottle
    python scripts/run_empirical_trial.py --scene shifted_com --no-viewer
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from empirical_grasp_sampler import (
    run_trials, get_object_info, reset_scene,
    HOME_Q, DEFAULT_OBJ_POS
)
from empirical_heatmap import EmpiricalHeatmap
import mujoco


# ─────────────────────────────────────────────────────────────────────────────
# SCENE CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

SCENE_CONFIGS = {
    "bottle": {
        "xml":            "scene_bottle.xml",
        "obj_name":       "cube",
        "n_top":          15,
        "n_side_heights": 8,
        "n_side_angles":  16,
        "tag":            "bottle",
        "description":    "Irregular bottle: friction zones + shifted CoM",
    },
    "shifted_com": {
        "xml":            "scene_shifted_com.xml",
        "obj_name":       "cube",
        "n_top":          15,
        "n_side_heights": 8,
        "n_side_angles":  16,
        "tag":            "shifted_com_empirical",
        "description":    "Shifted CoM cylinder: empirical validation",
    },
    "cylinder": {
        "xml":            "scene_grasp_cylinder.xml",
        "obj_name":       "cube",
        "n_top":          15,
        "n_side_heights": 8,
        "n_side_angles":  16,
        "tag":            "cylinder_empirical",
        "description":    "Standard cylinder: baseline comparison",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run empirical grasp trials and build affordance heatmap.")
    parser.add_argument("--scene", choices=list(SCENE_CONFIGS.keys()),
                        default="bottle",
                        help="Which scene to run (default: bottle)")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Run without MuJoCo viewer (headless, faster)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for contact point sampling")
    args = parser.parse_args()

    ROOT       = Path(__file__).resolve().parent.parent
    SCENES_DIR = ROOT / "assets" / "scenes"
    OUTPUT_DIR = ROOT / "outputs" / "empirical"

    cfg = SCENE_CONFIGS[args.scene]
    xml = SCENES_DIR / cfg["xml"]

    if not xml.exists():
        print(f"✗ Scene file not found: {xml}")
        print(f"  Make sure {cfg['xml']} is in {SCENES_DIR}")
        sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"  EMPIRICAL AFFORDANCE MAPPING")
    print(f"  Scene:       {cfg['xml']}")
    print(f"  Description: {cfg['description']}")
    print(f"  Top samples: {cfg['n_top']}")
    print(f"  Side grid:   {cfg['n_side_heights']} heights × "
          f"{cfg['n_side_angles']} angles = "
          f"{cfg['n_side_heights']*cfg['n_side_angles']} points")
    print(f"  Viewer:      {'OFF (headless)' if args.no_viewer else 'ON'}")
    print(f"{'#'*60}")

    # ── Run physical trials ───────────────────────────────────────────────────
    t_start = time.time()
    results = run_trials(
        xml_path       = str(xml),
        obj_name       = cfg["obj_name"],
        n_top          = cfg["n_top"],
        n_side_heights = cfg["n_side_heights"],
        n_side_angles  = cfg["n_side_angles"],
        show_viewer    = not args.no_viewer,
        verbose        = True,
        rng_seed       = args.seed,
    )
    t_elapsed = time.time() - t_start
    print(f"\n  Trials complete in {t_elapsed:.1f}s")

    # ── Get final object info for heatmap ─────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(xml))
    data  = mujoco.MjData(model)
    reset_scene(model, data, HOME_Q, cfg["obj_name"])
    mujoco.mj_forward(model, data)
    obj = get_object_info(model, data, cfg["obj_name"])

    # ── Build empirical heatmap ───────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Building empirical heatmap...")
    hm    = EmpiricalHeatmap(results, obj)
    paths = hm.save_all(str(OUTPUT_DIR), tag=cfg["tag"])

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EMPIRICAL MAPPING COMPLETE")
    print(f"{'='*60}")
    print(f"  Scene:         {cfg['xml']}")
    print(f"  Trials:        {len(results)}")
    print(f"  Attempted:     {len(hm.attempted)}")
    print(f"  Succeeded:     {len(hm.successes)}")
    print(f"  Success rate:  {hm.success_rate()*100:.1f}%")
    print(f"  Mean lift:     {hm.mean_lift()*100:.1f} cm (successes only)")
    print(f"  Time elapsed:  {t_elapsed:.1f}s")
    print(f"\n  Outputs saved to: {OUTPUT_DIR}/")
    for p in paths:
        print(f"    {p}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()