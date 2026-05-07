"""
Aggregate per-scene comparison JSONs into a single summary file.

Reads every outputs/subtask4/comparison_<scene>.json and writes an
aggregate outputs/subtask4/comparison_summary.json with per-scene
success rates, mean lift distances, and the improvement (MAP minus naive).

Run from the project root:
    python scripts/build_comparison_summary.py
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBTASK4_DIR = PROJECT_ROOT / "outputs" / "subtask4"


def main() -> None:
    rows = []
    for path in sorted(SUBTASK4_DIR.glob("comparison_scene_*.json")):
        with path.open() as f:
            data = json.load(f)
        scene = data.get("scene", path.stem)
        naive_sr = data.get("naive_success_rate", 0.0)
        map_sr = data.get("map_success_rate", 0.0)
        rows.append({
            "scene": scene,
            "object": data.get("object", ""),
            "com_offset_cm": round(data.get("com_offset_cm", 0.0), 3),
            "n_reachable_angles": data.get("n_reachable_angles", 0),
            "n_trials_per_planner": len(data.get("naive_trials", [])),
            "naive_success_rate": round(naive_sr, 3),
            "map_success_rate": round(map_sr, 3),
            "improvement_pp": round((map_sr - naive_sr) * 100, 1),
            "naive_avg_lift_cm": round(data.get("naive_avg_lift_cm", 0.0), 2),
            "map_avg_lift_cm": round(data.get("map_avg_lift_cm", 0.0), 2),
            "lift_improvement_cm": round(
                data.get("map_avg_lift_cm", 0.0) - data.get("naive_avg_lift_cm", 0.0), 2
            ),
        })

    summary = {
        "n_scenes": len(rows),
        "trials_per_planner_per_scene": rows[0]["n_trials_per_planner"] if rows else 0,
        "seed": 42,
        "per_scene": rows,
    }

    out_path = SUBTASK4_DIR / "comparison_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path} with {len(rows)} scenes.")


if __name__ == "__main__":
    main()