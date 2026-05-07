"""
Compute Pearson correlation between predicted Q and measured lift distance
for the empirical-trial JSON files. Used to validate the §4.2 numbers.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMP_DIR      = PROJECT_ROOT / "outputs" / "empirical"

def load_trials(path: Path):
    data = json.load(path.open())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("trials", [])
    return []

def get_q(t):
    for k in ("predicted_q", "quality", "q", "Q"):
        if k in t and t[k] is not None:
            return float(t[k])
    return None

def get_lift_cm(t):
    for k in ("lift_cm", "lift_distance_cm"):
        if k in t and t[k] is not None:
            return float(t[k])
    for k in ("lift_distance", "lift", "lift_m"):
        if k in t and t[k] is not None:
            v = float(t[k])
            return v * 100.0 if abs(v) < 1.5 else v
    return None

scenes = ["scene_grasp_cylinder", "scene_grasp_tallbox"]

all_q, all_L = [], []
for scene in scenes:
    candidates = [
        EMP_DIR / f"trial_outcomes_{scene}.json",
        EMP_DIR / f"empirical_trials_{scene}.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print(f"  No JSON for {scene}")
        continue

    trials = load_trials(path)
    if trials:
        print(f"\n  {path.name}: {len(trials)} trials, sample keys = {list(trials[0].keys())}")

    q, L = [], []
    for t in trials:
        qi = get_q(t); Li = get_lift_cm(t)
        if qi is not None and Li is not None:
            q.append(qi); L.append(Li)

    if len(q) < 3:
        print(f"  {scene}: only {len(q)} usable trials")
        continue

    r, p = pearsonr(q, L)
    print(f"  {scene}: n={len(q)}  r={r:.3f}  p={p:.4f}")
    all_q.extend(q); all_L.extend(L)

if len(all_q) >= 3:
    r, p = pearsonr(all_q, all_L)
    print(f"\n  POOLED: n={len(all_q)}  r={r:.3f}  p={p:.4f}")