import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMP_DIR      = PROJECT_ROOT / "outputs" / "empirical"

for fname in ["empirical_trials_scene_bottle.json",
              "trial_outcomes_scene_bottle.json"]:
    p = EMP_DIR / fname
    if not p.exists():
        continue
    data   = json.load(p.open())
    trials = data if isinstance(data, list) else data.get("trials", [])
    n      = len(trials)
    succ = 0
    for t in trials:
        if t.get("success") is True:
            succ += 1
        else:
            lift = t.get("lift_cm") or t.get("lift_distance_cm")
            if lift is None:
                lm = t.get("lift_distance") or t.get("lift_m")
                if lm is not None:
                    lift = float(lm) * 100.0 if abs(float(lm)) < 1.5 else float(lm)
            if lift is not None and float(lift) >= 7.0:
                succ += 1
    print(f"  {fname}: {n} trials, {succ} successes ({100*succ/n:.1f}%)")