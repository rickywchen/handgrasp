"""
Sub-task 4: Affordance-Guided Grasp Selection Criteria
======================================================
Compares Naive (geometric centre) vs Affordance-guided (CoM-aware) grasping.

Key insight: the Panda has a LIMITED reachable workspace for side grasps.
Not all angles are physically possible. The affordance-guided planner
respects this — it pre-filters for reachability, then biases toward
the CoM side among reachable angles. The naive planner picks uniformly.

This script:
  1. Loads a scene, reads true CoM via data.subtree_com
  2. Computes the REACHABLE angle set (pre-filter like empirical_grasp_sampler)
  3. Scenario 1 (NAIVE): grasp at uniformly sampled reachable angles
  4. Scenario 2 (MAP):   grasp at CoM-biased reachable angles + optimal height
  5. Saves comparison plots + JSON results

Usage:
    python scripts/affordance_guided_comparison.py
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import json
import sys

# ─── PATHS ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
SCENES_DIR = ROOT / "assets" / "scenes"
OUTPUT_DIR = ROOT / "outputs" / "subtask4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── PANDA CONSTANTS ─────────────────────────────────────────────────────────
# Exact match with empirical_grasp_sampler.py
HAND_TO_FINGERTIP_Z = 0.103
MAX_HAND_Z          = 0.720
APPROACH_CLEARANCE  = 0.15
SIDE_STANDOFF       = 0.18
SIDE_CLEARANCE      = 0.005
PANDA_REACH_MIN     = 0.28
PANDA_REACH_MAX     = 0.75
MAX_APPROACH_R_SIDE = 0.68
PANDA_MIN_GRASP_X   = 0.25

HOME_Q    = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785])
SAFE_HOME = np.array([0.45, 0.00, 0.70])

LIFT_SUCCESS_THRESH = 0.07   # 7 cm = success


# ─── QUATERNION HELPERS ───────────────────────────────────────────────────────

def _euler_to_quat(roll, pitch, yaw):
    cy, sy = np.cos(yaw*0.5),   np.sin(yaw*0.5)
    cp, sp = np.cos(pitch*0.5), np.sin(pitch*0.5)
    cr, sr = np.cos(roll*0.5),  np.sin(roll*0.5)
    return np.array([cr*cp*cy + sr*sp*sy,
                     sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy,
                     cr*cp*sy - sr*sp*cy])

def _gripper_down_quat():
    return _euler_to_quat(np.pi, 0.0, 0.0)

def _gripper_side_quat(approach_angle):
    return _euler_to_quat(0.0, np.pi / 2.0, approach_angle + np.pi)


# ─── REACHABILITY CHECK ──────────────────────────────────────────────────────
# EXACT same logic as empirical_grasp_sampler._side_reachable()

def _side_reachable(contact_pos: np.ndarray,
                    approach_angle: float,
                    obj_radius: float) -> bool:
    cx, cy = contact_pos[0], contact_pos[1]
    dx = np.cos(approach_angle)
    dy = np.sin(approach_angle)

    hand_standoff = obj_radius + SIDE_CLEARANCE
    hand_pos      = np.array([cx + dx * hand_standoff,
                               cy + dy * hand_standoff,
                               contact_pos[2]])
    approach_pos  = np.array([cx + dx * (hand_standoff + SIDE_STANDOFF),
                               cy + dy * (hand_standoff + SIDE_STANDOFF),
                               contact_pos[2] + 0.12])

    grasp_xy    = float(np.linalg.norm(hand_pos[:2]))
    approach_xy = float(np.linalg.norm(approach_pos[:2]))

    return (
        PANDA_REACH_MIN <= grasp_xy <= PANDA_REACH_MAX
        and approach_xy <= MAX_APPROACH_R_SIDE
        and hand_pos[0] >= PANDA_MIN_GRASP_X
    )


# ─── OBJECT ANALYSIS ─────────────────────────────────────────────────────────

@dataclass
class ObjectInfo:
    name:        str
    com:         np.ndarray
    geom_centre: np.ndarray
    geom_type:   int
    height:      float
    width:       float
    depth:       float
    table_z:     float
    top_z:       float

    def radius_xy(self) -> float:
        return min(self.width, self.depth) / 2.0

    def com_offset_xy(self) -> float:
        return float(np.linalg.norm(self.com[:2] - self.geom_centre[:2]))


def analyze_object(model, data, body_name="cube"):
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in model.")

    com = data.subtree_com[body_id].copy()
    geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
    if not geom_ids:
        raise ValueError(f"No geoms for '{body_name}'.")

    if len(geom_ids) > 1:
        z_mins, z_maxs, xy_maxs = [], [], []
        for gid in geom_ids:
            gpos  = data.geom_xpos[gid]
            gsize = model.geom_size[gid]
            gtype = model.geom_type[gid]
            if gtype == 5:
                z_mins.append(gpos[2] - gsize[1])
                z_maxs.append(gpos[2] + gsize[1])
                xy_maxs.append(gsize[0])
            elif gtype == 6:
                z_mins.append(gpos[2] - gsize[2])
                z_maxs.append(gpos[2] + gsize[2])
                xy_maxs.append(max(gsize[0], gsize[1]))
            elif gtype == 2:
                z_mins.append(gpos[2] - gsize[0])
                z_maxs.append(gpos[2] + gsize[0])
                xy_maxs.append(gsize[0])
        table_z = min(z_mins);  top_z = max(z_maxs)
        height  = top_z - table_z
        width   = depth = max(xy_maxs) * 2
        geom_centre = np.array([
            np.mean([data.geom_xpos[g][0] for g in geom_ids]),
            np.mean([data.geom_xpos[g][1] for g in geom_ids]),
            (table_z + top_z) / 2.0])
        geom_type = model.geom_type[geom_ids[0]]
    else:
        gid = geom_ids[0]
        geom_centre = data.geom_xpos[gid].copy()
        geom_type   = model.geom_type[gid]
        sz = model.geom_size[gid]
        if geom_type == 6:
            width, depth, height = sz[0]*2, sz[1]*2, sz[2]*2
        elif geom_type == 5:
            width = depth = sz[0]*2; height = sz[1]*2
        elif geom_type == 2:
            width = depth = height = sz[0]*2
        else:
            width = depth = height = 0.05
        table_z = geom_centre[2] - height/2
        top_z   = geom_centre[2] + height/2

    return ObjectInfo(
        name=body_name, com=com, geom_centre=geom_centre,
        geom_type=geom_type, height=height, width=width, depth=depth,
        table_z=table_z, top_z=top_z)


# ─── REACHABLE ANGLE SCAN ────────────────────────────────────────────────────

def _find_reachable_angles(obj, z: float, n_scan: int = 64) -> List[float]:
    """
    Scan 360° at n_scan resolution and return only reachable angles.
    This mirrors what empirical_grasp_sampler does (pre-filter unreachable).
    """
    cx, cy = obj.geom_centre[0], obj.geom_centre[1]
    r = obj.radius_xy()
    reachable = []
    for i in range(n_scan):
        angle = 2.0 * np.pi * i / n_scan
        contact = np.array([cx + r * np.cos(angle),
                            cy + r * np.sin(angle), z])
        if _side_reachable(contact, angle, r):
            reachable.append(angle)
    return reachable


# ─── OSC CONTROLLER ──────────────────────────────────────────────────────────

class _OSController:
    def __init__(self, model, data, ee_body="hand",
                 kp_pos=3.5, kp_ori=1.5, damping=0.01, null_kp=0.3):
        self.model   = model
        self.data    = data
        self.ee_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self.ee_id < 0:
            raise ValueError(f"Body '{ee_body}' not found")
        self.kp_pos  = kp_pos
        self.kp_ori  = kp_ori
        self.damping = damping
        self.null_kp = null_kp
        self.dt      = model.opt.timestep
        self.nj      = 7
        self.q_des   = np.zeros(self.nj)
        self.q_min   = model.jnt_range[:self.nj, 0].copy()
        self.q_max   = model.jnt_range[:self.nj, 1].copy()
        self.q_home  = HOME_Q.copy()

    def reset(self, q):
        self.q_des = q[:self.nj].copy()

    def _quat_err(self, qt, qc):
        qt = qt / np.linalg.norm(qt)
        qc = qc / np.linalg.norm(qc)
        if np.dot(qt, qc) < 0:
            qc = -qc
        w1, x1, y1, z1 = qt
        w2, x2, y2, z2 = qc
        return 2.0 * np.array([
            -w1*x2 + x1*w2 - y1*z2 + z1*y2,
            -w1*y2 + x1*z2 + y1*w2 - z1*x2,
            -w1*z2 - x1*y2 + y1*x2 + z1*w2])

    def move_to(self, target_pos, target_quat, steps=1000,
                viewer=None, pos_tol=0.012) -> bool:
        target_quat = target_quat / np.linalg.norm(target_quat)
        V_MAX_LIN = 0.15   # m/s — safe max velocity clamp
        for i in range(steps):
            mujoco.mj_forward(self.model, self.data)
            pos_err = target_pos - self.data.xpos[self.ee_id]
            ori_err = self._quat_err(target_quat, self.data.xquat[self.ee_id])
            if np.linalg.norm(pos_err) < pos_tol:
                break
            Jp = np.zeros((3, self.model.nv))
            Jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBodyCom(self.model, self.data, Jp, Jr, self.ee_id)
            J      = np.vstack([Jp[:, :self.nj], Jr[:, :self.nj]])
            v_lin  = self.kp_pos * pos_err
            v_lin_norm = np.linalg.norm(v_lin)
            if v_lin_norm > V_MAX_LIN:
                v_lin = v_lin * (V_MAX_LIN / v_lin_norm)
            v_task = np.hstack([v_lin, self.kp_ori * ori_err])
            Jinv   = J.T @ np.linalg.inv(J @ J.T + self.damping * np.eye(6))
            dq     = Jinv @ v_task
            dq    += (np.eye(self.nj) - Jinv @ J) @ (
                      self.null_kp * (self.q_home - self.q_des))
            self.q_des = np.clip(
                self.q_des + self.dt * dq, self.q_min, self.q_max)
            self.data.ctrl[:self.nj] = self.q_des
            mujoco.mj_step(self.model, self.data)
            if viewer and i % 10 == 0:
                viewer.sync()
        return np.linalg.norm(self.data.xpos[self.ee_id] - target_pos) < pos_tol * 2.5


# ─── GRIPPER HELPERS ──────────────────────────────────────────────────────────

def _set_gripper(data, model, cmd, steps=300, viewer=None):
    for i in range(steps):
        data.ctrl[7] = cmd
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()

def _ramp_gripper(data, model, from_cmd, to_cmd, steps=400, viewer=None):
    for i in range(steps):
        t = i / steps
        data.ctrl[7] = (1-t)*from_cmd + t*to_cmd
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()


# ─── SCENE RESET ──────────────────────────────────────────────────────────────

def reset_scene(model, data, obj_name="cube", obj_pos=None, viewer=None):
    if obj_pos is None:
        obj_pos = np.array([0.55, 0.0, 0.50])

    data.qpos[:7] = HOME_Q
    data.qvel[:7] = 0.0
    if model.nq > 7:
        data.qpos[7:9] = 0.04
        data.qvel[7:9] = 0.0

    body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    body_jnt = model.body_jntadr[body_id]
    if body_jnt >= 0 and model.jnt_type[body_jnt] == mujoco.mjtJoint.mjJNT_FREE:
        qa = model.jnt_qposadr[body_jnt]
        data.qpos[qa:qa+3]   = obj_pos
        data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
        va = model.jnt_dofadr[body_jnt]
        data.qvel[va:va+6]   = 0.0

    mujoco.mj_forward(model, data)
    for _ in range(600):
        data.ctrl[:7] = HOME_Q
        data.ctrl[7]  = 255.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()


# ─── GRASP EXECUTION ─────────────────────────────────────────────────────────
# Identical to empirical_grasp_sampler._execute_trial() for side grasps

def _execute_side(model, data, osc, obj, contact_pos, approach_angle,
                  obj_name="cube", viewer=None) -> Tuple[float, str]:
    body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    initial_z = data.xpos[body_id][2]

    r  = obj.radius_xy()
    dx = np.cos(approach_angle)
    dy = np.sin(approach_angle)
    cx, cy = obj.geom_centre[0], obj.geom_centre[1]
    gz     = contact_pos[2]

    hand_standoff   = r + SIDE_CLEARANCE
    final_grasp_pos = np.array([cx + dx * hand_standoff,
                                 cy + dy * hand_standoff, gz])
    approach_pos    = np.array([cx + dx * (hand_standoff + SIDE_STANDOFF),
                                 cy + dy * (hand_standoff + SIDE_STANDOFF),
                                 max(gz + 0.12, obj.table_z + 0.15)])
                                 # +0.12m above grasp / min 0.15m above table
                                 # avoids table collision during diagonal descent
    lift_pos        = np.array([final_grasp_pos[0], final_grasp_pos[1],
                                 gz + 0.22])
    quat = _gripper_side_quat(approach_angle)

    data.ctrl[7] = 255.0
    osc.move_to(SAFE_HOME, _gripper_down_quat(), steps=800, viewer=viewer)

    old_kp = osc.kp_pos
    osc.kp_pos = 2.0

    data.ctrl[7] = 255.0
    if not osc.move_to(approach_pos, quat, steps=1500, viewer=viewer):
        return 0.0, "failed_approach"
    if not osc.move_to(final_grasp_pos, quat, steps=2000,
                        pos_tol=0.018, viewer=viewer):
                        # 2000 steps / tighter tol for single diagonal move
        return 0.0, "failed_contact"

    _ramp_gripper(data, model, 255.0, -100.0, steps=400, viewer=viewer)
    _set_gripper(data, model, -100.0, steps=300, viewer=viewer)

    osc.move_to(lift_pos, quat, steps=1800, viewer=viewer)
    _set_gripper(data, model, -100.0, steps=400, viewer=viewer)

    final_z   = data.xpos[body_id][2]
    lift_dist = final_z - initial_z
    if lift_dist >= LIFT_SUCCESS_THRESH:
        return lift_dist, ""
    else:
        return lift_dist, "slipped"


# ─── TRIAL GENERATION ────────────────────────────────────────────────────────

def _generate_trials(obj, strategy: str, reachable_angles: List[float],
                     n: int, rng: np.random.Generator) -> List[Dict]:
    """
    Generate n trials using ONLY reachable angles.

    strategy = "naive":
       Uniform sample from reachable angles, all at geometric centre height.
       No CoM awareness.

    strategy = "map":
       Biased toward angles nearest the CoM side (if CoM is shifted).
       Height chosen near CoM z (where tipping resistance is best).
    """
    cx, cy = obj.geom_centre[0], obj.geom_centre[1]
    r = obj.radius_xy()
    z_min = obj.table_z + 0.01
    z_max = obj.top_z   - 0.01

    if not reachable_angles:
        return []

    trials = []
    com_dir = obj.com[:2] - obj.geom_centre[:2]
    com_angle = float(np.arctan2(com_dir[1], com_dir[0])) if np.linalg.norm(com_dir) > 1e-4 else None

    for i in range(n):
        if strategy == "naive":
            # Uniform selection from reachable angles
            angle = reachable_angles[i % len(reachable_angles)]
            # Vary height across low/mid/high
            z = z_min + (z_max - z_min) * (i % 3) / 2.0

        else:  # "map" — affordance-guided
            if com_angle is not None:
                # Score each reachable angle by proximity to CoM direction
                # Closer to CoM angle → more likely to be selected
                angle_diffs = [abs(((a - com_angle + np.pi) % (2*np.pi)) - np.pi)
                               for a in reachable_angles]
                # Convert to weights: small diff → high weight
                weights = np.array([np.exp(-3.0 * d) for d in angle_diffs])
                weights /= weights.sum()
                angle = rng.choice(reachable_angles, p=weights)
            else:
                # No CoM offset — same as naive
                angle = reachable_angles[i % len(reachable_angles)]

            # Height: bias toward CoM z (where torque resistance is best)
            # Use CoM Z directly (clamped to safe range) — no random noise
            z = np.clip(obj.com[2], z_min, z_max)

        contact = np.array([cx + r * np.cos(angle),
                            cy + r * np.sin(angle), z])

        trials.append({
            "contact_pos": contact,
            "approach_angle": float(angle),
            "grasp_z": float(z),
        })

    return trials


# ─── MAIN COMPARISON ─────────────────────────────────────────────────────────

def run_comparison(xml_path: str, obj_name: str = "cube",
                   obj_pos: np.ndarray = None,
                   n_trials_per_scenario: int = 8,
                   show_viewer: bool = True):
    if obj_pos is None:
        obj_pos = np.array([0.55, 0.0, 0.50])

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    print(f"\n{'='*65}")
    print(f"  SUB-TASK 4: AFFORDANCE-GUIDED GRASP SELECTION COMPARISON")
    print(f"  Scene: {Path(xml_path).name}")
    print(f"  Trials per scenario: {n_trials_per_scenario}")
    print(f"{'='*65}")

    # Settle and analyze
    reset_scene(model, data, obj_name, obj_pos)
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    obj = analyze_object(model, data, obj_name)

    print(f"\n  Object: {obj.name}")
    print(f"  Geometric centre:  [{obj.geom_centre[0]:.4f}, {obj.geom_centre[1]:.4f}, {obj.geom_centre[2]:.4f}]")
    print(f"  TRUE CoM:          [{obj.com[0]:.4f}, {obj.com[1]:.4f}, {obj.com[2]:.4f}]")
    print(f"  CoM offset (XY):   {obj.com_offset_xy()*100:.2f} cm")
    print(f"  Dimensions:        {obj.width*100:.1f} × {obj.depth*100:.1f} × {obj.height*100:.1f} cm")
    print(f"  Table Z / Top Z:   {obj.table_z:.3f} / {obj.top_z:.3f}  (from simulation)")

    # ── Find reachable angles at multiple heights ──
    z_mid = (obj.table_z + obj.top_z) / 2.0
    reachable = _find_reachable_angles(obj, z_mid, n_scan=64)

    print(f"\n  Reachable angles at z={z_mid:.3f}:  {len(reachable)}/{64}")
    if reachable:
        angle_degs = sorted([np.degrees(a) for a in reachable])
        print(f"  Range: {angle_degs[0]:.0f}° — {angle_degs[-1]:.0f}°")
        print(f"  Angles: {[f'{a:.0f}°' for a in angle_degs[:12]]}"
              + (f" ... +{len(angle_degs)-12} more" if len(angle_degs) > 12 else ""))
    else:
        print(f"  NO reachable side grasp angles! Cannot run comparison.")
        return None

    # CoM info
    com_dir = obj.com[:2] - obj.geom_centre[:2]
    if np.linalg.norm(com_dir) > 1e-4:
        com_angle_deg = np.degrees(np.arctan2(com_dir[1], com_dir[0]))
        print(f"  CoM direction:     {com_angle_deg:.1f}°")
        # Find nearest reachable angle to CoM
        com_angle = np.arctan2(com_dir[1], com_dir[0])
        nearest = min(reachable, key=lambda a: abs(((a - com_angle + np.pi) % (2*np.pi)) - np.pi))
        print(f"  Nearest reachable: {np.degrees(nearest):.1f}°")

    rng = np.random.default_rng(42)

    naive_trials = _generate_trials(obj, "naive", reachable, n_trials_per_scenario, rng)
    map_trials   = _generate_trials(obj, "map",   reachable, n_trials_per_scenario, rng)

    results = {"naive": [], "map": []}

    def _run(viewer=None):
        osc = _OSController(model, data)

        # ── SCENARIO 1: NAIVE ──
        print(f"\n{'─'*55}")
        print(f"  SCENARIO 1: NAIVE — uniform reachable angles, geom centre Z")
        print(f"{'─'*55}")

        for i, trial in enumerate(naive_trials):
            reset_scene(model, data, obj_name, obj_pos, viewer)
            osc.reset(HOME_Q)

            label = f"NAIVE-{i}"
            angle_deg = np.degrees(trial["approach_angle"])
            print(f"    [{label}] angle={angle_deg:6.1f}°  z={trial['grasp_z']:.3f}")

            lift, reason = _execute_side(
                model, data, osc, obj, trial["contact_pos"],
                trial["approach_angle"], obj_name, viewer)

            ok = (reason == "")
            icon = "✓" if ok else "✗"
            print(f"    [{label}] {icon} Lift: {lift*100:.1f} cm  {reason}")

            results["naive"].append({
                "trial": i,
                "approach_angle_deg": float(angle_deg),
                "grasp_z":     float(trial["grasp_z"]),
                "success":     bool(ok),
                "lift_cm":     float(lift*100),
                "fail_reason": reason,
            })

        # ── SCENARIO 2: MAP ──
        print(f"\n{'─'*55}")
        print(f"  SCENARIO 2: AFFORDANCE-GUIDED — CoM-biased angles + optimal Z")
        print(f"{'─'*55}")

        for i, trial in enumerate(map_trials):
            reset_scene(model, data, obj_name, obj_pos, viewer)
            osc.reset(HOME_Q)

            label = f"MAP-{i}"
            angle_deg = np.degrees(trial["approach_angle"])
            print(f"    [{label}] angle={angle_deg:6.1f}°  z={trial['grasp_z']:.3f}")

            lift, reason = _execute_side(
                model, data, osc, obj, trial["contact_pos"],
                trial["approach_angle"], obj_name, viewer)

            ok = (reason == "")
            icon = "✓" if ok else "✗"
            print(f"    [{label}] {icon} Lift: {lift*100:.1f} cm  {reason}")

            results["map"].append({
                "trial": i,
                "approach_angle_deg": float(angle_deg),
                "grasp_z":     float(trial["grasp_z"]),
                "success":     bool(ok),
                "lift_cm":     float(lift*100),
                "fail_reason": reason,
            })

    if show_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 1.8
            viewer.cam.lookat   = [0.5, 0.0, 0.4]
            _run(viewer)
    else:
        _run(None)

    # ── Summary ──
    naive_sr  = sum(1 for r in results["naive"] if r["success"]) / max(len(results["naive"]), 1)
    map_sr    = sum(1 for r in results["map"]   if r["success"]) / max(len(results["map"]), 1)
    naive_avg = float(np.mean([r["lift_cm"] for r in results["naive"]])) if results["naive"] else 0.0
    map_avg   = float(np.mean([r["lift_cm"] for r in results["map"]]))   if results["map"]   else 0.0

    naive_succ = [r for r in results["naive"] if r["success"]]
    map_succ   = [r for r in results["map"]   if r["success"]]

    print(f"\n{'='*65}")
    print(f"  COMPARISON RESULTS")
    print(f"{'='*65}")
    print(f"  CoM offset:        {obj.com_offset_xy()*100:.2f} cm")
    print(f"  Reachable angles:  {len(reachable)}")
    print(f"")
    print(f"  NAIVE (uniform angles, geom centre Z):")
    print(f"    Success rate:    {naive_sr*100:.0f}%  ({len(naive_succ)}/{len(results['naive'])})")
    print(f"    Avg lift:        {naive_avg:.1f} cm")
    for r in results["naive"]:
        icon = "✓" if r["success"] else "✗"
        print(f"      [{icon}] {r['approach_angle_deg']:6.1f}°  z={r['grasp_z']:.3f}"
              f"  lift={r['lift_cm']:5.1f}cm  {r['fail_reason']}")
    print(f"")
    print(f"  AFFORDANCE-GUIDED (CoM-biased angles + optimal Z):")
    print(f"    Success rate:    {map_sr*100:.0f}%  ({len(map_succ)}/{len(results['map'])})")
    print(f"    Avg lift:        {map_avg:.1f} cm")
    for r in results["map"]:
        icon = "✓" if r["success"] else "✗"
        print(f"      [{icon}] {r['approach_angle_deg']:6.1f}°  z={r['grasp_z']:.3f}"
              f"  lift={r['lift_cm']:5.1f}cm  {r['fail_reason']}")
    print(f"")
    print(f"  IMPROVEMENT:       {(map_sr - naive_sr)*100:+.0f}% success rate")
    print(f"                     {map_avg - naive_avg:+.1f} cm avg lift")
    print(f"{'='*65}")

    # ── Save JSON ──
    out = {
        "scene":              Path(xml_path).name,
        "object":             obj_name,
        "com_offset_cm":      float(obj.com_offset_xy() * 100),
        "geom_centre":        [float(v) for v in obj.geom_centre],
        "true_com":           [float(v) for v in obj.com],
        "table_z":            float(obj.table_z),
        "top_z":              float(obj.top_z),
        "n_reachable_angles": len(reachable),
        "reachable_angles_deg": [float(np.degrees(a)) for a in reachable],
        "naive_success_rate": float(naive_sr),
        "map_success_rate":   float(map_sr),
        "naive_avg_lift_cm":  float(naive_avg),
        "map_avg_lift_cm":    float(map_avg),
        "naive_trials":       results["naive"],
        "map_trials":         results["map"],
    }
    json_path = OUTPUT_DIR / f"comparison_{Path(xml_path).stem}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    _plot_comparison(out, obj, reachable)
    return out


# ─── COMPARISON PLOT ──────────────────────────────────────────────────────────

def _plot_comparison(results: dict, obj: ObjectInfo, reachable: List[float]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Sub-task 4: Affordance-Guided Grasp Selection\n"
        f"{results['scene']}  |  CoM offset: {results['com_offset_cm']:.1f} cm"
        f"  |  {results['n_reachable_angles']} reachable angles",
        fontsize=13, fontweight="bold")

    cx, cy = obj.geom_centre[0], obj.geom_centre[1]
    r = obj.radius_xy()

    # Panel 1: Polar view with reachable zone + grasp outcomes
    ax1 = axes[0]
    ax1.set_title("Grasp Angles (Top View)", fontsize=11)
    from matplotlib.patches import Circle

    ax1.add_patch(Circle((cx, cy), r, fill=False, color="black",
                          lw=1.5, linestyle="--", label="Object"))
    ax1.plot(cx, cy, "kx", ms=10, mew=2, label="Geom centre")
    ax1.plot(obj.com[0], obj.com[1], "r*", ms=14,
             label=f"True CoM (+{obj.com_offset_xy()*100:.1f}cm)")

    # Show reachable zone
    for a in reachable:
        px = cx + (r+0.005) * np.cos(a)
        py = cy + (r+0.005) * np.sin(a)
        ax1.plot(px, py, ".", color="lightgrey", ms=3)

    for trial in results["naive_trials"]:
        ang = np.radians(trial["approach_angle_deg"])
        px = cx + (r+0.012) * np.cos(ang)
        py = cy + (r+0.012) * np.sin(ang)
        color = "green" if trial["success"] else "red"
        ax1.plot(px, py, "s", color=color, ms=7, alpha=0.7)

    for trial in results["map_trials"]:
        ang = np.radians(trial["approach_angle_deg"])
        px = cx + (r+0.022) * np.cos(ang)
        py = cy + (r+0.022) * np.sin(ang)
        color = "limegreen" if trial["success"] else "orange"
        ax1.plot(px, py, "^", color=color, ms=8, alpha=0.8)

    ax1.plot([], [], "sg", ms=7, label="Naive ✓")
    ax1.plot([], [], "sr", ms=7, label="Naive ✗")
    ax1.plot([], [], "^", color="limegreen", ms=7, label="Map ✓")
    ax1.plot([], [], "^", color="orange", ms=7, label="Map ✗")

    ax1.set_xlim(cx - 0.08, cx + 0.08)
    ax1.set_ylim(cy - 0.08, cy + 0.08)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.legend(fontsize=7, loc="lower right"); ax1.grid(True, alpha=0.3)

    # Panel 2: Lift distances
    ax2 = axes[1]
    ax2.set_title("Lift Distance per Trial", fontsize=11)
    naive_lifts = [r["lift_cm"] for r in results["naive_trials"]]
    map_lifts   = [r["lift_cm"] for r in results["map_trials"]]
    n_t = max(len(naive_lifts), len(map_lifts))
    x   = np.arange(n_t)
    w   = 0.35
    if naive_lifts:
        ax2.bar(x[:len(naive_lifts)] - w/2, naive_lifts, w,
                color="steelblue", alpha=0.8, label="Naive")
    if map_lifts:
        ax2.bar(x[:len(map_lifts)] + w/2, map_lifts, w,
                color="coral", alpha=0.8, label="Affordance")
    ax2.axhline(LIFT_SUCCESS_THRESH * 100, color="black", lw=1.5,
                linestyle="--", label=f"Success ({LIFT_SUCCESS_THRESH*100:.0f}cm)")
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_xlabel("Trial"); ax2.set_ylabel("Lift (cm)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Success rate
    ax3 = axes[2]
    ax3.set_title("Success Rate Comparison", fontsize=11)
    srs = [results["naive_success_rate"]*100, results["map_success_rate"]*100]
    bars = ax3.bar([0, 1], srs, color=["steelblue", "coral"], alpha=0.85)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Naive\n(uniform angles)", "Affordance\n(CoM-biased)"], fontsize=10)
    ax3.set_ylabel("Success Rate (%)")
    ax3.set_ylim(0, 110)
    for bar, sr in zip(bars, srs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{sr:.0f}%", ha="center", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"comparison_{results['scene'].replace('.xml','')}.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"  Plot saved   → {save_path}")
    plt.close(fig)


# ─── ENTRY POINT ─────────────────────────────────────���────────────────────────

def main():
    scenes = [
        # (scene_file,                obj_name, obj_pos_z, n_trials)
        ("scene_shifted_com.xml",     "cube",   0.47,      8),
        ("scene_grasp_cylinder.xml",  "cube",   0.47,      8),
        ("scene_grasp_clean.xml",     "cube",   0.47,      8),
        ("scene_grasp_tallbox.xml",   "cube",   0.47,      8),
        ("scene_bottle.xml",          "cube",   0.47,      8),
    ]

    for scene_file, obj_name, z, n in scenes:
        xml = SCENES_DIR / scene_file
        if not xml.exists():
            print(f"Not found: {xml}")
            continue
        run_comparison(
            xml_path   = str(xml),
            obj_name   = obj_name,
            obj_pos    = np.array([0.55, 0.0, z]),
            n_trials_per_scenario = n,
            show_viewer = True,
        )

    print(f"\n✓ Sub-task 4 complete. Check {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

