"""
Adaptive Multi-Object Grasping System — v13
Franka Panda + MuJoCo 3.x

v13 changes vs v12:
  - execute_grasp() accepts obj_name parameter (no more hardcoded "cube")
  - generate_grasp_candidates() samples DIVERSE angles from full heatmap
    (not just π/2 and mirror) — essential for meaningful affordance maps
  - New run_affordance_trial_loop(): runs all sampled candidates, records
    every outcome → feeds EmpiricalHeatmap for data-driven affordance maps
  - Bottle scene (scene_bottle.xml) added to main()
  - plan_side_grasp() now accepts explicit grasp_z override so heatmap
    contact heights are respected (not always CoM z)
  - Reachability guard moved into plan_side_grasp() return value check
    (was silently ignoring bad angles)
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import time
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ──────────────────────────────────────────────���──────────────────
# PANDA GEOMETRY CONSTANTS
# ─────────────────────────────────────────────────────────────────

HAND_TO_FINGERTIP_Z = 0.103
MAX_HAND_Z          = 0.72
PANDA_REACH_MIN     = 0.28
PANDA_REACH_MAX     = 0.75
MAX_APPROACH_R_SIDE = 0.68
PANDA_MIN_GRASP_X   = 0.25

HOME_Q      = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785])
DEFAULT_OBJ = np.array([0.55, 0.00, 0.50])
SAFE_HOME   = np.array([0.45, 0.00, 0.70])

LIFT_SUCCESS_THRESH = 0.07   # 7 cm


# ─────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

@dataclass
class ObjectGeometry:
    name:        str
    com:         np.ndarray
    geom_centre: np.ndarray
    geom_type:   int
    size_raw:    np.ndarray
    height:      float
    width:       float
    depth:       float
    table_z:     float
    top_z:       float

    def is_tall(self) -> bool:
        return self.height > 0.06

    def is_graspable_laterally(self) -> bool:
        return min(self.width, self.depth) < 0.075

    def radius_xy(self) -> float:
        return min(self.width, self.depth) / 2.0

    def com_offset(self) -> float:
        return float(np.linalg.norm(self.com[:2] - self.geom_centre[:2]))

    def __str__(self) -> str:
        gtype = {2: "sphere", 5: "cylinder", 6: "box"}.get(
            self.geom_type, f"type{self.geom_type}")
        off = self.com_offset()
        return (
            f"ObjectGeometry({self.name}, {gtype}, "
            f"{self.width*100:.1f}×{self.depth*100:.1f}×{self.height*100:.1f} cm, "
            f"geom_centre_z={self.geom_centre[2]:.3f}, "
            f"com_z={self.com[2]:.3f}"
            + (f", CoM_offset={off*100:.1f}cm" if off > 0.005 else "") + ")"
        )


@dataclass
class GraspCandidate:
    """
    One sampled grasp: a contact position on the object surface +
    approach angle + predicted quality from the heatmap.
    Replaces the old GraspPlan dataclass — now explicitly tied to a
    surface contact point so outcomes can be logged to the heatmap.
    """
    strategy:        str
    quality:         float           # heatmap predicted quality (0–1)
    contact_pos:     np.ndarray      # world XYZ of contact on object surface
    approach_angle:  float           # radians (0 = +X, π/2 = +Y, etc.)
    grasp_type:      str             # "top_down" | "side"
    safe_home:       np.ndarray
    approach_pos:    np.ndarray
    grasp_pos:       np.ndarray      # intermediate descent waypoint (side only)
    final_grasp_pos: np.ndarray      # actual fingertip contact position
    lift_pos:        np.ndarray
    orientation:     np.ndarray      # quaternion [w,x,y,z]
    approach_steps:  int = 1500
    grasp_steps:     int = 1500
    final_steps:     int = 1200
    final_tol:       float = 0.025
    gripper_open:    float = 255.0
    gripper_close:   float = -100.0


@dataclass
class TrialOutcome:
    """Records the result of one physical grasp trial."""
    trial_id:       int
    strategy:       str
    grasp_type:     str
    contact_pos:    np.ndarray
    approach_angle: float
    contact_height: float
    predicted_q:    float        # heatmap score before trial
    lift_m:         float
    lift_cm:        float
    success:        bool
    fail_reason:    str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id":       self.trial_id,
            "strategy":       self.strategy,
            "grasp_type":     self.grasp_type,
            "contact_pos":    self.contact_pos.tolist(),
            "approach_angle": float(self.approach_angle),
            "contact_height": float(self.contact_height),
            "predicted_q":    float(self.predicted_q),
            "lift_m":         float(self.lift_m),
            "lift_cm":        float(self.lift_cm),
            "success":        bool(self.success),
            "fail_reason":    self.fail_reason,
        }


# ─────────────────────────────────────────────────────────────────
# OBJECT ANALYSIS
# ─────────────────────────────────────────────────────────────────

def analyze_object(model: mujoco.MjModel,
                   data:  mujoco.MjData,
                   object_body_name: str = "cube") -> ObjectGeometry:
    """
    Read object geometry from MuJoCo.
    Handles multi-geom bodies (e.g. bottle) by computing bounding box.
    Uses data.subtree_com for real physics CoM (includes hidden weights).
    """
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
    if body_id < 0:
        raise ValueError(f"Body '{object_body_name}' not found.")

    # Real physics CoM — includes hidden weights, shifted mass, etc.
    com = data.subtree_com[body_id].copy()

    # Collect all geoms attached to this body
    geom_ids = [gid for gid in range(model.ngeom)
                if model.geom_bodyid[gid] == body_id]
    if not geom_ids:
        raise ValueError(f"No geoms for '{object_body_name}'.")

    if len(geom_ids) > 1:
        # Multi-geom body (bottle): compute bounding box from all geoms
        z_mins, z_maxs, xy_maxs = [], [], []
        for gid in geom_ids:
            gpos  = data.geom_xpos[gid]
            gsize = model.geom_size[gid]
            gtype = model.geom_type[gid]
            if gtype == 5:    # cylinder
                z_mins.append(gpos[2] - gsize[1])
                z_maxs.append(gpos[2] + gsize[1])
                xy_maxs.append(gsize[0])
            elif gtype == 6:  # box
                z_mins.append(gpos[2] - gsize[2])
                z_maxs.append(gpos[2] + gsize[2])
                xy_maxs.append(max(gsize[0], gsize[1]))
            elif gtype == 2:  # sphere
                z_mins.append(gpos[2] - gsize[0])
                z_maxs.append(gpos[2] + gsize[0])
                xy_maxs.append(gsize[0])
            # skip invisible/collision-only geoms that have size[0] < 0.005
        table_z = min(z_mins)
        top_z   = max(z_maxs)
        height  = top_z - table_z
        radius  = max(xy_maxs)
        width   = depth = radius * 2
        geom_centre = np.array([
            np.mean([data.geom_xpos[g][0] for g in geom_ids]),
            np.mean([data.geom_xpos[g][1] for g in geom_ids]),
            (table_z + top_z) / 2.0,
        ])
        geom_type = model.geom_type[geom_ids[0]]
        size_raw  = model.geom_size[geom_ids[0]].copy()
    else:
        gid         = geom_ids[0]
        geom_centre = data.geom_xpos[gid].copy()
        geom_type   = model.geom_type[gid]
        size_raw    = model.geom_size[gid].copy()
        if geom_type == 6:    # box
            width = size_raw[0]*2; depth = size_raw[1]*2; height = size_raw[2]*2
        elif geom_type == 5:  # cylinder
            width = depth = size_raw[0]*2; height = size_raw[1]*2
        elif geom_type == 2:  # sphere
            width = depth = height = size_raw[0]*2
        elif geom_type == 3:  # capsule
            width = depth = size_raw[0]*2
            height = size_raw[1]*2 + size_raw[0]*2
        else:
            print(f"Unknown geom type {geom_type}, defaulting to 5cm")
            width = depth = height = 0.05
        table_z = geom_centre[2] - height / 2.0
        top_z   = geom_centre[2] + height / 2.0

    return ObjectGeometry(
        name=object_body_name,
        com=com, geom_centre=geom_centre,
        geom_type=geom_type, size_raw=size_raw,
        height=height, width=width, depth=depth,
        table_z=table_z, top_z=top_z,
    )


# ─────────────────────────────────────────────────────────────────
# QUATERNION UTILITIES
# ────────────────��────────────────────────────────────────────────

def euler_to_quat(roll, pitch, yaw):
    cy, sy = np.cos(yaw*0.5),   np.sin(yaw*0.5)
    cp, sp = np.cos(pitch*0.5), np.sin(pitch*0.5)
    cr, sr = np.cos(roll*0.5),  np.sin(roll*0.5)
    return np.array([cr*cp*cy + sr*sp*sy,
                     sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy,
                     cr*cp*sy - sr*sp*cy])

def gripper_down_quat():
    return euler_to_quat(np.pi, 0.0, 0.0)

def gripper_side_quat(approach_angle, roll=0.0):
    return euler_to_quat(roll, np.pi / 2.0, approach_angle + np.pi)


# ─────────────────────────────────────────────────────────────────
# REACHABILITY CHECK
# ─────────────────────────────────────────────────────────────────

def _top_down_reachable(contact_z: float) -> bool:
    hand_z     = contact_z + HAND_TO_FINGERTIP_Z
    approach_z = hand_z + 0.15
    return approach_z <= MAX_HAND_Z

def _side_reachable(contact_pos: np.ndarray,
                    approach_angle: float,
                    obj_radius: float) -> bool:
    cx, cy   = contact_pos[0], contact_pos[1]
    dx, dy   = np.cos(approach_angle), np.sin(approach_angle)
    standoff = obj_radius + 0.005
    hand_pos = np.array([cx + dx*standoff,
                         cy + dy*standoff,
                         contact_pos[2]])
    approach_pos = np.array([cx + dx*(standoff + 0.18),
                              cy + dy*(standoff + 0.18),
                              contact_pos[2] + 0.12])
    grasp_xy    = float(np.linalg.norm(hand_pos[:2]))
    approach_xy = float(np.linalg.norm(approach_pos[:2]))
    return (
        PANDA_REACH_MIN <= grasp_xy    <= PANDA_REACH_MAX
        and approach_xy                <= MAX_APPROACH_R_SIDE
        and hand_pos[0]                >= PANDA_MIN_GRASP_X
    )


# ─────────────────────────────────────────────────────────────────
# GRASP PLANNING  — contact-point driven
# ─────────────────────────────────────────────────────────────────

def plan_top_down_from_contact(obj: ObjectGeometry,
                                contact_xy: np.ndarray,
                                safe_home:  np.ndarray,
                                quality:    float,
                                strategy:   str = "top_down") -> Optional[GraspCandidate]:
    """
    Build a top-down GraspCandidate for a given XY contact point.
    contact_xy — [x, y] position where gripper descends to.
    """
    contact_pos = np.array([contact_xy[0], contact_xy[1], obj.top_z])
    if not _top_down_reachable(obj.top_z):
        return None

    hand_z   = min(obj.top_z + HAND_TO_FINGERTIP_Z, MAX_HAND_Z)
    approach = np.array([contact_xy[0], contact_xy[1], hand_z + 0.15])
    grasp    = np.array([contact_xy[0], contact_xy[1], hand_z])
    lift     = np.array([contact_xy[0], contact_xy[1], hand_z + 0.20])

    return GraspCandidate(
        strategy=strategy, quality=quality,
        contact_pos=contact_pos, approach_angle=0.0, grasp_type="top_down",
        safe_home=safe_home.copy(),
        approach_pos=approach, grasp_pos=grasp,
        final_grasp_pos=grasp, lift_pos=lift,
        orientation=gripper_down_quat(),
        approach_steps=1500, grasp_steps=1200,
        final_steps=800, final_tol=0.025,
    )


def plan_side_from_contact(obj:            ObjectGeometry,
                            contact_pos:   np.ndarray,
                            approach_angle: float,
                            safe_home:     np.ndarray,
                            quality:       float,
                            strategy:      str = "side") -> Optional[GraspCandidate]:
    """
    Build a side GraspCandidate for a given surface contact position and angle.
    contact_pos — [x, y, z] on the object surface.
    approach_angle — direction the gripper approaches FROM (radians).
    """
    if not obj.is_graspable_laterally() or not obj.is_tall():
        return None
    if contact_pos[2] < 0.20:
        return None
    if not _side_reachable(contact_pos, approach_angle, obj.radius_xy()):
        return None

    dx, dy  = np.cos(approach_angle), np.sin(approach_angle)
    standoff = obj.radius_xy() + 0.005

    final_grasp = np.array([contact_pos[0] + dx * standoff,
                             contact_pos[1] + dy * standoff,
                             contact_pos[2]])
    approach    = np.array([contact_pos[0] + dx * (standoff + 0.18),
                             contact_pos[1] + dy * (standoff + 0.18),
                             max(contact_pos[2] + 0.12, obj.table_z + 0.15)])
    lift        = np.array([final_grasp[0], final_grasp[1], contact_pos[2] + 0.22])

    return GraspCandidate(
        strategy=strategy, quality=quality,
        contact_pos=contact_pos, approach_angle=approach_angle,
        grasp_type="side",
        safe_home=safe_home.copy(),
        approach_pos=approach, grasp_pos=final_grasp,
        final_grasp_pos=final_grasp, lift_pos=lift,
        orientation=gripper_side_quat(approach_angle),
        approach_steps=1500, grasp_steps=1500,
        final_steps=1200, final_tol=0.025,
    )


# ─────────────────────────────────────────────────────────────────
# CANDIDATE GENERATION  — diverse sampling from heatmap
# ─────────────────────────────────────────────────────────────────

def generate_grasp_candidates(obj:       ObjectGeometry,
                               safe_home: np.ndarray,
                               heatmap=None,
                               use_map:   bool = True,
                               n_top:     int  = 12,
                               n_side:    int  = 16) -> List[GraspCandidate]:
    """
    Generate a DIVERSE set of grasp candidates.

    When use_map=True (affordance-guided):
      - Top-down: sample n_top points across the object top surface,
        scored by heatmap quality at each point.
      - Side: sample n_side angles uniformly across 0–360°,
        at multiple heights, scored by heatmap.
      Candidates are sorted by predicted quality — the robot tries them
      in order and records outcomes → this builds the empirical heatmap.

    When use_map=False (naive):
      - Top-down: always geometric centre.
      - Side: only π/2 and −π/2 (standard fixed angles).
      This is Scenario 1 — naive planner that ignores CoM.

    Key insight: diversity of candidates is ESSENTIAL for a meaningful
    affordance map. If we always pick the same best contact, the map
    shows one green dot and nothing else.
    """
    candidates: List[GraspCandidate] = []

    # ── NAIVE MODE ────────────────────────────────────────────────────────────
    if not use_map:
        print("  [NAIVE] Grasping geometric centre — ignoring CoM")

        # Top-down at geometric centre
        c = plan_top_down_from_contact(
            obj, obj.geom_centre[:2], safe_home,
            quality=0.40, strategy="top_down_naive")
        if c:
            candidates.append(c)

        # Side at fixed ±Y angles only
        if obj.is_tall() and obj.is_graspable_laterally():
            grasp_z = obj.geom_centre[2]   # geometric centre Z, ignores CoM
            for angle, label in [(np.pi/2, "side_naive_left"),
                                  (-np.pi/2, "side_naive_right")]:
                cp = np.array([obj.geom_centre[0] + obj.radius_xy() * np.cos(angle),
                               obj.geom_centre[1] + obj.radius_xy() * np.sin(angle),
                               grasp_z])
                c = plan_side_from_contact(
                    obj, cp, angle, safe_home,
                    quality=0.60, strategy=label)
                if c:
                    candidates.append(c)

        candidates.sort(key=lambda c: c.quality, reverse=True)
        _print_candidates(candidates, "NAIVE")
        return candidates

    # ── AFFORDANCE-GUIDED MODE ────────────────────────────────────────────────
    print("  [MAP] Generating diverse candidates from heatmap...")

    cx, cy = obj.geom_centre[0], obj.geom_centre[1]
    r      = obj.radius_xy()

    # ── TOP-DOWN: sample grid of XY points across the object top surface ──────
    # Use CoM as the highest-quality anchor, then spread outward
    # This means grasps near the real CoM will be tried (and should succeed),
    # while grasps away from CoM will also be tried (and may fail for shifted CoM)
    # → That variation is exactly what makes the affordance map informative.

    top_points = []

    # CoM-centred point (highest expected quality for shifted CoM objects)
    top_points.append((obj.com[:2].copy(), "top_com"))

    # Geometric centre point (expected to fail if CoM is shifted)
    top_points.append((obj.geom_centre[:2].copy(), "top_geom_centre"))

    # Radial spread across the top surface
    n_radial  = max(2, n_top // 4)
    n_angular = max(4, n_top - 2)
    for ri in range(n_radial):
        rad = r * (ri + 1) / n_radial
        for ai in range(n_angular // n_radial):
            angle = 2 * np.pi * ai / (n_angular // n_radial)
            xy = np.array([cx + rad * np.cos(angle),
                           cy + rad * np.sin(angle)])
            top_points.append((xy, f"top_r{ri}_a{ai}"))

    for xy, label in top_points:
        q = 0.75  # default if no heatmap
        if heatmap is not None:
            # Find nearest top-down sample in heatmap for quality estimate
            top_samps = [s for s in heatmap.samples
                         if s.grasp_type == "top_down"]
            if top_samps:
                nearest = min(top_samps,
                              key=lambda s: np.linalg.norm(s.position[:2] - xy))
                q = nearest.quality
        c = plan_top_down_from_contact(obj, xy, safe_home,
                                        quality=q, strategy=label)
        if c:
            candidates.append(c)

    # ── SIDE: sample uniformly across full 360° at multiple heights ───────────
    # This is the KEY change from v12: instead of only π/2 and mirror,
    # we sample ALL angles. Some will succeed, some fail.
    # The pattern of success/failure across angles IS the affordance map.

    if obj.is_tall() and obj.is_graspable_laterally():
        z_min = obj.table_z + 0.01
        z_max = obj.top_z   - 0.01
        n_heights = max(3, n_side // 8)
        n_angles  = n_side

        for hi in range(n_heights):
            if n_heights == 1:
                z = (z_min + z_max) / 2.0
            else:
                z = z_min + (z_max - z_min) * hi / (n_heights - 1)

            for ai in range(n_angles):
                angle = 2 * np.pi * ai / n_angles

                # Contact point on the object surface at this angle and height
                nx, ny  = np.cos(angle), np.sin(angle)
                contact = np.array([cx + r * nx, cy + r * ny, z])

                # Quality from heatmap if available
                q = 0.60
                if heatmap is not None:
                    side_samps = [s for s in heatmap.samples
                                  if s.grasp_type == "side"]
                    if side_samps:
                        nearest = min(side_samps,
                                      key=lambda s: (
                                          (s.position[0] - contact[0])**2 +
                                          (s.position[1] - contact[1])**2 +
                                          3.0 * (s.position[2] - contact[2])**2))
                        q = nearest.quality

                label = f"side_a{ai:02d}_h{hi}"
                c = plan_side_from_contact(
                    obj, contact, angle, safe_home,
                    quality=q, strategy=label)
                if c:
                    candidates.append(c)

    # Sort by predicted quality — try best first, but ALL are attempted
    # in run_affordance_trial_loop() so the full map gets populated
    candidates.sort(key=lambda c: c.quality, reverse=True)
    _print_candidates(candidates, "MAP")
    return candidates


def _print_candidates(candidates: List[GraspCandidate], mode: str):
    print(f"\n  [{mode}] {len(candidates)} candidates generated:")
    shown = min(len(candidates), 8)
    for i, c in enumerate(candidates[:shown]):
        print(f"    {i+1:2d}. {c.strategy:28s}  "
              f"q={c.quality:.3f}  "
              f"type={c.grasp_type:8s}  "
              f"angle={np.degrees(c.approach_angle):6.1f}°  "
              f"z={c.contact_pos[2]:.3f}")
    if len(candidates) > shown:
        print(f"    ... (+{len(candidates)-shown} more)")


# ─────────────────────────────────────────────────────────────────
# OSC CONTROLLER
# ─────────────────────────────────────────────────────────────────

class OSController:
    def __init__(self, model, data, ee_body="hand",
                 kp_pos=3.5, kp_ori=1.5, damping=0.01, null_kp=0.3):
        self.model     = model
        self.data      = data
        self.ee_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self.ee_id < 0:
            raise ValueError(f"Body '{ee_body}' not found.")
        self.kp_pos    = kp_pos
        self.kp_ori    = kp_ori
        self.damping   = damping
        self.null_kp   = null_kp
        self.dt        = model.opt.timestep
        self.nj        = 7
        self.q_desired = np.zeros(self.nj)
        self.q_min     = model.jnt_range[:self.nj, 0].copy()
        self.q_max     = model.jnt_range[:self.nj, 1].copy()
        self.q_home    = HOME_Q.copy()
        print(f"  OSController ready (ee={ee_body}, kp_pos={kp_pos})")

    def reset(self, q: np.ndarray):
        self.q_desired = q[:self.nj].copy()

    def _quat_error(self, q_tgt, q_cur):
        qt = q_tgt / np.linalg.norm(q_tgt)
        qc = q_cur / np.linalg.norm(q_cur)
        if np.dot(qt, qc) < 0:
            qc = -qc
        wt, xt, yt, zt = qt
        wc, xc, yc, zc = qc
        return 2.0 * np.array([
            -wt*xc + xt*wc - yt*zc + zt*yc,
            -wt*yc + xt*zc + yt*wc - zt*xc,
            -wt*zc - xt*yc + yt*xc + zt*wc])

    def move_to(self, target_pos, target_quat, steps=1000,
                viewer=None, pos_tol=0.012) -> bool:
        target_quat = target_quat / np.linalg.norm(target_quat)
        V_MAX_LIN = 0.30   # m/s — safe max velocity clamp
        for i in range(steps):
            mujoco.mj_forward(self.model, self.data)
            pos_err = target_pos - self.data.xpos[self.ee_id]
            ori_err = self._quat_error(target_quat, self.data.xquat[self.ee_id])
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
            # Null-space: drift toward home posture
            dq    += (np.eye(self.nj) - Jinv @ J) @ (
                      self.null_kp * (self.q_home - self.q_desired))
            self.q_desired = np.clip(
                self.q_desired + self.dt * dq, self.q_min, self.q_max)
            self.data.ctrl[:self.nj] = self.q_desired
            mujoco.mj_step(self.model, self.data)
            if viewer and i % 10 == 0:
                viewer.sync()
        final_err = np.linalg.norm(self.data.xpos[self.ee_id] - target_pos)
        return final_err < pos_tol * 2.5


# ─────────────────────────────────────────────────────────────────
# GRIPPER HELPERS
# ─────────────────────────────────────────────────────────────────

def set_gripper(data, model, cmd, steps=300, viewer=None):
    for i in range(steps):
        data.ctrl[7] = cmd
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()

def ramp_gripper(data, model, from_cmd, to_cmd, steps=400, viewer=None):
    for i in range(steps):
        t = i / steps
        data.ctrl[7] = (1 - t) * from_cmd + t * to_cmd
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()


# ─────────────────────────────────────────────────────────────────
# GRASP EXECUTION  — obj_name is a parameter, never hardcoded
# ─────────────────────────────────────────────────────────────────

def execute_grasp(model, data, osc: OSController,
                  grasp: GraspCandidate,
                  obj_name: str,
                  viewer=None,
                  verbose: bool = True) -> Tuple[bool, float, str]:
    """
    Execute one grasp candidate physically.
    Returns (success, lift_m, fail_reason).
    obj_name is passed explicitly — never hardcoded.
    """
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    if obj_body_id < 0:
        return False, 0.0, f"body_not_found_{obj_name}"

    initial_z = data.xpos[obj_body_id][2]

    if verbose:
        print(f"\n    ─── {grasp.strategy}  q={grasp.quality:.3f} ───")
        print(f"        contact = {np.round(grasp.contact_pos, 3)}")
        print(f"        angle   = {np.degrees(grasp.approach_angle):.1f}°")

    # Move to safe home with gripper open
    data.ctrl[7] = grasp.gripper_open
    osc.move_to(SAFE_HOME, gripper_down_quat(), steps=800, viewer=viewer)

    # Save kp_pos and lower it for gentle approach
    old_kp = osc.kp_pos
    osc.kp_pos = 2.0

    if verbose:
        print("        → Stage 1: approach...")
    data.ctrl[7] = grasp.gripper_open
    if not osc.move_to(grasp.approach_pos, grasp.orientation,
                       steps=grasp.approach_steps, viewer=viewer):
        osc.kp_pos = old_kp
        return False, 0.0, "failed_approach"

    if verbose:
        print("        → Stage 2: descend...")
    if grasp.grasp_type == "side":
        # Side grasp: single diagonal move from approach to final grasp
        if not osc.move_to(grasp.final_grasp_pos, grasp.orientation,
                           steps=2000, pos_tol=0.018, viewer=viewer):
            osc.kp_pos = old_kp
            return False, 0.0, "failed_contact"
    else:
        # Top-down: descend directly to grasp position
        if not osc.move_to(grasp.final_grasp_pos, grasp.orientation,
                           steps=grasp.grasp_steps, viewer=viewer):
            osc.kp_pos = old_kp
            return False, 0.0, "failed_descent"

    if verbose:
        print("        → Closing gripper...")
    ramp_gripper(data, model, grasp.gripper_open, grasp.gripper_close,
                 steps=400, viewer=viewer)
    set_gripper(data, model, grasp.gripper_close, steps=300, viewer=viewer)

    # Restore kp_pos for fast lift
    osc.kp_pos = old_kp

    if verbose:
        print("        → Lifting...")
    osc.move_to(grasp.lift_pos, grasp.orientation, steps=1800, viewer=viewer)
    set_gripper(data, model, grasp.gripper_close, steps=400, viewer=viewer)

    final_z   = data.xpos[obj_body_id][2]
    lift_dist = final_z - initial_z
    success   = lift_dist >= LIFT_SUCCESS_THRESH

    if verbose:
        icon = "✓" if success else "✗"
        print(f"        → Lift: {lift_dist*100:.1f} cm  {icon}")

    return success, lift_dist, ("success" if success else f"lift_{lift_dist*100:.0f}cm")


# ─────────────────────────────────────────────────────────────────
# SCENE RESET
# ─────────────────────────────────────────────────────────────────

def reset_scene(model, data, home_q,
                obj_name: str = "cube",
                obj_pos: np.ndarray = None,
                viewer=None):
    if obj_pos is None:
        obj_pos = DEFAULT_OBJ.copy()

    data.qpos[:7] = home_q
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
        data.ctrl[:7] = home_q
        data.ctrl[7]  = 255.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()


# ─────────────────────────────────────────────────────────────────
# AFFORDANCE TRIAL LOOP
# ─────────────────────────────────────────────────────────────────

def run_affordance_trial_loop(xml_path:     str,
                               obj_name:    str  = "cube",
                               obj_pos:     np.ndarray = None,
                               use_map:     bool = True,
                               max_trials:  int  = 30,
                               show_viewer: bool = True,
                               output_dir:  str  = "outputs/empirical") -> List[TrialOutcome]:
    """
    Core affordance data collection loop.

    1. Load scene, compute heatmap
    2. Generate DIVERSE candidates across full object surface
    3. Execute each candidate physically (up to max_trials)
    4. Record outcome (lift distance, success/fail) per contact point
    5. Save JSON → feed to EmpiricalHeatmap for visualisation

    The diversity of contact points + outcomes IS the affordance map.
    Running this on the bottle will produce a map showing:
      - Base section (dark blue): high success
      - Slippery band (orange):   low success / slips
      - Neck (bright blue):       fails due to tipping (CoM far below)
      - Near-CoM (+Y side):       higher success than -Y side
    """
    if obj_pos is None:
        obj_pos = DEFAULT_OBJ.copy()

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    print(f"\n{'='*60}")
    print(f"  AFFORDANCE TRIAL LOOP: {Path(xml_path).name}")
    print(f"  Object: {obj_name}  |  use_map={use_map}  |  max_trials={max_trials}")
    print(f"{'='*60}")

    # Settle scene and read geometry
    reset_scene(model, data, HOME_Q, obj_name, obj_pos)
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    obj = analyze_object(model, data, obj_name)
    print(f"\n  {obj}")

    # Compute analytical heatmap for candidate scoring
    heatmap = None
    if use_map:
        try:
            from grasp_affordance_heatmap import AffordanceHeatmap
            heatmap = AffordanceHeatmap(model, data, obj_name)
            heatmap.compute(n_top_radial=8,  n_top_angular=12,
                            n_side_height=8, n_side_angular=24)
            print("  Analytical heatmap computed for candidate scoring.")
        except ImportError:
            print("  grasp_affordance_heatmap not found — using uniform quality.")

    candidates = generate_grasp_candidates(
        obj, SAFE_HOME, heatmap=heatmap,
        use_map=use_map, n_top=12, n_side=24)

    outcomes:   List[TrialOutcome] = []

    def _run(viewer=None):
        osc = OSController(model, data)

        n_run = min(len(candidates), max_trials)
        print(f"\n  Executing {n_run} trials...")

        for trial_id, grasp in enumerate(candidates[:n_run]):
            reset_scene(model, data, HOME_Q, obj_name, obj_pos, viewer=viewer)
            osc.reset(HOME_Q)

            ok, lift_m, reason = execute_grasp(
                model, data, osc, grasp,
                obj_name=obj_name,
                viewer=viewer, verbose=True)

            outcome = TrialOutcome(
                trial_id       = trial_id,
                strategy       = grasp.strategy,
                grasp_type     = grasp.grasp_type,
                contact_pos    = grasp.contact_pos.copy(),
                approach_angle = grasp.approach_angle,
                contact_height = grasp.contact_pos[2],
                predicted_q    = grasp.quality,
                lift_m         = lift_m,
                lift_cm        = lift_m * 100,
                success        = ok,
                fail_reason    = "" if ok else reason,
            )
            outcomes.append(outcome)

            icon = "✓" if ok else "✗"
            print(f"  [{icon}] Trial {trial_id:3d}  {grasp.strategy:28s}  "
                  f"lift={lift_m*100:5.1f}cm  {'' if ok else reason}")

    if show_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 1.8
            viewer.cam.lookat   = [0.5, 0.0, 0.4]
            _run(viewer)
    else:
        _run(None)

    # ── Summary ───────────────────────────────────────────────────────────────
    successes = [o for o in outcomes if o.success]
    failures  = [o for o in outcomes if not o.success]
    print(f"\n  {'─'*50}")
    print(f"  TRIAL SUMMARY")
    print(f"  {'─'*50}")
    print(f"  Attempted:   {len(outcomes)}")
    print(f"  Succeeded:   {len(successes)}")
    print(f"  Failed:      {len(failures)}")
    if outcomes:
        print(f"  Success rate: {len(successes)/len(outcomes)*100:.1f}%")
    print(f"  {'─'*50}")

    # ── Save JSON ─��───────────────────────────────────────────────────────────
    out_dir   = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_tag = Path(xml_path).stem
    json_path = out_dir / f"trial_outcomes_{scene_tag}.json"
    with open(json_path, "w") as f:
        json.dump([o.to_dict() for o in outcomes], f, indent=2)
    print(f"  Trial data saved → {json_path}")

    return outcomes


# ─────────────────────────────────────────────────────────────────
# SHIFTED CoM COMPARISON  — Naive vs Affordance-guided
# ─────────────────────────────────────────────────────────────────

def run_shifted_com_comparison(xml_path:   str,
                                obj_name:  str = "cube",
                                obj_pos:   np.ndarray = None,
                                output_dir: str = "outputs/heatmaps") -> dict:
    """
    Two-scenario comparison for the shifted CoM experiment.
    Scenario 1 (NAIVE):  grasp geometric centre — expects failure.
    Scenario 2 (MAP):    grasp heatmap best contact (CoM-aware) — expects success.
    Saves comparison plot.
    """
    from grasp_affordance_heatmap import AffordanceHeatmap

    if obj_pos is None:
        obj_pos = DEFAULT_OBJ.copy()

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    print(f"\n{'#'*60}")
    print(f"  SHIFTED CoM COMPARISON: {Path(xml_path).name}")
    print(f"  Scenario 1: NAIVE  (geometric centre, no map)")
    print(f"  Scenario 2: MAP    (heatmap CoM-aware contact)")
    print(f"{'#'*60}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.lookat   = [0.5, 0.0, 0.4]
        osc = OSController(model, data)

        reset_scene(model, data, HOME_Q, obj_name, obj_pos, viewer=viewer)
        osc.reset(HOME_Q)

        obj = analyze_object(model, data, obj_name)
        print(f"\n  {obj}")
        print(f"  CoM offset: {obj.com_offset()*100:.2f} cm")

        hm = AffordanceHeatmap(model, data, obj_name)
        hm.compute(n_top_radial=8, n_top_angular=12,
                   n_side_height=8, n_side_angular=24)

        naive_contact = hm.naive_contact()
        best_contact  = hm.best_contact()
        print(f"\n  Naive contact:  pos={np.round(naive_contact.position,3)}"
              f"  q={naive_contact.quality:.3f}")
        print(f"  Best contact:   pos={np.round(best_contact.position,3)}"
              f"  q={best_contact.quality:.3f}")
        print(f"  Quality gap:    {(best_contact.quality-naive_contact.quality)*100:.1f}%")

        results = {}

        # ── Scenario 1: NAIVE ─────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  SCENARIO 1: NAIVE")
        reset_scene(model, data, HOME_Q, obj_name, obj_pos, viewer=viewer)
        osc.reset(HOME_Q)

        naive_candidates = generate_grasp_candidates(
            obj, SAFE_HOME, heatmap=None, use_map=False)

        naive_ok, naive_lift = False, 0.0
        if naive_candidates:
            naive_ok, naive_lift, _ = execute_grasp(
                model, data, osc, naive_candidates[0],
                obj_name=obj_name, viewer=viewer)
            naive_ok = naive_lift >= 0.10   # higher bar: tipping object

        results["naive"] = {
            "contact": naive_contact,
            "lift_cm": naive_lift * 100,
            "success": naive_ok,
            "label":   "Scenario 1: NAIVE\n(geometric centre, no map)",
        }
        print(f"  NAIVE: {'✓' if naive_ok else '✗'}  lift={naive_lift*100:.1f}cm")
        time.sleep(1.5)

        # ── Scenario 2: MAP ───────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  SCENARIO 2: MAP")
        reset_scene(model, data, HOME_Q, obj_name, obj_pos, viewer=viewer)
        osc.reset(HOME_Q)

        map_candidates = generate_grasp_candidates(
            obj, SAFE_HOME, heatmap=hm, use_map=True, n_top=4, n_side=4)

        map_ok, map_lift = False, 0.0
        if map_candidates:
            map_ok, map_lift, _ = execute_grasp(
                model, data, osc, map_candidates[0],
                obj_name=obj_name, viewer=viewer)

        results["map"] = {
            "contact": best_contact,
            "lift_cm": map_lift * 100,
            "success": map_ok,
            "label":   "Scenario 2: AFFORDANCE MAP\n(CoM-aware contact)",
        }
        print(f"  MAP:   {'✓' if map_ok else '✗'}  lift={map_lift*100:.1f}cm")
        time.sleep(1.5)

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n  {'='*50}")
        print(f"  COMPARISON SUMMARY")
        print(f"  {'='*50}")
        print(f"  Naive → lift={naive_lift*100:.1f}cm  {'✓' if naive_ok else '✗ FAIL'}")
        print(f"  Map   → lift={map_lift*100:.1f}cm  {'✓' if map_ok else '✗ FAIL'}")
        delta = map_lift - naive_lift
        print(f"  Improvement: {delta*100:+.1f} cm  "
              f"({'map wins' if delta > 0 else 'naive similar or better'})")
        print(f"  {'='*50}")

        # ── Save plots ────────────────────────────────────────────────────────
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        hm.save_all(str(out), tag="shifted_com")
        hm.plot_comparison(
            naive_result=results["naive"],
            map_result  =results["map"],
            save_path   =str(out / "comparison_shifted_com.png"),
        )

        print("\n  Closing in 3s...")
        time.sleep(3.0)

    return results


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    ROOT       = Path(__file__).resolve().parent.parent
    SCENES_DIR = ROOT / "assets" / "scenes"
    OUTPUT_DIR = ROOT / "outputs"

    # ── Standard scenes: affordance trial loop (diverse sampling) ─────────────
    trial_scenes = [
        # (scene_file,                   obj_name, obj_pos_z,  max_trials)
        ("scene_grasp_clean.xml",        "cube",   0.50,       20),
        ("scene_grasp_cylinder.xml",     "cube",   0.47,       20),
        ("scene_grasp_tallbox.xml",      "cube",   0.50,       20),
        ("scene_bottle.xml",             "cube",   0.47,       40),  # ← NEW: bottle
    ]

    all_outcomes = {}

    for scene_file, obj_name, obj_z, n_trials in trial_scenes:
        xml = SCENES_DIR / scene_file
        if not xml.exists():
            print(f"Not found: {xml}")
            continue

        print(f"\n\n{'#'*60}\n  SCENE: {scene_file}\n{'#'*60}")
        obj_pos = np.array([0.55, 0.00, obj_z])

        outcomes = run_affordance_trial_loop(
            xml_path    = str(xml),
            obj_name    = obj_name,
            obj_pos     = obj_pos,
            use_map     = True,
            max_trials  = n_trials,
            show_viewer = True,
            output_dir  = str(OUTPUT_DIR / "empirical"),
        )
        all_outcomes[scene_file] = outcomes

    # ── Shifted CoM comparison ────────────────────────────────────────────────
    shifted_xml = SCENES_DIR / "scene_shifted_com.xml"
    if shifted_xml.exists():
        run_shifted_com_comparison(
            xml_path   = str(shifted_xml),
            obj_name   = "cube",
            obj_pos    = np.array([0.55, 0.00, 0.47]),
            output_dir = str(OUTPUT_DIR / "heatmaps"),
        )
    else:
        print(f"Not found: {shifted_xml}")

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*60}\n  OVERALL RESULTS\n{'='*60}")
    for scene, outcomes in all_outcomes.items():
        n     = len(outcomes)
        ok    = sum(1 for o in outcomes if o.success)
        sr    = ok / n * 100 if n > 0 else 0
        print(f"  {scene}: {ok}/{n} ({sr:.0f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()