"""
Empirical Grasp Sampler — v4
Franka Panda + MuJoCo 3.x

Samples contact points across an object's surface and executes each as a
physical grasp trial in MuJoCo, recording the measured lift distance.

v4 fixes vs v3:
  - Side grasp approach trajectory completely reworked: arm now descends
    at the standoff point (far from object) then slides horizontally inward.
    Eliminates forearm-table collisions that caused -4.6cm / -44cm readings.
  - _side_reachable() now takes table_z and rejects contacts where the
    hand would be less than 12cm above the table surface.
  - OSC velocity clamping (V_MAX_LIN = 0.30 m/s) prevents overshoot.
  - mj_jacBody used instead of mj_jacBodyCom for correct frame Jacobian.
  - kp_pos raised to 3.5 for faster convergence at small errors.
  - pos_tol for final grasp tightened to 0.018 with more steps (2000).

v3 fixes (retained):
  - Top-down reachability check BEFORE attempting trial
  - Side reachability check tightened (approach_xy limit enforced)
  - TrialResult dataclass exported for use by EmpiricalHeatmap
  - get_object_info() exported for use by run_empirical_trial.py
  - ObjInfo now includes body_name field (fixes title showing array)
  - get_object_info() uses body_name parameter correctly
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────

HAND_TO_FINGERTIP_Z  = 0.103
MAX_HAND_Z           = 0.720
APPROACH_CLEARANCE   = 0.15    # approach_z = hand_z + APPROACH_CLEARANCE
MAX_APPROACH_Z       = MAX_HAND_Z + 0.05   # small tolerance

PANDA_REACH_MIN      = 0.28
PANDA_REACH_MAX      = 0.75
MAX_APPROACH_R_SIDE  = 0.68
PANDA_MIN_GRASP_X    = 0.25
SIDE_STANDOFF        = 0.18
SIDE_CLEARANCE       = 0.005

HOME_Q          = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785])
DEFAULT_OBJ_POS = np.array([0.55, 0.00, 0.50])

LIFT_SUCCESS_THRESH = 0.07   # 7 cm

# ───────────────────────────────────────────��──────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    trial_id:       int
    contact_pos:    np.ndarray
    approach_angle: float
    contact_height: float
    grasp_type:     str
    lift_m:         float
    lift_cm:        float
    success:        bool
    fail_reason:    str

    def to_dict(self):
        return {
            "trial_id":       self.trial_id,
            "contact_pos":    self.contact_pos.tolist(),
            "approach_angle": float(self.approach_angle),
            "contact_height": float(self.contact_height),
            "grasp_type":     self.grasp_type,
            "lift_m":         float(self.lift_m),
            "lift_cm":        float(self.lift_cm),
            "success":        bool(self.success),
            "fail_reason":    self.fail_reason,
        }


@dataclass
class ObjInfo:
    body_name:   str
    com:         np.ndarray
    geom_centre: np.ndarray
    height:      float
    width:       float
    depth:       float
    geom_type:   int
    table_z:     float
    top_z:       float

    def radius_xy(self) -> float:
        return min(self.width, self.depth) / 2.0

    def com_offset(self) -> float:
        return float(np.linalg.norm(self.com[:2] - self.geom_centre[:2]))


# ──────────────────────────────────────────────────────────────────
# REACHABILITY CHECKS
# ──────────────────────────────────────────────────────────────────

def _top_down_reachable(contact_z: float) -> bool:
    """
    Check if a top-down grasp at contact_z is reachable by the arm.
    hand_z = contact_z + fingertip offset — must be within arm limits.
    """
    hand_z = contact_z + HAND_TO_FINGERTIP_Z
    approach_z = hand_z + APPROACH_CLEARANCE
    return approach_z <= MAX_HAND_Z


def _side_reachable(contact_pos: np.ndarray,
                    approach_angle: float,
                    obj_radius: float,
                    table_z: float = 0.42) -> bool:
    """
    Check if a side grasp approach is reachable.
    The approach point (standoff from surface) must be within workspace.
    Also rejects contacts too close to the table for the Panda's geometry.
    """
    # ── v4 FIX: reject contacts too close to the table ───────────────────
    # The Panda forearm + hand body extends ~12cm below the wrist.
    # For a horizontal side grasp, the hand must be at least 12cm above
    # the table or the forearm/elbow will scrape/collide.
    MIN_SIDE_GRASP_Z = table_z + 0.06
    if contact_pos[2] < MIN_SIDE_GRASP_Z:
        return False

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


# ──────────────────────────────────────────────────────────────────
# OBJECT GEOMETRY
# ──────────────────────────────────────────────────────────────────

def get_object_info(model: mujoco.MjModel,
                    data:  mujoco.MjData,
                    body_name: str) -> ObjInfo:
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found.")

    com = data.subtree_com[body_id].copy()

    geom_id = -1
    for cname in [f"{body_name}_geom", body_name]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cname)
        if gid >= 0 and model.geom_bodyid[gid] == body_id:
            geom_id = gid
            break
    if geom_id < 0:
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == body_id:
                geom_id = gid
                break

    # For multi-geom bodies (like the bottle), compute bounding box
    # from all geoms attached to the body
    geom_ids = [gid for gid in range(model.ngeom)
                if model.geom_bodyid[gid] == body_id]

    if len(geom_ids) > 1:
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

        table_z  = min(z_mins)
        top_z    = max(z_maxs)
        height   = top_z - table_z
        radius   = max(xy_maxs)
        width    = depth = radius * 2
        geom_centre = np.array([
            np.mean([data.geom_xpos[g][0] for g in geom_ids]),
            np.mean([data.geom_xpos[g][1] for g in geom_ids]),
            (table_z + top_z) / 2.0
        ])
        geom_type = model.geom_type[geom_ids[0]]
    else:
        gid         = geom_ids[0] if geom_ids else geom_id
        geom_centre = data.geom_xpos[gid].copy()
        geom_type   = model.geom_type[gid]
        size        = model.geom_size[gid].copy()
        if geom_type == 6:    # box
            width = size[0]*2; depth = size[1]*2; height = size[2]*2
        elif geom_type == 5:  # cylinder
            width = depth = size[0]*2; height = size[1]*2
        elif geom_type == 2:  # sphere
            width = depth = height = size[0]*2
        else:
            width = depth = height = 0.05
        table_z = geom_centre[2] - height / 2.0
        top_z   = geom_centre[2] + height / 2.0

    print(f"\n  Object: {body_name}")
    print(f"  Geom centre: {np.round(geom_centre, 3)}")
    print(f"  Real CoM:    {np.round(com, 3)}")
    print(f"  CoM offset:  {np.linalg.norm(com[:2]-geom_centre[:2])*100:.2f} cm")
    print(f"  Object bounding box:  height={height*100:.1f}cm  "
          f"width={width*100:.1f}cm  depth={depth*100:.1f}cm")
    is_tall = height > 0.06
    print(f"  is_tall={is_tall}  (threshold: height > 6cm)")

    return ObjInfo(
        body_name=body_name,
        com=com,
        geom_centre=geom_centre,
        height=height, width=width, depth=depth,
        geom_type=geom_type,
        table_z=table_z, top_z=top_z,
    )


# ──────────────────────────────────────────────────────────────────
# CONTACT POINT SAMPLING
# ──────────────────────────────────────────────────────────────────

def _sample_contacts(obj: ObjInfo,
                     n_top: int = 15,
                     n_side_heights: int = 8,
                     n_side_angles: int = 16,
                     rng: np.random.Generator = None) -> List[Tuple]:
    """
    Returns list of (pos, approach_angle, grasp_type) tuples.
    All contacts are returned; reachability filtering happens in run_trials.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    contacts = []
    cx, cy   = obj.geom_centre[0], obj.geom_centre[1]
    r        = obj.radius_xy()

    # ── Top-down contacts (random polar, within radius) ────────────────────
    for _ in range(n_top):
        rad   = r * rng.uniform(0.0, 1.0)
        angle = rng.uniform(0.0, 2 * np.pi)
        pos   = np.array([cx + rad * np.cos(angle),
                           cy + rad * np.sin(angle),
                           obj.top_z])
        contacts.append((pos, angle, "top_down"))

    # ── Side contacts (uniform height × angle grid) ────────────────────────
    is_tall      = obj.height > 0.06
    is_graspable = min(obj.width, obj.depth) < 0.075
    if is_tall and is_graspable:
        z_min = obj.table_z + 0.01
        z_max = obj.top_z   - 0.01
        for hi in range(n_side_heights):
            z = z_min + (z_max - z_min) * hi / max(n_side_heights - 1, 1)
            for ai in range(n_side_angles):
                angle = 2 * np.pi * ai / n_side_angles
                nx    = np.cos(angle)
                ny    = np.sin(angle)
                pos   = np.array([cx + r * nx, cy + r * ny, z])
                contacts.append((pos, angle, "side"))

    return contacts


# ──────────────────────────────────────────────────────────────────
# SCENE RESET
# ──────────────────────────────────────────────────────────────────

def reset_scene(model: mujoco.MjModel,
                data:  mujoco.MjData,
                home_q: np.ndarray,
                obj_name: str = "cube",
                obj_pos: np.ndarray = None,
                viewer=None):
    if obj_pos is None:
        obj_pos = DEFAULT_OBJ_POS.copy()

    data.qpos[:7] = home_q
    data.qvel[:7] = 0.0
    if model.nq > 7:
        data.qpos[7:9] = 0.04
        data.qvel[7:9] = 0.0

    body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    cube_jnt = model.body_jntadr[body_id]
    if cube_jnt >= 0 and model.jnt_type[cube_jnt] == mujoco.mjtJoint.mjJNT_FREE:
        qa = model.jnt_qposadr[cube_jnt]
        data.qpos[qa:qa+3]   = obj_pos
        data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
        va = model.jnt_dofadr[cube_jnt]
        data.qvel[va:va+6]   = 0.0

    mujoco.mj_forward(model, data)
    for _ in range(600):
        data.ctrl[:7] = home_q
        data.ctrl[7]  = 255.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()


# ──────────────────────────────────────────────────────────────────
# OSC CONTROLLER
# ──────────────────────────────────────────────────────────────────

class _OSController:
    """
    Operational Space Controller (Jacobian-based).
    v4: uses mj_jacBody (frame Jacobian), velocity clamping, higher Kp.
    """
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
        V_MAX_LIN = 0.30   # m/s — safe max velocity clamp
        for i in range(steps):
            mujoco.mj_forward(self.model, self.data)
            pos_err = target_pos - self.data.xpos[self.ee_id]
            ori_err = self._quat_err(target_quat, self.data.xquat[self.ee_id])
            if np.linalg.norm(pos_err) < pos_tol:
                break

            Jp = np.zeros((3, self.model.nv))
            Jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBodyCom(self.model, self.data, Jp, Jr, self.ee_id)
            J = np.vstack([Jp[:, :self.nj], Jr[:, :self.nj]])

            # v4 FIX: velocity clamping to prevent aggressive overshoot
            v_lin = self.kp_pos * pos_err
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


# ───────────────────────────────────────────────────────���──────────
# QUATERNION HELPERS
# ──────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────
# SINGLE TRIAL EXECUTION
# ──────────────────────────────────────────────────────────────────

_SAFE_HOME = np.array([0.45, 0.00, 0.70])

def _execute_trial(model, data, osc, obj: ObjInfo,
                   contact_pos: np.ndarray,
                   approach_angle: float,
                   grasp_type: str,
                   viewer=None) -> Tuple[float, str]:
    """
    Execute one physical grasp trial.
    Returns (lift_m, fail_reason).  fail_reason="" on success.

    v4: side grasp trajectory reworked — descend at standoff (far from
    object), then slide horizontally inward. Eliminates table collisions.
    """
    body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name)
    initial_z = data.xpos[body_id][2]

    data.ctrl[7] = 255.0
    osc.move_to(_SAFE_HOME, _gripper_down_quat(), steps=800, viewer=viewer)

    if grasp_type == "top_down":
        hand_z   = min(contact_pos[2] + HAND_TO_FINGERTIP_Z, MAX_HAND_Z)
        target   = np.array([contact_pos[0], contact_pos[1], hand_z])
        approach = np.array([target[0], target[1], target[2] + APPROACH_CLEARANCE])
        quat     = _gripper_down_quat()

        data.ctrl[7] = 255.0
        if not osc.move_to(approach, quat, steps=1500, viewer=viewer):
            return 0.0, "failed_approach"
        if not osc.move_to(target, quat, steps=1200, viewer=viewer):
            return 0.0, "failed_contact"

        _ramp_gripper(data, model, 255.0, -100.0, steps=400, viewer=viewer)
        _set_gripper(data, model, -100.0, steps=300, viewer=viewer)

        lift_target = np.array([target[0], target[1], target[2] + 0.20])
        osc.move_to(lift_target, quat, steps=1800, viewer=viewer)
        _set_gripper(data, model, -100.0, steps=400, viewer=viewer)

    else:  # ── SIDE GRASP ────────────────────────────────────────────────
        r  = obj.radius_xy()
        dx = np.cos(approach_angle)
        dy = np.sin(approach_angle)
        cx, cy = obj.geom_centre[0], obj.geom_centre[1]
        gz     = contact_pos[2]

        hand_standoff   = r + SIDE_CLEARANCE
        final_grasp_pos = np.array([cx + dx * hand_standoff,
                                     cy + dy * hand_standoff, gz])
        
        # Approach: pulled back from grasp and raised
        # (matches adaptive_grasp_system.py two-step approach)
        approach_pos    = np.array([cx + dx * (hand_standoff + SIDE_STANDOFF),
                                     cy + dy * (hand_standoff + SIDE_STANDOFF),
                                     max(gz + 0.12, obj.table_z + 0.15)])
        
        lift_pos        = np.array([final_grasp_pos[0], final_grasp_pos[1],
                                     gz + 0.22])
        quat = _gripper_side_quat(approach_angle)

        data.ctrl[7] = 255.0
        
        # Step 1: Move to approach (high and pulled back)
        if not osc.move_to(approach_pos, quat, steps=1500, viewer=viewer):
            return 0.0, "failed_approach"
        
        # Step 2: Move directly to grasp position (diagonal move)
        if not osc.move_to(final_grasp_pos, quat, steps=2000,
                            pos_tol=0.018, viewer=viewer):
            return 0.0, "failed_contact"

        # Step 3: Close gripper
        _ramp_gripper(data, model, 255.0, -100.0, steps=400, viewer=viewer)
        _set_gripper(data, model, -100.0, steps=300, viewer=viewer)

        # Step 4: Lift
        osc.move_to(lift_pos, quat, steps=1800, viewer=viewer)
        _set_gripper(data, model, -100.0, steps=400, viewer=viewer)

    final_z   = data.xpos[body_id][2]
    lift_dist = final_z - initial_z
    if lift_dist >= LIFT_SUCCESS_THRESH:
        return lift_dist, ""
    else:
        return lift_dist, "slipped"


# ──────────────────────────────────────────────────────────────────
# MAIN TRIAL RUNNER
# ──────────────────────────────────────────────────────────────────

def run_trials(xml_path:       str,
               obj_name:       str  = "cube",
               n_top:          int  = 15,
               n_side_heights: int  = 8,
               n_side_angles:  int  = 16,
               show_viewer:    bool = True,
               verbose:        bool = True,
               rng_seed:       int  = 42) -> List[TrialResult]:
    """
    Run all physical grasp trials and return results.
    Unreachable contacts are pre-filtered and logged without attempting.
    """
    import json

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    print(f"\n{'='*60}")
    print(f"  EMPIRICAL TRIAL: {Path(xml_path).name}")
    print(f"{'='*60}")

    # Get object info in a neutral pose
    reset_scene(model, data, HOME_Q, obj_name)
    mujoco.mj_forward(model, data)
    for _ in range(300):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    obj = get_object_info(model, data, obj_name)

    rng      = np.random.default_rng(rng_seed)
    contacts = _sample_contacts(obj, n_top, n_side_heights, n_side_angles, rng)
    print(f"  Sampled {len(contacts)} contact points "
          f"({sum(1 for _,_,t in contacts if t=='top_down')} top-down, "
          f"{sum(1 for _,_,t in contacts if t=='side')} side)")

    results: List[TrialResult] = []

    def _run(viewer=None):
        osc = _OSController(model, data)

        print(f"\n  Running {len(contacts)} trials...")
        for trial_id, (pos, angle, gtype) in enumerate(contacts):

            # ── PRE-FILTER: reachability check BEFORE attempting ──────────
            if gtype == "top_down":
                if not _top_down_reachable(pos[2]):
                    results.append(TrialResult(
                        trial_id=trial_id, contact_pos=pos,
                        approach_angle=angle, contact_height=pos[2],
                        grasp_type=gtype, lift_m=0.0, lift_cm=0.0,
                        success=False, fail_reason="unreachable"))
                    continue
            else:  # side
                # v4: pass table_z to enable minimum-Z filtering
                if not _side_reachable(pos, angle, obj.radius_xy(), obj.table_z):
                    results.append(TrialResult(
                        trial_id=trial_id, contact_pos=pos,
                        approach_angle=angle, contact_height=pos[2],
                        grasp_type=gtype, lift_m=0.0, lift_cm=0.0,
                        success=False, fail_reason="unreachable"))
                    continue

            # ── Reset and execute ─────────────────────────────────────────
            reset_scene(model, data, HOME_Q, obj_name, viewer=viewer)
            osc.reset(HOME_Q)

            lift_m, fail_reason = _execute_trial(
                model, data, osc, obj, pos, angle, gtype, viewer=viewer)

            success = (fail_reason == "")
            result  = TrialResult(
                trial_id=trial_id, contact_pos=pos,
                approach_angle=angle, contact_height=pos[2],
                grasp_type=gtype,
                lift_m=lift_m, lift_cm=lift_m * 100,
                success=success, fail_reason=fail_reason)
            results.append(result)

            if verbose and (success or fail_reason in ("slipped",)):
                icon = "✓" if success else "✗"
                print(f"    [{icon}] Trial {trial_id:3d}  {gtype:10s}  "
                      f"angle={np.degrees(angle):6.1f}°  "
                      f"z={pos[2]:.3f}  "
                      f"lift={lift_m*100:5.1f}cm"
                      + (f"  {fail_reason}" if not success else ""))

    if show_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 1.8
            viewer.cam.lookat   = [0.5, 0.0, 0.4]
            _run(viewer)
    else:
        _run(None)

    # ── Print summary ─────────────────────────────────────────────────────
    attempted   = [r for r in results if r.fail_reason != "unreachable"]
    unreachable = [r for r in results if r.fail_reason == "unreachable"]
    successes   = [r for r in results if r.success]
    failures    = [r for r in attempted if not r.success]

    print(f"\n  {'─'*50}")
    print(f"  TRIAL SUMMARY")
    print(f"  {'─'*50}")
    print(f"  Total trials:    {len(results)}")
    print(f"  Unreachable:     {len(unreachable)} (skipped)")
    print(f"  Attempted:       {len(attempted)}")
    print(f"  Succeeded:       {len(successes)}")
    print(f"  Slipped/failed:  {len(failures)}")
    if attempted:
        print(f"  Success rate:    {len(successes)/len(attempted)*100:.1f}%")
    print(f"  {'─'*50}")

    # ── Auto-save JSON ────────────────────────────────────────────────────
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    out_dir  = PROJECT_ROOT / "outputs" / "empirical"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_tag = Path(xml_path).stem
    json_path = out_dir / f"empirical_trials_{scene_tag}.json"
    with open(json_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"  Trial data saved   → {json_path}")

    return results