"""
Grasp Affordance Heatmap — v5
Franka Panda + MuJoCo 3.x

Changes in v5:
  9.  plot_mesh()       — coloured surface mesh painted with quality scores
  10. plot_comparison() — side-by-side naive vs affordance-guided comparison
  11. best_contact()    — returns single best ContactSample across all types
  12. naive_contact()   — returns contact closest to geometric centre (ignores CoM)
  13. scene_shifted_com added to main()
"""

import numpy as np
import mujoco
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ─────────────────────────────────────────────────────────────────────────────
# PANDA WORKSPACE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PANDA_BASE_XY        = np.array([0.0, 0.0])
PANDA_REACH_MIN      = 0.28
PANDA_REACH_MAX      = 0.82
MAX_APPROACH_R_SIDE  = 0.68
PANDA_MAX_HAND_Z     = 0.72
PANDA_MIN_GRASP_X    = 0.25
MAX_GRIPPER_WIDTH    = 0.076
HAND_TO_FINGERTIP_Z  = 0.103


# ─────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid_fc(obj_width: float,
                gripper_max: float = MAX_GRIPPER_WIDTH,
                midpoint: float = 0.15,
                steepness: float = 10.0) -> float:
    frac = float(np.clip((gripper_max - obj_width) / gripper_max, 0.0, 1.0))
    return float(1.0 / (1.0 + np.exp(-steepness * (frac - midpoint))))


def _bell_clearance(z: float,
                    table_z: float,
                    top_z: float,
                    width: float = 0.35) -> float:
    if z < table_z + 0.02:
        return 0.0
    height = max(top_z - table_z, 1e-6)
    mid_z  = table_z + height / 2.0
    sigma  = width * height
    score  = float(np.exp(-0.5 * ((z - mid_z) / sigma) ** 2))
    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContactSample:
    position: np.ndarray
    normal: np.ndarray
    grasp_type: str
    quality: float
    reachability: float
    force_closure: float
    clearance: float
    approach_angle: float


@dataclass
class ObjGeom:
    com: np.ndarray          # real physics CoM (from MuJoCo subtree_com)
    geom_centre: np.ndarray  # geometric centre (from geom position)
    height: float
    width: float
    depth: float
    geom_type: int
    table_z: float
    top_z: float

    def radius_xy(self) -> float:
        return min(self.width, self.depth) / 2.0

    def is_tall(self) -> bool:
        return self.height > 0.06

    def is_graspable_laterally(self) -> bool:
        return min(self.width, self.depth) < 0.075

    def com_offset(self) -> np.ndarray:
        """XY offset of real CoM from geometric centre."""
        return self.com[:2] - self.geom_centre[:2]

    def com_offset_magnitude(self) -> float:
        return float(np.linalg.norm(self.com_offset()))


# ─────────────────────────────────────────────────────────────────────────────
# OBJECT GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def _get_object_geometry(model: mujoco.MjModel,
                         data: mujoco.MjData,
                         body_name: str) -> ObjGeom:
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found.")

    # Real physics CoM (already aggregated over the body subtree)
    com = data.subtree_com[body_id].copy()

    # Collect every geom that belongs to this body
    geom_ids = [gid for gid in range(model.ngeom)
                if model.geom_bodyid[gid] == body_id]
    if not geom_ids:
        raise ValueError(f"No geom for '{body_name}'.")

    # Drop invisible geoms (rgba alpha == 0) so the bounding box reflects
    # the *visible* object rather than hidden ballast spheres.
    visible = [gid for gid in geom_ids if model.geom_rgba[gid, 3] > 0.0]
    if visible:
        geom_ids = visible

    # Axis-aligned bounding box across all (visible) geoms.
    mins, maxs = [], []
    for gid in geom_ids:
        pos   = data.geom_xpos[gid]
        size  = model.geom_size[gid]
        gtype = model.geom_type[gid]
        if   gtype == mujoco.mjtGeom.mjGEOM_BOX:
            half = size.copy()
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            half = np.array([size[0], size[0], size[1]])
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = np.array([size[0], size[0], size[0]])
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = np.array([size[0], size[0], size[1] + size[0]])
        else:
            half = np.array([0.025, 0.025, 0.025])
        mins.append(pos - half)
        maxs.append(pos + half)
    mins = np.min(np.stack(mins), axis=0)
    maxs = np.max(np.stack(maxs), axis=0)

    width       = float(maxs[0] - mins[0])
    depth       = float(maxs[1] - mins[1])
    height      = float(maxs[2] - mins[2])
    geom_centre = (mins + maxs) / 2.0

    # Pick the representative geom for outline drawing — largest visible geom,
    # so the dashed XY outline (circle vs rectangle) makes sense.
    primary   = max(geom_ids, key=lambda g: float(np.prod(model.geom_size[g])))
    geom_type = model.geom_type[primary]

    return ObjGeom(
        com         = com,
        geom_centre = geom_centre,
        height      = height,
        width       = width,
        depth       = depth,
        geom_type   = geom_type,
        table_z     = geom_centre[2] - height / 2.0,
        top_z       = geom_centre[2] + height / 2.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SURFACE SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def _sample_top_surface(obj: ObjGeom,
                        n_radial: int = 8,
                        n_angular: int = 12) -> List[Tuple]:
    samples = []
    cx, cy  = obj.geom_centre[0], obj.geom_centre[1]
    r_max   = obj.radius_xy()

    for ri in range(n_radial):
        r = r_max * (ri + 1) / n_radial
        for ai in range(n_angular):
            angle = 2 * np.pi * ai / n_angular
            pos   = np.array([cx + r * np.cos(angle),
                               cy + r * np.sin(angle),
                               obj.top_z])
            samples.append((pos, np.array([0.0, 0.0, 1.0]), 0.0))

    samples.append((np.array([cx, cy, obj.top_z]),
                    np.array([0.0, 0.0, 1.0]), 0.0))
    return samples


def _sample_side_surface(obj: ObjGeom,
                         n_height: int = 8,
                         n_angular: int = 24) -> List[Tuple]:
    if not obj.is_tall():
        return []

    samples = []
    cx, cy  = obj.geom_centre[0], obj.geom_centre[1]
    r       = obj.radius_xy()
    z_min   = obj.table_z + 0.01
    z_max   = obj.top_z   - 0.01

    for hi in range(n_height):
        z = z_min + (z_max - z_min) * hi / max(n_height - 1, 1)
        for ai in range(n_angular):
            angle = 2 * np.pi * ai / n_angular
            nx    = np.cos(angle)
            ny    = np.sin(angle)
            pos   = np.array([cx + r * nx, cy + r * ny, z])
            samples.append((pos, np.array([nx, ny, 0.0]), angle))

    return samples


# ───���─────────────────────────────────────────────────────────────────────────
# QUALITY SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _score_top_down_contact(pos: np.ndarray,
                             normal: np.ndarray,
                             obj: ObjGeom) -> ContactSample:
    hand_z         = pos[2] + HAND_TO_FINGERTIP_Z
    hand_xy        = pos[:2]
    dist_from_base = np.linalg.norm(hand_xy - PANDA_BASE_XY)

    reach_ok = (
        hand_z <= PANDA_MAX_HAND_Z
        and PANDA_REACH_MIN <= dist_from_base <= PANDA_REACH_MAX
        and hand_xy[0] >= PANDA_MIN_GRASP_X
    )

    if not reach_ok:
        reachability = 0.0
    else:
        z_margin     = (PANDA_MAX_HAND_Z - hand_z) / PANDA_MAX_HAND_Z
        r_margin     = 1.0 - abs(dist_from_base - 0.55) / 0.27
        reachability = float(np.clip(0.5 * z_margin + 0.5 * r_margin, 0.0, 1.0))

    # Force closure uses REAL CoM — contact near CoM scores higher
    dist_fc       = np.linalg.norm(pos[:2] - obj.com[:2])
    force_closure = float(1.0 - np.clip(dist_fc / (obj.radius_xy() + 1e-6), 0.0, 1.0))
    clearance     = 1.0

    if reachability < 0.01:
        return ContactSample(pos, normal, "not_feasible", 0.0,
                             reachability, force_closure, clearance, 0.0)

    quality = 0.50 * reachability + 0.35 * force_closure + 0.15 * clearance
    return ContactSample(pos, normal, "top_down", float(quality),
                         reachability, force_closure, clearance, 0.0)


def _score_side_contact(pos: np.ndarray,
                        normal: np.ndarray,
                        approach_angle: float,
                        obj: ObjGeom) -> ContactSample:
    CLEARANCE_MM = 0.005
    STANDOFF     = 0.18

    hand_pos     = pos + normal * CLEARANCE_MM
    approach_pos = (pos + normal * (CLEARANCE_MM + STANDOFF)).copy()
    approach_pos[2] += 0.12

    grasp_xy    = float(np.linalg.norm(hand_pos[:2]))
    approach_xy = float(np.linalg.norm(approach_pos[:2]))

    grasp_ok = (
        PANDA_REACH_MIN <= grasp_xy <= 0.75
        and approach_xy <= MAX_APPROACH_R_SIDE
        and hand_pos[0] >= PANDA_MIN_GRASP_X
    )

    if not grasp_ok:
        reachability = 0.0
    else:
        y_preference = abs(float(np.sin(approach_angle)))
        r_margin     = 1.0 - abs(grasp_xy - 0.55) / 0.27

        # CoM proximity bonus — approach direction toward real CoM scores higher
        com_dir  = obj.com[:2] - obj.geom_centre[:2]
        com_dist = np.linalg.norm(com_dir)
        if com_dist > 1e-4:
            approach_vec  = np.array([np.cos(approach_angle), np.sin(approach_angle)])
            com_alignment = float(np.dot(approach_vec, com_dir / com_dist))
            com_bonus     = 0.5 * (1.0 + com_alignment)   # 0→1
        else:
            com_bonus = 0.5   # symmetric object — no preference

        reachability = float(np.clip(
            0.3 * y_preference + 0.4 * r_margin + 0.3 * com_bonus,
            0.0, 1.0))

    obj_width     = min(obj.width, obj.depth)
    force_closure = _sigmoid_fc(obj_width)
    clearance     = _bell_clearance(pos[2], obj.table_z, obj.top_z)

    if reachability < 0.01 or not obj.is_graspable_laterally():
        return ContactSample(pos, normal, "not_feasible", 0.0,
                             reachability, force_closure, clearance, approach_angle)

    quality = 0.45 * reachability + 0.35 * force_closure + 0.20 * clearance
    return ContactSample(pos, normal, "side", float(quality),
                         reachability, force_closure, clearance, approach_angle)


# ─────────────────────────────────────────────────────────────────────────────
# DRAW HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _draw_object_outline_xy(ax, obj: ObjGeom) -> None:
    from matplotlib.patches import Circle, Rectangle
    cx, cy = obj.geom_centre[0], obj.geom_centre[1]
    r      = obj.radius_xy()
    if obj.geom_type in (5, 2):
        ax.add_patch(Circle((cx, cy), r, fill=False, color="black",
                            lw=1.5, linestyle="--", zorder=4))
    else:
        ax.add_patch(Rectangle((cx - obj.width/2, cy - obj.depth/2),
                                obj.width, obj.depth,
                                fill=False, color="black",
                                lw=1.5, linestyle="--", zorder=4))
    ax.set_xlim(cx - 0.12, cx + 0.12)
    ax.set_ylim(cy - 0.10, cy + 0.10)


def _draw_wireframe_3d(ax, obj: ObjGeom, alpha: float = 0.25) -> None:
    cx, cy = obj.geom_centre[0], obj.geom_centre[1]
    r      = obj.radius_xy()
    tz     = obj.table_z
    tz_top = obj.top_z
    color  = "dimgrey"

    if obj.geom_type in (5, 2):
        theta = np.linspace(0, 2 * np.pi, 64)
        for z in [tz, tz_top]:
            ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                    np.full_like(theta, z), color=color, alpha=alpha, lw=0.8)
        for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            ax.plot([cx + r * np.cos(t), cx + r * np.cos(t)],
                    [cy + r * np.sin(t), cy + r * np.sin(t)],
                    [tz, tz_top], color=color, alpha=alpha, lw=0.8)
    else:
        hw = obj.width  / 2
        hd = obj.depth  / 2
        corners = np.array([[cx-hw, cy-hd], [cx+hw, cy-hd],
                             [cx+hw, cy+hd], [cx-hw, cy+hd]])
        for z in [tz, tz_top]:
            for i in range(4):
                j = (i + 1) % 4
                ax.plot([corners[i,0], corners[j,0]],
                        [corners[i,1], corners[j,1]],
                        [z, z], color=color, alpha=alpha, lw=0.8)
        for i in range(4):
            ax.plot([corners[i,0], corners[i,0]],
                    [corners[i,1], corners[i,1]],
                    [tz, tz_top], color=color, alpha=alpha, lw=0.8)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AffordanceHeatmap:
    """
    Compute and visualise a grasp affordance heatmap for a MuJoCo object.

    Quality model:
        Top-down weights: 0.50 / 0.35 / 0.15
        Side weights:     0.45 / 0.35 / 0.20

    v5 additions:
        - ObjGeom now tracks both real CoM and geometric centre separately
        - Side contact scoring includes CoM proximity bonus
        - plot_mesh(): coloured surface mesh visualisation
        - plot_comparison(): naive vs affordance-guided side-by-side
        - best_contact() / naive_contact() query methods
    """

    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 object_body_name: str = "cube"):
        self.model     = model
        self.data      = data
        self.body_name = object_body_name
        self.obj: Optional[ObjGeom]       = None
        self.samples: List[ContactSample] = []

    # ── COMPUTE ───────────────────────────────────────────────────────────────

    def compute(self,
                n_top_radial:   int = 8,
                n_top_angular:  int = 12,
                n_side_height:  int = 8,
                n_side_angular: int = 24) -> "AffordanceHeatmap":
        mujoco.mj_forward(self.model, self.data)
        self.obj     = _get_object_geometry(self.model, self.data, self.body_name)
        self.samples = []

        for pos, normal, angle in _sample_top_surface(
                self.obj, n_top_radial, n_top_angular):
            self.samples.append(_score_top_down_contact(pos, normal, self.obj))

        for pos, normal, angle in _sample_side_surface(
                self.obj, n_side_height, n_side_angular):
            self.samples.append(
                _score_side_contact(pos, normal, angle, self.obj))

        n_top  = sum(1 for s in self.samples if s.grasp_type == "top_down")
        n_side = sum(1 for s in self.samples if s.grasp_type == "side")
        n_nf   = sum(1 for s in self.samples if s.grasp_type == "not_feasible")
        qs     = [s.quality for s in self.samples if s.quality > 0]

        com_off = self.obj.com_offset_magnitude()
        print(f"AffordanceHeatmap: {len(self.samples)} contact samples computed")
        print(f"  Geom centre:       [{self.obj.geom_centre[0]:.3f}, "
              f"{self.obj.geom_centre[1]:.3f}, {self.obj.geom_centre[2]:.3f}]")
        print(f"  Real CoM:          [{self.obj.com[0]:.3f}, "
              f"{self.obj.com[1]:.3f}, {self.obj.com[2]:.3f}]")
        print(f"  CoM offset (XY):   {com_off*100:.1f} cm  "
              f"{'SHIFTED' if com_off > 0.005 else '(symmetric)'}")
        print(f"  Top-down feasible: {n_top}")
        print(f"  Side feasible:     {n_side}")
        print(f"  Not feasible:      {n_nf}")
        if qs:
            print(f"  Quality: min={min(qs):.2f}  "
                  f"max={max(qs):.2f}  mean={np.mean(qs):.2f}")
        return self

    # ── QUERY METHODS ─────────────────────────────────────────────────────────

    def best_contact(self) -> Optional[ContactSample]:
        """Best contact sample across all feasible grasp types."""
        feasible = [s for s in self.samples
                    if s.grasp_type in ("top_down", "side")]
        if not feasible:
            return None
        return max(feasible, key=lambda s: s.quality)

    def naive_contact(self) -> Optional[ContactSample]:
        """
        Contact closest to geometric centre — simulates naive planner
        that ignores real CoM. This is what a robot WITHOUT the
        affordance map would choose.
        """
        feasible = [s for s in self.samples
                    if s.grasp_type in ("top_down", "side")]
        if not feasible:
            return None
        gc = self.obj.geom_centre
        return min(feasible,
                   key=lambda s: np.linalg.norm(s.position[:2] - gc[:2]))

    def best_side_approach(self, n_best: int = 3) -> List[Tuple[float, float]]:
        side_samps = [s for s in self.samples if s.grasp_type == "side"]
        if not side_samps:
            return []
        angle_quality: dict = {}
        for s in side_samps:
            deg = round(np.degrees(s.approach_angle) / 5.0) * 5.0
            rad = np.radians(deg)
            if rad not in angle_quality or s.quality > angle_quality[rad]:
                angle_quality[rad] = s.quality
        ranked = sorted(angle_quality.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n_best]

    def top_down_quality(self) -> float:
        top_samps = [s for s in self.samples if s.grasp_type == "top_down"]
        if not top_samps:
            return 0.85
        return float(np.mean([s.quality for s in top_samps]))

    # ── 2D PLOT ───────────────────────────────────────────────────────────────

    def plot_2d(self,
                save_path: Optional[str] = None,
                show: bool = False) -> str:
        if not self.samples:
            raise RuntimeError("Call compute() before plot_2d().")
        if save_path is None:
            save_path = f"affordance_heatmap_{self.body_name}.png"

        obj       = self.obj
        cmap      = plt.cm.RdYlGn
        gtype_lbl = {5: "cylinder", 6: "box", 2: "sphere"}.get(
            obj.geom_type, "object")
        com_off   = obj.com_offset_magnitude()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Grasp Affordance Heatmap — {self.body_name}  ({gtype_lbl})\n"
            f"{obj.width*100:.1f} × {obj.depth*100:.1f} × {obj.height*100:.1f} cm"
            + (f"  |  CoM offset: {com_off*100:.1f} cm" if com_off > 0.005 else ""),
            fontsize=14, fontweight="bold")

        ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

        # ── Panel 1: Top view XY ──────────────────────────────────────────────
        ax1.set_title("Top View  (top-down grasp quality)", fontsize=11)
        top_samps = [s for s in self.samples if s.grasp_type == "top_down"]
        nf_top    = [s for s in self.samples
                     if s.grasp_type == "not_feasible"
                     and s.position[2] >= obj.top_z - 0.01]

        if top_samps:
            sc = ax1.scatter(
                [s.position[0] for s in top_samps],
                [s.position[1] for s in top_samps],
                c=[s.quality   for s in top_samps],
                cmap=cmap, vmin=0, vmax=1, s=90, edgecolors="none", zorder=3)
            plt.colorbar(sc, ax=ax1, label="Grasp quality")
        if nf_top:
            ax1.scatter([s.position[0] for s in nf_top],
                        [s.position[1] for s in nf_top],
                        c="lightgrey", s=40, edgecolors="none",
                        alpha=0.5, zorder=2, label="Not feasible")

        # Draw CoM marker if shifted
        if com_off > 0.005:
            ax1.plot(obj.com[0], obj.com[1], "r*", ms=14,
                     zorder=6, label=f"Real CoM (offset={com_off*100:.1f}cm)")
            ax1.plot(obj.geom_centre[0], obj.geom_centre[1], "kx", ms=10,
                     zorder=6, label="Geom centre")

        _draw_object_outline_xy(ax1, obj)
        ax1.plot(*PANDA_BASE_XY, "k^", ms=10, label="Robot base", zorder=5)
        ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
        ax1.set_aspect("equal"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

        # ── Panel 2: Unwrapped side contact map ───────────────────────────────
        ax2.set_title("Side Contact Map  (unwrapped, angle vs height)", fontsize=11)
        side_samps = [s for s in self.samples if s.grasp_type == "side"]
        nf_side    = [s for s in self.samples
                      if s.grasp_type == "not_feasible"
                      and s.position[2] < obj.top_z - 0.01]

        if side_samps:
            ang_deg = [np.degrees(s.approach_angle) % 360 for s in side_samps]
            sc2 = ax2.scatter(ang_deg,
                              [s.position[2] for s in side_samps],
                              c=[s.quality   for s in side_samps],
                              cmap=cmap, vmin=0, vmax=1,
                              s=55, edgecolors="none", zorder=3)
            plt.colorbar(sc2, ax=ax2, label="Grasp quality")
        if nf_side:
            ax2.scatter([np.degrees(s.approach_angle) % 360 for s in nf_side],
                        [s.position[2] for s in nf_side],
                        c="lightgrey", s=25, edgecolors="none",
                        alpha=0.45, zorder=2, label="Not feasible")

        # CoM direction marker
        if com_off > 0.005:
            com_angle_deg = np.degrees(
                np.arctan2(obj.com[1] - obj.geom_centre[1],
                           obj.com[0] - obj.geom_centre[0])) % 360
            ax2.axvline(com_angle_deg, color="red", lw=1.5,
                        linestyle="-.", alpha=0.8, label=f"CoM dir ({com_angle_deg:.0f}°)")

        for centre in [90, 270]:
            ax2.axvspan(centre - 20, centre + 20,
                        color="limegreen", alpha=0.12, zorder=1)
            ax2.axvline(centre, color="green", lw=1.0, linestyle="--", alpha=0.6,
                        label=f"±Y ({centre}°)" if centre == 90 else None)

        ax2.set_xlim(0, 360)
        ax2.set_ylim(obj.table_z - 0.02, obj.top_z + 0.05)
        ax2.set_xlabel("Approach angle (°)"); ax2.set_ylabel("Contact height Z (m)")
        ax2.set_xticks([0, 90, 180, 270, 360])
        ax2.set_xticklabels(["0°\n(+X far)", "90°\n(+Y left)",
                              "180°\n(−X robot)", "270°\n(−Y right)", "360°"])
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

        # ── Panel 3: Quality histogram ────────────────────────────────────────
        ax3.set_title("Quality Distribution", fontsize=11)
        top_qs  = [s.quality for s in self.samples if s.grasp_type == "top_down"]
        side_qs = [s.quality for s in self.samples if s.grasp_type == "side"]
        bins    = np.linspace(0, 1, 21)
        if top_qs:
            ax3.hist(top_qs,  bins=bins, alpha=0.7,
                     color="steelblue", label=f"Top-down (n={len(top_qs)})")
        if side_qs:
            ax3.hist(side_qs, bins=bins, alpha=0.7,
                     color="coral", label=f"Side (n={len(side_qs)})")
        all_q = top_qs + side_qs
        if all_q:
            ax3.axvline(np.mean(all_q), color="k", lw=1.5, linestyle="--",
                        label=f"Mean = {np.mean(all_q):.2f}")
        ax3.set_xlabel("Grasp quality score")
        ax3.set_ylabel("Number of contact samples")
        ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

        # ── Panel 4: Sub-score breakdown ─────────────────────────────��────────
        ax4.set_title("Mean Sub-score Breakdown", fontsize=11)
        sub_labels = ["Reachability", "Force\nClosure", "Clearance"]
        bar_width  = 0.25
        x          = np.arange(len(sub_labels))
        has_data   = False

        if top_samps:
            ax4.bar(x - bar_width / 2,
                    [np.mean([s.reachability  for s in top_samps]),
                     np.mean([s.force_closure for s in top_samps]),
                     np.mean([s.clearance     for s in top_samps])],
                    bar_width, color="steelblue", alpha=0.8,
                    label=f"Top-down (n={len(top_samps)})")
            has_data = True
        if side_samps:
            ax4.bar(x + bar_width / 2,
                    [np.mean([s.reachability  for s in side_samps]),
                     np.mean([s.force_closure for s in side_samps]),
                     np.mean([s.clearance     for s in side_samps])],
                    bar_width, color="coral", alpha=0.8,
                    label=f"Side (n={len(side_samps)})")
            has_data = True

        if has_data:
            ax4.set_xticks(x); ax4.set_xticklabels(sub_labels, fontsize=10)
            ax4.set_ylim(0, 1.05); ax4.set_ylabel("Mean score")
            ax4.axhline(1.0, color="grey", lw=0.8, linestyle=":")
            ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3, axis="y")
        else:
            ax4.text(0.5, 0.5, "No feasible grasps",
                     ha="center", va="center", transform=ax4.transAxes,
                     fontsize=12, color="grey")

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  2D Heatmap saved → {save_path}")
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    # ── 3D PLOT ───────────────────────────────────────────────────────────────

    def plot_3d(self,
                save_path: Optional[str] = None,
                show: bool = False) -> str:
        if not self.samples:
            raise RuntimeError("Call compute() before plot_3d().")
        if save_path is None:
            save_path = f"affordance_heatmap_3d_{self.body_name}.png"

        obj  = self.obj
        cmap = plt.cm.RdYlGn
        norm = mcolors.Normalize(vmin=0, vmax=1)

        xs    = np.array([s.position[0] for s in self.samples])
        ys    = np.array([s.position[1] for s in self.samples])
        zs    = np.array([s.position[2] for s in self.samples])
        qs    = np.array([s.quality     for s in self.samples])
        types = [s.grasp_type           for s in self.samples]

        mask_top  = np.array([t == "top_down"    for t in types])
        mask_side = np.array([t == "side"         for t in types])
        mask_nf   = np.array([t == "not_feasible" for t in types])

        fig = plt.figure(figsize=(12, 9))
        ax  = fig.add_subplot(111, projection="3d")
        _draw_wireframe_3d(ax, obj, alpha=0.30)

        if np.any(mask_top):
            ax.scatter(xs[mask_top], ys[mask_top], zs[mask_top],
                       c=qs[mask_top], cmap=cmap, norm=norm,
                       s=60, marker="o", alpha=0.85,
                       label=f"Top-down (n={mask_top.sum()})")
        if np.any(mask_side):
            ax.scatter(xs[mask_side], ys[mask_side], zs[mask_side],
                       c=qs[mask_side], cmap=cmap, norm=norm,
                       s=40, marker="s", alpha=0.75,
                       label=f"Side (n={mask_side.sum()})")
        if np.any(mask_nf):
            ax.scatter(xs[mask_nf], ys[mask_nf], zs[mask_nf],
                       c="lightgrey", s=15, marker="x", alpha=0.3,
                       label=f"Not feasible (n={mask_nf.sum()})")

        # CoM marker
        if obj.com_offset_magnitude() > 0.005:
            ax.scatter([obj.com[0]], [obj.com[1]], [obj.com[2]],
                       c="red", s=120, marker="*", zorder=10,
                       label=f"Real CoM (offset={obj.com_offset_magnitude()*100:.1f}cm)")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.55, label="Grasp quality score")
        ax.scatter([0], [0], [0], c="black", s=100, marker="^", label="Robot base")

        cx, cy, cz = obj.geom_centre
        r      = obj.radius_xy()
        half_h = obj.height / 2.0
        span   = max(r * 2.5, half_h * 2.5, 0.12)
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_zlim(cz - span, cz + span)

        gtype_lbl = {5: "cylinder", 6: "box", 2: "sphere"}.get(obj.geom_type, "object")
        ax.set_title(
            f"3D Affordance Heatmap — {self.body_name}  ({gtype_lbl})\n"
            f"{obj.width*100:.1f} × {obj.depth*100:.1f} × {obj.height*100:.1f} cm",
            fontsize=12)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  3D Heatmap saved → {save_path}")
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    # ── MESH PLOT ─────────────────────────────────────────────────────────────

    def plot_mesh(self,
                  save_path: Optional[str] = None,
                  show: bool = False) -> str:
        """
        Coloured surface mesh — object surface painted with quality scores.
        Cylinder → tube + top disc.
        Box      → all 5 visible faces.
        Most intuitive visualisation: green = grasp here, red = avoid.
        """
        if not self.samples:
            raise RuntimeError("Call compute() before plot_mesh().")
        if save_path is None:
            save_path = f"affordance_mesh_{self.body_name}.png"

        obj       = self.obj
        cmap      = plt.cm.RdYlGn
        norm      = mcolors.Normalize(vmin=0, vmax=1)
        gtype_lbl = {5: "cylinder", 6: "box", 2: "sphere"}.get(
            obj.geom_type, "object")

        fig = plt.figure(figsize=(13, 9))
        ax  = fig.add_subplot(111, projection="3d")

        cx, cy = obj.geom_centre[0], obj.geom_centre[1]
        r      = obj.radius_xy()

        side_samps = [s for s in self.samples if s.grasp_type == "side"]
        top_samps  = [s for s in self.samples if s.grasp_type == "top_down"]

        def _nearest_q(samps, px, py, pz, w_xy=1.0, w_z=3.0):
            """Quality of nearest sample weighted by XY and Z distance."""
            if not samps:
                return 0.0
            best = min(samps, key=lambda s:
                w_xy * ((s.position[0]-px)**2 + (s.position[1]-py)**2)
                + w_z * (s.position[2]-pz)**2)
            return best.quality

        if obj.geom_type == 5:   # ── CYLINDER ─────────────────────────────
            n_a = 80
            n_h = 50
            angles  = np.linspace(0, 2 * np.pi, n_a)
            heights = np.linspace(obj.table_z, obj.top_z, n_h)

            Q_tube = np.zeros((n_h, n_a))
            for hi, z in enumerate(heights):
                for ai, ang in enumerate(angles):
                    px = cx + r * np.cos(ang)
                    py = cy + r * np.sin(ang)
                    Q_tube[hi, ai] = _nearest_q(side_samps, px, py, z)

            A, H = np.meshgrid(angles, heights)
            X = cx + r * np.cos(A)
            Y = cy + r * np.sin(A)
            Z = H
            ax.plot_surface(X, Y, Z, facecolors=cmap(norm(Q_tube)),
                            rstride=1, cstride=1,
                            antialiased=True, shade=False, alpha=0.93)

            # Top disc
            n_r, n_ta = 25, 80
            radii = np.linspace(0, r, n_r)
            tang  = np.linspace(0, 2 * np.pi, n_ta)
            Q_top = np.zeros((n_r, n_ta))
            for ri2, rad in enumerate(radii):
                for ai2, ang in enumerate(tang):
                    px = cx + rad * np.cos(ang)
                    py = cy + rad * np.sin(ang)
                    Q_top[ri2, ai2] = _nearest_q(top_samps, px, py, obj.top_z)
            R2, A2 = np.meshgrid(radii, tang, indexing="ij")
            Xt = cx + R2 * np.cos(A2)
            Yt = cy + R2 * np.sin(A2)
            ax.plot_surface(Xt, Yt, np.full_like(Xt, obj.top_z),
                            facecolors=cmap(norm(Q_top)),
                            rstride=1, cstride=1,
                            antialiased=True, shade=False, alpha=0.93)

        else:   # ── BOX ──────────────────────────────────────────────────────
            hw = obj.width  / 2
            hd = obj.depth  / 2
            tz = obj.table_z
            tz_top = obj.top_z
            n_u, n_v = 35, 35

            def _face(X, Y, Z, samps):
                Q = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Q[i,j] = _nearest_q(samps, X[i,j], Y[i,j], Z[i,j])
                ax.plot_surface(X, Y, Z, facecolors=cmap(norm(Q)),
                                rstride=1, cstride=1,
                                antialiased=True, shade=False, alpha=0.93)

            us  = np.linspace(cx-hw, cx+hw, n_u)
            vs  = np.linspace(tz, tz_top, n_v)
            us2 = np.linspace(cy-hd, cy+hd, n_u)

            U,  V  = np.meshgrid(us,  vs)
            U2, V2 = np.meshgrid(us2, vs)
            U3, V3 = np.meshgrid(us,  us2)

            _face(U,  np.full_like(U,  cy+hd), V,  side_samps)  # +Y
            _face(U,  np.full_like(U,  cy-hd), V,  side_samps)  # -Y
            _face(np.full_like(U2, cx+hw), U2, V2, side_samps)  # +X
            _face(np.full_like(U2, cx-hw), U2, V2, side_samps)  # -X
            _face(U3, V3, np.full_like(U3, tz_top), top_samps)  # top

        # ── CoM marker ────────────────────────────────────────────────────────
        if obj.com_offset_magnitude() > 0.005:
            ax.scatter([obj.com[0]], [obj.com[1]], [obj.com[2]],
                       c="red", s=200, marker="*", zorder=10,
                       label=f"Real CoM\n(offset={obj.com_offset_magnitude()*100:.1f}cm)")
            ax.scatter([obj.geom_centre[0]], [obj.geom_centre[1]],
                       [obj.geom_centre[2]],
                       c="black", s=100, marker="x", zorder=10,
                       label="Geom centre\n(naive grasp)")
            # Arrow from geom centre to CoM
            ax.quiver(obj.geom_centre[0], obj.geom_centre[1], obj.geom_centre[2],
                      obj.com[0]-obj.geom_centre[0],
                      obj.com[1]-obj.geom_centre[1],
                      obj.com[2]-obj.geom_centre[2],
                      color="red", linewidth=2, arrow_length_ratio=0.4)

        # ── Colorbar + axes ───────────────────────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.1,
                          label="Grasp quality score")
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb.set_ticklabels(["0.0\n(avoid)", "0.25", "0.5", "0.75", "1.0\n(best)"])

        cx2, cy2, cz2 = obj.geom_centre
        span = max(r * 3.0, obj.height * 0.9, 0.15)
        ax.set_xlim(cx2 - span, cx2 + span)
        ax.set_ylim(cy2 - span, cy2 + span)
        ax.set_zlim(obj.table_z - 0.02, obj.top_z + span * 0.4)

        com_str = (f"  |  CoM offset: {obj.com_offset_magnitude()*100:.1f} cm"
                   if obj.com_offset_magnitude() > 0.005 else "")
        ax.set_title(
            f"Affordance Surface Mesh — {self.body_name}  ({gtype_lbl})\n"
            f"{obj.width*100:.1f} × {obj.depth*100:.1f} × {obj.height*100:.1f} cm"
            + com_str,
            fontsize=13, fontweight="bold")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.view_init(elev=28, azim=-55)
        if obj.com_offset_magnitude() > 0.005:
            ax.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Mesh plot saved  → {save_path}")
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    # ── COMPARISON PLOT ────────────────────────────────────────────────���──────

    def plot_comparison(self,
                        naive_result: dict,
                        map_result: dict,
                        save_path: Optional[str] = None,
                        show: bool = False) -> str:
        """
        Side-by-side comparison: Naive planner vs Affordance-guided planner.

        naive_result / map_result dicts:
            {
              "contact":    ContactSample,
              "lift_cm":    float,
              "success":    bool,
              "label":      str,
            }
        """
        if save_path is None:
            save_path = f"affordance_comparison_{self.body_name}.png"

        obj  = self.obj
        cmap = plt.cm.RdYlGn
        norm = mcolors.Normalize(vmin=0, vmax=1)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            f"Naive vs Affordance-Guided Grasp — {self.body_name}\n"
            f"CoM offset: {obj.com_offset_magnitude()*100:.1f} cm  "
            f"(geom centre ≠ real CoM)",
            fontsize=14, fontweight="bold")

        side_samps = [s for s in self.samples if s.grasp_type == "side"]

        for ax, result, title_col in zip(
                axes,
                [naive_result, map_result],
                ["#d9534f", "#5cb85c"]):

            contact = result["contact"]
            lift    = result["lift_cm"]
            success = result["success"]
            label   = result["label"]

            # Background: unwrapped side contact map
            if side_samps:
                ang_deg = [np.degrees(s.approach_angle) % 360 for s in side_samps]
                sc = ax.scatter(ang_deg,
                                [s.position[2] for s in side_samps],
                                c=[s.quality   for s in side_samps],
                                cmap=cmap, norm=norm,
                                s=60, edgecolors="none",
                                alpha=0.6, zorder=2)

            # Mark the chosen contact point
            if contact is not None:
                chosen_deg = np.degrees(contact.approach_angle) % 360
                ax.scatter([chosen_deg], [contact.position[2]],
                           c="white", s=350, zorder=5,
                           edgecolors="black", linewidths=2)
                ax.scatter([chosen_deg], [contact.position[2]],
                           c=[contact.quality], cmap=cmap, norm=norm,
                           s=200, zorder=6, marker="*")
                ax.annotate(
                    f"q = {contact.quality:.3f}",
                    xy=(chosen_deg, contact.position[2]),
                    xytext=(chosen_deg + 25, contact.position[2] + 0.02),
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black"),
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="black"))

            # CoM direction line
            if obj.com_offset_magnitude() > 0.005:
                com_angle_deg = np.degrees(
                    np.arctan2(obj.com[1] - obj.geom_centre[1],
                               obj.com[0] - obj.geom_centre[0])) % 360
                ax.axvline(com_angle_deg, color="red", lw=2,
                           linestyle="-.", alpha=0.9,
                           label=f"Real CoM dir ({com_angle_deg:.0f}°)")

            for centre in [90, 270]:
                ax.axvspan(centre-20, centre+20,
                           color="limegreen", alpha=0.08, zorder=1)

            result_str = f"✓  Lift: {lift:.1f} cm" if success else f"✗  Lift: {lift:.1f} cm  (FAIL)"
            ax.set_title(
                f"{label}\n{result_str}",
                fontsize=13, fontweight="bold", color=title_col)
            ax.set_xlim(0, 360)
            ax.set_ylim(obj.table_z - 0.02, obj.top_z + 0.05)
            ax.set_xlabel("Approach angle (°)", fontsize=11)
            ax.set_ylabel("Contact height Z (m)", fontsize=11)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_xticklabels(["0°", "90°\n(+Y)", "180°", "270°\n(−Y)", "360°"])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                     ax=axes, label="Grasp quality score",
                     shrink=0.7, pad=0.02)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Comparison plot saved → {save_path}")
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    # ── TEXT REPORT ───────────────────────────────────────────────────────────

    def report(self) -> str:
        if not self.samples or self.obj is None:
            return "No data — call compute() first."

        obj          = self.obj
        top_samps    = [s for s in self.samples if s.grasp_type == "top_down"]
        side_samps   = [s for s in self.samples if s.grasp_type == "side"]
        all_feasible = top_samps + side_samps
        gtype_lbl    = {5: "cylinder", 6: "box", 2: "sphere"}.get(
            obj.geom_type, "unknown")

        lines = [
            "", "=" * 60,
            f"  AFFORDANCE REPORT: {self.body_name}",
            "=" * 60,
            f"  Object type:      {gtype_lbl}",
            f"  Dimensions:       {obj.width*100:.1f} × {obj.depth*100:.1f} × {obj.height*100:.1f} cm",
            f"  Geom centre:      [{obj.geom_centre[0]:.3f}, {obj.geom_centre[1]:.3f}, {obj.geom_centre[2]:.3f}]",
            f"  Real CoM:         [{obj.com[0]:.3f}, {obj.com[1]:.3f}, {obj.com[2]:.3f}]",
            f"  CoM offset (XY):  {obj.com_offset_magnitude()*100:.1f} cm",
            f"  Table Z:          {obj.table_z:.3f} m",
            "",
            f"  Total samples:    {len(self.samples)}",
            f"  Top-down:         {len(top_samps)} feasible",
            f"  Side (antip.):    {len(side_samps)} feasible",
            f"  Not feasible:     {len(self.samples) - len(all_feasible)}",
            "",
        ]

        if top_samps:
            tq       = [s.quality for s in top_samps]
            best_top = max(top_samps, key=lambda s: s.quality)
            lines += [
                f"  Top-down quality:  min={min(tq):.2f}  max={max(tq):.2f}  mean={np.mean(tq):.2f}",
                f"  Best top contact:  pos={np.round(best_top.position, 3)}  q={best_top.quality:.3f}",
                "",
            ]

        if side_samps:
            sq        = [s.quality for s in side_samps]
            best_side = max(side_samps, key=lambda s: s.quality)
            naive_c   = self.naive_contact()
            best_c    = self.best_contact()
            lines += [
                f"  Side quality:      min={min(sq):.2f}  max={max(sq):.2f}  mean={np.mean(sq):.2f}",
                f"  Best contact:      pos={np.round(best_side.position, 3)}  q={best_side.quality:.3f}",
                f"    approach={np.degrees(best_side.approach_angle):.0f}°",
                "",
                f"  NAIVE contact:     pos={np.round(naive_c.position, 3)}  q={naive_c.quality:.3f}"
                if naive_c else "",
                f"  MAP contact:       pos={np.round(best_c.position, 3)}  q={best_c.quality:.3f}"
                if best_c else "",
                f"  Quality gap:       {(best_c.quality - naive_c.quality)*100:.1f}%"
                if (naive_c and best_c) else "",
                "",
            ]

        lines.append("=" * 60)
        report_str = "\n".join(lines)
        print(report_str)
        return report_str

    # ── SAVE ALL ──────────────────────────────────────────────────────────────

    def save_all(self, output_dir: str = ".", tag: str = "") -> List[str]:
        out    = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = tag if tag else self.body_name

        p2d   = self.plot_2d  (str(out / f"heatmap_2d_{suffix}.png"))
        p3d   = self.plot_3d  (str(out / f"heatmap_3d_{suffix}.png"))
        pmesh = self.plot_mesh(str(out / f"heatmap_mesh_{suffix}.png"))

        report_str  = self.report()
        report_path = out / f"affordance_report_{suffix}.txt"
        report_path.write_text(report_str, encoding="utf-8")
        print(f"  Report saved     → {report_path}")

        return [p2d, p3d, pmesh, str(report_path)]


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_heatmap_analysis(xml_path: str,
                         obj_name: str = "cube",
                         output_dir: str = "outputs/heatmaps",
                         tag: str = "") -> "AffordanceHeatmap":
    scene_tag = tag if tag else Path(xml_path).stem
    print(f"\n{'='*60}")
    print(f"  AFFORDANCE HEATMAP: {Path(xml_path).name}")
    print(f"{'='*60}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hm = AffordanceHeatmap(model, data, obj_name)
    hm.compute(n_top_radial=10, n_top_angular=16,
               n_side_height=10, n_side_angular=32)
    paths = hm.save_all(output_dir, tag=scene_tag)
    print(f"\n  Saved {len(paths)} files to {output_dir}/")
    return hm


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ROOT       = Path(__file__).resolve().parent.parent
    SCENES_DIR = ROOT / "assets" / "scenes"
    OUTPUT_DIR = ROOT / "outputs" / "heatmaps"

    scenes = [
        ("scene_grasp_clean.xml",    "cube"),
        ("scene_grasp_cylinder.xml", "cube"),
        ("scene_grasp_tallbox.xml",  "cube"),
        ("scene_shifted_com.xml",    "cube"),
        ("scene_bottle.xml",         "cube"),
    ]

    for scene_file, obj_name in scenes:
        xml = SCENES_DIR / scene_file
        if not xml.exists():
            print(f"Not found: {xml}")
            continue
        run_heatmap_analysis(str(xml), obj_name, str(OUTPUT_DIR))

    print(f"\n✓ Heatmap analysis complete. Check {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()