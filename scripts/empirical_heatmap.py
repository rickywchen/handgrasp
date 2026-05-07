"""
Empirical Grasp Affordance Heatmap — v4 (final)
Franka Panda + MuJoCo 3.x

v4 fixes vs v3:
  - plot_mesh(): mesh now tapers like the real bottle (uses per-height radius
    from bottle_sections geometry, not a single max radius cylinder)
  - plot_2d():   slippery band uses hard-coded XML Z range (0.478–0.522)
    instead of data-driven range that spanned the whole bottle height
  - ObjInfo:     optional bottle_sections field for tapered mesh
  - Grey zones:  unchanged from v3 (no-data zones stay grey)
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

LIFT_SUCCESS_THRESH = 0.07    # 7 cm
LIFT_SCORE_MAX      = 0.25    # 25 cm = score 1.0

# Slippery band world-Z bounds (from scene_bottle.xml, geom slippery_band)
# pos z_rel=0.030, half-height=0.020 → world z = 0.47+0.030 ± 0.020
SLIP_BAND_BOT = 0.478
SLIP_BAND_TOP = 0.522


# ─────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

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


@dataclass
class ObjInfo:
    """Minimal geometry info for visualisation."""
    body_name:   str
    com:         np.ndarray
    geom_centre: np.ndarray
    height:      float
    width:       float
    depth:       float
    geom_type:   int       # 5=cylinder, 6=box, 2=sphere
    table_z:     float
    top_z:       float
    # Optional: list of (z_bot, z_top, radius) tuples for tapered mesh.
    # If provided, plot_mesh() draws the correct bottle profile instead of
    # a uniform fat cylinder.
    bottle_sections: Optional[List[Tuple[float, float, float]]] = None

    def radius_xy(self) -> float:
        return min(self.width, self.depth) / 2.0

    def com_offset(self) -> float:
        return float(np.linalg.norm(self.com[:2] - self.geom_centre[:2]))

    def radius_at_z(self, z: float) -> float:
        """Return the bottle surface radius at world height z."""
        if self.bottle_sections:
            for (zb, zt, r) in self.bottle_sections:
                if zb <= z <= zt:
                    return r
            # Clamp to nearest section if out of range
            if z < self.bottle_sections[0][0]:
                return self.bottle_sections[0][2]
            return self.bottle_sections[-1][2]
        return self.radius_xy()


def _lift_score(lift_m: float) -> float:
    return float(np.clip(max(lift_m, 0.0) / LIFT_SCORE_MAX, 0.0, 1.0))


def _from_dict(d: Dict[str, Any]) -> TrialResult:
    return TrialResult(
        trial_id       = d["trial_id"],
        contact_pos    = np.array(d["contact_pos"]),
        approach_angle = float(d["approach_angle"]),
        contact_height = float(d["contact_height"]),
        grasp_type     = d["grasp_type"],
        lift_m         = float(d["lift_m"]),
        lift_cm        = float(d["lift_cm"]),
        success        = bool(d["success"]),
        fail_reason    = d.get("fail_reason", ""),
    )


# ─────────────────────────────────────────────────────────────���───
# DRAW HELPERS
# ─────────────────────────────────────────────────────────────────

def _draw_object_outline_xy(ax, obj: ObjInfo) -> None:
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


def _draw_wireframe_3d(ax, obj: ObjInfo, alpha: float = 0.20) -> None:
    cx, cy = obj.geom_centre[0], obj.geom_centre[1]
    tz     = obj.table_z
    tz_top = obj.top_z
    color  = "dimgrey"
    theta  = np.linspace(0, 2 * np.pi, 64)

    if obj.geom_type in (5, 2):
        if obj.bottle_sections:
            # Draw each section as a separate ring pair + verticals
            for (zb, zt, r) in obj.bottle_sections:
                for z in [zb, zt]:
                    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                            np.full_like(theta, z),
                            color=color, alpha=alpha, lw=0.8)
                for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    ax.plot([cx + r * np.cos(t)] * 2,
                            [cy + r * np.sin(t)] * 2,
                            [zb, zt], color=color, alpha=alpha, lw=0.8)
        else:
            r = obj.radius_xy()
            for z in [tz, tz_top]:
                ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                        np.full_like(theta, z), color=color, alpha=alpha, lw=0.8)
            for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                ax.plot([cx + r * np.cos(t)] * 2,
                        [cy + r * np.sin(t)] * 2,
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
            ax.plot([corners[i,0]] * 2,
                    [corners[i,1]] * 2,
                    [tz, tz_top], color=color, alpha=alpha, lw=0.8)


# ─────────────────────────────────────────────────────────────────
# INTERPOLATION HELPERS
# ─────────────────────────────────────────────────────────────────

def _estimate_grid_spacing(side_r: List[TrialResult],
                            obj_radius: float) -> float:
    if not side_r:
        return 0.05
    angles_used  = sorted(set(round(r.approach_angle, 3) for r in side_r))
    heights_used = sorted(set(round(r.contact_pos[2],  3) for r in side_r))
    n_ang = max(len(angles_used), 1)
    n_hgt = max(len(heights_used), 1)
    angular_spacing = 2 * np.pi * obj_radius / n_ang
    height_spacing  = ((heights_used[-1] - heights_used[0]) / (n_hgt - 1)
                       if n_hgt > 1 else 0.03)
    return 1.5 * max(angular_spacing, height_spacing)


def _nearest_score_with_dist(samps: List[TrialResult],
                              px: float, py: float, pz: float,
                              w_xy: float = 1.0,
                              w_z:  float = 3.0) -> Tuple[float, float]:
    """Return (score, distance) using inverse-distance-weighted average
    of nearby samples instead of single nearest neighbour."""
    if not samps:
        return 0.0, float("inf")

    scores_dists = []
    for s in samps:
        d2 = (w_xy * ((s.contact_pos[0] - px)**2 + (s.contact_pos[1] - py)**2)
              + w_z * (s.contact_pos[2] - pz)**2)
        raw = float(np.sqrt(
            (s.contact_pos[0] - px)**2 +
            (s.contact_pos[1] - py)**2 +
            (s.contact_pos[2] - pz)**2))
        scores_dists.append((_lift_score(s.lift_m), d2, raw))

    # Sort by weighted distance
    scores_dists.sort(key=lambda x: x[1])
    closest_raw = scores_dists[0][2]

    # Use up to K nearest neighbours, weighted by 1/d²
    K = min(5, len(scores_dists))
    eps = 1e-8
    w_sum = 0.0
    s_sum = 0.0
    for i in range(K):
        sc, d2, _ = scores_dists[i]
        w = 1.0 / (d2 + eps)
        w_sum += w
        s_sum += w * sc

    return s_sum / w_sum, closest_raw


# ─────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────

class EmpiricalHeatmap:

    def __init__(self, results: List[TrialResult], obj: ObjInfo):
        self.results      = results
        self.obj          = obj
        self.attempted    = [r for r in results if r.fail_reason != "unreachable"]
        self.successes    = [r for r in results if r.success]
        self.failures     = [r for r in self.attempted if not r.success]
        self.top_results  = [r for r in self.attempted if r.grasp_type == "top_down"]
        self.side_results = [r for r in self.attempted if r.grasp_type == "side"]

    @classmethod
    def from_json(cls, json_path: str, obj: ObjInfo) -> "EmpiricalHeatmap":
        import json
        with open(json_path, "r") as f:
            raw = json.load(f)
        return cls([_from_dict(d) for d in raw], obj)

    def success_rate(self) -> float:
        if not self.attempted:
            return 0.0
        return len(self.successes) / len(self.attempted)

    def mean_lift(self) -> float:
        if not self.successes:
            return 0.0
        return float(np.mean([r.lift_m for r in self.successes]))

    # ── 2D PLOT ──────────────────────────────────────────────────────────────

    def plot_2d(self, save_path: Optional[str] = None,
                show: bool = False) -> str:
        if save_path is None:
            save_path = "empirical_heatmap_2d.png"

        obj     = self.obj
        cmap    = plt.cm.RdYlGn
        norm    = mcolors.Normalize(vmin=0, vmax=1)
        com_off = obj.com_offset()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Empirical Grasp Affordance Map — {obj.body_name}\n"
            f"From {len(self.attempted)} physical trials  |  "
            f"Success rate: {self.success_rate()*100:.1f}%  |  "
            f"CoM offset: {com_off*100:.1f} cm",
            fontsize=13, fontweight="bold")

        ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

        # ── Panel 1: Top view XY ─────────────────────────────────────────────
        ax1.set_title("Top View — Top-Down Grasps\n(colour = lift distance)",
                      fontsize=10)
        if self.top_results:
            sc = ax1.scatter(
                [r.contact_pos[0] for r in self.top_results],
                [r.contact_pos[1] for r in self.top_results],
                c=[_lift_score(r.lift_m) for r in self.top_results],
                cmap=cmap, norm=norm,
                s=120, edgecolors="black", linewidths=0.5, zorder=4)
            plt.colorbar(sc, ax=ax1, label="Lift score (0=0cm, 1=25cm+)")
        if com_off > 0.005:
            ax1.plot(obj.com[0], obj.com[1], "r*", ms=14, zorder=6,
                     label=f"Real CoM (+{com_off*100:.1f}cm)")
            ax1.plot(obj.geom_centre[0], obj.geom_centre[1], "kx", ms=10,
                     zorder=6, label="Geom centre")
        _draw_object_outline_xy(ax1, obj)
        ax1.plot(0, 0, "k^", ms=10, label="Robot base", zorder=5)
        ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
        ax1.set_aspect("equal"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

        # ── Panel 2: Unwrapped side contact map ──────────────────────────────
        ax2.set_title("Side Contact Map — Unwrapped\n"
                      "(angle vs height, colour = lift)", fontsize=10)

        # Grippy base zone — from XML: base_section z_world=0.42..0.44
        ax2.axhspan(obj.table_z, obj.table_z + 0.02,
                    color="limegreen", alpha=0.12, zorder=1,
                    label="Grippy base zone")

        # Slippery band — FIXED: hard-coded from XML, not data-driven
        ax2.axhspan(SLIP_BAND_BOT, SLIP_BAND_TOP,
                    color="tomato", alpha=0.10, zorder=1,
                    label=f"Slippery band (µ=0.15, Z={SLIP_BAND_BOT:.2f}–{SLIP_BAND_TOP:.2f}m)")

        if self.side_results:
            ang_deg = [np.degrees(r.approach_angle) % 360
                       for r in self.side_results]
            sc2 = ax2.scatter(
                ang_deg,
                [r.contact_pos[2] for r in self.side_results],
                c=[_lift_score(r.lift_m) for r in self.side_results],
                cmap=cmap, norm=norm,
                s=60, edgecolors="black", linewidths=0.3,
                alpha=0.85, zorder=3)
            plt.colorbar(sc2, ax=ax2, label="Lift score")
            succ = [r for r in self.side_results if r.success]
            if succ:
                ax2.scatter(
                    [np.degrees(r.approach_angle) % 360 for r in succ],
                    [r.contact_pos[2] for r in succ],
                    facecolors="none", edgecolors="black", s=80,
                    linewidths=1.5, zorder=5)

        if com_off > 0.005:
            com_angle_deg = np.degrees(
                np.arctan2(obj.com[1] - obj.geom_centre[1],
                           obj.com[0] - obj.geom_centre[0])) % 360
            ax2.axvline(com_angle_deg, color="red", lw=2,
                        linestyle="-.", alpha=0.9,
                        label=f"CoM direction ({com_angle_deg:.0f}°)")

        ax2.set_xlim(0, 360)
        ax2.set_ylim(obj.table_z - 0.02, obj.top_z + 0.05)
        ax2.set_xlabel("Approach angle (°)")
        ax2.set_ylabel("Contact height Z (m)")
        ax2.set_xticks([0, 90, 180, 270, 360])
        ax2.set_xticklabels(["0°", "90°\n(+Y)", "180°", "270°\n(-Y)", "360°"])
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

        # ── Panel 3: Lift histogram ──────────────────────────────────────────
        ax3.set_title("Lift Distance Distribution\n(all attempted trials)",
                      fontsize=10)
        top_lifts  = [r.lift_cm for r in self.top_results]
        side_lifts = [r.lift_cm for r in self.side_results]
        bins = np.linspace(-5, 30, 36)
        if top_lifts:
            ax3.hist(top_lifts,  bins=bins, alpha=0.7, color="steelblue",
                     label=f"Top-down (n={len(top_lifts)})")
        if side_lifts:
            ax3.hist(side_lifts, bins=bins, alpha=0.7, color="coral",
                     label=f"Side (n={len(side_lifts)})")
        ax3.axvline(LIFT_SUCCESS_THRESH * 100, color="black",
                    lw=1.8, linestyle="--",
                    label=f"Success threshold ({LIFT_SUCCESS_THRESH*100:.0f}cm)")
        ax3.set_xlabel("Lift distance (cm)")
        ax3.set_ylabel("Number of trials")
        ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

        # ── Panel 4: Success rate by height ──────────────────────────────────
        ax4.set_title("Success Rate by Contact Height\n(side grasps only)",
                      fontsize=10)
        if self.side_results:
            heights = sorted(set(round(r.contact_height, 3)
                                 for r in self.side_results))
            srs, ns, zs = [], [], []
            for h in heights:
                grp = [r for r in self.side_results
                       if abs(r.contact_height - h) < 0.002]
                sr  = sum(1 for r in grp if r.success) / max(len(grp), 1)
                srs.append(sr); ns.append(len(grp)); zs.append(h)

            colors = []
            for z in zs:
                if z <= obj.table_z + 0.02:
                    colors.append("limegreen")
                elif SLIP_BAND_BOT <= z <= SLIP_BAND_TOP:
                    colors.append("tomato")
                else:
                    colors.append("steelblue")

            bars = ax4.barh(zs, srs, height=0.012, color=colors, alpha=0.85)
            for bar, n in zip(bars, ns):
                ax4.text(bar.get_width() + 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f"n={n}", va="center", fontsize=8)
            ax4.axvline(0.5, color="grey", lw=1, linestyle=":",
                        label="50% threshold")
            ax4.set_xlim(0, 1.15)
            ax4.set_xlabel("Success rate")
            ax4.set_ylabel("Contact height Z (m)")
            from matplotlib.patches import Patch
            ax4.legend(handles=[
                Patch(facecolor="limegreen", alpha=0.85, label="Grippy base"),
                Patch(facecolor="tomato",    alpha=0.85, label="Slippery band"),
                Patch(facecolor="steelblue", alpha=0.85, label="Normal body"),
            ], fontsize=8)
        else:
            ax4.text(0.5, 0.5, "No side grasps", ha="center", va="center",
                     transform=ax4.transAxes, fontsize=12, color="grey")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Empirical 2D heatmap saved → {save_path}")
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    # ── 3D PLOT ──────────────────────────────────────────────────────────────

    def plot_3d(self, save_path: Optional[str] = None,
                show: bool = False) -> str:
        if save_path is None:
            save_path = "empirical_heatmap_3d.png"

        obj  = self.obj
        cmap = plt.cm.RdYlGn
        norm = mcolors.Normalize(vmin=0, vmax=1)

        fig = plt.figure(figsize=(12, 9))
        ax  = fig.add_subplot(111, projection="3d")
        _draw_wireframe_3d(ax, obj, alpha=0.20)

        top_r  = [r for r in self.attempted if r.grasp_type == "top_down"]
        side_r = [r for r in self.attempted if r.grasp_type == "side"]

        if top_r:
            ax.scatter(
                [r.contact_pos[0] for r in top_r],
                [r.contact_pos[1] for r in top_r],
                [r.contact_pos[2] for r in top_r],
                c=[_lift_score(r.lift_m) for r in top_r],
                cmap=cmap, norm=norm, s=80, marker="o", alpha=0.85,
                label=f"Top-down (n={len(top_r)})")
        if side_r:
            ax.scatter(
                [r.contact_pos[0] for r in side_r],
                [r.contact_pos[1] for r in side_r],
                [r.contact_pos[2] for r in side_r],
                c=[_lift_score(r.lift_m) for r in side_r],
                cmap=cmap, norm=norm, s=45, marker="s", alpha=0.80,
                label=f"Side (n={len(side_r)})")

        if obj.com_offset() > 0.005:
            ax.scatter([obj.com[0]], [obj.com[1]], [obj.com[2]],
                       c="red", s=180, marker="*", zorder=10,
                       label=f"Real CoM (offset={obj.com_offset()*100:.1f}cm)")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.55,
                     label=f"Lift score  (1.0 = {LIFT_SCORE_MAX*100:.0f}cm+)")

        cx, cy = obj.geom_centre[0], obj.geom_centre[1]
        r      = obj.radius_xy()
        span   = max(r * 3.0, obj.height * 0.8, 0.15)
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_zlim(obj.table_z - 0.02, obj.top_z + span * 0.4)

        gtype_lbl = {5: "cylinder", 6: "box", 2: "sphere"}.get(obj.geom_type, "obj")
        ax.set_title(
            f"Empirical 3D Affordance Map — {obj.body_name}  ({gtype_lbl})\n"
            f"{len(self.attempted)} trials  |  "
            f"{len(self.successes)} successes  |  "
            f"SR={self.success_rate()*100:.1f}%",
            fontsize=12)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Empirical 3D heatmap saved → {save_path}")
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    # ── MESH PLOT ─────────────────────────────────────────────────────────────

    def plot_mesh(self, save_path: Optional[str] = None,
                  show: bool = False) -> str:
        """
        Colour the object surface mesh by empirical lift score.

        v4: mesh now tapers to match the actual bottle profile (base=3cm,
        shoulder=2.2cm, neck=1.5cm) using bottle_sections geometry.
        Grey zones = no trial data nearby (v3 fix, retained).
        """
        if save_path is None:
            save_path = "empirical_heatmap_mesh.png"

        obj       = self.obj
        cmap      = plt.cm.RdYlGn
        norm      = mcolors.Normalize(vmin=0, vmax=1)
        gtype_lbl = {5: "cylinder", 6: "box", 2: "sphere"}.get(
            obj.geom_type, "object")
        UNKNOWN_RGBA = np.array([0.78, 0.78, 0.78, 1.0])

        fig = plt.figure(figsize=(13, 9))
        ax  = fig.add_subplot(111, projection="3d")

        cx, cy = obj.geom_centre[0], obj.geom_centre[1]
        r      = obj.radius_xy()

        side_r = [r2 for r2 in self.attempted if r2.grasp_type == "side"]
        top_r  = [r2 for r2 in self.attempted if r2.grasp_type == "top_down"]

        max_interp_dist = _estimate_grid_spacing(side_r, r)

        if obj.geom_type == 5:   # ── CYLINDER / BOTTLE ─────────────────────
            n_a = 120
            n_h = 80
            angles  = np.linspace(0, 2 * np.pi, n_a)
            heights = np.linspace(obj.table_z, obj.top_z, n_h)

            # Per-height radius — uses bottle_sections if available,
            # otherwise falls back to uniform max radius
            radii_profile = np.array([obj.radius_at_z(z) for z in heights])

            C_tube = np.tile(UNKNOWN_RGBA, (n_h, n_a, 1))

            for hi, z in enumerate(heights):
                rz = radii_profile[hi]
                for ai, ang in enumerate(angles):
                    px = cx + rz * np.cos(ang)
                    py = cy + rz * np.sin(ang)
                    score, dist = _nearest_score_with_dist(side_r, px, py, z)
                    if dist <= max_interp_dist:
                        C_tube[hi, ai, :] = cmap(norm(score))

            # Build tapered surface grid
            A, H       = np.meshgrid(angles, heights)
            R_grid     = radii_profile[:, np.newaxis] * np.ones((1, n_a))
            X          = cx + R_grid * np.cos(A)
            Y          = cy + R_grid * np.sin(A)
            Z          = H
            ax.plot_surface(X, Y, Z, facecolors=C_tube,
                            rstride=1, cstride=1,
                            antialiased=True, shade=False, alpha=0.93)

            # Top disc
            n_r2, n_ta   = 20, 80
            radii_disc   = np.linspace(0, obj.radius_at_z(obj.top_z), n_r2)
            tang         = np.linspace(0, 2 * np.pi, n_ta)
            top_max_dist = obj.radius_at_z(obj.top_z) * 0.8
            C_top        = np.tile(UNKNOWN_RGBA, (n_r2, n_ta, 1))
            if top_r:
                for ri2, rad in enumerate(radii_disc):
                    for ai2, ang in enumerate(tang):
                        px = cx + rad * np.cos(ang)
                        py = cy + rad * np.sin(ang)
                        score, dist = _nearest_score_with_dist(
                            top_r, px, py, obj.top_z, w_xy=1.0, w_z=0.0)
                        if dist <= top_max_dist:
                            C_top[ri2, ai2, :] = cmap(norm(score))
            R2, A2 = np.meshgrid(radii_disc, tang, indexing="ij")
            Xt = cx + R2 * np.cos(A2)
            Yt = cy + R2 * np.sin(A2)
            ax.plot_surface(Xt, Yt, np.full_like(Xt, obj.top_z),
                            facecolors=C_top,
                            rstride=1, cstride=1,
                            antialiased=True, shade=False, alpha=0.93)

        else:   # ── BOX ────────────────────────────────────────────────────
            hw     = obj.width  / 2
            hd     = obj.depth  / 2
            tz     = obj.table_z
            tz_top = obj.top_z
            n_u, n_v = 35, 35

            def _face(X, Y, Z, samps):
                C = np.tile(UNKNOWN_RGBA, (*X.shape, 1))
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        score, dist = _nearest_score_with_dist(
                            samps, X[i,j], Y[i,j], Z[i,j])
                        if dist <= max_interp_dist:
                            C[i, j, :] = cmap(norm(score))
                ax.plot_surface(X, Y, Z, facecolors=C,
                                rstride=1, cstride=1,
                                antialiased=True, shade=False, alpha=0.93)

            us  = np.linspace(cx - hw, cx + hw, n_u)
            vs  = np.linspace(tz, tz_top, n_v)
            us2 = np.linspace(cy - hd, cy + hd, n_u)
            U,  V  = np.meshgrid(us,  vs)
            U2, V2 = np.meshgrid(us2, vs)
            U3, V3 = np.meshgrid(us,  us2)
            _face(U,  np.full_like(U,  cy + hd), V,  side_r)
            _face(U,  np.full_like(U,  cy - hd), V,  side_r)
            _face(np.full_like(U2, cx + hw), U2, V2, side_r)
            _face(np.full_like(U2, cx - hw), U2, V2, side_r)
            _face(U3, V3, np.full_like(U3, tz_top), top_r)

        # ── Markers ───────────────────────────────────────────────────────────
        if obj.com_offset() > 0.005:
            ax.scatter([obj.com[0]], [obj.com[1]], [obj.com[2]],
                       c="red", s=200, marker="*", zorder=10,
                       label=f"Real CoM\n(offset={obj.com_offset()*100:.1f}cm)")
            ax.scatter([obj.geom_centre[0]], [obj.geom_centre[1]],
                       [obj.geom_centre[2]],
                       c="black", s=100, marker="x", zorder=10,
                       label="Geom centre")

        # ── Colorbar ─────────────────────────────────────────────────────────
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.1,
                          label="Empirical lift score")
        cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb.set_ticklabels(["0\n(fail)", "25%", "50%", "75%",
                           f"100%\n(≥{LIFT_SCORE_MAX*100:.0f}cm)"])

        span = max(r * 3.0, obj.height * 0.9, 0.15)
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
        ax.set_zlim(obj.table_z - 0.02, obj.top_z + span * 0.4)

        ax.set_title(
            f"Empirical Affordance Surface — {obj.body_name}  ({gtype_lbl})\n"
            f"{len(self.attempted)} physical trials  |  "
            f"SR={self.success_rate()*100:.1f}%",
            fontsize=13, fontweight="bold")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
        ax.view_init(elev=28, azim=-55)

        from matplotlib.patches import Patch
        from matplotlib.lines  import Line2D
        handles = [Patch(facecolor=UNKNOWN_RGBA[:3], label="No trial data (grey)")]
        if obj.com_offset() > 0.005:
            handles += [
                Line2D([0],[0], marker="*", color="w", markerfacecolor="red",
                       markersize=10,
                       label=f"Real CoM (offset={obj.com_offset()*100:.1f}cm)"),
                Line2D([0],[0], marker="x", color="black",
                       markersize=8, label="Geom centre"),
            ]
        ax.legend(handles=handles, loc="upper left", fontsize=8)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Empirical mesh plot saved  → {save_path}")
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    # ── TEXT REPORT ───────────────────────────────────────────────────────────

    def report(self) -> str:
        obj   = self.obj
        lines = [
            "", "=" * 60,
            f"  EMPIRICAL AFFORDANCE REPORT: {obj.body_name}",
            "=" * 60,
            f"  Total trials:      {len(self.results)}",
            f"  Unreachable:       {len(self.results) - len(self.attempted)}",
            f"  Attempted:         {len(self.attempted)}",
            f"  Successes:         {len(self.successes)}",
            f"  Failures:          {len(self.failures)}",
            f"  Success rate:      {self.success_rate()*100:.1f}%",
            f"  Mean lift (succ):  {self.mean_lift()*100:.1f} cm",
            "",
            f"  Object CoM:        {np.round(obj.com, 3)}",
            f"  Geom centre:       {np.round(obj.geom_centre, 3)}",
            f"  CoM offset:        {obj.com_offset()*100:.2f} cm",
            "",
        ]
        if self.top_results:
            top_s = sum(1 for r in self.top_results if r.success)
            top_f = len(self.top_results) - top_s
            lines += [
                "  TOP-DOWN GRASPS:",
                f"    Attempted: {len(self.top_results)}"
                f"  Success: {top_s}  Fail: {top_f}",
                f"    SR: {top_s/max(len(self.top_results),1)*100:.1f}%", "",
            ]
        if self.side_results:
            side_s = sum(1 for r in self.side_results if r.success)
            side_f = len(self.side_results) - side_s
            fail_counts: Dict[str, int] = {}
            for r in self.side_results:
                if not r.success and r.fail_reason:
                    fail_counts[r.fail_reason] = fail_counts.get(r.fail_reason, 0) + 1
            lines += [
                "  SIDE GRASPS:",
                f"    Attempted: {len(self.side_results)}"
                f"  Success: {side_s}  Fail: {side_f}",
                f"    SR: {side_s/max(len(self.side_results),1)*100:.1f}%",
                f"    Fail breakdown: {fail_counts}", "",
            ]
        lines.append("=" * 60)
        report_str = "\n".join(lines)
        print(report_str)
        return report_str

    # ── SAVE ALL ──────────────────────────────────────────────────────────────

    def save_all(self, output_dir: str = ".", tag: str = "") -> List[str]:
        out    = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = tag if tag else "empirical"
        p2d    = self.plot_2d  (str(out / f"empirical_2d_{suffix}.png"))
        p3d    = self.plot_3d  (str(out / f"empirical_3d_{suffix}.png"))
        pmesh  = self.plot_mesh(str(out / f"empirical_mesh_{suffix}.png"))
        report_str  = self.report()
        report_path = out / f"empirical_report_{suffix}.txt"
        report_path.write_text(report_str, encoding="utf-8")
        print(f"  Report saved       → {report_path}")
        return [p2d, p3d, pmesh, str(report_path)]