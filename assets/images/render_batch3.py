#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vector-style chapter banners for《Agent 从零到专家》 Part 3 (CH.09-CH.12).

Unified design system:
  - figsize (16, 6) @ dpi 100 -> 1600x600 px, axis off
  - vertical gradient #0d1b2a (top, deep blue) -> #1b263b (bottom, deep purple)
  - scattered low-alpha dots
  - flat shapes only (Circle / FancyBboxPatch / FancyArrowPatch), glow via
    low-alpha stacked circles
  - palette: cyan #00b4d8 / gold #ffd166 / coral #ef476f / mint #06d6a0 /
    lavender #a9a4f5
  - minimal English text only
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import (
    Circle, Ellipse, Rectangle, Polygon, FancyBboxPatch, FancyArrowPatch,
)
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms

# ---------------------------------------------------------------- palette ---
BLUE   = "#0d1b2a"
PURPLE = "#1b263b"
CYAN   = "#00b4d8"
GOLD   = "#ffd166"
CORAL  = "#ef476f"
MINT   = "#06d6a0"
LAV    = "#a9a4f5"
WHITE  = "#ffffff"

OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/"


# ------------------------------------------------------------ background ---
def new_canvas(seed):
    fig = plt.figure(figsize=(16, 6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # vertical gradient: deep blue (top) -> deep purple (bottom)
    cmap = mcolors.LinearSegmentedColormap.from_list("bg", [PURPLE, BLUE])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, cmap=cmap, extent=(0, 16, 0, 6), origin="lower",
              aspect="auto", zorder=-10, interpolation="bicubic")

    # scattered low-alpha dots
    rng = np.random.default_rng(seed)
    n = 110
    xs = rng.uniform(0.15, 15.85, n)
    ys = rng.uniform(0.15, 5.85, n)
    base = np.array([
        mcolors.to_rgba(WHITE), mcolors.to_rgba(CYAN),
        mcolors.to_rgba(GOLD), mcolors.to_rgba(LAV),
    ])
    pick = rng.integers(0, len(base), n)
    rgba = base[pick].copy()
    rgba[:, 3] = rng.uniform(0.03, 0.11, n)
    sizes = rng.uniform(3, 22, n)
    ax.scatter(xs, ys, c=rgba, s=sizes, edgecolors="none", zorder=-5)

    return fig, ax


def chapter_tag(ax, num, sub):
    ax.text(0.55, 5.52, f"CH.{num}", fontsize=13, fontweight="bold",
            color=GOLD, alpha=0.92, zorder=20)
    ax.text(0.57, 5.14, sub, fontsize=8, color=WHITE, alpha=0.45,
            zorder=20, fontfamily="DejaVu Sans")


def glow(ax, x, y, r, color, layers=3, z=0):
    for k, a in ((0.85, 0.05), (0.55, 0.08), (0.3, 0.12))[:layers]:
        ax.add_patch(Circle((x, y), r * (1 + k), facecolor=color,
                            alpha=a, edgecolor="none", zorder=z))


def node_bg(ax, x, y, r, color, z=6):
    glow(ax, x, y, r, color, z=1)
    ax.add_patch(Circle((x, y), r, facecolor=WHITE, alpha=0.07,
                        edgecolor=color, lw=2.5, zorder=z))


# ----------------------------------------------------------------- icons ---
def icon_eye(ax, x, y, s, color, z=8):
    ax.add_patch(Ellipse((x, y), 1.75 * s, 1.05 * s, facecolor=WHITE,
                         alpha=0.92, edgecolor=color, lw=2.5, zorder=z))
    ax.add_patch(Circle((x, y), 0.42 * s, facecolor=color, zorder=z + 1))
    ax.add_patch(Circle((x, y), 0.19 * s, facecolor=BLUE, zorder=z + 2))
    ax.add_patch(Circle((x + 0.12 * s, y + 0.12 * s), 0.07 * s,
                        facecolor=WHITE, zorder=z + 3))


def icon_gear(ax, x, y, s, color, z=8):
    for a in range(8):
        t = mtransforms.Affine2D().rotate_deg_around(x, y, a * 45)
        ax.add_patch(Rectangle((x - 0.10 * s, y + 0.48 * s), 0.20 * s,
                               0.26 * s, facecolor=color, edgecolor="none",
                               transform=t + ax.transData, zorder=z))
    ax.add_patch(Circle((x, y), 0.56 * s, facecolor=color, zorder=z + 1))
    ax.add_patch(Circle((x, y), 0.23 * s, facecolor=BLUE, zorder=z + 2))


def icon_wrench(ax, x, y, s, color, z=8, rot=35):
    """Ring-end spanner: handle + ring head with dark hole."""
    t = mtransforms.Affine2D().rotate_deg_around(x, y, rot)
    tr = t + ax.transData
    ax.add_patch(Rectangle((x - 0.68 * s, y - 0.11 * s), 1.05 * s, 0.22 * s,
                           facecolor=color, edgecolor="none",
                           transform=tr, zorder=z))
    ax.add_patch(Circle((x + 0.45 * s, y), 0.30 * s, facecolor=color,
                        edgecolor="none", transform=tr, zorder=z))
    ax.add_patch(Circle((x + 0.45 * s, y), 0.13 * s, facecolor=BLUE,
                        edgecolor="none", transform=tr, zorder=z + 1))
    ax.add_patch(Circle((x - 0.68 * s, y), 0.15 * s, facecolor=color,
                        edgecolor="none", transform=tr, zorder=z))


def icon_brain(ax, x, y, s, color, z=8):
    """Blobby brain: overlapping lobes + central fissure + squiggles."""
    lobes = [(-0.26, 0.10, 0.34), (0.26, 0.10, 0.34), (-0.15, -0.20, 0.31),
             (0.15, -0.20, 0.31), (0.0, 0.0, 0.32)]
    for dx, dy, r in lobes:
        ax.add_patch(Circle((x + dx * s, y + dy * s), r * s, facecolor=color,
                            edgecolor="none", zorder=z))
    ax.add_line(Line2D([x, x], [y - 0.40 * s, y + 0.42 * s], lw=2.2,
                       color=BLUE, alpha=0.85, zorder=z + 1,
                       solid_capstyle="round"))
    for dx, dy in ((-0.30, 0.20), (0.32, 0.14)):
        ax.add_patch(Circle((x + dx * s, y + dy * s), 0.05 * s,
                            facecolor=WHITE, alpha=0.5, zorder=z + 1))


def icon_mag(ax, x, y, s, color, z=8):
    ax.add_patch(Circle((x - 0.10 * s, y + 0.12 * s), 0.36 * s,
                        facecolor=color, alpha=0.20, edgecolor="none",
                        zorder=z))
    ax.add_patch(Circle((x - 0.10 * s, y + 0.12 * s), 0.36 * s,
                        facecolor="none", edgecolor=color, lw=5, zorder=z))
    ax.add_line(Line2D([x + 0.16 * s, x + 0.52 * s],
                       [y - 0.14 * s, y - 0.52 * s], lw=6, color=color,
                       solid_capstyle="round", zorder=z))


def icon_robot(ax, x, y, s, color, z=8):
    ax.add_line(Line2D([x, x], [y + 0.45 * s, y + 0.78 * s], lw=3,
                       color=color, zorder=z))
    ax.add_patch(Circle((x, y + 0.90 * s), 0.10 * s, facecolor=GOLD,
                        zorder=z + 1))
    ax.add_patch(FancyBboxPatch(
        (x - 0.62 * s, y - 0.46 * s), 1.24 * s, 0.92 * s,
        boxstyle=f"round,pad=0.04,rounding_size={0.20 * s}",
        facecolor=color, edgecolor="none", zorder=z))
    # ears
    for dx in (-1, 1):
        ax.add_patch(Rectangle((x + dx * 0.70 * s - 0.05 * s, y - 0.10 * s),
                               0.10 * s, 0.24 * s, facecolor=color,
                               edgecolor="none", zorder=z))
    for dx in (-1, 1):
        ax.add_patch(Circle((x + dx * 0.24 * s, y + 0.10 * s), 0.15 * s,
                            facecolor=WHITE, zorder=z + 1))
        ax.add_patch(Circle((x + dx * 0.24 * s, y + 0.10 * s), 0.065 * s,
                            facecolor=BLUE, zorder=z + 2))
    ax.add_line(Line2D([x - 0.18 * s, x + 0.18 * s], [y - 0.20 * s,
                                                      y - 0.20 * s],
                       lw=3.5, color=WHITE, solid_capstyle="round",
                       zorder=z + 1))


def icon_calc(ax, x, y, s, color, z=8):
    ax.add_patch(FancyBboxPatch(
        (x - 0.50 * s, y - 0.62 * s), 1.0 * s, 1.24 * s,
        boxstyle=f"round,pad=0.04,rounding_size={0.16 * s}",
        facecolor=color, edgecolor="none", zorder=z))
    ax.add_patch(FancyBboxPatch(
        (x - 0.34 * s, y + 0.30 * s), 0.68 * s, 0.22 * s,
        boxstyle=f"round,pad=0.02,rounding_size={0.06 * s}",
        facecolor=BLUE, alpha=0.9, edgecolor="none", zorder=z + 1))
    for i in range(3):
        for j in range(3):
            cx = x - 0.28 * s + j * 0.28 * s
            cy = y + 0.02 * s - i * 0.24 * s
            ax.add_patch(Circle((cx, cy), 0.065 * s, facecolor=WHITE,
                                alpha=0.9, zorder=z + 1))


def icon_cloud(ax, x, y, s, color, z=8):
    ax.add_patch(Rectangle((x - 0.48 * s, y - 0.30 * s), 0.96 * s, 0.22 * s,
                           facecolor=color, edgecolor="none", zorder=z))
    for cx, cy, r in ((-0.30, -0.02, 0.30), (0.02, 0.14, 0.40),
                      (0.34, -0.02, 0.30)):
        ax.add_patch(Circle((x + cx * s, y + cy * s), r * s,
                            facecolor=color, edgecolor="none", zorder=z))
    # inner sparkle
    ax.add_patch(Circle((x + 0.02 * s, y + 0.16 * s), 0.12 * s,
                        facecolor=WHITE, alpha=0.35, zorder=z + 1))


def icon_db(ax, x, y, s, color, z=8):
    ax.add_patch(Rectangle((x - 0.42 * s, y - 0.45 * s), 0.84 * s, 0.9 * s,
                           facecolor=color, edgecolor="none", zorder=z))
    ax.add_patch(Ellipse((x, y + 0.45 * s), 0.84 * s, 0.30 * s,
                         facecolor=color, edgecolor="none", zorder=z))
    ax.add_patch(Ellipse((x, y + 0.45 * s), 0.84 * s, 0.30 * s,
                         facecolor="none", edgecolor=BLUE, lw=2, zorder=z + 1))
    for dy in (0.05, -0.35):
        ax.add_patch(Ellipse((x, y + dy * s), 0.84 * s, 0.30 * s,
                             facecolor="none", edgecolor=BLUE, lw=2,
                             zorder=z + 1))


def icon_folder(ax, x, y, s, color, z=8):
    ax.add_patch(Rectangle((x - 0.55 * s, y + 0.22 * s), 0.32 * s, 0.16 * s,
                           facecolor=color, edgecolor="none", zorder=z))
    ax.add_patch(FancyBboxPatch(
        (x - 0.55 * s, y - 0.42 * s), 1.10 * s, 0.64 * s,
        boxstyle=f"round,pad=0.02,rounding_size={0.08 * s}",
        facecolor=color, edgecolor="none", zorder=z))
    ax.add_patch(Rectangle((x - 0.40 * s, y - 0.30 * s), 0.80 * s, 0.10 * s,
                           facecolor=WHITE, alpha=0.25, zorder=z + 1))


def icon_globe(ax, x, y, s, color, z=8):
    ax.add_patch(Circle((x, y), 0.52 * s, facecolor=color, alpha=0.25,
                        edgecolor=color, lw=2.5, zorder=z))
    ax.add_patch(Ellipse((x, y), 0.52 * s, 1.04 * s, facecolor="none",
                         edgecolor=color, lw=2, zorder=z + 1))
    ax.add_line(Line2D([x - 0.52 * s, x + 0.52 * s], [y, y], lw=2,
                       color=color, zorder=z + 1))
    for dy in (0.26, -0.26):
        w = 0.52 * s * np.sqrt(max(1.0 - (dy / (0.52 * s)) ** 2, 0.0))
        ax.add_line(Line2D([x - w, x + w], [y + dy * s, y + dy * s], lw=2,
                           color=color, zorder=z + 1))


def icon_flag(ax, x, y, s, color, z=8):
    ax.add_line(Line2D([x, x], [y - 0.62 * s, y + 0.72 * s], lw=4,
                       color=WHITE, alpha=0.9, zorder=z,
                       solid_capstyle="round"))
    ax.add_patch(Polygon([(x, y + 0.72 * s), (x + 0.80 * s, y + 0.46 * s),
                          (x, y + 0.20 * s)], closed=True,
                         facecolor=color, edgecolor="none", zorder=z + 1))


def icon_doc(ax, x, y, s, color, z=8):
    ax.add_patch(FancyBboxPatch(
        (x - 0.46 * s, y - 0.62 * s), 0.92 * s, 1.24 * s,
        boxstyle=f"round,pad=0.04,rounding_size={0.12 * s}",
        facecolor=WHITE, alpha=0.95, edgecolor=color, lw=2.5, zorder=z))
    ax.text(x, y - 0.02 * s, "{ }", fontsize=15 * s, fontweight="bold",
            color=BLUE, ha="center", va="center", zorder=z + 2,
            fontfamily="DejaVu Sans Mono")
    for dy in (0.34, -0.38):
        ax.add_line(Line2D([x - 0.26 * s, x + 0.26 * s],
                           [y + dy * s, y + dy * s], lw=2, color=BLUE,
                           alpha=0.45, zorder=z + 2,
                           solid_capstyle="round"))


def bubble(ax, x, y, w, h, color, z=8, tail=(0, -1)):
    tx, ty = tail
    n = np.hypot(tx, ty)
    tx, ty = tx / n, ty / n
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.03,rounding_size={h * 0.35}",
        facecolor=WHITE, alpha=0.92, edgecolor=color, lw=2.5, zorder=z))
    # tail: base on bubble edge, tip pointing along (tx, ty)
    bx, by = x + tx * w * 0.22, y + ty * h * 0.55
    tipx, tipy = bx + tx * 0.30, by + ty * 0.30
    px, py = -ty, tx
    ax.add_patch(Polygon(
        [(bx + px * 0.17, by + py * 0.17),
         (bx - px * 0.17, by - py * 0.17),
         (tipx, tipy)], closed=True, facecolor=WHITE, edgecolor=color,
        lw=2.5, zorder=z + 1))


def arrow(ax, a, b, color, rad=0.0, lw=3.5, ms=24, z=7, dashed=False,
          alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
        mutation_scale=ms, lw=lw, color=color, zorder=z, alpha=alpha,
        linestyle=(0, (5, 4)) if dashed else "solid",
        shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(OUT + name, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", OUT + name)


# ============================================================ CH.09 ========
def ch09():
    fig, ax = new_canvas(41)
    chapter_tag(ax, "09", "AGENT CORE LOOP")

    cx, cy, r = 8.0, 3.0, 2.05

    # dashed orbit
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=LAV,
                        lw=1.6, linestyle=(0, (4, 5)), alpha=0.35, zorder=2))

    # central robot (the agent)
    glow(ax, cx, cy, 0.85, LAV)
    icon_robot(ax, cx, cy, 1.05, LAV, z=6)

    nodes = [
        (90,  icon_eye,   CYAN,  "PERCEPTION"),
        (0,   icon_gear,  GOLD,  "PLANNING"),
        (-90, icon_wrench, CORAL, "ACTION"),
        (180, icon_mag,   MINT,  "OBSERVATION"),
    ]
    pts = {}
    for ang, ic, col, lab in nodes:
        a = np.deg2rad(ang)
        x, y = cx + r * np.cos(a), cy + r * np.sin(a)
        pts[ang] = (x, y, col)
        node_bg(ax, x, y, 0.66, col)
        ic(ax, x, y, 0.62, col, z=9)

    # loop arrows between neighbouring nodes (clockwise)
    order = [90, 0, -90, 180]
    for i in range(4):
        a1, a2 = order[i], order[(i + 1) % 4]
        p1 = pts[a1]
        p2 = pts[a2]
        x1, y1, c1 = p1
        x2, y2, c2 = p2
        s1 = (cx + (r + 0.10) * np.cos(np.deg2rad(a1 - 18)),
              cy + (r + 0.10) * np.sin(np.deg2rad(a1 - 18)))
        s2 = (cx + (r + 0.10) * np.cos(np.deg2rad(a2 + 18)),
              cy + (r + 0.10) * np.sin(np.deg2rad(a2 + 18)))
        arrow(ax, s1, s2, c2, rad=0.30, lw=3.5, ms=22, z=5)

    save(fig, "ch09-banner.png")


# ============================================================ CH.10 ========
def ch10():
    fig, ax = new_canvas(7)
    chapter_tag(ax, "10", "REACT  ·  THOUGHT / ACTION / OBSERVATION")

    y_hi, y_lo = 4.30, 1.70
    xs = [2.90 + i * 1.95 for i in range(6)]
    ys = [y_hi if i % 2 == 0 else y_lo for i in range(6)]

    # serpentine guide curve (ends at the last block)
    gx = np.linspace(2.0, 12.9, 400)
    gy = 3.0 + 1.30 * np.sin(np.pi / 2 + np.pi * (gx - xs[0]) / 1.95)
    ax.plot(gx, gy, color=LAV, lw=1.8, linestyle=(0, (5, 6)), alpha=0.30,
            zorder=1)

    seq = [
        ("T", icon_brain, LAV, "Thought"),
        ("A", icon_wrench, CORAL, "Action"),
        ("O", icon_mag, MINT, "Observation"),
    ]

    for i, (x, y) in enumerate(zip(xs, ys)):
        kind, ic, col, lab = seq[i % 3]
        glow(ax, x, y, 0.62, col, z=1)
        ax.add_patch(FancyBboxPatch(
            (x - 0.56, y - 0.56), 1.12, 1.12,
            boxstyle="round,pad=0.04,rounding_size=0.24",
            facecolor=col, alpha=0.20, edgecolor=col, lw=2.5, zorder=6))
        ic(ax, x, y, 0.62, col, z=8)
        ax.text(x, y - 0.86, lab, fontsize=6.5, color=WHITE, alpha=0.6,
                ha="center", va="top", zorder=9)

    # arrows along the snake
    for i in range(5):
        x1, y1, x2, y2 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        rad = 0.28 if y2 > y1 else -0.28
        a = (x1 + 0.66 * (x2 - x1) / abs(x2 - x1 + 0.001) * 0 + 0.66, y1)
        # start just right of current block, end just left of next block
        ax_p = (x1 + 0.66, y1)
        bx_p = (x2 - 0.66, y2)
        arrow(ax, ax_p, bx_p, WHITE, rad=rad, lw=2.6, ms=18, z=5, alpha=0.85)

    # final dash to the flag
    arrow(ax, (xs[-1] + 0.66, ys[-1]), (13.85, 1.95), GOLD, rad=-0.25,
          lw=2.8, ms=20, z=5)
    icon_flag(ax, 14.25, 2.0, 1.15, GOLD, z=8)

    save(fig, "ch10-banner.png")


# ============================================================ CH.11 ========
def ch11():
    fig, ax = new_canvas(19)
    chapter_tag(ax, "11", "TOOL CALLING")

    # --- LLM brain panel (left)
    px, py = 3.15, 3.15
    glow(ax, px, py, 1.5, LAV, z=0)
    ax.add_patch(FancyBboxPatch(
        (px - 1.15, py - 1.35), 2.3, 2.7,
        boxstyle="round,pad=0.06,rounding_size=0.30",
        facecolor=LAV, alpha=0.16, edgecolor=LAV, lw=3, zorder=6))
    icon_robot(ax, px, py + 0.28, 0.95, LAV, z=8)
    ax.text(px, py - 0.85, "LLM", fontsize=12, fontweight="bold",
            color=WHITE, alpha=0.9, ha="center", zorder=9)

    # --- structured tool-call document (middle)
    dx, dy = 7.25, 3.15
    glow(ax, dx, dy, 0.85, CYAN, z=0)
    icon_doc(ax, dx, dy, 1.05, CYAN, z=8)
    ax.text(dx, dy - 1.15, "TOOL CALL", fontsize=7, color=WHITE, alpha=0.55,
            ha="center", zorder=9)

    # --- three tools (right)
    tools = [
        (icon_calc, CYAN, 12.55, 4.65),
        (icon_cloud, LAV, 12.55, 3.15),
        (icon_db, GOLD, 12.55, 1.65),
    ]
    for ic, col, tx, ty in tools:
        node_bg(ax, tx, ty, 0.66, col)
        ic(ax, tx, ty, 0.62, col, z=9)

    # forward calls
    arrow(ax, (px + 1.25, py), (dx - 0.75, dy), CYAN, rad=0.0, lw=3.5,
          ms=24, z=7)
    arrow(ax, (dx + 0.75, dy + 0.1), (11.85, 4.55), CYAN, rad=0.22, lw=3,
          ms=20, z=7)
    arrow(ax, (dx + 0.75, dy), (11.85, 3.15), LAV, rad=0.0, lw=3, ms=20,
          z=7)
    arrow(ax, (dx + 0.75, dy - 0.1), (11.85, 1.75), GOLD, rad=-0.22, lw=3,
          ms=20, z=7)

    # result feedback (dashed, mint)
    arrow(ax, (11.85, 5.15), (px + 1.05, py + 1.15), MINT, rad=-0.30,
          lw=2.4, ms=16, z=6, dashed=True, alpha=0.9)
    arrow(ax, (11.85, 2.55), (px + 1.05, py - 0.75), MINT, rad=0.42,
          lw=2.4, ms=16, z=6, dashed=True, alpha=0.9)
    arrow(ax, (11.85, 1.05), (px + 0.35, py - 1.45), MINT, rad=0.30,
          lw=2.4, ms=16, z=6, dashed=True, alpha=0.9)
    ax.text(8.3, 0.52, "RESULTS", fontsize=6.5, color=MINT, alpha=0.6,
            ha="center", zorder=9)

    save(fig, "ch11-banner.png")


# ============================================================ CH.12 ========
def ch12():
    fig, ax = new_canvas(23)
    chapter_tag(ax, "12", "MCP  ·  UNIVERSAL PROTOCOL")

    cx, cy = 7.55, 3.05

    # ---- central USB-C style universal slot (soft, tight glow)
    for kr, ka in ((1.45, 0.04), (1.15, 0.06)):
        ax.add_patch(Circle((cx, cy), 1.05 * kr, facecolor=GOLD, alpha=ka,
                            edgecolor="none", zorder=0))
    ax.add_patch(FancyBboxPatch(
        (cx - 0.95, cy - 0.52), 1.9, 1.04,
        boxstyle="round,pad=0.05,rounding_size=0.45",
        facecolor=GOLD, edgecolor="none", zorder=6))
    ax.add_patch(FancyBboxPatch(
        (cx - 0.72, cy - 0.32), 1.44, 0.64,
        boxstyle="round,pad=0.02,rounding_size=0.31",
        facecolor=BLUE, edgecolor="none", zorder=7))
    for k in range(4):
        ax.add_patch(Rectangle((cx - 0.36 + k * 0.22, cy - 0.10), 0.09,
                               0.20, facecolor=GOLD, edgecolor="none",
                               zorder=8))
    ax.text(cx, cy - 1.05, "MCP", fontsize=9, fontweight="bold", color=GOLD,
            alpha=0.9, ha="center", zorder=9)

    # ---- left: heterogeneous plugs, each adapted to the unified interface
    plugs = [
        (icon_db, GOLD, 3.05, 4.55),
        (icon_folder, CORAL, 2.95, 3.05),
        (icon_globe, CYAN, 3.05, 1.55),
    ]
    for ic, col, px, py in plugs:
        node_bg(ax, px, py, 0.62, col)
        ic(ax, px, py, 0.58, col, z=9)
        # short colored cable to a small unified tip
        tipx = px + 1.35
        ax.add_line(Line2D([px + 0.66, tipx - 0.16], [py, py], lw=3,
                           color=col, alpha=0.8, zorder=5,
                           solid_capstyle="round"))
        # unified tip (mini USB-C pill)
        ax.add_patch(FancyBboxPatch(
            (tipx - 0.16, py - 0.14), 0.34, 0.28,
            boxstyle="round,pad=0.01,rounding_size=0.13",
            facecolor=GOLD, edgecolor="none", zorder=6))
        # tip -> central slot
        dy_ = (cy - py)
        arrow(ax, (tipx + 0.20, py), (cx - 1.02, cy + dy_ * 0.10),
              GOLD, rad=0.16 if py < cy else -0.16, lw=2.6, ms=17, z=5)

    # ---- right: two agents talking via A2A bubbles
    icon_robot(ax, 13.35, 4.55, 0.80, CYAN, z=8)
    icon_robot(ax, 13.35, 1.55, 0.80, MINT, z=8)
    bubble(ax, 11.45, 4.30, 1.5, 0.78, CYAN, z=8, tail=(0.7, -1))
    bubble(ax, 11.45, 1.80, 1.5, 0.78, MINT, z=8, tail=(0.7, 1))
    ax.text(11.45, 4.30, "A2A", fontsize=8, fontweight="bold", color=CYAN,
            ha="center", va="center", zorder=10)
    ax.text(11.45, 1.80, "A2A", fontsize=8, fontweight="bold",
            color=MINT, ha="center", va="center", zorder=10)
    # dialogue exchange between the two bubbles
    arrow(ax, (11.05, 3.86), (11.05, 2.28), CYAN, rad=0.0, lw=2.2, ms=15,
          z=7)
    arrow(ax, (11.85, 2.28), (11.85, 3.86), MINT, rad=0.0, lw=2.2, ms=15,
          z=7)

    save(fig, "ch12-banner.png")


if __name__ == "__main__":
    ch09()
    ch10()
    ch11()
    ch12()
    print("done")
