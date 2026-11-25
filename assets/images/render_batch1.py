#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_batch1.py — 《Agent 从零到专家》第一批矢量风格插图（5 张）
统一设计系统:
  - 横幅 figsize=(16,6) dpi=100 (1600x600), 封面 (10,15) dpi=160
  - 背景: 竖直渐变 深蓝 #0d1b2a -> 深紫 #1b263b + 低透明度星点
  - 扁平化主元素 + 轻微发光感, 配色: 青/金/珊瑚/薄荷/薰衣草
  - 文字仅少量英文 (DejaVu Sans)
可复现: python3 assets/images/render_batch1.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyBboxPatch, FancyArrowPatch,
                                 Arc, Polygon)
from matplotlib.colors import LinearSegmentedColormap, to_rgb

OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images"

# ---------------- palette ----------------
BG1, BG2   = "#0d1b2a", "#1b263b"   # gradient low -> high
CYAN       = "#00b4d8"
GOLD       = "#ffd166"
CORAL      = "#ef476f"
MINT       = "#06d6a0"
LAV        = "#a9a4f5"
INK        = "#ffffff"
SLATE      = "#3a5068"

def mix(c1, c2, t):
    """linear blend between two hex colors, t in [0,1]"""
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple(float(x) for x in a + (b - a) * t)

# ---------------- canvas helpers ----------------
def new_canvas(figsize, dpi, w, h):
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=BG1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax

def gradient(ax, w, h, z=0, n=600):
    cmap = LinearSegmentedColormap.from_list("bg", [BG1, BG2])
    rgba = cmap(np.linspace(0, 1, n))[:, None, :]
    ax.imshow(rgba, extent=[0, w, 0, h], aspect="auto", origin="lower",
              zorder=z, interpolation="nearest")

def grain(ax, w, h, seed, z=1, n=360, alpha=0.04):
    """ultra-faint noise texture: enriches background, avoids flat banding"""
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 1, (n, max(2, int(n * w / h))))
    ax.imshow(img, extent=[0, w, 0, h], aspect="auto", cmap="gray",
              vmin=0, vmax=1, alpha=alpha, zorder=z, interpolation="nearest")

def stars(ax, w, h, seed, n=150, z=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.15, w - 0.15, n)
    y = rng.uniform(0.15, h - 0.15, n)
    s = rng.uniform(3, 26, n)
    a = rng.uniform(0.05, 0.15, n)
    ax.scatter(x, y, s=s, c=[(1, 1, 1, float(ai)) for ai in a],
               linewidths=0, zorder=z)

def rbox(ax, x, y, w, h, fc, r=0.12, ec="none", lw=0, alpha=1.0, z=5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))

def glow_circle(ax, x, y, r, color, z=4, core_fc=None, core_ec="none",
                lw=0, text=None, tcolor=INK, tsize=13, rings=((1.75, 0.10), (1.38, 0.20))):
    for f, a in rings:
        ax.add_patch(Circle((x, y), r * f, color=color, alpha=a, lw=0, zorder=z))
    ax.add_patch(Circle((x, y), r, fc=core_fc or color, ec=core_ec, lw=lw, zorder=z + 1))
    if text:
        ax.text(x, y, text, color=tcolor, fontsize=tsize, fontweight="bold",
                ha="center", va="center", zorder=z + 2)

def arrow(ax, p1, p2, color, lw=3, ms=18, z=7, ls="-", alpha=1.0, rad=0.0,
          shrinkA=0, shrinkB=0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                 color=color, lw=lw, zorder=z, linestyle=ls, alpha=alpha,
                 shrinkA=shrinkA, shrinkB=shrinkB,
                 connectionstyle=f"arc3,rad={rad}"))

def glow_line(ax, xs, ys, color, lw=3, z=6, alpha_g=0.14, lw_g=None, ls="-"):
    xs, ys = np.asarray(xs), np.asarray(ys)
    ax.plot(xs, ys, color=color, lw=lw_g or lw * 3.2, alpha=alpha_g, zorder=z - 1,
            solid_capstyle="round", ls=ls)
    ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="round", ls=ls)

def tag(ax, x, y, ch, kw):
    ax.text(x, y, ch, color=INK, fontsize=14, fontweight="bold",
            ha="left", va="center", zorder=20)
    ax.text(x + 0.82, y, kw, color=GOLD, fontsize=11, ha="left", va="center",
            zorder=20, alpha=0.95)

def orb(ax, x, y, r, color, alpha=0.06):
    ax.add_patch(Circle((x, y), r, color=color, alpha=alpha, lw=0, zorder=2))

def save(fig, name, dpi):
    fig.savefig(f"{OUT}/{name}", dpi=dpi, facecolor=BG1)
    plt.close(fig)
    print("saved", name)

# =========================================================================
# 1) BOOK COVER  (10 x 15 in, dpi 160)
# =========================================================================
def fig_cover():
    W, H = 10, 15
    fig, ax = new_canvas((10, 15), 160, W, H)
    gradient(ax, W, H, n=800)
    grain(ax, W, H, seed=8)
    stars(ax, W, H, seed=7, n=300)
    orb(ax, 1.2, 13.6, 1.3, CYAN, 0.05)
    orb(ax, 9.0, 1.7, 1.0, LAV, 0.06)
    ax.add_patch(Circle((9.0, 6.3), 1.0, fc="none", ec=LAV, lw=1.2,
                 alpha=0.12, zorder=2))
    ax.add_patch(Circle((1.5, 6.2), 0.7, fc="none", ec=CYAN, lw=1.2,
                 alpha=0.10, zorder=2))

    # ---- title block ----
    ax.text(5.0, 13.35, "AGENT", color=INK, fontsize=54, fontweight="bold",
            ha="center", va="center", zorder=20)
    ax.text(5.0, 12.5, "FROM ZERO TO HERO", color=GOLD, fontsize=19,
            ha="center", va="center", zorder=20)
    rbox(ax, 3.7, 12.02, 2.5, 0.045, LAV, r=0.02, alpha=0.55, z=20)

    # ---- gold rising polyline ----
    pts = [(1.6, 2.05), (2.5, 4.3), (3.7, 6.4), (5.0, 7.8),
           (6.25, 9.2), (7.78, 9.62)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    glow_line(ax, xs, ys, GOLD, lw=3.2, z=5, alpha_g=0.16, lw_g=10)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.075, fc=GOLD, ec="none", zorder=6))

    # ---- bottom-left: neuron tree ----
    edges = [((2.2, 1.9), (1.55, 2.75)), ((2.2, 1.9), (3.0, 2.65)),
             ((3.0, 2.65), (2.55, 3.55)), ((3.0, 2.65), (3.85, 3.5))]
    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [y1, y2], color=LAV, lw=2, alpha=0.9, zorder=7,
                solid_capstyle="round")
    for (sx, sy) in [(2.2, 1.9), (1.55, 2.75), (3.0, 2.65), (2.55, 3.55), (3.85, 3.5)]:
        ax.add_patch(Circle((sx, sy), 0.24, color=MINT, alpha=0.18, lw=0, zorder=7))
        ax.add_patch(Circle((sx, sy), 0.15, fc=MINT, ec="none", zorder=8))
        for ang in (120, 90, 150):
            r = np.deg2rad(ang)
            ax.plot([sx, sx + 0.30 * np.cos(r)], [sy, sy + 0.30 * np.sin(r)],
                    color=MINT, lw=1.3, alpha=0.85, zorder=8)
    ax.text(2.25, 1.4, "NEURON", color=INK, fontsize=10, alpha=0.75,
            ha="center", va="center", zorder=20)

    # ---- center: glowing brain ----
    bx, by = 5.0, 7.8
    ax.add_patch(Circle((bx, by), 2.15, color=LAV, alpha=0.07, lw=0, zorder=6))
    ax.add_patch(Circle((bx, by), 1.85, color=LAV, alpha=0.13, lw=0, zorder=6))
    ax.add_patch(Circle((bx, by), 1.5, fc="#152035", ec=LAV, lw=2.5, zorder=7))
    ax.add_patch(Circle((bx, by), 1.12, fc="none", ec=CYAN, lw=2.0, alpha=0.75, zorder=8))
    ax.add_patch(Circle((bx, by), 0.72, fc="none", ec=MINT, lw=1.8, alpha=0.65, zorder=8))
    ax.add_patch(Circle((bx, by), 0.14, fc=GOLD, ec="none", zorder=9))
    sulci = [((bx - 0.4, by + 0.3), 1.8, 1.8, -40, 100, LAV),
             ((bx + 0.4, by - 0.3), 1.6, 1.6, 120, 260, CYAN),
             ((bx, by), 1.35, 1.35, 200, 340, MINT),
             ((bx + 0.35, by + 0.35), 1.0, 1.0, 60, 170, GOLD)]
    for (cent, dw, dh, t1, t2, col) in sulci:
        ax.add_patch(Arc(cent, dw, dh, theta1=t1, theta2=t2, color=col,
                         lw=2.2, alpha=0.85, zorder=9))
    ax.text(5.0, 5.85, "BRAIN", color=INK, fontsize=10, alpha=0.75,
            ha="center", va="center", zorder=20)

    # ---- top-right: robot silhouette + tool icons ----
    # torso + head
    ax.add_patch(Circle((8.2, 10.6), 1.5, color=LAV, alpha=0.08, lw=0, zorder=7))
    rbox(ax, 7.78, 9.35, 0.84, 0.7, LAV, r=0.16, z=8)
    rbox(ax, 7.6, 10.15, 1.2, 0.9, LAV, r=0.22, z=8)
    ax.plot([8.2, 8.2], [10.05, 10.35], color=LAV, lw=4, zorder=8,
            solid_capstyle="round")  # neck
    ax.plot([8.2, 8.2], [11.05, 11.5], color=LAV, lw=2.6, zorder=8)  # antenna
    ax.add_patch(Circle((8.2, 11.6), 0.1, fc=GOLD, ec="none", zorder=9))
    ax.add_patch(Circle((7.95, 10.6), 0.1, fc=BG1, ec="none", zorder=9))
    ax.add_patch(Circle((8.45, 10.6), 0.1, fc=BG1, ec="none", zorder=9))
    ax.plot([8.05, 8.35], [10.35, 10.35], color=BG1, lw=2, zorder=9)  # mouth

    # tool icons
    # gear
    gx, gy = 6.55, 11.85
    for k in range(8):
        r = np.deg2rad(k * 45)
        ax.plot([gx + 0.15 * np.cos(r), gx + 0.27 * np.cos(r)],
                [gy + 0.15 * np.sin(r), gy + 0.27 * np.sin(r)],
                color=GOLD, lw=3.4, solid_capstyle="round", zorder=9)
    ax.add_patch(Circle((gx, gy), 0.17, fc=GOLD, ec="none", zorder=9))
    ax.add_patch(Circle((gx, gy), 0.06, fc=BG1, ec="none", zorder=10))
    # lightning bolt
    bolt = [(0.16, 0.5), (-0.20, 0.06), (-0.04, 0.06), (-0.16, -0.5),
            (0.20, -0.04), (0.04, -0.04)]
    bpts = [(9.12 + p[0], 11.42 + p[1]) for p in bolt]
    ax.add_patch(Polygon([(9.12 + p[0] * 1.25, 11.42 + p[1] * 1.25) for p in bolt],
                         closed=True, fc=CORAL, alpha=0.16, lw=0, zorder=8))
    ax.add_patch(Polygon(bpts, closed=True, fc=CORAL, ec="none", zorder=9))
    # 2x2 grid icon
    for ix in (9.18, 9.36):
        for iy in (9.32, 9.50):
            rbox(ax, ix, iy, 0.13, 0.13, MINT, r=0.03, z=9)
    # dotted connectors robot -> icons
    for (p1, p2) in [((7.72, 10.9), (6.78, 11.68)),
                     ((8.62, 10.95), (9.02, 11.32)),
                     ((8.58, 10.2), (9.32, 9.62))]:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=LAV, lw=1.5, alpha=0.6,
                ls=(0, (1.2, 2.2)), zorder=8)
        ax.add_patch(Circle(p2, 0.05, fc=LAV, ec="none", alpha=0.9, zorder=9))
    ax.text(8.2, 8.75, "AGENT + TOOLS", color=INK, fontsize=10, alpha=0.75,
            ha="center", va="center", zorder=20)

    # ---- bottom caption + frame ----
    ax.text(5.0, 0.9, "A VISUAL JOURNEY THROUGH INTELLIGENCE",
            color=INK, fontsize=11, alpha=0.7, ha="center", va="center", zorder=20)
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 9.0, 14.0,
                 boxstyle="round,pad=0,rounding_size=0.25", fc="none",
                 ec=GOLD, lw=1.4, alpha=0.32, zorder=30))
    save(fig, "book-cover.png", 160)

# =========================================================================
# 2) CH.01 — PERCEPTRON
# =========================================================================
def fig_ch01():
    W, H = 16, 6
    fig, ax = new_canvas((16, 6), 100, W, H)
    gradient(ax, W, H)
    grain(ax, W, H, seed=12)
    stars(ax, W, H, seed=11, n=150)
    tag(ax, 1.05, 5.42, "CH.01", "THE PERCEPTRON")
    orb(ax, 15.3, 5.0, 0.8, CYAN, 0.06)
    orb(ax, 0.7, 0.6, 0.9, LAV, 0.05)

    sx, sy, r = 4.6, 3.05, 0.6
    # dendrites
    for ang in (140, 170, 110, 200, 75):
        rad = np.deg2rad(ang)
        c1, s1 = np.cos(rad), np.sin(rad)
        x0, y0 = sx + 0.58 * c1, sy + 0.58 * s1
        x1, y1 = sx + 1.72 * c1, sy + 1.72 * s1
        ax.plot([x0, x1], [y0, y1], color=MINT, lw=2.8, alpha=0.9,
                zorder=6, solid_capstyle="round")
        bx, by = sx + 1.15 * c1, sy + 1.15 * s1
        for d in (-24, 24):
            r2 = np.deg2rad(ang + d)
            ax.plot([bx, bx + 0.55 * np.cos(r2)], [by, by + 0.55 * np.sin(r2)],
                    color=MINT, lw=1.5, alpha=0.8, zorder=6)
    glow_circle(ax, sx, sy, r, MINT, z=6)
    # axon -> perceptron
    glow_line(ax, [sx + 0.6, 8.85], [sy, 3.0], CYAN, lw=3.4, z=6)
    ax.add_patch(Circle((8.85, 3.0), 0.1, fc=CYAN, ec="none", zorder=7))
    # inputs
    ys_in = [1.75, 3.0, 4.35]
    for yy in ys_in:
        arrow(ax, (8.85, 3.0), (10.0, yy), LAV, lw=2.2, ms=15, z=7,
              shrinkB=20)
        glow_circle(ax, 10.0, yy, 0.28, CYAN, z=6)
    # output node
    glow_circle(ax, 13.1, 3.0, 0.44, GOLD, z=6, text="Σ", tsize=17)
    for yy in ys_in:
        arrow(ax, (10.32, yy), (12.62, 3.0), LAV, lw=2.2, ms=15, z=7,
              shrinkA=4, shrinkB=26)
    arrow(ax, (13.58, 3.0), (14.5, 3.0), GOLD, lw=3.2, ms=20, z=7)
    ax.text(14.1, 3.45, "y", color=GOLD, fontsize=12, fontweight="bold",
            ha="center", va="center", zorder=10)
    ax.text(3.9, 1.5, "NEURON", color=MINT, fontsize=10, alpha=0.85,
            ha="center", va="center", zorder=20)
    ax.text(12.1, 1.5, "PERCEPTRON", color=GOLD, fontsize=10, alpha=0.85,
            ha="center", va="center", zorder=20)
    save(fig, "ch01-banner.png", 100)

# =========================================================================
# 3) CH.02 — CONVOLUTION
# =========================================================================
def fig_ch02():
    W, H = 16, 6
    fig, ax = new_canvas((16, 6), 100, W, H)
    gradient(ax, W, H)
    grain(ax, W, H, seed=24)
    stars(ax, W, H, seed=23, n=150)
    tag(ax, 1.05, 5.42, "CH.02", "CONVOLUTION")
    orb(ax, 15.2, 1.0, 0.8, MINT, 0.06)

    # ---- input 5x5 grid ----
    x0, y0, c = 2.1, 1.75, 0.55
    for i in range(5):
        for j in range(5):
            t = (np.cos(i * 0.8) + np.sin(j * 0.9) + 2) / 4
            rbox(ax, x0 + i * c + 0.03, y0 + j * c + 0.03, c - 0.06, c - 0.06,
                 mix("#13355c", CYAN, t), r=0.04, z=5)
    # sliding window (highlight) at cells (1,1)-(3,3)
    wx, wy, ws = x0 + c, y0 + c, 3 * c
    rbox(ax, wx - 0.02, wy - 0.02, ws + 0.04, ws + 0.04, GOLD, r=0.10,
         alpha=0.10, z=6)
    ax.add_patch(FancyBboxPatch((wx - 0.02, wy - 0.02), ws + 0.04, ws + 0.04,
                 boxstyle="round,pad=0,rounding_size=0.10", fc="none",
                 ec=GOLD, lw=6.5, alpha=0.16, zorder=7))
    ax.add_patch(FancyBboxPatch((wx - 0.02, wy - 0.02), ws + 0.04, ws + 0.04,
                 boxstyle="round,pad=0,rounding_size=0.10", fc="none",
                 ec=GOLD, lw=2.8, zorder=8))
    # ghost window (previous scan position, bottom-right)
    gx, gy = x0 + 2 * c, y0
    ax.add_patch(FancyBboxPatch((gx, gy), 3 * c, 3 * c,
                 boxstyle="round,pad=0,rounding_size=0.10", fc="none",
                 ec=LAV, lw=1.8, ls=(0, (3, 2)), alpha=0.38, zorder=7))
    ax.text(3.47, 1.4, "INPUT", color=INK, fontsize=9, alpha=0.7,
            ha="center", va="center", zorder=20)

    # ---- arrows ----
    arrow(ax, (5.15, 3.0), (7.0, 3.0), CYAN, lw=3.4, ms=22, z=7)
    arrow(ax, (8.85, 3.0), (10.6, 3.0), CORAL, lw=3.4, ms=22, z=7)

    # ---- 3x3 kernel mini-grid ----
    kx, ky, kc = 7.25, 2.42, 0.4
    weights = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    for i in range(3):
        for j in range(3):
            wv = weights[j][i]
            col = MINT if wv == 1 else (CORAL if wv == -1 else SLATE)
            al = 0.9 if wv != 0 else 0.55
            rbox(ax, kx + i * kc + 0.03, ky + j * kc + 0.03, kc - 0.06,
                 kc - 0.06, col, r=0.04, z=6, alpha=al)
    ax.add_patch(FancyBboxPatch((kx, ky), 3 * kc, 3 * kc,
                 boxstyle="round,pad=0,rounding_size=0.10", fc="none",
                 ec=GOLD, lw=1.8, ls=(0, (4, 2.5)), alpha=0.8, zorder=7))
    ax.text(7.85, 2.1, "3×3 KERNEL", color=INK, fontsize=9, alpha=0.75,
            ha="center", va="center", zorder=20)

    # ---- feature map (3x3, smaller) ----
    fx, fy, fc = 11.05, 2.35, 0.42
    for i in range(3):
        for j in range(3):
            t = abs(i - (2 - j)) / 2
            rbox(ax, fx + i * fc + 0.025, fy + j * fc + 0.025, fc - 0.05,
                 fc - 0.05, mix("#274b6d", GOLD, t), r=0.035, z=6)
    ax.add_patch(FancyBboxPatch((fx - 0.05, fy - 0.05), 3 * fc + 0.1,
                 3 * fc + 0.1, boxstyle="round,pad=0,rounding_size=0.08",
                 fc="none", ec=MINT, lw=2, alpha=0.75, zorder=7))
    ax.text(11.66, 1.98, "FEATURE MAP", color=INK, fontsize=9, alpha=0.7,
            ha="center", va="center", zorder=20)
    # small deco dots on the right
    for k, (dx, dy, al) in enumerate([(13.3, 3.5, 0.9), (13.9, 3.05, 0.6), (14.5, 2.6, 0.35)]):
        ax.add_patch(Circle((dx, dy), 0.07, fc=MINT, ec="none", alpha=al, zorder=6))
    save(fig, "ch02-banner.png", 100)

# =========================================================================
# 4) CH.03 — BACKPROPAGATION
# =========================================================================
def fig_ch03():
    W, H = 16, 6
    fig, ax = new_canvas((16, 6), 100, W, H)
    gradient(ax, W, H)
    grain(ax, W, H, seed=32)
    stars(ax, W, H, seed=31, n=140)
    tag(ax, 1.05, 5.42, "CH.03", "BACKPROPAGATION")
    orb(ax, 15.3, 5.1, 0.8, CORAL, 0.05)
    orb(ax, 0.7, 0.9, 0.9, CYAN, 0.05)

    fy, by = 4.35, 1.65
    fx = [3.6, 8.0, 12.4]
    # forward chain
    for k, (x, lab) in enumerate(zip(fx, ["x", "h", "y"])):
        glow_circle(ax, x, fy, 0.42, CYAN, z=6, text=lab, tsize=14)
    arrow(ax, (fx[0] + 0.48, fy), (fx[1] - 0.48, fy), CYAN, lw=3.2, ms=20, z=7)
    arrow(ax, (fx[1] + 0.48, fy), (fx[2] - 0.48, fy), CYAN, lw=3.2, ms=20, z=7)
    ax.text(8.0, 5.05, "FORWARD PASS", color=INK, fontsize=11, alpha=0.85,
            ha="center", va="center", zorder=20)
    # backward chain (hollow, coral, right -> left)
    for x in fx:
        ax.add_patch(Circle((x, by), 0.44, color=CORAL, alpha=0.12, lw=0, zorder=5))
        ax.add_patch(Circle((x, by), 0.30, fc=BG2, ec=CORAL, lw=2.2, zorder=6))
        ax.text(x, by, "∇", color=CORAL, fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=7)
    arrow(ax, (fx[2] - 0.38, by), (fx[1] + 0.38, by), CORAL, lw=2.6, ms=16,
          z=7, ls=(0, (4, 3)))
    arrow(ax, (fx[1] - 0.38, by), (fx[0] + 0.38, by), CORAL, lw=2.6, ms=16,
          z=7, ls=(0, (4, 3)))
    ax.text(8.0, 0.98, "GRADIENT FLOW", color=CORAL, fontsize=11, alpha=0.9,
            ha="center", va="center", zorder=20)
    # closing loop arcs
    arrow(ax, (fx[2], fy - 0.48), (fx[2], by + 0.36), CORAL, lw=2.4, ms=15,
          z=7, rad=0.5, alpha=0.85)
    ax.text(13.5, 3.0, "LOSS", color=CORAL, fontsize=8.5, alpha=0.8,
            ha="left", va="center", zorder=20)
    arrow(ax, (fx[0], by + 0.36), (fx[0], fy - 0.48), GOLD, lw=2.4, ms=15,
          z=7, rad=0.5)
    ax.text(2.62, 3.0, "UPDATE", color=GOLD, fontsize=8.5, alpha=0.8,
            ha="center", va="center", zorder=20)
    save(fig, "ch03-banner.png", 100)

# =========================================================================
# 5) CH.04 — TOKENS & EMBEDDINGS
# =========================================================================
def fig_ch04():
    W, H = 16, 6
    fig, ax = new_canvas((16, 6), 100, W, H)
    gradient(ax, W, H)
    grain(ax, W, H, seed=42)
    stars(ax, W, H, seed=41, n=150)
    tag(ax, 1.05, 5.42, "CH.04", "TOKENS & EMBEDDINGS")

    cols = [CYAN, GOLD, CORAL, MINT, LAV, CYAN]
    xs_tok = [1.7, 3.02, 4.34, 5.66, 6.98, 8.3]
    tw, th = 1.05, 0.55
    ty = 4.07  # bottom of token blocks
    rng = np.random.default_rng(4)
    heat = LinearSegmentedColormap.from_list(
        "heat", ["#0e2440", "#00b4d8", "#ffd166", "#ef476f"])

    for cx, col in zip(xs_tok, cols):
        # token block + glow
        rbox(ax, cx - tw / 2 - 0.055, ty - 0.055, tw + 0.11, th + 0.11,
             col, r=0.17, alpha=0.16, z=5)
        rbox(ax, cx - tw / 2, ty, tw, th, col, r=0.14, z=6)
        # projection dashed line
        ax.plot([cx, cx], [ty - 0.02, 3.05], color=col, lw=1.5, alpha=0.75,
                ls=(0, (2, 3)), zorder=5)
        # embedding bar (5 segments)
        vals = rng.uniform(0.0, 1.0, 5)
        for j, v in enumerate(vals):
            yy = 3.0 - 0.15 - j * 0.185
            rbox(ax, cx - 0.31, yy, 0.62, 0.15, heat(v), r=0.03, z=6)
    # baseline under bars
    ax.plot([1.35, 9.55], [2.02, 2.02], color=LAV, lw=1.2, alpha=0.4, zorder=5)
    ax.text(5.45, 1.62, "EMBEDDINGS", color=INK, fontsize=10, alpha=0.7,
            ha="center", va="center", zorder=20)

    # ---- clock (positional metaphor) ----
    cx, cy, R = 12.55, 4.2, 0.52
    ax.add_patch(Circle((cx, cy), 0.82, color=LAV, alpha=0.10, lw=0, zorder=4))
    ax.add_patch(Circle((cx, cy), R, fc="none", ec=LAV, lw=3, zorder=6))
    for k in range(12):
        a = np.deg2rad(k * 30)
        ax.plot([cx + 0.40 * np.cos(a), cx + 0.50 * np.cos(a)],
                [cy + 0.40 * np.sin(a), cy + 0.50 * np.sin(a)],
                color=LAV, lw=1.6, alpha=0.85, zorder=6)
    ah = np.deg2rad(150)  # hour hand ~ 10 o'clock
    am = np.deg2rad(30)   # minute hand ~ 2 o'clock
    ax.plot([cx, cx + 0.24 * np.cos(ah)], [cy, cy + 0.24 * np.sin(ah)],
            color=GOLD, lw=3, zorder=7, solid_capstyle="round")
    ax.plot([cx, cx + 0.38 * np.cos(am)], [cy, cy + 0.38 * np.sin(am)],
            color=GOLD, lw=2.2, zorder=7, solid_capstyle="round")
    ax.add_patch(Circle((cx, cy), 0.055, fc=GOLD, ec="none", zorder=8))

    # dotted arrow tokens -> clock
    arrow(ax, (9.55, 4.35), (11.9, 4.28), LAV, lw=1.7, ms=11, z=6,
          ls=(0, (1.2, 2.2)), alpha=0.55)

    # ---- sine waves (position encoding) ----
    xw = np.linspace(10.75, 15.25, 400)
    ax.plot(xw, 1.95 + 0.5 * np.sin((xw - 10.75) * 1.7), color=GOLD, lw=8,
            alpha=0.13, zorder=5, solid_capstyle="round")
    ax.plot(xw, 1.95 + 0.5 * np.sin((xw - 10.75) * 1.7), color=GOLD, lw=2.8,
            zorder=6, solid_capstyle="round")
    ax.plot(xw, 1.95 + 0.5 * np.sin((xw - 10.75) * 1.7 + 1.2), color=CYAN,
            lw=1.8, alpha=0.45, zorder=6)
    ax.text(13.0, 0.95, "POSITIONAL ENCODING", color=INK, fontsize=10,
            alpha=0.8, ha="center", va="center", zorder=20)
    save(fig, "ch04-banner.png", 100)

if __name__ == "__main__":
    fig_cover()
    fig_ch01()
    fig_ch02()
    fig_ch03()
    fig_ch04()
    print("batch 1 done.")
