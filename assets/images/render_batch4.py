#!/usr/bin/env python3
"""Batch 4 chapter banners (ch13-ch16) for《Agent 从零到专家》.

Vector-style flat banners, 1600x600 px, dark blue -> purple vertical gradient.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import (Circle, FancyBboxPatch, FancyArrowPatch,
                                Ellipse, Polygon, Rectangle)

CY, GOLD, CORAL, MINT, LAV = ("#00b4d8", "#ffd166", "#ef476f",
                               "#06d6a0", "#a9a4f5")
INK = "#e8f1ff"
W, H = 16, 6
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images"


def new_ax():
    fig = plt.figure(figsize=(16, 6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    # vertical gradient  deep blue -> deep purple
    top = np.array(mcolors.to_rgb("#0d1b2a"))
    bot = np.array(mcolors.to_rgb("#1b263b"))
    n = 256
    rows = np.linspace(0, 1, n)[:, None, None]
    grad = top[None, None, :] * (1 - rows) + bot[None, None, :] * rows
    ax.imshow(grad.reshape(n, 1, 3), extent=[0, W, 0, H], aspect="auto",
              origin="upper", zorder=-10)
    # scattered low-alpha dots
    rng = np.random.default_rng(7)
    xs = rng.uniform(0.2, W - 0.2, 70)
    ys = rng.uniform(0.2, H - 0.2, 70)
    ss = rng.uniform(4, 26, 70)
    cols = [CY, GOLD, CORAL, MINT, LAV]
    for x, y, s in zip(xs, ys, ss):
        ax.add_patch(Circle((x, y), s / 300, color=cols[int(rng.integers(0, 5))],
                            alpha=0.06, lw=0, zorder=-5))
    return fig, ax


def chapter_tag(ax, num):
    ax.text(0.55, 5.5, f"CH.{num}", fontsize=15, fontweight="bold",
            color=GOLD, alpha=0.95, family="monospace", zorder=20)
    ax.text(W - 0.45, 0.35, f"{num}", fontsize=150, fontweight="bold",
            color=LAV, alpha=0.05, ha="right", va="bottom", zorder=-4)


def glow_circle(ax, x, y, r, color, z=3, layers=4, step=0.3, a=0.05):
    for i in range(layers, 0, -1):
        ax.add_patch(Circle((x, y), r * (1 + step * i), color=color,
                            alpha=a * (layers - i + 1) / layers, lw=0, zorder=z))


def rbox(ax, x, y, w, h, ec, fc, lw=2, ra=0.18, z=5, a=1, glow=False):
    if glow:
        for i in range(3, 0, -1):
            ax.add_patch(FancyBboxPatch((x - 0.03 * i, y - 0.03 * i),
                                        w + 0.06 * i, h + 0.06 * i,
                                        boxstyle=f"round,pad=0,rounding_size={ra + 0.05 * i}",
                                        fc="none", ec=ec, lw=lw + 2 * i,
                                        alpha=0.05 + 0.03 * (4 - i), zorder=z - 1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={ra}",
                                fc=fc, ec=ec, lw=lw, alpha=a, zorder=z))


def arrow(ax, p1, p2, color, rad=0.2, lw=2.6, z=6, style="-|>", ls="-", ms=22, glow=True):
    if glow:
        for glw, ga in ((lw * 3.2, 0.06), (lw * 1.8, 0.10)):
            ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style,
                                         mutation_scale=ms, lw=glw, color=color,
                                         alpha=ga, linestyle=ls,
                                         connectionstyle=f"arc3,rad={rad}", zorder=z - 1))
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                                  lw=lw, color=color, linestyle=ls,
                                  connectionstyle=f"arc3,rad={rad}", zorder=z))


def spark(ax, x, y, s, color, z=8, a=0.9):
    pts = [(x, y + s), (x + 0.28 * s, y + 0.28 * s), (x + s, y),
           (x + 0.28 * s, y - 0.28 * s), (x, y - s),
           (x - 0.28 * s, y - 0.28 * s), (x - s, y),
           (x - 0.28 * s, y + 0.28 * s)]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", alpha=a, zorder=z))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=100)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- ch13 RAG
def ch13():
    fig, ax = new_ax()
    chapter_tag(ax, 13)

    # -- document stack (left)
    for dx, dy, ec in ((0.55, 2.05, LAV), (0.75, 2.25, LAV), (0.95, 2.45, "#e8f1ff")):
        rbox(ax, dx, dy, 1.5, 1.9, ec, "#141c33", lw=1.8, ra=0.12, z=4, a=0.9)
        for k in range(5):
            ax.add_patch(Rectangle((dx + 0.18, dy + 0.32 + k * 0.3),
                                   1.14 * (1 - 0.16 * (k % 2)), 0.075,
                                   fc=ec, ec="none", alpha=0.35, zorder=5))
    rbox(ax, 1.15, 2.65, 1.5, 1.9, GOLD, "#141c33", lw=2.2, ra=0.12, z=6, a=1, glow=True)
    for k in range(5):
        ax.add_patch(Rectangle((1.38, 2.97 + k * 0.3), 1.05 * (1 - 0.15 * (k % 2)), 0.075,
                               fc=GOLD, ec="none", alpha=0.5, zorder=7))

    # -- chunk slices flying out
    rng = np.random.default_rng(11)
    for i in range(8):
        x = 3.15 + (i % 4) * 0.34
        y = 1.7 + 0.42 * i
        col = [CY, MINT, GOLD, CY, MINT, LAV, CY, MINT][i]
        ax.add_patch(FancyBboxPatch((x, y), 0.95, 0.16,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=col, ec="none", alpha=0.75, zorder=4))
    arrow(ax, (2.75, 3.4), (4.15, 3.35), CY, rad=0.12, lw=2.8)

    # -- vector space (2D axes)
    ox, oy = 6.35, 1.55
    arrow(ax, (ox, oy), (ox, oy + 3.1), INK, rad=0.0, lw=1.4, ms=14, glow=False)
    ax.add_patch(FancyArrowPatch((ox, oy), (ox + 3.6, oy), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.4, color=INK, alpha=0.45, zorder=3))
    clusters = [((7.3, 3.9), CY), ((8.9, 2.5), MINT), ((7.1, 2.35), LAV)]
    pts = []
    for (cx, cy), col in clusters:
        for _ in range(8):
            p = (cx + rng.normal(0, 0.33), cy + rng.normal(0, 0.28))
            pts.append(p)
            ax.add_patch(Circle(p, rng.uniform(0.05, 0.10), color=col,
                                alpha=0.85, lw=0, zorder=5))
    for _ in range(7):  # strays
        p = (rng.uniform(6.6, 9.9), rng.uniform(1.9, 4.4))
        ax.add_patch(Circle(p, 0.05, color=LAV, alpha=0.35, lw=0, zorder=4))

    # -- query point + nearest neighbours
    qx, qy = 8.55, 3.75
    near = [(7.45, 3.95), (8.9, 3.55), (8.3, 2.6)]
    for nx, ny in near:
        for r, a in ((0.30, 0.10), (0.21, 0.28)):
            ax.add_patch(Circle((nx, ny), r, fc="none", ec=CORAL, lw=1.6,
                                alpha=a, zorder=6))
        ax.add_patch(FancyArrowPatch((qx, qy), (nx, ny), arrowstyle="-",
                                     lw=1.1, color=CORAL, linestyle=(0, (3, 3)),
                                     alpha=0.8, zorder=6))
    glow_circle(ax, qx, qy, 0.13, GOLD, z=7)
    ax.add_patch(Circle((qx, qy), 0.13, fc=GOLD, ec="#fff", lw=1.2, zorder=8))

    # -- answer bubble
    rbox(ax, 12.3, 3.0, 2.9, 1.85, MINT, "#0f2b26", lw=2.4, ra=0.35, z=6, glow=True)
    ax.add_patch(Polygon([(12.3, 3.45), (11.75, 3.2), (12.3, 3.05)],
                         closed=True, fc="#0f2b26", ec=MINT, lw=2.0, zorder=6))
    for k, wl in enumerate((2.2, 1.6, 2.4)):
        ax.add_patch(Rectangle((12.6, 4.3 - k * 0.42), wl, 0.10, fc=MINT,
                               ec="none", alpha=0.7, zorder=7))
    arrow(ax, (10.2, 3.9), (11.7, 3.9), GOLD, rad=-0.15, lw=2.8)
    save(fig, "ch13-banner.png")


# ------------------------------------------------------------- ch14 Memory
def ch14():
    fig, ax = new_ax()
    chapter_tag(ax, 14)

    # -- central glowing workbench (context window)
    rbox(ax, 5.9, 1.95, 4.2, 2.5, MINT, "#0e2a2a", lw=2.6, ra=0.30, z=6, glow=True)
    ax.text(8.0, 4.15, "CONTEXT WINDOW", fontsize=9.5, color=MINT, alpha=0.85,
            ha="center", family="monospace", zorder=8)
    # capacity bar (partially full -> limited)
    ax.add_patch(FancyBboxPatch((6.45, 2.35), 3.1, 0.3,
                                 boxstyle="round,pad=0,rounding_size=0.15",
                                 fc="#12332f", ec=MINT, lw=1.4, zorder=7))
    ax.add_patch(FancyBboxPatch((6.45, 2.35), 1.9, 0.3,
                                 boxstyle="round,pad=0,rounding_size=0.15",
                                 fc=GOLD, ec="none", alpha=0.9, zorder=8))
    for k in range(3):  # workbench content lines
        ax.add_patch(Rectangle((6.6, 3.0 + k * 0.32), 2.6 - 0.5 * k, 0.09,
                               fc=MINT, ec="none", alpha=0.55, zorder=7))
    for sx, sy in ((6.3, 4.75), (9.7, 4.9), (8.0, 1.6), (7.0, 1.45)):
        spark(ax, sx, sy, 0.12, MINT, a=0.8)

    # -- RAM stick (short-term, left)
    rx, ry, rw, rh = 1.95, 2.1, 1.05, 2.6
    rbox(ax, rx, ry, rw, rh, GOLD, "#2a2415", lw=2.2, ra=0.08, z=5, glow=True)
    for k in range(4):  # chips
        ax.add_patch(Rectangle((rx + 0.14, ry + 1.15 + k * 0.32), 0.77, 0.22,
                                fc=GOLD, ec="none", alpha=0.35, zorder=6))
    for k in range(9):  # gold contact pins
        ax.add_patch(Rectangle((rx + 0.05 + k * 0.115, ry - 0.16), 0.06, 0.16,
                               fc=GOLD, ec="none", alpha=0.85, zorder=6))
    ax.text(rx + rw / 2, ry + rh + 0.28, "RAM", fontsize=9.5, color=GOLD,
            alpha=0.85, ha="center", family="monospace", zorder=8)

    # -- disk cylinder (long-term store, right)
    cx, cy, cwd, chh = 13.35, 2.8, 2.1, 0.42
    for k in range(4):
        y = cy + k * 0.5
        ax.add_patch(Ellipse((cx, y), cwd, chh, fc=LAV, ec=LAV, lw=1.6,
                             alpha=0.14 + 0.06 * k, zorder=5))
    ax.plot([cx - cwd / 2, cx - cwd / 2], [cy, cy + 1.5], color=LAV, lw=1.8, zorder=6)
    ax.plot([cx + cwd / 2, cx + cwd / 2], [cy, cy + 1.5], color=LAV, lw=1.8, zorder=6)
    ax.add_patch(Ellipse((cx, cy), cwd, chh, fc="#1c1c33", ec=LAV, lw=2.2, zorder=6))
    glow_circle(ax, cx, cy + 0.75, 0.9, LAV, z=4)
    ax.text(cx, cy + 2.1, "LONG-TERM", fontsize=9.5, color=LAV, alpha=0.85,
            ha="center", family="monospace", zorder=8)

    # -- swap in / out double arrows
    arrow(ax, (3.35, 3.6), (5.75, 3.7), GOLD, rad=0.22, lw=2.4)      # RAM -> CTX (load)
    arrow(ax, (5.75, 2.9), (3.35, 2.8), CORAL, rad=0.22, lw=2.0, ms=18)  # evict
    arrow(ax, (10.25, 3.7), (12.2, 3.9), MINT, rad=0.22, lw=2.4)     # retrieve
    arrow(ax, (12.2, 2.9), (10.25, 2.7), CORAL, rad=0.22, lw=2.0, ms=18)  # write back
    save(fig, "ch14-banner.png")


# -------------------------------------------------------------- ch15 Skill
def ch15():
    fig, ax = new_ax()
    chapter_tag(ax, 15)

    # -- agent robot (left)
    bx, by = 2.5, 1.9
    rbox(ax, bx, by, 1.5, 1.9, CY, "#10283a", lw=2.4, ra=0.28, z=5, glow=True)
    rbox(ax, bx + 0.22, by + 1.55, 1.06, 0.75, CY, "#10283a", lw=2.0, ra=0.2, z=5)
    for ex in (bx + 0.48, bx + 0.98):  # eyes
        glow_circle(ax, ex, by + 1.93, 0.10, CY, z=7)
        ax.add_patch(Circle((ex, by + 1.93), 0.10, fc="#dff6ff", ec=CY, lw=1.0, zorder=8))
    ax.plot([bx + 0.75, bx + 0.75], [by + 2.3, by + 2.75], color=CY, lw=2.0, zorder=6)
    glow_circle(ax, bx + 0.75, by + 2.82, 0.10, CORAL, z=7)
    ax.add_patch(Circle((bx + 0.75, by + 2.82), 0.10, fc=CORAL, ec="#fff", lw=1.0, zorder=8))
    ax.add_patch(Circle((bx + 0.75, by + 0.85), 0.16, fc=GOLD, ec="none",
                        alpha=0.9, zorder=7))  # chest light
    ax.text(bx + 0.75, by - 0.3, "AGENT", fontsize=9.5, color=CY, alpha=0.85,
            ha="center", family="monospace", zorder=8)

    # -- small skill cards (floating)
    def icon_bolt(x, y, s, color, z):
        ax.add_patch(Polygon([(x - 0.1 * s, y + s), (x + 0.35 * s, y + 0.1 * s),
                              (x + 0.05 * s, y + 0.1 * s), (x + 0.25 * s, y - s),
                              (x - 0.25 * s, y - 0.05 * s), (x + 0.02 * s, y - 0.05 * s),
                              (x - 0.3 * s, y + s)], closed=True, fc=color, ec="none", zorder=z))
    def icon_target(x, y, s, color, z):
        for r, a in ((s, 0.9), (0.62 * s, 1)):
            ax.add_patch(Circle((x, y), r, fc="none", ec=color, lw=1.8, alpha=a, zorder=z))
        ax.add_patch(Circle((x, y), 0.16 * s, fc=color, ec="none", zorder=z))
    def icon_code(x, y, s, color, z):
        for d in (-1, 1):
            ax.add_patch(Polygon([(x + d * 0.15 * s, y + 0.55 * s), (x + d * 0.55 * s, y),
                                  (x + d * 0.15 * s, y - 0.55 * s)], closed=True,
                                 fc="none", ec=color, lw=1.8, zorder=z))
        ax.plot([x - 0.12 * s, x + 0.12 * s], [y + 0.7 * s, y - 0.7 * s], color=color,
                lw=1.8, zorder=z)

    small = [((5.15, 3.9), LAV, icon_code), ((5.55, 1.85), MINT, icon_target),
             ((6.9, 4.35), CY, icon_bolt), ((7.15, 2.25), CORAL, icon_target)]
    for (x, y), ec, icon in small:
        rbox(ax, x, y, 1.25, 1.6, ec, "#141c33", lw=1.7, ra=0.16, z=5, a=0.92)
        icon(x + 0.62, y + 1.05, 0.24, ec, 7)
        for k in range(2):
            ax.add_patch(Rectangle((x + 0.2, y + 0.32 + k * 0.22), 0.85 - 0.2 * k, 0.06,
                                   fc=ec, ec="none", alpha=0.35, zorder=6))
        ax.add_patch(FancyArrowPatch((bx + 1.5, by + 1.0), (x, y + 0.8),
                                     arrowstyle="-", lw=1.0, color=ec,
                                     linestyle=(0, (2, 3)), alpha=0.35,
                                     connectionstyle="arc3,rad=-0.15", zorder=3))

    # -- selected card (lit + enlarged)
    sx, sy, sw, sh = 10.1, 1.85, 2.1, 2.7
    glow_circle(ax, sx + sw / 2, sy + sh / 2, 1.55, GOLD, z=3, layers=5, step=0.18, a=0.05)
    rbox(ax, sx, sy, sw, sh, GOLD, "#2a2312", lw=2.8, ra=0.24, z=6, glow=True)
    icon_bolt(sx + sw / 2, sy + 1.85, 0.5, GOLD, 8)
    for k in range(3):
        ax.add_patch(Rectangle((sx + 0.35, sy + 0.55 + k * 0.32), 1.4 - 0.35 * k, 0.10,
                               fc=GOLD, ec="none", alpha=0.6, zorder=7))
    ax.text(sx + sw / 2, sy + sh + 0.32, "SKILL LOADED", fontsize=9.5, color=GOLD,
            alpha=0.9, ha="center", family="monospace", zorder=8)
    for px, py in ((sx - 0.35, sy + sh + 0.15), (sx + sw + 0.35, sy + sh + 0.2),
                   (sx - 0.5, sy - 0.25), (sx + sw + 0.45, sy + 0.35)):
        spark(ax, px, py, 0.14, GOLD)
    arrow(ax, (bx + 1.55, by + 1.6), (sx - 0.15, sy + 1.5), GOLD, rad=-0.22, lw=2.8)
    save(fig, "ch15-banner.png")


# ------------------------------------------------------ ch16 Multi-agent
def ch16():
    fig, ax = new_ax()
    chapter_tag(ax, 16)

    sx, sy = 8.0, 4.75
    # -- supervisor node (conductor)
    glow_circle(ax, sx, sy, 0.55, GOLD, z=3)
    ax.add_patch(Circle((sx, sy), 0.55, fc="#2a2415", ec=GOLD, lw=2.6, zorder=5))
    ax.add_patch(Polygon([(sx - 0.42, sy + 0.62), (sx - 0.28, sy + 0.95), (sx - 0.14, sy + 0.62),
                          (sx, sy + 0.98), (sx + 0.14, sy + 0.62), (sx + 0.28, sy + 0.95),
                          (sx + 0.42, sy + 0.62)], closed=True, fc=GOLD, ec="none",
                         alpha=0.95, zorder=6))
    ax.text(sx, sy - 0.02, "SUP", fontsize=9.5, color=GOLD, alpha=0.95,
            ha="center", va="center", family="monospace", fontweight="bold", zorder=7)

    # -- worker agents
    workers = [(5.0, 2.55, CY, "W1"), (8.0, 2.35, CORAL, "W2"), (11.0, 2.55, MINT, "W3")]
    for wx, wy, ec, name in workers:
        arrow(ax, (sx, sy - 0.55), (wx, wy + 0.52), ec, rad=-0.28, lw=2.2, ms=18)
    for (x1, y1, c1, _), (x2, y2, c2, _) in zip(workers, workers[1:]):
        ax.add_patch(FancyArrowPatch((x1 + 0.55, y1), (x2 - 0.55, y2), arrowstyle="-",
                                     lw=1.6, color=LAV, linestyle=(0, (4, 4)),
                                     alpha=0.75, connectionstyle="arc3,rad=0.12", zorder=4))
    for wx, wy, ec, name in workers:
        glow_circle(ax, wx, wy, 0.5, ec, z=3)
        ax.add_patch(Circle((wx, wy), 0.5, fc="#141c33", ec=ec, lw=2.4, zorder=5))
        ax.add_patch(Circle((wx, wy), 0.16, fc=ec, ec="none", zorder=6))
        ax.text(wx, wy - 0.78, name, fontsize=9, color=ec, alpha=0.85,
                ha="center", family="monospace", zorder=7)

    # -- converging output
    ox, oy = 8.0, 0.95
    for wx, wy, ec, _ in workers:
        arrow(ax, (wx, wy - 0.55), (ox - 0.5 + 0.25 * (wx - 8), oy + 0.45), ec,
              rad=0.18, lw=1.9, ms=16)
    glow_circle(ax, ox, oy + 0.1, 0.85, MINT, z=3)
    rbox(ax, ox - 0.95, oy - 0.35, 1.9, 0.78, MINT, "#0e2a2a", lw=2.4, ra=0.34, z=6, glow=True)
    ax.plot([ox - 0.35, ox - 0.08, ox + 0.38], [oy + 0.12, oy - 0.14, oy + 0.28],
            color=MINT, lw=2.6, solid_capstyle="round", zorder=7)  # check mark
    ax.text(ox, oy - 0.85, "OUTPUT", fontsize=9, color=MINT, alpha=0.8,
            ha="center", family="monospace", zorder=7)
    save(fig, "ch16-banner.png")


if __name__ == "__main__":
    ch13()
    ch14()
    ch15()
    ch16()
    print("done")
