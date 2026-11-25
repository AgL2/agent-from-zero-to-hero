"""Batch-2 chapter banners for 《Agent 从零到专家》.
Vector-style flat banners, 1600x600, dark blue->purple gradient, starfield.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtr
from matplotlib.path import Path
from matplotlib.patches import (Circle, FancyBboxPatch, FancyArrowPatch,
                                Polygon, Rectangle, PathPatch)

BLUE_D, PURPLE_D = "#0d1b2a", "#1b263b"
CYAN, GOLD, CORAL, MINT, LAV = "#00b4d8", "#ffd166", "#ef476f", "#06d6a0", "#a9a4f5"
PALETTE = [CYAN, GOLD, CORAL, MINT, LAV]
C1 = np.array(mcolors.to_rgb(BLUE_D))
C2 = np.array(mcolors.to_rgb(PURPLE_D))


def new_canvas():
    fig, ax = plt.subplots(figsize=(16, 6), dpi=100)
    ax.set_xlim(0, 16); ax.set_ylim(0, 6)
    ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    t = np.linspace(0, 1, 256).reshape(-1, 1, 1)   # row0 = bottom
    img = C2 * (1 - t) + C1 * t                    # bottom purple -> top blue
    ax.imshow(img, extent=[0, 16, 0, 6], aspect="auto",
              origin="lower", zorder=-10, interpolation="bilinear")
    starfield(ax)
    return fig, ax


def starfield(ax, n=130, seed=42):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0.2, 15.8, n), rng.uniform(0.2, 5.8, n)
    s = rng.uniform(2, 22, n)
    for i in range(n):
        col = "white" if rng.random() < 0.82 else PALETTE[rng.integers(0, 5)]
        ax.scatter(x[i], y[i], s=s[i], c=col, alpha=rng.uniform(0.05, 0.28),
                   linewidths=0, zorder=0)


def chapter_tag(ax, num, subtitle, color):
    ax.text(0.78, 5.32, f"CH.{num}", fontsize=17, fontweight="bold",
            color="white", alpha=0.95, family="DejaVu Sans", zorder=30)
    ax.text(0.80, 4.90, subtitle, fontsize=8.5, fontweight="bold",
            color=color, alpha=0.8, family="DejaVu Sans", zorder=30)


def glow_circle(ax, x, y, r, color, alpha=0.95, zorder=5):
    """Stacked halo keeps the hue instead of turning muddy on the dark bg."""
    for s, a in ((1.42, 0.10), (1.20, 0.16)):
        ax.add_patch(Circle((x, y), r * s, fc=color, ec="none", alpha=a, zorder=zorder))
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", alpha=alpha, zorder=zorder + 1))


def rrect(ax, cx, cy, w, h, color, alpha=0.95, zorder=6, rs=0.16):
    for s, a in ((1.16, 0.10), (1.07, 0.16)):
        ax.add_patch(FancyBboxPatch((cx - w / 2 * s, cy - h / 2 * s), w * s, h * s,
                                    boxstyle=f"round,pad=0.02,rounding_size={rs * s}",
                                    fc=color, ec="none", alpha=a, zorder=zorder - 1))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0.02,rounding_size={rs}",
                                fc=color, ec="none", alpha=alpha, zorder=zorder))


def star_path(cx, cy, r, points=5, inner=0.42):
    verts = []
    for i in range(points * 2):
        ang = np.pi / 2 + i * np.pi / points
        rr = r if i % 2 == 0 else r * inner
        verts.append((cx + rr * np.cos(ang), cy + rr * np.sin(ang)))
    verts.append(verts[0])
    return Path(verts, [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1))


def arrow(ax, a, b, color, lw=3, alpha=0.9, rad=0.0, style="-|>", ms=16, zorder=4, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms,
                                 lw=lw, alpha=alpha, color=color, zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}",
                                 linestyle=ls, shrinkA=6, shrinkB=6))


def weight_label(ax, x, y, txt, col):
    ax.text(x, y, txt, fontsize=8, color=col, fontweight="bold", ha="center",
            va="center", zorder=9,
            bbox=dict(boxstyle="round,pad=0.18", fc=BLUE_D, ec="none", alpha=0.75))


# ---------------------------------------------------------------- CH.05 ----
def ch05():
    fig, ax = new_canvas()
    chapter_tag(ax, "05", "TRANSFORMER  ATTENTION", CYAN)

    # token nodes (left) with radiating attention edges to the focus
    tokens = [
        (1.05, LAV,  2.2, 0.35, "0.22", ".",    -0.20),
        (2.15, MINT, 3.6, 0.60, "0.47", "sat",   0.18),
        (3.25, CYAN, 6.0, 0.92, "0.83", "cat",  -0.12),
        (4.35, GOLD, 3.2, 0.50, "0.31", "the",   0.24),
    ]
    for ty, col, lw, al, wt, word, rad in tokens:
        rrect(ax, 2.35, ty, 1.75, 0.72, col, alpha=0.93, rs=0.18)
        ax.text(2.35, ty, word, fontsize=9.5, fontweight="bold",
                color=BLUE_D, ha="center", va="center", zorder=8)
        arrow(ax, (3.35, ty), (9.55, 2.95), col, lw=lw, alpha=al, rad=rad, ms=18)

    weight_label(ax, 6.15, 1.30, "0.22", LAV)
    weight_label(ax, 6.35, 2.10, "0.47", MINT)
    weight_label(ax, 6.55, 3.35, "0.83", CYAN)
    weight_label(ax, 6.05, 3.85, "0.31", GOLD)

    # Q / K / V small chips, dashed projection lines to the focus
    for i, (lab, col) in enumerate(zip(["Q", "K", "V"], [CORAL, GOLD, MINT])):
        gx, gy = 6.55 + i * 0.95, 5.05
        ax.add_patch(Circle((gx, gy), 0.42, fc=col, ec="none", alpha=0.15, zorder=3))
        ax.add_patch(Circle((gx, gy), 0.30, fc=col, ec="none", alpha=0.95, zorder=4))
        ax.text(gx, gy, lab, fontsize=10, fontweight="bold", color=BLUE_D,
                ha="center", va="center", zorder=5)
        ax.plot([gx, 10.35 + i * 0.15], [gy - 0.34, 3.62], color=col, lw=0.9,
                alpha=0.40, ls=(0, (2, 3)), zorder=2)

    # focus node
    rrect(ax, 10.6, 2.95, 2.45, 1.05, GOLD, alpha=0.97, rs=0.2)
    ax.text(10.6, 2.95, "FOCUS", fontsize=11, fontweight="bold",
            color=BLUE_D, ha="center", va="center", zorder=8)

    # output flow to the right
    arrow(ax, (11.95, 2.95), (13.55, 2.95), GOLD, lw=3, alpha=0.85, ms=20)
    ax.add_patch(Circle((14.15, 2.95), 0.24, fc=GOLD, ec="none", alpha=0.15, zorder=4))
    ax.add_patch(Circle((14.15, 2.95), 0.19, fc=GOLD, ec="none", alpha=0.9, zorder=5))
    ax.add_patch(Circle((14.95, 2.95), 0.10, fc=GOLD, ec="none", alpha=0.55, zorder=5))

    fig.savefig("ch05-banner.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- CH.06 ----
def ch06():
    fig, ax = new_canvas()
    chapter_tag(ax, "06", "FINE-TUNING  &  ALIGNMENT", MINT)

    # pre-trained model (big circle)
    glow_circle(ax, 2.9, 3.15, 1.5, LAV, alpha=0.95)
    ax.add_patch(Circle((2.9, 3.15), 0.95, fc=BLUE_D, ec="none", alpha=0.25, zorder=7))
    ax.text(2.9, 3.15, "LLM", fontsize=12, fontweight="bold", color="white",
            ha="center", va="center", alpha=0.95, zorder=8)
    ax.text(2.9, 1.15, "PRE-TRAINED", fontsize=7.5, color=LAV, alpha=0.75,
            ha="center", fontweight="bold", zorder=8)

    arrow(ax, (4.62, 3.15), (5.7, 3.15), "white", lw=2.5, alpha=0.5, ms=16)

    # SFT funnel
    ax.add_patch(Polygon([(5.9, 4.5), (7.7, 4.5), (7.03, 2.6), (6.57, 2.6)],
                         closed=True, fc=GOLD, ec="none", alpha=0.35, zorder=4))
    ax.add_patch(Polygon([(5.75, 4.62), (7.85, 4.62), (7.18, 2.6), (6.42, 2.6)],
                         closed=True, fc=GOLD, ec="none", alpha=0.95, zorder=5))
    ax.add_patch(Polygon([(6.42, 2.6), (7.18, 2.6), (7.18, 2.05), (6.42, 2.05)],
                         closed=True, fc=GOLD, ec="none", alpha=0.95, zorder=5))
    ax.text(6.8, 5.0, "SFT", fontsize=11, fontweight="bold", color=GOLD,
            ha="center", alpha=0.95, zorder=8)

    arrow(ax, (8.0, 3.15), (8.85, 3.15), "white", lw=2.5, alpha=0.5, ms=16)

    # tuned model: chat-card shape
    rrect(ax, 9.85, 3.15, 1.85, 1.85, MINT, alpha=0.95, rs=0.22)
    for i, w in enumerate([1.15, 0.9, 0.65]):
        ax.add_patch(FancyBboxPatch((9.85 - w / 2, 3.72 - i * 0.42), w, 0.14,
                                    boxstyle="round,pad=0.01,rounding_size=0.07",
                                    fc=BLUE_D, ec="none", alpha=0.45, zorder=7))
    ax.text(9.85, 1.15, "TUNED", fontsize=7.5, color=MINT, alpha=0.75,
            ha="center", fontweight="bold", zorder=8)

    # DPO balance scale (right)
    px, piv, half, tilt = 13.15, 4.65, 1.8, 0.16
    ax.add_patch(Polygon([(px, piv - 0.1), (px - 0.28, 1.75), (px + 0.28, 1.75)],
                         closed=True, fc=LAV, ec="none", alpha=0.85, zorder=5))
    ax.add_patch(FancyBboxPatch((px - 0.72, 1.55), 1.44, 0.2,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=LAV, ec="none", alpha=0.95, zorder=5))
    ax.text(px, 1.05, "DPO", fontsize=10, fontweight="bold", color=LAV,
            ha="center", alpha=0.9, zorder=8)

    beam = FancyBboxPatch((px - half, piv - 0.08), 2 * half, 0.16,
                          boxstyle="round,pad=0.01,rounding_size=0.08",
                          fc="white", ec="none", alpha=0.85, zorder=6)
    beam.set_transform(mtr.Affine2D().rotate_around(px, piv, tilt) + ax.transData)
    ax.add_patch(beam)
    glow_circle(ax, px, piv, 0.19, GOLD, zorder=7)

    for sgn, col, lab in ((-1, MINT, "CHOSEN"), (1, CORAL, "REJECTED")):
        ex = px + sgn * half * np.cos(tilt)
        ey = piv + sgn * half * np.sin(tilt)
        py = ey - 1.0
        ax.plot([ex, ex], [ey, py + 0.26], color="white", lw=1.1, alpha=0.6, zorder=5)
        ax.add_patch(Polygon([(ex - 0.6, py + 0.26), (ex + 0.6, py + 0.26),
                              (ex + 0.28, py - 0.14), (ex - 0.28, py - 0.14)],
                             closed=True, fc=col, ec="none", alpha=0.95, zorder=6))
        ax.text(ex, py + 0.5, lab, fontsize=6.5, fontweight="bold", color=col,
                alpha=0.9, ha="center", zorder=8)

    fig.savefig("ch06-banner.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- CH.07 ----
def ch07():
    fig, ax = new_canvas()
    chapter_tag(ax, "07", "PROMPT  ENGINEERING", LAV)

    # speech bubble
    bx, by, bw, bh = 4.6, 3.3, 4.3, 3.0
    for s, a in ((1.10, 0.08), (1.05, 0.13)):
        ax.add_patch(FancyBboxPatch((bx - bw / 2 * s, by - bh / 2 * s), bw * s, bh * s,
                                    boxstyle="round,pad=0.02,rounding_size=0.55",
                                    fc=LAV, ec="none", alpha=a, zorder=3))
    ax.add_patch(FancyBboxPatch((bx - bw / 2, by - bh / 2), bw, bh,
                                boxstyle="round,pad=0.02,rounding_size=0.5",
                                fc=LAV, ec="none", alpha=0.15, zorder=4))
    ax.add_patch(FancyBboxPatch((bx - bw / 2, by - bh / 2), bw, bh,
                                boxstyle="round,pad=0.02,rounding_size=0.5",
                                fc="none", ec=LAV, lw=2.2, alpha=0.9, zorder=5))
    ax.add_patch(Polygon([(bx - 1.05, by - bh / 2 + 0.03), (bx - 0.25, by - bh / 2 + 0.03),
                          (bx - 1.05, by - bh / 2 - 0.6)],
                         closed=True, fc=LAV, ec="none", alpha=0.9, zorder=5))
    ax.text(bx + 0.85, 1.15, "PROMPT", fontsize=7.5, color=LAV, alpha=0.75,
            ha="center", fontweight="bold", zorder=8)

    # gear inside bubble
    gx, gy, gr = 3.75, 3.55, 0.6
    for s, a in ((1.45, 0.10), (1.22, 0.16)):
        ax.add_patch(Circle((gx, gy), gr * s, fc=GOLD, ec="none", alpha=a, zorder=5))
    for i in range(8):
        ang = i * np.pi / 4
        tx, ty_ = gx + gr * 0.92 * np.cos(ang), gy + gr * 0.92 * np.sin(ang)
        tooth = Rectangle((tx - 0.09, ty_ - 0.09), 0.18, 0.18,
                          fc=GOLD, ec="none", alpha=0.95, zorder=6)
        tooth.set_transform(mtr.Affine2D().rotate_around(tx, ty_, ang) + ax.transData)
        ax.add_patch(tooth)
    ax.add_patch(Circle((gx, gy), gr, fc=GOLD, ec="none", alpha=0.95, zorder=6))
    ax.add_patch(Circle((gx, gy), 0.26, fc=BLUE_D, ec="none", alpha=0.55, zorder=7))

    # lever inside bubble
    lx, ly = 5.35, 2.7
    ax.plot([lx, lx + 0.8], [ly, ly + 0.8], color=CORAL, lw=4.5,
            solid_capstyle="round", alpha=0.95, zorder=6)
    glow_circle(ax, lx, ly, 0.2, CORAL, zorder=7)
    ax.add_patch(Circle((lx + 0.8, ly + 0.8), 0.3, fc="white", ec="none",
                        alpha=0.9, zorder=7))

    # radiating output paths converging to the star
    src = (bx + bw / 2 - 0.15, by + 0.1)
    for i, (rad, col, al, lw) in enumerate([(-0.26, CYAN, 0.8, 2.6),
                                            (-0.10, MINT, 0.6, 2.1),
                                            (0.08, CORAL, 0.6, 2.1),
                                            (0.28, GOLD, 0.8, 2.6)]):
        arrow(ax, src, (12.9, 2.95), col, lw=lw, alpha=al, rad=rad, ms=13, zorder=3)

    # the star (good answer)
    sx, sy, sr = 13.75, 2.95, 0.72
    for s, a in ((1.5, 0.20), (1.25, 0.32)):
        ax.add_patch(PathPatch(star_path(sx, sy, sr * s), fc=GOLD, ec="none",
                               alpha=a, zorder=4))
    ax.add_patch(PathPatch(star_path(sx, sy, sr), fc=GOLD, ec="none",
                           alpha=0.98, zorder=5))
    ax.text(sx, sy - 1.4, "BEST  ANSWER", fontsize=7.5, color=GOLD, alpha=0.8,
            ha="center", fontweight="bold", zorder=8)

    fig.savefig("ch07-banner.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- CH.08 ----
def ch08():
    fig, ax = new_canvas()
    chapter_tag(ax, "08", "INFERENCE  &  DEPLOYMENT", CYAN)

    # prefill wide arrow
    pts = np.array([(0.95, 2.35), (4.15, 2.35), (4.15, 1.95),
                    (5.45, 3.0), (4.15, 4.05), (4.15, 3.65), (0.95, 3.65)])
    ax.add_patch(Polygon(pts, closed=True, fc=CYAN, ec="none", alpha=0.95, zorder=4))
    ax.text(2.5, 3.0, "PREFILL", fontsize=11, fontweight="bold", color=BLUE_D,
            ha="center", va="center", zorder=6)

    # decode: autoregressive tokens popping out one by one
    xs = np.arange(6.15, 10.7, 0.93)
    for i, x in enumerate(xs):
        f = i / (len(xs) - 1)
        al = 1.0 - 0.5 * f
        rrect(ax, x, 3.0, 0.66, 0.66, MINT, alpha=al, rs=0.12)
        if i == len(xs) - 1:
            ax.add_patch(FancyBboxPatch((x - 0.4, 2.6), 0.8, 0.8,
                                        boxstyle="round,pad=0.02,rounding_size=0.14",
                                        fc="none", ec=MINT, lw=1.4,
                                        alpha=0.8, ls=(0, (3, 3)), zorder=7))
        if i < len(xs) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.42, 3.0), (x + 0.55, 3.0),
                                         arrowstyle="-|>", mutation_scale=9,
                                         lw=1.4, color=MINT, alpha=0.5, zorder=5))
    ax.text(8.4, 4.1, "DECODE", fontsize=9, fontweight="bold", color=MINT,
            alpha=0.9, ha="center", zorder=8)
    ax.text(8.4, 1.95, "AUTOREGRESSIVE", fontsize=6.5, color=MINT, alpha=0.6,
            ha="center", fontweight="bold", zorder=8)

    # quantization: big circle -> small circle
    glow_circle(ax, 11.55, 3.85, 0.68, CORAL, alpha=0.9)
    ax.text(11.55, 3.85, "W16", fontsize=7.5, fontweight="bold", color="white",
            ha="center", va="center", zorder=8)
    arrow(ax, (12.42, 3.5), (13.72, 2.4), CORAL, lw=2.6, alpha=0.85, ms=16, rad=-0.15)
    ax.add_patch(Circle((14.15, 2.12), 0.62, fc=CORAL, ec="none", alpha=0.14, zorder=4))
    ax.add_patch(Circle((14.15, 2.12), 0.44, fc=CORAL, ec="none", alpha=0.95, zorder=5))
    ax.text(14.15, 2.12, "W4", fontsize=7, fontweight="bold", color="white",
            ha="center", va="center", zorder=8)
    ax.text(13.35, 5.05, "QUANT", fontsize=8.5, fontweight="bold", color=CORAL,
            alpha=0.9, ha="center", zorder=8)

    # server / deployment chip
    rrect(ax, 14.15, 4.45, 1.35, 0.55, "white", alpha=0.88, rs=0.1)
    for i in range(3):
        ax.add_patch(Circle((13.72 + i * 0.26, 4.45), 0.07, fc=CYAN, ec="none",
                            alpha=0.9, zorder=8))

    fig.savefig("ch08-banner.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    ch05(); ch06(); ch07(); ch08()
    print("done")
