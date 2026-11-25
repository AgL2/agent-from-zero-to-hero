#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《Agent 从零到专家》章节题图 · 批次 5
ch17 评估与安全 / ch18 Mini-Agent 框架 / ch19 知识库 Agent / ch20 多 Agent 流水线
统一设计系统: 1600x600, 深蓝->深紫渐变背景, 扁平矢量图形
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyBboxPatch, FancyArrowPatch,
                                Polygon, Rectangle, Wedge, Ellipse, PathPatch)
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap

# ---------- 调色板 ----------
CYAN, GOLD, CORAL, MINT, LAV = "#00b4d8", "#ffd166", "#ef476f", "#06d6a0", "#a9a4f5"
INK, PANEL = "#0d1b2a", "#152238"
LIGHT = "#eaf2ff"

BG_CMAP = LinearSegmentedColormap.from_list("bg", ["#0d1b2a", "#1b263b"])
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images"


# ---------- 基础工具 ----------
def new_canvas():
    fig = plt.figure(figsize=(16, 6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")
    grad = np.linspace(0, 1, 600).reshape(-1, 1)
    ax.imshow(grad, extent=[0, 16, 0, 6], aspect="auto", cmap=BG_CMAP,
              origin="lower", zorder=-10)
    return fig, ax


def scatter_dots(ax, seed, n=46):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x, y = rng.uniform(0, 16), rng.uniform(0, 6)
        r = rng.uniform(0.014, 0.05)
        c = rng.choice(["#ffffff", CYAN, LAV, GOLD])
        ax.add_patch(Circle((x, y), r, fc=c, alpha=rng.uniform(0.06, 0.25),
                            lw=0, zorder=-3))


def chapter_tag(ax, label, x=0.75, y=5.52):
    ax.text(x, y, label, fontsize=15, fontweight="bold", color=GOLD,
            family="DejaVu Sans", zorder=20)
    ax.plot([x, x + 1.15], [y - 0.34, y - 0.34], color=GOLD, lw=2,
            alpha=0.85, zorder=20, solid_capstyle="round")


def rrect(ax, cx, cy, w, h, fc, ec="none", lw=0, rs=0.12, alpha=1.0, z=3):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rs}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(p)
    return p


def glow_rrect(ax, cx, cy, w, h, color, rs=0.12):
    for pad, al in ((0.09, 0.14), (0.22, 0.07), (0.40, 0.035)):
        rrect(ax, cx, cy, w + 2 * pad, h + 2 * pad, fc=color,
              rs=rs + pad, alpha=al, z=1)


def glow_circle(ax, cx, cy, r, color):
    for k, al in ((1.35, 0.08), (1.8, 0.05), (2.4, 0.03)):
        ax.add_patch(Circle((cx, cy), r * k, fc=color, alpha=al, lw=0, zorder=1))


def txt(ax, x, y, s, size=9, color=LIGHT, weight="bold", ha="center", z=20):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va="center", family="DejaVu Sans", zorder=z)


def doc_icon(ax, cx, cy, w, h, lines=3, z=6, fc="#f5f7ff", ec="none"):
    rrect(ax, cx, cy, w, h, fc=fc, ec=ec, rs=0.06, z=z)
    for i in range(lines):
        ww = w * (0.68 - 0.09 * (i % 2))
        ax.add_patch(Rectangle((cx - ww / 2, cy + h / 2 - 0.28 - i * 0.24),
                               ww, 0.075, fc="#8fa3c8", alpha=0.85, lw=0, zorder=z + 1))


# ================================================================
# CH.17  评估与安全 —— 仪表盘 + 盾牌
# ================================================================
def shield_path(cx, cy, w, h):
    x0, x1, xc = cx - w / 2, cx + w / 2, cx
    yt, ym, yb = cy + h / 2, cy + h * 0.12, cy - h / 2
    verts = [(x0, yt),
             (x0, yt + 0.12 * h), (x1, yt + 0.12 * h), (x1, yt),
             (x1, ym), (xc + 0.20 * w, yb + 0.16 * h), (xc, yb),
             (xc - 0.20 * w, yb + 0.16 * h), (x0, ym), (x0, yt),
             (0, 0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    return Path(verts, codes)


def warn_triangle(ax, cx, cy, s, color=CORAL, z=6):
    ax.add_patch(Polygon([(cx, cy + s), (cx - 0.92 * s, cy - 0.68 * s),
                          (cx + 0.92 * s, cy - 0.68 * s)],
                         fc=color, ec="none", zorder=z))
    ax.add_patch(Rectangle((cx - 0.05 * s, cy - 0.18 * s), 0.10 * s, 0.55 * s,
                           fc="#ffffff", lw=0, zorder=z + 1))
    ax.add_patch(Circle((cx, cy - 0.38 * s), 0.07 * s, fc="#ffffff",
                        lw=0, zorder=z + 1))


def draw_ch17():
    fig, ax = new_canvas()
    scatter_dots(ax, seed=17)
    chapter_tag(ax, "CH.17")

    # ---- 左侧: 评估仪表盘面板 ----
    rrect(ax, 4.2, 3.0, 6.3, 4.5, fc=PANEL, ec="#2e3f63", lw=1.5, rs=0.28,
          alpha=0.62, z=2)
    txt(ax, 1.6, 4.85, "METRICS", size=8, color=LAV, ha="left")

    # 半圆仪表(准确率)
    gx, gy, gr = 2.9, 2.35, 1.15
    ax.add_patch(Wedge((gx, gy), gr, 180, 360, width=0.24, fc="#223350",
                       ec="none", zorder=4))
    val = 0.92
    ax.add_patch(Wedge((gx, gy), gr, 180, 180 + 180 * val, width=0.24,
                       fc=MINT, ec="none", zorder=5))
    for th in (180, 225, 270, 315, 360):
        t = np.deg2rad(th)
        ax.plot([gx + 0.82 * gr * np.cos(t), gx + 1.02 * gr * np.cos(t)],
                [gy + 0.82 * gr * np.sin(t), gy + 1.02 * gr * np.sin(t)],
                color="#54689a", lw=1.6, zorder=6)
    nt = np.deg2rad(180 + 180 * val)
    ax.plot([gx, gx + 0.70 * gr * np.cos(nt)], [gy, gy + 0.70 * gr * np.sin(nt)],
            color=GOLD, lw=3.5, zorder=7, solid_capstyle="round")
    ax.add_patch(Circle((gx, gy), 0.09, fc=GOLD, ec="none", zorder=8))
    txt(ax, gx, gy + 0.22, "ACC", size=8, color=LIGHT)
    txt(ax, gx, gy - 0.5, "92%", size=12, color=MINT)

    # 柱状图(延迟/成本/错误)
    base = 1.32
    for bx, bh, c, lab in ((5.05, 1.5, CYAN, "LAT"),
                           (5.95, 0.95, GOLD, "COST"),
                           (6.85, 0.42, CORAL, "ERR")):
        glow_rrect(ax, bx, base + bh / 2, 0.56, bh, c, rs=0.07)
        rrect(ax, bx, base + bh / 2, 0.56, bh, fc=c, rs=0.07, z=6)
        txt(ax, bx, base - 0.30, lab, size=6.5, color="#9fb4dc")
    ax.plot([4.7, 7.2], [base, base], color="#54689a", lw=1.5, zorder=5)

    # ---- 右侧: 安全盾牌 ----
    scx, scy = 11.15, 3.25
    glow_circle(ax, scx, scy, 1.5, CYAN)
    ax.add_patch(PathPatch(shield_path(scx, scy, 2.7, 3.3), fc=CYAN,
                           ec="#8fe9ff", lw=2.5, alpha=0.92, zorder=6))
    ax.add_patch(PathPatch(shield_path(scx, scy, 1.85, 2.3), fc="none",
                           ec="#0d1b2a", lw=2.0, alpha=0.55, zorder=7))
    ax.plot([scx - 0.62, scx - 0.16, scx + 0.68],
            [scy - 0.10, scy - 0.55, scy + 0.55],
            color=MINT, lw=6, zorder=8, solid_capstyle="round",
            solid_joinstyle="round")

    # 注入攻击: 警告三角 + 冲击轨迹
    for tx, ty, ts in ((13.75, 4.75, 0.42), (14.85, 3.45, 0.60),
                       (14.05, 2.05, 0.48), (15.15, 5.05, 0.30)):
        warn_triangle(ax, tx, ty, ts)
        for k in (0.30, 0.55, 0.80):
            ax.plot([tx + ts + k, tx + ts + k + 0.22], [ty, ty],
                    color=CORAL, lw=1.6, alpha=0.45 - k * 0.3, zorder=5)
    # 冲击火花(从盾牌边缘向外)
    for ang, ln in ((18, 0.55), (48, 0.4), (-12, 0.5), (-42, 0.35)):
        a = np.deg2rad(ang)
        ax.plot([scx + 1.50 * np.cos(a), scx + (1.50 + ln) * np.cos(a)],
                [scy + 1.50 * np.sin(a), scy + (1.50 + ln) * np.sin(a)],
                color=GOLD, lw=2.2, alpha=0.85, zorder=7)
    ax.add_patch(FancyArrowPatch((14.3, 3.75), (12.85, 3.4),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=CORAL, lw=2, linestyle=(0, (4, 3)),
                                 alpha=0.8, zorder=5))

    fig.savefig(f"{OUT}/ch17-banner.png", dpi=100)
    plt.close(fig)


# ================================================================
# CH.18  Mini-Agent 框架 —— 发光积木拼成小机器人
# ================================================================
def block(ax, cx, cy, w, h, color, label=None, ls=9, studs=True):
    glow_rrect(ax, cx, cy, w, h, color, rs=0.14)
    rrect(ax, cx, cy, w, h, fc=color, ec="none", rs=0.14, z=6)
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.14",
                                fc="none", ec="#ffffff", lw=1.2,
                                alpha=0.35, zorder=7))
    if studs:
        for sx in (cx - w / 4, cx + w / 4):
            ax.add_patch(Circle((sx, cy + h / 2), 0.09, fc=color,
                                ec="#ffffff", lw=1, alpha=0.95, zorder=8))
    if label:
        txt(ax, cx, cy - h * 0.22, label, size=ls, color="#0d1b2a")


def draw_ch18():
    fig, ax = new_canvas()
    scatter_dots(ax, seed=18)
    chapter_tag(ax, "CH.18", x=0.75, y=5.35)
    txt(ax, 2.0, 4.85, "MINI-AGENT", size=11, color=LAV, ha="left")

    # 地面阴影
    ax.add_patch(Ellipse((9.9, 1.05), 7.2, 0.55, fc="#000000", alpha=0.35,
                         lw=0, zorder=0))

    # 电路连线(左)
    ax.plot([1.3, 5.6], [3.1, 3.1], color=LAV, lw=1.8,
            linestyle=(0, (2, 4)), alpha=0.6, zorder=2)
    for px in (1.3, 3.45, 5.6):
        ax.add_patch(Circle((px, 3.1), 0.09, fc=LAV, alpha=0.9, zorder=3))

    # 头 LLM
    block(ax, 9.9, 4.5, 2.3, 1.0, CYAN, "LLM", ls=8)
    for ex in (9.35, 10.45):
        ax.add_patch(Circle((ex, 4.62), 0.155, fc="#ffffff", zorder=9))
        ax.add_patch(Circle((ex + 0.03, 4.60), 0.065, fc="#0d1b2a", zorder=10))
    # 天线
    ax.plot([9.9, 9.9], [5.0, 5.45], color=GOLD, lw=2.5, zorder=6)
    glow_circle(ax, 9.9, 5.55, 0.13, GOLD)
    ax.add_patch(Circle((9.9, 5.55), 0.13, fc=GOLD, zorder=7))

    # 身体 CORE
    block(ax, 9.9, 2.85, 2.85, 1.55, GOLD, "CORE", ls=10, studs=False)

    # 左臂 TOOLS / 右臂 MEMORY
    block(ax, 7.35, 3.28, 1.95, 0.78, MINT, "TOOLS", ls=7.5)
    block(ax, 12.45, 3.28, 1.95, 0.78, LAV, "MEMORY", ls=7.5)
    ax.add_patch(Rectangle((8.25, 3.16), 0.5, 0.24, fc="#223350", lw=0, zorder=5))
    ax.add_patch(Rectangle((11.05, 3.16), 0.5, 0.24, fc="#223350", lw=0, zorder=5))

    # 腿 LOOP
    block(ax, 9.05, 1.68, 0.78, 0.82, CORAL, "LOOP", ls=6.5, studs=False)
    block(ax, 10.75, 1.68, 0.78, 0.82, CORAL, None, studs=False)
    ax.plot([9.6, 10.2], [1.68, 1.68], color=CORAL, lw=0, zorder=5)

    # 左下散落积木
    block(ax, 2.1, 1.6, 1.1, 0.62, LAV, None, studs=False)
    block(ax, 3.6, 1.25, 0.9, 0.55, MINT, None, studs=False)
    block(ax, 1.2, 1.15, 0.8, 0.5, GOLD, None, studs=False)

    fig.savefig(f"{OUT}/ch18-banner.png", dpi=100)
    plt.close(fig)


# ================================================================
# CH.19  知识库 Agent —— 书架 + 放大镜 + 引用气泡
# ================================================================
def draw_ch19():
    fig, ax = new_canvas()
    scatter_dots(ax, seed=19)
    chapter_tag(ax, "CH.19", x=1.05, y=5.66)

    # ---- 左: 书架 ----
    rrect(ax, 3.1, 3.42, 4.3, 3.95, fc=PANEL, ec=LAV, lw=1.8, rs=0.18,
          alpha=0.75, z=2)
    palette = [CYAN, GOLD, CORAL, MINT, LAV, "#f7b2c1", "#7fd8ff"]
    for shelf_y in (2.25, 3.98):
        ax.add_patch(Rectangle((0.98, shelf_y), 4.24, 0.13, fc="#3a4a6b",
                               lw=0, zorder=3))
        rng = np.random.default_rng(int(shelf_y * 10))
        bx = 1.28
        i = 0
        while bx < 4.9:
            bw = rng.uniform(0.26, 0.46)
            bh = rng.uniform(1.15, 1.55)
            if shelf_y > 3:  # 上层书稍矮
                bh *= 0.70
            c = palette[i % len(palette)]
            if i == 4 and shelf_y < 3:  # 一本倾斜的书
                ax.add_patch(Polygon([(bx, 2.38), (bx + bw, 2.38),
                                      (bx + bw + 0.28, 2.38 + bh),
                                      (bx + 0.28, 2.38 + bh)],
                                     fc=c, alpha=0.92, lw=0, zorder=4))
            else:
                ax.add_patch(Rectangle((bx, shelf_y + 0.13), bw, bh, fc=c,
                                       alpha=0.92, lw=0, zorder=4))
                ax.add_patch(Rectangle((bx + 0.05, shelf_y + 0.13 + bh - 0.18),
                                       bw - 0.10, 0.05, fc="#0d1b2a",
                                       alpha=0.35, lw=0, zorder=5))
            bx += bw + 0.07
            i += 1

    # ---- 中: 文档页 + 放大镜 ----
    rrect(ax, 8.0, 3.0, 3.1, 3.8, fc="#f5f7ff", rs=0.10, z=4)
    glow_rrect(ax, 8.0, 3.0, 3.1, 3.8, "#f5f7ff", rs=0.10)
    for i, yy in enumerate((4.45, 4.18, 3.91, 3.64)):
        ax.add_patch(Rectangle((6.65, yy), 2.3 - 0.25 * (i % 2), 0.075,
                               fc="#8fa3c8", alpha=0.8, lw=0, zorder=5))
    # 命中行(金色高亮)
    rrect(ax, 7.82, 3.28, 2.15, 0.30, fc=GOLD, rs=0.05, alpha=0.95, z=5)
    for yy in (2.9, 2.63, 2.36, 2.09, 1.82, 1.55):
        ax.add_patch(Rectangle((6.65, yy), 2.3 - 0.3 * (yy * 3 % 2 > 1), 0.075,
                               fc="#8fa3c8", alpha=0.8, lw=0, zorder=5))

    mx, my, mr = 8.35, 3.55, 0.92
    glow_circle(ax, mx, my, mr, GOLD)
    ax.add_patch(Circle((mx, my), mr, fc=CYAN, alpha=0.16, ec="none", zorder=8))
    ax.add_patch(Circle((mx, my), mr, fc="none", ec=GOLD, lw=7, zorder=9))
    a = np.deg2rad(-45)
    ax.plot([mx + mr * np.cos(a) - 0.06, mx + (mr + 0.95) * np.cos(a)],
            [my + mr * np.sin(a) - 0.06, my + (mr + 0.95) * np.sin(a)],
            color=GOLD, lw=9, zorder=8, solid_capstyle="round")
    ax.plot([mx - 0.35, mx - 0.05], [my + 0.55, my + 0.68],
            color="#ffffff", lw=3, alpha=0.8, zorder=10,
            solid_capstyle="round")

    # 页 -> 气泡
    ax.add_patch(FancyArrowPatch((9.65, 3.4), (11.15, 3.35),
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=MINT, lw=2.4, linestyle=(0, (5, 3)),
                                 zorder=5))

    # ---- 右: 对话气泡 + 引用角标 ----
    bx, by, bw, bh = 12.85, 3.35, 3.05, 2.0
    glow_rrect(ax, bx, by, bw, bh, MINT, rs=0.22)
    rrect(ax, bx, by, bw, bh, fc=MINT, rs=0.22, z=6)
    ax.add_patch(Polygon([(bx - 0.75, by - 0.75), (bx - 0.15, by - 1.0),
                          (bx - 0.35, by - 0.25)], fc=MINT, lw=0, zorder=6))
    for i, (yy, ww) in enumerate(((3.85, 2.2), (3.52, 2.35), (3.19, 1.7))):
        ax.add_patch(Rectangle((bx - ww / 2, yy - 0.045), ww, 0.09,
                               fc="#ffffff", alpha=0.8, lw=0, zorder=7))
    # 引用角标 [1]
    bx2, by2 = 13.85, 2.62
    rrect(ax, bx2, by2, 0.46, 0.46, fc=GOLD, rs=0.08, z=8)
    txt(ax, bx2, by2, "1", size=9, color="#0d1b2a")
    ax.plot([bx2, bx2], [by2 + 0.23, 3.05], color=GOLD, lw=1.4,
            linestyle=(0, (2, 2)), alpha=0.7, zorder=7)

    fig.savefig(f"{OUT}/ch19-banner.png", dpi=100)
    plt.close(fig)


# ================================================================
# CH.20  多 Agent 流水线 —— 传送带四工位 -> 发光报告
# ================================================================
def station_panel(ax, cx, color):
    rrect(ax, cx, 3.62, 2.05, 2.55, fc=PANEL, ec=color, lw=1.6, rs=0.2,
          alpha=0.62, z=2)
    for sx in (cx - 0.72, cx + 0.72):
        ax.add_patch(Rectangle((sx - 0.05, 1.78), 0.10, 0.62, fc=color,
                               alpha=0.5, lw=0, zorder=2))


def draw_ch20():
    fig, ax = new_canvas()
    scatter_dots(ax, seed=20)
    chapter_tag(ax, "CH.20", x=0.75, y=5.45)

    # 传送带
    rrect(ax, 7.05, 1.55, 11.7, 0.42, fc="#223350", ec="#3a4a6b", lw=1.5,
          rs=0.2, z=3)
    for wx in np.arange(1.8, 12.9, 1.35):
        ax.add_patch(Circle((wx, 1.32), 0.20, fc="#3a4a6b", ec="none", zorder=2))
        ax.add_patch(Circle((wx, 1.32), 0.07, fc="#54689a", ec="none", zorder=3))
    for ax_ in (3.6, 6.2, 8.9, 11.5):
        ax.add_patch(FancyArrowPatch((ax_ - 0.18, 1.55), (ax_ + 0.22, 1.55),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color="#7fd8ff", lw=1.6, alpha=0.55,
                                     zorder=4))

    # 工位 1: 收集漏斗
    cx = 2.6
    station_panel(ax, cx, CYAN)
    doc_icon(ax, cx, 4.5, 0.52, 0.62, lines=2, z=5)
    ax.plot([cx, cx], [4.1, 3.85], color=CYAN, lw=1.5,
            linestyle=(0, (2, 2)), alpha=0.6, zorder=4)
    ax.add_patch(Polygon([(cx - 0.72, 3.85), (cx + 0.72, 3.85),
                          (cx + 0.16, 3.05), (cx - 0.16, 3.05)],
                         fc=CYAN, ec="none", zorder=6))
    ax.add_patch(Rectangle((cx - 0.16, 2.82), 0.32, 0.24, fc=CYAN, lw=0, zorder=6))

    # 工位 2: 筛选筛子
    cx = 5.4
    station_panel(ax, cx, MINT)
    rrect(ax, cx, 3.5, 1.32, 0.92, fc="none", ec=MINT, lw=3.5, rs=0.08, z=6)
    for k in (-0.22, 0, 0.22):
        ax.plot([cx - 0.58, cx + 0.58], [3.5 + k, 3.5 + k], color=MINT,
                lw=1.4, alpha=0.8, zorder=7)
        ax.plot([cx + k, cx + k], [3.10, 3.90], color=MINT, lw=1.4,
                alpha=0.8, zorder=7)
    for dx, dy in ((-0.3, 2.72), (0.05, 2.58), (0.38, 2.74)):
        ax.add_patch(Circle((cx + dx, dy), 0.07, fc=MINT, alpha=0.9, zorder=6))

    # 工位 3: 书写笔
    cx = 8.2
    station_panel(ax, cx, GOLD)
    ax.plot([cx - 0.55, cx + 0.42], [2.95, 3.92], color=GOLD, lw=11,
            zorder=6, solid_capstyle="round")
    ax.add_patch(Polygon([(cx - 0.55, 2.95), (cx - 0.30, 3.02),
                          (cx - 0.63, 3.10)], fc="#0d1b2a", lw=0, zorder=7))
    ax.plot([cx - 0.72, cx - 0.30, cx + 0.15, cx + 0.62],
            [2.62, 2.72, 2.55, 2.68], color="#ffffff", lw=2, alpha=0.85,
            zorder=5, solid_capstyle="round")
    ax.scatter([cx + 0.85], [4.28], marker="*", s=90, color=GOLD, zorder=6)

    # 工位 4: 排版印章
    cx = 11.0
    station_panel(ax, cx, LAV)
    ax.add_patch(Circle((cx, 4.28), 0.16, fc=LAV, ec="none", zorder=6))
    ax.add_patch(Rectangle((cx - 0.09, 3.82), 0.18, 0.34, fc=LAV, lw=0, zorder=6))
    ax.add_patch(Polygon([(cx - 0.62, 3.82), (cx + 0.62, 3.82),
                          (cx + 0.48, 3.36), (cx - 0.48, 3.36)],
                         fc=LAV, ec="none", zorder=6))
    for a_ in (-30, 0, 30):
        a = np.deg2rad(a_)
        ax.plot([cx + 0.68 * np.cos(a), cx + 0.98 * np.cos(a)],
                [3.20 + 0.68 * np.sin(a), 3.20 + 0.98 * np.sin(a)],
                color=LAV, lw=2, alpha=0.6, zorder=5)

    # 传送带上的文档(加工中)
    doc_icon(ax, 3.98, 2.15, 0.72, 0.85, lines=3)
    doc_icon(ax, 6.75, 2.15, 0.72, 0.85, lines=2)
    ax.plot([6.5, 6.74, 6.98], [1.85, 2.04, 1.85], color=MINT, lw=2.6,
            zorder=8, solid_capstyle="round")
    doc_icon(ax, 9.55, 2.15, 0.72, 0.85, lines=3)
    ax.scatter([9.9], [2.55], marker="*", s=70, color=GOLD, zorder=7)

    # 最终发光报告
    rx, ry = 14.15, 2.85
    glow_circle(ax, rx, ry, 1.25, GOLD)
    rrect(ax, rx, ry, 1.9, 2.45, fc="#f8f6ee", ec=GOLD, lw=3, rs=0.08, z=6)
    rrect(ax, rx, ry + 0.82, 1.35, 0.30, fc=GOLD, rs=0.04, z=7)
    for i, yy in enumerate((2.75, 2.45, 2.15, 1.85)):
        ax.add_patch(Rectangle((rx - 0.62, yy), 1.24 - 0.2 * (i % 2), 0.07,
                               fc="#8fa3c8", lw=0, zorder=7))
    ax.scatter([rx + 0.62], [ry + 1.05], marker="*", s=110, color=GOLD, zorder=8)
    for a_ in range(0, 360, 45):
        a = np.deg2rad(a_)
        ax.plot([rx + 1.06 * np.cos(a), rx + 1.42 * np.cos(a)],
                [ry + 1.06 * np.sin(a), ry + 1.42 * np.sin(a)],
                color=GOLD, lw=2.2, alpha=0.5, zorder=4)
    ax.add_patch(FancyArrowPatch((13.0, 1.55), (13.65, 2.3),
                                 arrowstyle="-|>", mutation_scale=20,
                                 color=MINT, lw=2.6, zorder=5))

    fig.savefig(f"{OUT}/ch20-banner.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    draw_ch17()
    draw_ch18()
    draw_ch19()
    draw_ch20()
    print("done: ch17-ch20 banners rendered")
