"""第 17 章一张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系，对齐 ch01 模板）
图 1 trace-span.png  Trace 与 Span：一次请求的结构化全记录（17.5）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

NAVY, INDIGO = "#0d1b2a", "#1b263b"
PANEL = "#16324f"
CYAN, AMBER, CORAL, MINT, VIOLET = "#22d3ee", "#fbbf24", "#fb7185", "#34d399", "#a78bfa"
TXT = "#e8f1f8"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"


def new_canvas(w=16, h=6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[0, w, 0, h], aspect="auto", origin="lower",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg", [NAVY, INDIGO]),
              zorder=0)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    rng = np.random.default_rng(3)
    ax.scatter(rng.uniform(0, w, 90), rng.uniform(0, h, 90),
               s=rng.uniform(2, 14, 90), c="white", alpha=0.07, zorder=1)
    return fig, ax


def box(ax, x, y, w, h, text, color, fs=13, glow=True):
    if glow:
        ax.add_patch(FancyBboxPatch((x - 0.06, y - 0.06), w + 0.12, h + 0.12,
                                    boxstyle="round,pad=0.06", fc=color, ec="none", alpha=0.15, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=PANEL, ec=color, lw=2, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=TXT, fontsize=fs, zorder=5, linespacing=1.5)


def arrow(ax, p1, p2, color=CYAN, lw=2.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


def label(ax, x, y, text, color=TXT, fs=12, ha="center", bold=False, alpha=1.0):
    ax.text(x, y, text, ha=ha, va="center", color=color, fontsize=fs,
            fontweight="bold" if bold else "normal", alpha=alpha, zorder=5, linespacing=1.5)


# ============ 图 1：Trace 与 Span（17.5） ============
fig, ax = new_canvas(16, 6.4)
label(ax, 8.0, 6.0, "Trace：一次完整请求的结构化全记录（树状）", TXT, 17, bold=True)

# 根节点
box(ax, 0.8, 4.0, 4.6, 1.3, "Trace（根）\n“北京今天多少度”\n总耗时 3.2s", CYAN, 12.5)

# 三个 Span 子节点
spans = [
    ("Span[llm]  plan(...)\n1.4s · 850 tok\n“调用 get_temp”", CYAN, 6.2),
    ("Span[tool] get_temp(北京)\n0.3s → “晴 12℃”", AMBER, 9.4),
    ("Span[llm]  respond(...)\n1.5s · 620 tok\n“北京今天晴，12℃”", CYAN, 12.6),
]
for t, c, x in spans:
    box(ax, x, 4.0, 2.9, 1.3, t, c, 11)
    arrow(ax, (5.5, 4.65), (x, 4.65), c, 1.6)

# 每个 Span 记录什么
label(ax, 8.0, 3.35, "每个 Span 记录：输入 · 输出 · 耗时 · token 数 · 异常", AMBER, 13, bold=True)
arrow(ax, (10.85, 3.9), (10.85, 3.6), AMBER, 1.4)

# 指标条
ax.add_patch(FancyBboxPatch((0.8, 1.3), 14.7, 1.35, boxstyle="round,pad=0.08",
                            fc=INDIGO, ec=MINT, lw=2.2, zorder=3))
label(ax, 8.15, 1.97, "从 Trace 直接统计出的指标：2 次 LLM 调用 / 1 次工具 / 1470 token / P50-P95 延迟",
      MINT, 13.5, bold=True)
label(ax, 8.15, 1.6, "出问题时可精确回放：第几步、哪个工具、传了什么参数（17.2 的“成本与延迟”尺子从这里来）",
      TXT, 12, alpha=0.9)
arrow(ax, (8.15, 1.3), (8.15, 0.85), MINT, 1.4)
label(ax, 8.0, 0.45, "评估、可观测、成本预算共用这一套基建，而不是三套报表", CORAL, 12.5, bold=True)
fig.savefig(OUT + "trace-span.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 1 figure rendered ->", OUT)
