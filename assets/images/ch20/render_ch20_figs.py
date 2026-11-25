"""第 20 章一张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系）
「每日 AI 资讯简报」流水线：Supervisor 编排 + 4 个专职 Agent + 共享 state + @trace 计时
"""
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
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch20/"


def new_canvas(w=15, h=8.6):
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


def arrow(ax, p1, p2, color=CYAN, lw=2.2, ls="-", ms=18):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


# ============ 多 Agent 资讯流水线 ============
fig, ax = new_canvas(15, 8.6)

# ---- Supervisor：只编排，不懂业务 ----
box(ax, 0.9, 7.15, 4.6, 1.1, "Supervisor.run(state)\n只编排、不懂业务：顺序即流程", CORAL, fs=12.5)
arrow(ax, (3.2, 7.15), (3.2, 6.35), CORAL)

# ---- 四个专职 Agent（串行链）----
AG_X, AG_W, AG_H = 0.9, 4.6, 1.15
rows = [
    (5.15, "CollectorAgent\n采集：拉取原始资讯\n@trace 24.9 ms", CYAN),
    (3.70, "FilterAgent\n筛选：关键词打分 ≥ 3\n@trace 0.1 ms", AMBER),
    (2.25, "WriterAgent\n摘要 + 编辑点评\n@trace 0.0 ms", MINT),
    (0.80, "EditorAgent\n排版 Markdown 落盘\n@trace 0.3 ms", VIOLET),
]
for y, text, color in rows:
    box(ax, AG_X, y, AG_W, AG_H, text, color, fs=12)
for y0, y1 in ((5.15, 4.85), (3.70, 3.40), (2.25, 1.95)):
    arrow(ax, (3.2, y0), (3.2, y1), CYAN, lw=1.8)

# ---- 共享 state：每个 Agent 的产出键即契约 ----
ST_X, ST_W, ST_H = 7.6, 6.8, 1.05
state_rows = [
    (5.2, "state[\"raw\"]\n10 条原始资讯（含 3 条非 AI 噪声）", CYAN),
    (3.75, "state[\"kept\"]\n7 条达标入选（MAX_KEEP=8）", AMBER),
    (2.3, "state[\"summaries\"]\n逐条摘要 + 点评", MINT),
    (0.85, "state[\"report_path\"] → report.md\n产物是文件，不是返回值", VIOLET),
]
for (ay, _, a_color), (sy, text, _) in zip(rows, state_rows):
    box(ax, ST_X, sy, ST_W, ST_H, text, a_color, fs=12)
    arrow(ax, (AG_X + AG_W, ay + AG_H / 2), (ST_X, sy + ST_H / 2), a_color, lw=1.8)

# ---- state 提示 + Trace 统计脚注 ----
ax.text(11.0, 6.9, "共享 state 字典：键名即契约", color=TXT, fontsize=13,
        fontweight="bold", ha="center", va="center", alpha=0.95)
ax.text(7.5, 0.18, "Trace 统计：Supervisor 总计 25.3 ms，Collector 一项占 98%——瓶颈自己举手（20.10 并行化实验正冲它去）",
        color=TXT, fontsize=11, ha="center", va="bottom", alpha=0.85)

fig.savefig(OUT + "news-pipeline.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)
print("OK: ch20 figures rendered")
