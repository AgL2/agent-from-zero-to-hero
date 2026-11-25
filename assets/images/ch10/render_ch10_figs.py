"""第 10 章两张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系）
图 1：三层范式分工（Reflexion / Plan-and-Execute / ReAct 可叠加层级）
图 2：Reflexion 迭代循环（执行 → 评估 → 语言反思 → 教训入记忆 → 重试）
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
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch10/"


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


def arrow(ax, p1, p2, color=CYAN, lw=2.2, ls="-", ms=18):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


# ============ 图 1：三个范式的可叠加层级 ============
fig, ax = new_canvas(14, 7)
ax.text(7.0, 6.55, "三个范式的分工：不是三选一，是可叠加的三层", color=TXT,
        fontsize=16, fontweight="bold", ha="center")
# 任务入口
ax.text(7.0, 5.95, "任务", color=MINT, fontsize=14, fontweight="bold", ha="center")
arrow(ax, (7.0, 5.8), (7.0, 5.35), MINT, lw=2.6)
layers = [
    (4.05, "Reflexion：外层迭代闭环\n（失败信号 → 教训 → 重试）", VIOLET, "外层", "这次错了，下次聪明点"),
    (2.45, "Plan-and-Execute：大结构\n（Planner 出计划，Executor 执行）", AMBER, "中层", "先画图纸，再施工"),
    (0.85, "ReAct：小步引擎\n（Thought ⇄ Action ⇄ Observation）", CYAN, "内层", "边想边做，每步看最新证据"),
]
for y, text, color, tag, note in layers:
    box(ax, 3.6, y, 6.8, 1.25, text, color, fs=13.5)
    ax.text(3.2, y + 0.62, tag, color=color, fontsize=13, fontweight="bold", ha="right")
    ax.text(10.9, y + 0.62, note, color=color, fontsize=12, ha="left", alpha=0.95)
# 层间箭头
arrow(ax, (7.0, 4.05), (7.0, 3.78), TXT, lw=2.0, ms=15)
arrow(ax, (7.0, 2.45), (7.0, 2.18), TXT, lw=2.0, ms=15)
ax.text(0.55, 3.75, "外层管迭代\n中层管结构\n内层管单步", color=TXT, fontsize=12,
        alpha=0.85, ha="left", va="center", linespacing=1.9)
fig.savefig(OUT + "pattern-layers.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：Reflexion 迭代循环 ============
fig, ax = new_canvas(16, 7)
ax.text(8.0, 6.55, "Reflexion 迭代循环：失败信号 → 语言教训 → 带着教训重试", color=TXT,
        fontsize=16, fontweight="bold", ha="center")
# ① 执行
box(ax, 1.6, 4.4, 3.4, 1.3, "① 执行\nAgent 试解任务", CYAN)
# ② 评估
box(ax, 7.2, 4.4, 3.4, 1.3, "② 评估\n测试通过？", AMBER)
ax.text(8.9, 6.15, "信号来源：单元测试 / 编译\n环境反馈 / 人工打分", color=AMBER,
        fontsize=11.5, ha="center", va="center", linespacing=1.5, alpha=0.95)
# 成功出口
box(ax, 12.4, 4.4, 2.6, 1.3, "通过 ✓\n任务完成", MINT)
# ③ 自我反思
box(ax, 6.4, 1.3, 5.0, 1.3, "③ 自我反思：这次为什么失败？\n生成一段文字「教训」", CORAL)
# ④ 教训存入 episodic memory
box(ax, 0.6, 1.3, 4.6, 1.3, "④ 教训存入\nepisodic memory", VIOLET)
# 循环箭头
arrow(ax, (5.0, 5.05), (7.2, 5.05), CYAN, lw=2.4)                       # ①→② 轨迹
ax.text(6.1, 5.3, "轨迹", color=TXT, fontsize=11, ha="center")
arrow(ax, (10.6, 5.05), (12.4, 5.05), MINT, lw=2.2)                     # ②→成功
ax.text(11.5, 5.3, "通过", color=MINT, fontsize=11, ha="center")
arrow(ax, (8.9, 4.4), (8.9, 2.6), CORAL, lw=2.4)                        # ②→③ 失败
ax.text(9.25, 3.5, "失败", color=CORAL, fontsize=12, ha="left")
arrow(ax, (6.4, 1.95), (5.2, 1.95), CORAL, lw=2.2)                      # ③→④
arrow(ax, (2.9, 2.6), (3.0, 4.4), VIOLET, lw=2.4)                       # ④→①
ax.text(2.15, 3.5, "回到 ①：\n带着全部历史教训重试", color=VIOLET, fontsize=11.5,
        ha="center", va="center", linespacing=1.5)
fig.savefig(OUT + "reflexion-loop.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)
print("OK: ch10 2 figures rendered")
