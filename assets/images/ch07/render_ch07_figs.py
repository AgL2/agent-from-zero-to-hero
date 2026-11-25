"""第 7 章两张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系，风格对齐 ch01 模板）"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["font.family"] = ["Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

NAVY, INDIGO = "#0d1b2a", "#1b263b"
PANEL = "#16324f"
CYAN, AMBER, CORAL, MINT, VIOLET = "#22d3ee", "#fbbf24", "#fb7185", "#34d399", "#a78bfa"
TXT = "#e8f1f8"
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch07/"


def new_canvas(w=16, h=6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[0, w, 0, h], aspect="auto", origin="lower",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg", [NAVY, INDIGO]),
              zorder=0)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    rng = np.random.default_rng(3)
    ax.scatter(rng.uniform(0, w, int(w * h * 6)), rng.uniform(0, h, int(w * h * 6)),
               s=rng.uniform(2, 14, int(w * h * 6)), c="white", alpha=0.07, zorder=1)
    return fig, ax


def box(ax, x, y, w, h, text, color, fs=13, glow=True):
    if glow:
        ax.add_patch(FancyBboxPatch((x - 0.06, y - 0.06), w + 0.12, h + 0.12,
                                    boxstyle="round,pad=0.06", fc=color, ec="none", alpha=0.15, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=PANEL, ec=color, lw=2, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=TXT, fontsize=fs, zorder=5, linespacing=1.5)


def arrow(ax, p1, p2, color=CYAN, lw=2.2, style="-|>", ls="-",
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                                 color=color, lw=lw, linestyle=ls, zorder=3,
                                 connectionstyle=connectionstyle))


def diamond(ax, x, y, r, text, color, fs=12):
    ax.add_patch(Circle((x, y), r * 1.6, fc=color, ec="none", alpha=0.12, zorder=3))
    ax.add_patch(Circle((x, y), r, fc=PANEL, ec=color, lw=2, zorder=4))
    ax.text(x, y, text, ha="center", va="center", color=TXT, fontsize=fs, zorder=5)


# ============ 图 1：提示词调试迭代工作流 ============
fig, ax = new_canvas(16, 8)
ax.text(8.0, 7.55, "提示词调试工作流：和调试代码一模一样",
        color=TXT, fontsize=18, fontweight="bold", ha="center")

box(ax, 5.5, 6.2, 5.0, 0.85, "写下第一版提示词", CYAN, fs=14)
box(ax, 5.5, 4.8, 5.0, 0.85, "在 5~10 个代表性用例上跑", CYAN, fs=14)
diamond(ax, 8.0, 3.6, 0.6, "全对？", AMBER, fs=13)
box(ax, 12.0, 6.2, 3.7, 0.85, "上线，但保留用例集\n（回归测试资产）", MINT, fs=12.5)
box(ax, 5.5, 1.7, 5.0, 1.0, "错题分析：是格式错？知识错？\n推理错？没读懂要求？", CORAL, fs=12.5)
box(ax, 5.5, 0.4, 5.0, 0.85, "针对失败模式改提示\n（一次只改一处！）", VIOLET, fs=12.5)

arrow(ax, (8.0, 6.2), (8.0, 5.72), CYAN, lw=2.4)
arrow(ax, (8.0, 4.8), (8.0, 4.32), CYAN, lw=2.4)
arrow(ax, (8.0, 2.98), (8.0, 2.74), AMBER, lw=2.4)                       # 否 ↓
ax.text(8.42, 2.86, "否", color=AMBER, fontsize=13, fontweight="bold", zorder=5)
arrow(ax, (8.0, 1.7), (8.0, 1.29), CORAL, lw=2.4)
arrow(ax, (8.62, 3.85), (11.9, 6.45), MINT, lw=2.4, connectionstyle="arc3,rad=0.25")
ax.text(10.9, 5.5, "是", color=MINT, fontsize=13, fontweight="bold", zorder=5)
arrow(ax, (5.44, 0.82), (5.44, 5.24), VIOLET, lw=2.4, ls="--", connectionstyle="arc3,rad=-0.3")
ax.text(3.75, 3.0, "改完重跑", color=VIOLET, fontsize=12.5, rotation=90,
        ha="center", va="center", zorder=5)
fig.savefig(OUT + "prompt-debug-loop.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：提示策略决策树 ============
fig, ax = new_canvas(16, 9.6)
ax.text(8.0, 9.1, "拿到一个 LLM 任务，怎么选提示策略",
        color=TXT, fontsize=18, fontweight="bold", ha="center")

box(ax, 5.5, 7.6, 5.0, 0.9, "拿到一个 LLM 任务，先问四个问题", CYAN, fs=14)

questions = [
    (0.45, "是单步任务吗？\n（分类 / 抽取 / 翻译）"),
    (4.45, "需要多步推理吗？\n（数学 / 逻辑 / 规划）"),
    (8.45, "需要外部信息\n或工具吗？"),
    (12.45, "输出要被程序\n消费吗？"),
]
answers = [
    "zero-shot + 四要素写齐\n格式挑剔，加 2~5 个\n覆盖边界的 few-shot 示例",
    "zero-shot CoT：\n“让我们一步一步思考”\n答案唯一且错不起 →\nSelf-Consistency（k=5~20）\n需探索回溯 → ToT 思想（第 10 章）",
    "ReAct 格式，或原生\nfunction calling\n（第 10 章两种都写）",
    "schema 写进提示 +\nJSON mode / 约束解码\n仍偶发坏格式 →\npydantic 校验 + 错误回喂重试",
]
colors = [MINT, AMBER, VIOLET, CORAL]
rads = [0.25, 0.1, -0.1, -0.25]
for i, ((qx, qtext), atext, color) in enumerate(zip(questions, answers, colors)):
    box(ax, qx, 5.2, 3.1, 1.15, qtext, CYAN, fs=12)
    box(ax, qx, 2.35, 3.1, 2.3, atext, color, fs=11.5)
    arrow(ax, (8.0, 7.6), (qx + 1.55, 6.42), CYAN, lw=1.8,
          connectionstyle=f"arc3,rad={rads[i]}")
    arrow(ax, (qx + 1.55, 5.2), (qx + 1.55, 4.72), CYAN, lw=1.8)

ax.text(8.0, 1.15, "两条纪律不豁免：① 改提示前先建 5~10 条用例集，改时一次只改一处　② system / 模板 / user 三层分家",
        ha="center", color=TXT, fontsize=12.5, alpha=0.9)
fig.savefig(OUT + "prompt-decision-tree.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 2 figures rendered")
