"""第 18 章一张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系）
miniagent 整体架构：core.py 三步循环 + 三个可插拔模块 + 示例工具带
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
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch18/"


def new_canvas(w=16, h=7.2):
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


def arrow(ax, p1, p2, color=CYAN, lw=2.2, ls="-", ms=18, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


# ============ miniagent 整体架构 ============
fig, ax = new_canvas(16, 7.2)

# ---- 顶层：Agent 核心循环（core.py）----
ax.add_patch(FancyBboxPatch((1.2, 3.4), 13.6, 3.3, boxstyle="round,pad=0.06",
                            fc=INDIGO, ec=CYAN, lw=2.2, zorder=2))
ax.text(1.7, 6.38, "Agent（core.py）· run(task)：第 10 章的循环，直到给出答案",
        color=CYAN, fontsize=13.5, fontweight="bold", ha="left", va="center", zorder=5)

# 循环三步
box(ax, 1.9, 4.4, 3.2, 1.15, "① 思考\n记忆 + 工具 schema\n交给 LLM", CYAN, fs=12)
box(ax, 6.4, 4.4, 3.2, 1.15, "② 行动\n执行 tool_calls", AMBER, fs=12)
box(ax, 10.9, 4.4, 3.2, 1.15, "③ 观察\n结果写回记忆", MINT, fs=12)
arrow(ax, (5.1, 4.97), (6.4, 4.97), CYAN)
arrow(ax, (9.6, 4.97), (10.9, 4.97), AMBER)
# 回环：③ 顶部向上 -> 横向 -> 落回 ① 顶部
arrow(ax, (12.5, 5.55), (12.5, 5.97), VIOLET, lw=2.0, ls="--", style="-")
ax.plot([12.5, 3.5], [5.97, 5.97], color=VIOLET, lw=2.0, ls="--", zorder=3)
arrow(ax, (3.5, 5.97), (3.5, 5.55), VIOLET, lw=2.0, ls="--")
ax.text(11.2, 6.1, "带着观察回到 ①，循环往复", color=VIOLET, fontsize=11.5,
        ha="center", va="bottom", zorder=5)
# 出口分支
arrow(ax, (8.0, 4.4), (8.0, 3.92), CORAL, lw=2.0, ls="--")
ax.text(8.0, 3.68, "② 无 tool_calls → 跳出循环，返回最终答案；max_steps 是安全绳",
        color=CORAL, fontsize=11.5, ha="center", va="center", zorder=5)

# ---- 中层：三个可插拔模块 ----
box(ax, 1.0, 1.65, 4.2, 1.15, "llm.py\nMockLLM / OpenAILLM\n统一返回 LLMResponse", CYAN, fs=12)
box(ax, 5.9, 1.65, 4.2, 1.15, "memory.py\n对话记忆 · 滑动窗口\nsystem 提示词常驻", AMBER, fs=12)
box(ax, 10.8, 1.65, 4.2, 1.15, "tools.py\nToolRegistry + @register\nschema 从签名反射", MINT, fs=12)
arrow(ax, (3.05, 3.4), (3.05, 2.8), CYAN)
arrow(ax, (8.0, 3.4), (8.0, 2.8), AMBER)
arrow(ax, (12.9, 3.4), (12.9, 2.8), MINT)

# ---- 底层：示例工具带 ----
box(ax, 1.0, 0.25, 14.0, 0.85,
    "示例工具：calculator · web_search · file_reader —— @registry.register 一步入表，docstring 即契约",
    VIOLET, fs=12)
arrow(ax, (12.9, 1.65), (12.9, 1.1), VIOLET, lw=1.6)

fig.savefig(OUT + "miniagent-architecture.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)
print("OK: ch18 figures rendered")
