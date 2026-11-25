"""第 11 章一张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系）
图 1：Function Calling 五步全链路时序图（模型只出嘴，宿主才出手）
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
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch11/"


def new_canvas(w=16, h=8):
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[0, w, 0, h], aspect="auto", origin="lower",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg", [NAVY, INDIGO]),
              zorder=0)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    rng = np.random.default_rng(3)
    ax.scatter(rng.uniform(0, w, 100), rng.uniform(0, h, 100),
               s=rng.uniform(2, 14, 100), c="white", alpha=0.07, zorder=1)
    return fig, ax


def box(ax, x, y, w, h, text, color, fs=13, glow=True):
    if glow:
        ax.add_patch(FancyBboxPatch((x - 0.06, y - 0.06), w + 0.12, h + 0.12,
                                    boxstyle="round,pad=0.06", fc=color, ec="none", alpha=0.15, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=PANEL, ec=color, lw=2, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=TXT, fontsize=fs, zorder=5, linespacing=1.5)


def arrow(ax, p1, p2, color=CYAN, lw=2.2, ls="-", ms=16):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


fig, ax = new_canvas(16, 8)
ax.text(8.0, 7.6, "Function Calling 五步全链路：模型只出嘴，宿主才出手", color=TXT,
        fontsize=16, fontweight="bold", ha="center")

# 四条生命线
lines = [(2.4, "用户", MINT), (6.3, "宿主\n（你的 Python）", CYAN),
         (10.2, "LLM", VIOLET), (13.9, "本地工具函数", AMBER)]
for x, name, color in lines:
    box(ax, x - 1.35, 6.45, 2.7, 0.95, name, color, fs=12.5)
    ax.plot([x, x], [0.95, 6.45], color=color, lw=1.6, ls=(0, (4, 5)), alpha=0.55, zorder=2)
    ax.add_patch(plt.Circle((x, 0.95), 0.09, fc=color, ec="none", zorder=3))

U, H, L, T = 2.4, 6.3, 10.2, 13.9

# 消息序列（自上而下）
msgs = [
    (U, H, 5.75, "「北京多少度？」", MINT, None),
    (H, L, 5.15, "① messages + tools schema", CYAN, "①"),
    (L, H, 4.50, "② tool_calls: get_weather({\"city\":\"北京\"})", VIOLET, "②"),
    (H, T, 3.75, "③ json.loads → get_weather(...)", AMBER, "③"),
    (T, H, 3.05, '{"temperature": 22, ...}', AMBER, None),
    (H, L, 2.35, "④ role=\"tool\" 回填 messages", CYAN, "④"),
    (L, H, 1.70, "⑤ assistant: 「北京 22°C 晴」", VIOLET, "⑤"),
    (H, U, 1.05, "「北京 22°C 晴」", MINT, None),
]
for p1, p2, y, label, color, badge in msgs:
    x1 = p1 + (1.35 if p1 < p2 else -1.35)
    x2 = p2 + (-1.35 if p1 < p2 else 1.35)
    arrow(ax, (x1, y), (x2, y), color, lw=2.0)
    mid = (p1 + p2) / 2
    ax.text(mid, y + 0.13, label, color=TXT, fontsize=11.5, ha="center", va="bottom",
            linespacing=1.4, zorder=5)
    if badge:
        ax.add_patch(plt.Circle((min(x1, x2) + 0.42, y), 0.21, fc=color, ec="none", alpha=0.9, zorder=5))
        ax.text(min(x1, x2) + 0.42, y, badge[0], color=NAVY, fontsize=10.5,
                ha="center", va="center", fontweight="bold", zorder=6)

# 关键事实注释：模型输出的调用请求与宿主执行之间只是普通函数调用
ax.text(8.0, 0.35, "关键事实：步骤②与③之间没有任何魔法，模型只输出结构化「意图」；真正执行函数的，"
                   "始终是宿主程序（你的 Python 代码）",
        color=TXT, fontsize=11.5, ha="center", va="center", alpha=0.9)
fig.savefig(OUT + "tool-calling-seq.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)
print("OK: ch11 1 figure rendered")
