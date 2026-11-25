"""第 8 章两张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系，风格对齐 ch01 模板）"""
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
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch08/"


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


def chip(ax, x, y, w, h, text, color, fs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                fc=PANEL, ec=color, lw=1.6, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=TXT, fontsize=fs, zorder=5)


def arrow(ax, p1, p2, color=CYAN, lw=2.2, style="-|>", ls="-",
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                                 color=color, lw=lw, linestyle=ls, zorder=3,
                                 connectionstyle=connectionstyle))


# ============ 图 1：Prefill 与 Decode 两阶段 ============
fig, ax = new_canvas(16, 8)
ax.text(8.0, 7.5, "推理的两个阶段：Prefill 与 Decode",
        color=TXT, fontsize=18, fontweight="bold", ha="center")

ax.text(4.1, 6.85, "输入提示词“中国的首都是”（6 个 token）",
        color=TXT, fontsize=12.5, ha="center", alpha=0.9)
for i, ch in enumerate("中国的首都是"):
    chip(ax, 2.15 + i * 0.82, 6.05, 0.7, 0.55, ch, MINT, fs=13)
arrow(ax, (4.1, 6.02), (4.1, 5.58), CYAN, lw=2.4)

box(ax, 1.0, 3.45, 6.2, 2.1,
    "阶段一：Prefill 预填充\n一次前向，并行处理全部 6 个 token\n算出每个位置的 K/V 存入 KV Cache\n耗时 ≈ 一次大矩阵乘法",
    CYAN, fs=12.5)
box(ax, 8.8, 3.45, 6.2, 2.1,
    "阶段二：Decode 逐 token 解码\n每次只处理 1 个新 token\n从 Cache 读历史 K/V 算注意力\n重复 N 次，直到 EOS",
    AMBER, fs=12.5)
arrow(ax, (7.3, 4.5), (8.7, 4.5), CORAL, lw=2.8)
ax.text(8.0, 4.85, "第 1 个新 token", color=CORAL, fontsize=12,
        fontweight="bold", ha="center", zorder=5)

for i, ch in enumerate("北京是中国的首都"):
    chip(ax, 9.2 + i * 0.72, 2.15, 0.6, 0.55, ch, CORAL, fs=12)
arrow(ax, (11.9, 3.42), (11.9, 2.78), AMBER, lw=2.4)
ax.text(12.1, 3.1, "逐个吐出", color=AMBER, fontsize=11.5, ha="left", zorder=5)

ax.text(4.1, 2.35, "计算密集\nGPU 算力是瓶颈 → 决定 TTFT",
        color=CYAN, fontsize=12, ha="center", va="top", linespacing=1.6, zorder=5)
ax.text(11.9, 1.55, "显存带宽密集\n读权重是瓶颈 → 决定吞吐（tokens/秒）",
        color=AMBER, fontsize=12, ha="center", va="top", linespacing=1.6, zorder=5)

ax.text(8.0, 0.55, "“思考了一会儿，然后文字哗哗流出”：TTFT 由 prefill 决定，吞吐由 decode 决定——第 t 个字依赖前 t-1 个字，自回归无法并行",
        ha="center", color=TXT, fontsize=12, alpha=0.9)
fig.savefig(OUT + "prefill-decode.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：采样三旋钮管线 ============
fig, ax = new_canvas(16, 5.2)
ax.text(8.0, 4.75, "Decode 每一步：三个采样旋钮作用于同一条管线",
        color=TXT, fontsize=18, fontweight="bold", ha="center")

steps = [
    ("logits\n原始打分", CYAN),
    ("÷ temperature\n调分布锐度", AMBER),
    ("top-k 截断\n砍掉长尾", VIOLET),
    ("top-p 截断\n自适应砍长尾", MINT),
    ("softmax\n得到概率", CYAN),
    ("随机采样\n下一个 token", CORAL),
]
for i, (text, color) in enumerate(steps):
    x = 0.55 + i * 2.42
    box(ax, x, 1.75, 2.05, 1.15, text, color, fs=12)
    if i < 5:
        arrow(ax, (x + 2.08, 2.32), (x + 2.39, 2.32), TXT, lw=2.2)

ax.text(8.0, 0.85, "T→0 的极限是贪心解码（永远选最大概率）；Agent 调工具 / 生成 JSON 用低温求确定，创意写作用高温求多样",
        ha="center", color=TXT, fontsize=12.5, alpha=0.9)
fig.savefig(OUT + "sampling-pipeline.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 2 figures rendered")
