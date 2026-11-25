"""第 19 章一张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系）
个人知识库 Agent 双管线：离线索引管线 + 在线查询管线，共享 TfidfEmbedder 与 VectorIndex
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
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch19/"


def new_canvas(w=16, h=7):
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


# ============ 知识库 Agent 双管线 ============
fig, ax = new_canvas(16, 7)

# ---- 管线标题 ----
ax.text(4.0, 6.55, "索引管线（离线，跑一次）", color=MINT, fontsize=15,
        fontweight="bold", ha="center", va="center")
ax.text(12.0, 6.55, "查询管线（在线，每次提问）", color=CYAN, fontsize=15,
        fontweight="bold", ha="center", va="center")

# ---- 左列：索引管线 ----
LX, LW = 1.2, 5.6
box(ax, LX, 5.05, LW, 1.0, "notes/*.md\n几十篇 Markdown 笔记", MINT, fs=12.5)
box(ax, LX, 3.70, LW, 1.0, "递归切片 chunk=120\n段落 → 句子 → 硬切，重叠 20", AMBER, fs=12.5)
box(ax, LX, 2.35, LW, 1.0, "TF-IDF embedding\n字符 bigram + L2 归一化", VIOLET, fs=12.5)
box(ax, LX, 1.00, LW, 1.0, "kb_index.npz\n向量 + 切片元数据落盘", CORAL, fs=12.5)
for y0, y1 in ((5.05, 4.70), (3.70, 3.35), (2.35, 2.00)):
    arrow(ax, (4.0, y0), (4.0, y1), MINT)

# ---- 右列：查询管线 ----
RX, RW = 9.2, 5.6
box(ax, RX, 5.05, RW, 1.0, "用户问题\n支持“它有什么缺点？”式追问", CYAN, fs=12.5)
box(ax, RX, 3.70, RW, 1.0, "记忆追问改写\nChatMemory 拼上一轮问题", AMBER, fs=12.5)
box(ax, RX, 2.35, RW, 1.0, "top-k 检索\n点积即余弦，Hit@3 可评", VIOLET, fs=12.5)
box(ax, RX, 1.00, RW, 1.0, "带引用的回答\n出处来自检索元数据", MINT, fs=12.5)
for y0, y1 in ((5.05, 4.70), (3.70, 3.35), (2.35, 2.00)):
    arrow(ax, (12.0, y0), (12.0, y1), CYAN)

# ---- 共享组件：两条管线之间 ----
arrow(ax, (LX + LW, 2.85), (RX, 2.85), VIOLET, lw=2.0, ls="--")
ax.text(8.0, 3.42, "共享\nTfidfEmbedder\nVectorIndex", color=VIOLET, fontsize=10.5,
        ha="center", va="center", zorder=5, linespacing=1.4)
arrow(ax, (LX + LW, 1.50), (RX, 2.50), CORAL, lw=2.0, ls="--")
ax.text(7.9, 1.32, "启动时加载\n（秒级）", color=CORAL, fontsize=10.5, ha="center", va="center", zorder=5)

fig.savefig(OUT + "kb-pipelines.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)
print("OK: ch19 figures rendered")
