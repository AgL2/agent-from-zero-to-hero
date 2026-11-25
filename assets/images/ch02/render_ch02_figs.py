"""第 2 章两张正式配图（与第 1 章视觉系统一致：深蓝渐变底、五色体系）
替换正文中的 ASCII 教学示意图：卷积直觉图、典型 CNN block 流程图
数值与 2.2.2 节手算及 2.4 节完全一致。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = ["Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

NAVY, INDIGO = "#0d1b2a", "#1b263b"
PANEL = "#16324f"
CYAN, AMBER, CORAL, MINT, VIOLET = "#22d3ee", "#fbbf24", "#fb7185", "#34d399", "#a78bfa"
TXT = "#e8f1f8"
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch02/"


def new_canvas(w=16, h=6.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[0, 16, 0, 6.4], aspect="auto", origin="lower",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg", [NAVY, INDIGO]),
              zorder=0)
    ax.set_xlim(0, 16); ax.set_ylim(0, 6.4); ax.axis("off")
    rng = np.random.default_rng(3)
    ax.scatter(rng.uniform(0, 16, 90), rng.uniform(0, 6.4, 90),
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


def arrow(ax, p1, p2, color=CYAN, lw=2.2, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


def grid(ax, x, y, cell, M, color, fs=11, highlight=None, hl_color=AMBER, title=None,
         title_color=None):
    """画数值网格；highlight=(r0,c0,h,w) 用 hl_color 框出子窗口"""
    h, w = M.shape
    for i in range(h):
        for j in range(w):
            ax.add_patch(Rectangle((x + j * cell, y - (i + 1) * cell), cell, cell,
                                   fc=PANEL, ec="#2c4a6e", lw=1.0, zorder=4))
            ax.text(x + j * cell + cell / 2, y - (i + 1) * cell + cell / 2, str(M[i, j]),
                    ha="center", va="center", color=TXT, fontsize=fs, zorder=5)
    if highlight is not None:
        r0, c0, hh, ww = highlight
        ax.add_patch(Rectangle((x + c0 * cell, y - (r0 + hh) * cell), ww * cell, hh * cell,
                               fc="none", ec=hl_color, lw=2.6, zorder=6))
    if title is not None:
        ax.text(x + w * cell / 2, y + 0.28, title, ha="center", va="bottom",
                color=title_color or color, fontsize=13, fontweight="bold", zorder=6)
    return x + w * cell, y - h * cell  # 右边界与下边界


# ============ 图 1：卷积运算直觉（5x5 输入 ⊛ 3x3 核 → 3x3 特征图） ============
X = np.array([[1, 1, 1, 0, 0],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 1, 1],
              [0, 0, 1, 1, 0],
              [0, 1, 1, 0, 0]])
K = np.array([[1, 0, 1],
              [0, 1, 0],
              [1, 0, 1]])
F = np.array([[4, 3, 4],
              [2, 4, 3],
              [2, 3, 4]])

fig, ax = new_canvas()
cell = 0.62
top = 5.55
# 输入（左上角 3x3 窗口高亮）
grid(ax, 1.1, top, cell, X, CYAN, highlight=(0, 0, 3, 3), title="输入图像 5×5")
ax.text(1.1 + 2.5 * cell, top - 5 * cell - 0.30, "当前 3×3 窗口",
        ha="center", va="top", color=AMBER, fontsize=11.5, zorder=6)
# 卷积核
grid(ax, 6.3, top, cell, K, MINT, title="卷积核 3×3")
ax.text(6.3 + 1.5 * cell, top - 3 * cell - 0.30, "滑动扫过全图",
        ha="center", va="top", color=MINT, fontsize=11.5, zorder=6)
# ⊛
ax.text(5.35, 4.3, "⊛", ha="center", va="center", color=AMBER, fontsize=24, zorder=5)
# 输出特征图
grid(ax, 11.6, top, cell, F, AMBER, highlight=(0, 0, 1, 1), hl_color=CYAN,
     title="输出特征图 3×3")
ax.text(11.6 + 1.5 * cell, top - 3 * cell - 0.30, "左上角 = 4",
        ha="center", va="top", color=CYAN, fontsize=11.5, zorder=6)
# 箭头：输入→核，核→输出
arrow(ax, (4.5, 3.7), (6.0, 3.7), CYAN)
arrow(ax, (8.45, 3.7), (11.2, 3.7), AMBER)
# 底部说明
ax.text(8.0, 1.75, "每个输出 = 卷积核与对应 3×3 窗口逐元素相乘、再求和",
        ha="center", color=TXT, fontsize=14, zorder=5)
ax.text(8.0, 1.15, "左上角手算：1×1 + 1×0 + 1×1 + 0×0 + 1×1 + 1×0 + 0×1 + 0×0 + 1×1 = 4",
        ha="center", color=CYAN, fontsize=12.5, alpha=0.95, zorder=5)
ax.text(8.0, 0.55, "窗口滑一步、再算一次，直到扫完整张图（2.2.2 节完整手算）",
        ha="center", color=TXT, fontsize=12, alpha=0.8, zorder=5)
fig.savefig(OUT + "conv-intuition.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：典型 CNN block 流水线（空间换语义的漏斗） ============
fig, ax = new_canvas(16, 6.0)
y0 = 3.15
box(ax, 0.7, y0, 2.6, 1.1, "卷积 3×3 → ReLU\n通道 1→16，28×28", CYAN, fs=11.5)
box(ax, 3.9, y0, 2.2, 1.1, "MaxPool 2×2\n尺寸减半 14×14", MINT, fs=11.5)
box(ax, 6.7, y0, 2.6, 1.1, "卷积 3×3 → ReLU\n通道 16→32，14×14", CYAN, fs=11.5)
box(ax, 9.9, y0, 2.2, 1.1, "MaxPool 2×2\n尺寸减半 7×7", MINT, fs=11.5)
box(ax, 12.9, y0, 2.4, 1.1, "展平 → 全连接\n→ Softmax", AMBER, fs=11.5)
arrow(ax, (3.3, y0 + 0.55), (3.9, y0 + 0.55), CYAN)
arrow(ax, (6.1, y0 + 0.55), (6.7, y0 + 0.55), MINT)
arrow(ax, (9.3, y0 + 0.55), (9.9, y0 + 0.55), CYAN)
arrow(ax, (12.1, y0 + 0.55), (12.9, y0 + 0.55), AMBER)
ax.text(0.7, 5.15, "空间尺寸 28 → 14 → 7，一路缩；通道数 1 → 16 → 32，一路涨",
        color=AMBER, fontsize=14, fontweight="bold", ha="left", zorder=5)
ax.text(0.7, 4.55, "每个 block：先[卷积+ReLU]重复 2~3 次，再下采样一次（2.4 节）",
        color=TXT, fontsize=12.5, alpha=0.9, ha="left", zorder=5)
ax.text(0.7, 1.95, "设计哲学：越深，空间分辨率越低，语义抽象度越高",
        color=MINT, fontsize=13.5, ha="left", zorder=5)
ax.text(0.7, 1.35, "浅层学边缘纹理 → 中层学部件（眼睛、轮子） → 深层学整体概念（脸、车）",
        color=TXT, fontsize=12.5, alpha=0.85, ha="left", zorder=5)
ax.text(0.7, 0.75, "“空间换语义”的漏斗：池化没有可学习参数；现代网络也常用步长卷积顺手完成下采样",
        color=TXT, fontsize=12, alpha=0.7, ha="left", zorder=5)
fig.savefig(OUT + "cnn-block.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 2 figures rendered")
