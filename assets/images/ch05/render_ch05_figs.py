"""第 5 章三张正式配图（与第 1 章视觉系统一致：深蓝渐变底、五色体系）
1. Encoder-Decoder 完整架构（替换 5.6 节 ASCII 架构图）
2. 因果 mask：允许矩阵 → 置 -inf 后的分数矩阵（替换 5.6 节 ASCII，数值与正文一致）
3. 5.4 节手算注意力权重热图（数值与手算及 5.8 节实验 A 输出完全一致）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = ["Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

NAVY, INDIGO = "#0d1b2a", "#1b263b"
PANEL = "#16324f"
CYAN, AMBER, CORAL, MINT, VIOLET = "#22d3ee", "#fbbf24", "#fb7185", "#34d399", "#a78bfa"
TXT = "#e8f1f8"
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch05/"


def new_canvas(w=16, h=6.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[0, 16, 0, h], aspect="auto", origin="lower",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg", [NAVY, INDIGO]),
              zorder=0)
    ax.set_xlim(0, 16); ax.set_ylim(0, h); ax.axis("off")
    rng = np.random.default_rng(3)
    ax.scatter(rng.uniform(0, 16, 90), rng.uniform(0, h, 90),
               s=rng.uniform(2, 14, 90), c="white", alpha=0.07, zorder=1)
    return fig, ax


def box(ax, x, y, w, h, text, color, fs=12, glow=True, lw=2):
    if glow:
        ax.add_patch(FancyBboxPatch((x - 0.05, y - 0.05), w + 0.1, h + 0.1,
                                    boxstyle="round,pad=0.05", fc=color, ec="none", alpha=0.15, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=PANEL, ec=color, lw=lw, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=TXT, fontsize=fs, zorder=5, linespacing=1.4)


def arrow(ax, p1, p2, color=CYAN, lw=2.0, ls="-", ms=15):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


# ============ 图 1：Encoder-Decoder 完整架构 ============
fig, ax = new_canvas(16, 8.0)
# Encoder 外框
box(ax, 0.8, 1.7, 5.0, 4.9, "", CYAN, lw=2.2)
ax.text(3.3, 6.75, "Encoder × N", color=CYAN, fontsize=15, fontweight="bold", ha="center", zorder=5)
box(ax, 1.3, 5.0, 4.0, 0.8, "Multi-Head\nSelf-Attention", AMBER, fs=12)
box(ax, 1.3, 4.2, 4.0, 0.5, "Add & LayerNorm", MINT, fs=11)
box(ax, 1.3, 2.9, 4.0, 0.8, "FFN\n（两层全连接）", AMBER, fs=12)
box(ax, 1.3, 2.1, 4.0, 0.5, "Add & LayerNorm", MINT, fs=11)
arrow(ax, (3.3, 5.0), (3.3, 4.75), TXT, lw=1.4)
arrow(ax, (3.3, 4.2), (3.3, 3.75), TXT, lw=1.4)
arrow(ax, (3.3, 2.9), (3.3, 2.65), TXT, lw=1.4)
ax.text(3.3, 1.15, "输入句子 “我 爱 你”", color=CYAN, fontsize=12.5, ha="center", zorder=5)
arrow(ax, (3.3, 1.45), (3.3, 1.66), CYAN, lw=2.0)
# Decoder 外框
box(ax, 9.2, 0.7, 5.0, 7.1, "", CORAL, lw=2.2)
ax.text(11.7, 7.95, "Decoder × N", color=CORAL, fontsize=15, fontweight="bold", ha="center", zorder=5)
box(ax, 9.7, 6.1, 4.0, 0.8, "Masked Multi-Head\nSelf-Attention", AMBER, fs=12)
box(ax, 9.7, 5.3, 4.0, 0.5, "Add & LayerNorm", MINT, fs=11)
box(ax, 9.7, 3.9, 4.0, 0.8, "Cross-Attention\n（Q=Dec，K/V=Enc）", VIOLET, fs=11.5)
box(ax, 9.7, 3.1, 4.0, 0.5, "Add & LayerNorm", MINT, fs=11)
box(ax, 9.7, 1.8, 4.0, 0.8, "FFN\n（两层全连接）", AMBER, fs=12)
box(ax, 9.7, 1.0, 4.0, 0.5, "Add & LayerNorm", MINT, fs=11)
arrow(ax, (11.7, 6.1), (11.7, 5.85), TXT, lw=1.4)
arrow(ax, (11.7, 5.3), (11.7, 4.75), TXT, lw=1.4)
arrow(ax, (11.7, 3.9), (11.7, 3.65), TXT, lw=1.4)
arrow(ax, (11.7, 3.1), (11.7, 2.65), TXT, lw=1.4)
arrow(ax, (11.7, 1.8), (11.7, 1.55), TXT, lw=1.4)
ax.text(11.7, 7.35, "目标已生成部分 “<s> I”", color=CORAL, fontsize=12, ha="center", zorder=5)
arrow(ax, (11.7, 7.2), (11.7, 6.95), CORAL, lw=2.0)
arrow(ax, (11.7, 0.7), (11.7, 0.48), CORAL, lw=2.0)
ax.text(11.7, 0.2, "Linear + softmax → 下一个词", color=CORAL, fontsize=12.5, ha="center", zorder=5)
# 编码结果 → 交叉注意力
arrow(ax, (5.85, 4.3), (9.65, 4.3), CYAN, lw=2.4)
ax.text(7.75, 4.55, "编码结果（K / V）", color=CYAN, fontsize=11.5, ha="center", zorder=5)
# 底部注解
ax.text(0.8, 0.62, "残差 + LayerNorm：每个子层输出都加回输入（x + Sublayer(x)）再归一化",
        color=MINT, fontsize=11.5, ha="left", zorder=5)
ax.text(0.8, 0.22, "因果 mask：Decoder 每个位置只能看自己和左边，绝不能偷看未来",
        color=CORAL, fontsize=11.5, ha="left", zorder=5)
fig.savefig(OUT + "encoder-decoder.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：因果 mask（数值与 5.6 节正文一致） ============
fig, ax = new_canvas(16, 6.2)
allowed = [["✓", "✗", "✗"], ["✓", "✓", "✗"], ["✓", "✓", "✓"]]
scores = [["0.9", "-∞", "-∞"], ["0.3", "1.2", "-∞"], ["0.1", "0.4", "0.8"]]
cell = 1.0

def label_grid(x0, y0, title, color):
    ax.text(x0 + 1.5 * cell, y0 + 0.55, title, color=color, fontsize=13.5,
            fontweight="bold", ha="center", zorder=5)
    for j in range(3):
        ax.text(x0 + j * cell + cell / 2, y0 - 0.3, f"k{j+1}", color=TXT,
                fontsize=12, ha="center", va="center", zorder=5)
    for i in range(3):
        ax.text(x0 - 0.35, y0 - i * cell - cell / 2, f"q{i+1}", color=TXT,
                fontsize=12, ha="center", va="center", zorder=5)

# 左：允许矩阵
x0, y0 = 2.2, 4.9
label_grid(x0, y0, "允许看（含自身）", MINT)
for i in range(3):
    for j in range(3):
        ok = allowed[i][j] == "✓"
        ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell,
                               fc=MINT if ok else PANEL, alpha=0.25 if ok else 1.0,
                               ec="#2c4a6e", lw=1.0, zorder=4))
        ax.text(x0 + j * cell + cell / 2, y0 - (i + 1) * cell + cell / 2, allowed[i][j],
                color=MINT if ok else "#5a6f8c", fontsize=15, ha="center", va="center", zorder=5)
# 箭头
arrow(ax, (6.1, 3.4), (8.3, 3.4), AMBER, lw=2.6, ms=22)
ax.text(7.2, 3.75, "上三角置 -∞", color=AMBER, fontsize=12.5, ha="center", zorder=5)
# 右：分数矩阵
x1, y1 = 9.0, 4.9
label_grid(x1, y1, "被因果 mask 后的分数矩阵", AMBER)
for i in range(3):
    for j in range(3):
        masked = scores[i][j] == "-∞"
        ax.add_patch(Rectangle((x1 + j * cell, y1 - (i + 1) * cell), cell, cell,
                               fc=CORAL if masked else PANEL, alpha=0.28 if masked else 1.0,
                               ec="#2c4a6e", lw=1.0, zorder=4))
        ax.text(x1 + j * cell + cell / 2, y1 - (i + 1) * cell + cell / 2, scores[i][j],
                color=CORAL if masked else TXT, fontsize=13.5, ha="center", va="center", zorder=5)
ax.text(8.0, 1.35, "softmax 对 -∞ 输出 0：被挡住的位置等于从未存在，q1 永远看不到 k2、k3",
        color=TXT, fontsize=13, ha="center", zorder=5)
ax.text(8.0, 0.7, "训练时防作弊（偷看答案会穿帮），推理时逐词生成天然满足这一约束",
        color=TXT, fontsize=12, alpha=0.8, ha="center", zorder=5)
fig.savefig(OUT + "causal-mask.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 3：5.4 节手算注意力权重（数值与正文一致） ============
A = np.array([[0.40, 0.20, 0.40],
              [0.20, 0.40, 0.40],
              [0.25, 0.25, 0.50]])
AV = [["1.20", "1.00"], ["1.00", "1.20"], ["1.26", "1.26"]]
fig, ax = new_canvas(16, 6.4)
cell = 1.15
gx, gy = 1.6, 5.3
ax.text(gx + 1.5 * cell, gy + 0.5, "注意力权重矩阵 A（逐行 softmax）", color=AMBER,
        fontsize=13.5, fontweight="bold", ha="center", zorder=5)
for j in range(3):
    ax.text(gx + j * cell + cell / 2, gy - 0.32, f"token {j+1}", color=TXT, fontsize=11.5,
            ha="center", va="center", zorder=5)
for i in range(3):
    ax.text(gx - 0.42, gy - i * cell - cell / 2, f"token {i+1}", color=TXT, fontsize=11.5,
            ha="center", va="center", zorder=5)
    for j in range(3):
        v = A[i, j]
        ax.add_patch(Rectangle((gx + j * cell, gy - (i + 1) * cell), cell, cell,
                               fc=AMBER, alpha=0.12 + 0.75 * v, ec="#2c4a6e", lw=1.0, zorder=4))
        ax.text(gx + j * cell + cell / 2, gy - (i + 1) * cell + cell / 2, f"{v:.2f}",
                color=NAVY if v > 0.4 else TXT, fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=5)
# 右侧读图面板
bx = 7.6
ax.text(bx, 5.5, "读这张图", color=CYAN, fontsize=14, fontweight="bold", ha="left", zorder=5)
ax.text(bx, 4.85, "· token 3（比如“它”）是全场焦点：\n  前两个 token 各分它 0.40，它自留 0.50",
        color=TXT, fontsize=12.5, ha="left", va="top", zorder=5, linespacing=1.6)
ax.text(bx, 3.55, "· token 1、2 除看自己（0.40）和 token 3，\n  也互相投去一瞥（0.20）",
        color=TXT, fontsize=12.5, ha="left", va="top", zorder=5, linespacing=1.6)
ax.text(bx, 2.35, "· 每行和为 1（行随机）：输出是全场 V 的凸组合", color=TXT, fontsize=12.5,
        ha="left", va="top", zorder=5)
ax.text(bx, 1.55, "输出 AV =", color=MINT, fontsize=13, fontweight="bold", ha="left", zorder=5)
ax.text(bx + 1.9, 1.55, "[[1.20, 1.00],  [1.00, 1.20],  [1.26, 1.26]]", color=MINT,
        fontsize=12.5, ha="left", va="center", zorder=5, family="monospace")
ax.text(bx, 0.75, "与 5.8 节实验 A 的 PyTorch 复核逐位一致", color=TXT, fontsize=11.5,
        alpha=0.8, ha="left", zorder=5)
fig.savefig(OUT + "attention-weights.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 3 figures rendered")
