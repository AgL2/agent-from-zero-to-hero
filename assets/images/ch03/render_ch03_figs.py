"""第 3 章三张正式配图（与第 1 章视觉系统一致：深蓝渐变底、五色体系）
1. 链式法则计算图（替换正文 ASCII 示意图）
2. 2-2-1 最小网络的前向/反向全景（数值与 3.3 节手算完全一致）
3. 逐层梯度范数实测（数值与 3.9 节真实运行输出完全一致）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["font.family"] = ["Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"  # 对数刻度的 10^-6 等数学符号用 DejaVu，避免缺字形

NAVY, INDIGO = "#0d1b2a", "#1b263b"
PANEL = "#16324f"
CYAN, AMBER, CORAL, MINT, VIOLET = "#22d3ee", "#fbbf24", "#fb7185", "#34d399", "#a78bfa"
TXT = "#e8f1f8"
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch03/"


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


def box(ax, x, y, w, h, text, color, fs=12, glow=True):
    if glow:
        ax.add_patch(FancyBboxPatch((x - 0.05, y - 0.05), w + 0.1, h + 0.1,
                                    boxstyle="round,pad=0.05", fc=color, ec="none", alpha=0.15, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=PANEL, ec=color, lw=2, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=TXT, fontsize=fs, zorder=5, linespacing=1.4)


def circle_node(ax, x, y, r, text, color, fs=12):
    ax.add_patch(Circle((x, y), r * 1.4, fc=color, ec="none", alpha=0.13, zorder=3))
    ax.add_patch(Circle((x, y), r, fc=PANEL, ec=color, lw=2, zorder=4))
    ax.text(x, y, text, ha="center", va="center", color=TXT, fontsize=fs, zorder=5)


def arrow(ax, p1, p2, color=CYAN, lw=2.0, ls="-", style="-|>", ms=16):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


# ============ 图 1：链式法则的计算图（前向青色、反向珊瑚色） ============
fig, ax = new_canvas(16, 6.0)
# 前向主链
circle_node(ax, 1.3, 3.6, 0.42, "x", CYAN)
circle_node(ax, 1.3, 1.9, 0.42, "w", AMBER)
box(ax, 2.9, 2.9, 1.5, 0.9, "乘", AMBER, fs=13)
circle_node(ax, 5.5, 3.35, 0.42, "z", CYAN)
box(ax, 2.9, 0.7, 1.5, 0.9, "加 b", AMBER, fs=13)
box(ax, 7.2, 2.9, 1.5, 0.9, "σ", VIOLET, fs=13)
circle_node(ax, 9.9, 3.35, 0.42, "a", VIOLET)
box(ax, 11.4, 2.9, 2.0, 0.9, "求差平方", CORAL, fs=12)
circle_node(ax, 14.8, 3.35, 0.45, "L", CORAL)
circle_node(ax, 12.4, 1.2, 0.42, "y", MINT)
# 前向箭头
arrow(ax, (1.6, 3.75), (2.85, 3.5), CYAN)
arrow(ax, (1.6, 2.05), (2.85, 3.0), AMBER)
arrow(ax, (4.4, 3.35), (5.05, 3.35), CYAN)
arrow(ax, (4.4, 1.15), (5.5, 2.9), AMBER)
ax.text(4.75, 1.75, "b", color=TXT, fontsize=11, ha="center", zorder=5)
arrow(ax, (5.95, 3.35), (7.15, 3.35), CYAN)
arrow(ax, (8.7, 3.35), (9.45, 3.35), VIOLET)
arrow(ax, (10.35, 3.35), (11.35, 3.35), CORAL)
arrow(ax, (12.75, 1.35), (13.4, 2.85), MINT)
arrow(ax, (13.45, 3.35), (14.3, 3.35), CORAL)
# 反向链（珊瑚色虚线，自右向左，与对应节点区段对齐）
arrow(ax, (14.4, 4.35), (10.5, 4.35), CORAL, ls="--", lw=2.4)
ax.text(12.45, 4.72, "∂L/∂a = a - y", color=CORAL, fontsize=11.5, ha="center", zorder=5)
arrow(ax, (9.6, 4.35), (5.8, 4.35), CORAL, ls="--", lw=2.4)
ax.text(7.7, 4.72, "× σ′(z)", color=CORAL, fontsize=11.5, ha="center", zorder=5)
arrow(ax, (5.2, 4.35), (2.0, 4.35), CORAL, ls="--", lw=2.4)
ax.text(3.6, 4.72, "× x（回到 w）", color=CORAL, fontsize=11.5, ha="center", zorder=5)
# 说明
ax.text(0.7, 5.65, "前向传播（青色）：从输入到损失，缓存每个中间量",
        color=CYAN, fontsize=13.5, fontweight="bold", ha="left", zorder=5)
ax.text(0.7, 5.15, "反向传播（珊瑚色）：从损失倒着走，把沿途局部导数连乘",
        color=CORAL, fontsize=13.5, fontweight="bold", ha="left", zorder=5)
ax.text(0.7, 0.35, "∂L/∂w = ∂L/∂a · σ′(z) · x  ——三个局部导数相乘，就是链式法则的全部内容",
        color=TXT, fontsize=13, ha="left", zorder=5)
fig.savefig(OUT + "comp-graph.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：2-2-1 最小网络（数值与 3.3 节手算一致） ============
fig, ax = new_canvas(16, 6.6)
# 节点：输入 / 隐层 / 输出
circle_node(ax, 2.0, 4.3, 0.5, "x₁\n1.0", CYAN, fs=11)
circle_node(ax, 2.0, 1.9, 0.5, "x₂\n0.5", CYAN, fs=11)
circle_node(ax, 8.0, 4.6, 0.62, "a₁\n0.5866", AMBER, fs=10.5)
circle_node(ax, 8.0, 1.6, 0.62, "a₂\n0.5744", AMBER, fs=10.5)
circle_node(ax, 13.6, 3.1, 0.62, "ŷ\n0.6506", CORAL, fs=10.5)
ax.text(2.0, 5.75, "输入层", color=CYAN, fontsize=14, fontweight="bold", ha="center", zorder=5)
ax.text(8.0, 5.75, "隐层（sigmoid）", color=AMBER, fontsize=14, fontweight="bold", ha="center", zorder=5)
ax.text(13.6, 5.75, "输出层", color=CORAL, fontsize=14, fontweight="bold", ha="center", zorder=5)
# 前向箭头 + 权重标签
arrow(ax, (2.55, 4.35), (7.3, 4.55), CYAN, lw=1.8)
ax.text(4.9, 4.75, "W=0.15", color=TXT, fontsize=10.5, ha="center", zorder=5)
arrow(ax, (2.55, 2.2), (7.3, 4.3), CYAN, lw=1.8)
ax.text(6.4, 3.5, "W=0.20", color=TXT, fontsize=10.5, ha="center", zorder=5)
arrow(ax, (2.55, 4.0), (7.3, 1.9), CYAN, lw=1.8)
ax.text(3.5, 3.15, "W=0.25", color=TXT, fontsize=10.5, ha="center", zorder=5)
arrow(ax, (2.55, 1.85), (7.3, 1.65), CYAN, lw=1.8)
ax.text(4.9, 1.3, "W=0.30", color=TXT, fontsize=10.5, ha="center", zorder=5)
arrow(ax, (8.65, 4.4), (12.95, 3.3), AMBER, lw=2.0)
ax.text(10.8, 4.35, "W₂,₁=0.4", color=TXT, fontsize=10.5, ha="center", zorder=5)
arrow(ax, (8.65, 1.8), (12.95, 2.9), AMBER, lw=2.0)
ax.text(10.8, 1.85, "W₂,₂=0.5", color=TXT, fontsize=10.5, ha="center", zorder=5)
ax.text(8.0, 0.55, "h₁=0.35, h₂=0.30；z=0.6218", color=TXT, fontsize=11, ha="center",
        alpha=0.85, zorder=5)
# 反向（珊瑚色虚线）
arrow(ax, (12.9, 2.75), (8.7, 3.9), CORAL, ls="--", lw=2.0)
ax.text(10.9, 3.55, "δ_out = -0.0794", color=CORAL, fontsize=11, ha="center", zorder=5)
arrow(ax, (7.3, 4.35), (2.6, 4.15), CORAL, ls="--", lw=1.6)
ax.text(4.9, 3.95, "δ_h₁ = -0.0077", color=CORAL, fontsize=10.5, ha="center", zorder=5)
arrow(ax, (7.3, 1.85), (2.6, 2.05), CORAL, ls="--", lw=1.6)
ax.text(4.9, 2.15, "δ_h₂ = -0.0097", color=CORAL, fontsize=10.5, ha="center", zorder=5)
# 说明
ax.text(13.6, 1.45, "L = ½(1 - 0.6506)²\n≈ 0.0610", color=CORAL, fontsize=11,
        ha="center", va="top", zorder=5, linespacing=1.5)
ax.text(0.7, 6.05, "一遍前向（青→金→珊瑚）+ 一遍反向（珊瑚虚线），7 个参数的梯度全部到手",
        color=TXT, fontsize=13, ha="left", zorder=5)
fig.savefig(OUT + "net-221.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 3：逐层梯度范数实测（数值与 3.9 节输出一致） ============
sigmoid_norms = [2.0987e-06, 2.8800e-05, 1.0351e-04, 4.1038e-04,
                 2.2291e-03, 9.0173e-03, 3.5715e-02, 1.3297e-01]
relu_norms = [7.5610e-01, 1.2184e+00, 1.8101e+00, 2.2052e+00,
              3.1289e+00, 2.4933e+00, 1.8220e+00, 1.7611e+00]
layers = np.arange(1, 9)
fig, ax = plt.subplots(figsize=(16, 6.2), dpi=110, facecolor=NAVY)
grad = np.linspace(0, 1, 256).reshape(-1, 1)
ax.imshow(grad, extent=[0.4, 8.85, 0, 6.2], aspect="auto", origin="lower",
          cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg", [NAVY, INDIGO]),
          zorder=0)
ax.set_facecolor(NAVY)
bw = 0.38
ax.bar(layers - bw / 2, sigmoid_norms, width=bw, color=CORAL, label="sigmoid 网络（逐层缩水）", zorder=4)
ax.bar(layers + bw / 2, relu_norms, width=bw, color=MINT, label="ReLU 网络（同一量级）", zorder=4)
ax.set_yscale("log")
ax.set_ylim(1e-7, 10)
yticks = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
ylabels = ["$\\mathsf{10^{-7}}$", "$\\mathsf{10^{-6}}$", "$\\mathsf{10^{-5}}$", "$\\mathsf{10^{-4}}$",
           "$\\mathsf{10^{-3}}$", "$\\mathsf{10^{-2}}$", "$\\mathsf{10^{-1}}$", "1", "10"]
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
ax.set_xticks(layers)
ax.set_xticklabels([f"第{i}层" for i in layers], color=TXT, fontsize=12)
ax.tick_params(colors=TXT, labelsize=11, axis="y")
for s in ax.spines.values():
    s.set_color("#3a4a63")
ax.grid(True, axis="y", color="#ffffff", alpha=0.08, linewidth=0.8, zorder=2)
ax.set_ylabel("权重梯度 L2 范数（对数刻度）", color=TXT, fontsize=13)
ax.legend(loc="upper left", fontsize=13, facecolor=PANEL, edgecolor="#3a4a63",
          labelcolor=TXT, framealpha=0.9)
ax.annotate("sigmoid：首尾相差 63359 倍，第 1 层梯度仅 ≈ 2.1×10⁻⁶", xy=(1 - bw / 2, 2.0987e-06),
            xytext=(3.3, 5.3), color=CORAL, fontsize=12.5, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=1.8), zorder=6)
ax.annotate("首尾仅差 2.3 倍", xy=(1 + bw / 2, 7.5610e-01), xytext=(2.4, 3.2),
            color=MINT, fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=MINT, lw=1.8), zorder=6)
ax.set_title("同一个 8 层网络、同一份数据，只换激活函数：梯度消失肉眼可见（3.9 节实测）",
             color=TXT, fontsize=16, fontweight="bold", pad=14, zorder=6)
fig.tight_layout()
fig.savefig(OUT + "grad-norms.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 3 figures rendered")
