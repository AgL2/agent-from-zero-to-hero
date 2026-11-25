"""第 1 章四张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系）
对应 Mermaid 源文件见 assets/diagrams/ch01/*.mmd（可编辑源，沙箱无法启动 Chromium 时用本脚本出图）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe

plt.rcParams["font.family"] = ["Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

NAVY, INDIGO = "#0d1b2a", "#1b263b"
PANEL = "#16324f"
CYAN, AMBER, CORAL, MINT, VIOLET = "#22d3ee", "#fbbf24", "#fb7185", "#34d399", "#a78bfa"
TXT = "#e8f1f8"
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch01/"


def new_canvas(w=16, h=6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    # 竖直渐变背景
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[0, 16, 0, 6], aspect="auto", origin="lower",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg", [NAVY, INDIGO]),
              zorder=0)
    ax.set_xlim(0, 16); ax.set_ylim(0, 6); ax.axis("off")
    rng = np.random.default_rng(3)
    ax.scatter(rng.uniform(0, 16, 90), rng.uniform(0, 6, 90),
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


def arrow(ax, p1, p2, color=CYAN, lw=2.2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


def node(ax, x, y, r, text, color, fs=12):
    ax.add_patch(Circle((x, y), r * 1.5, fc=color, ec="none", alpha=0.12, zorder=3))
    ax.add_patch(Circle((x, y), r, fc=PANEL, ec=color, lw=2, zorder=4))
    ax.text(x, y, text, ha="center", va="center", color=TXT, fontsize=fs, zorder=5)


# ============ 图 1：生物神经元 → 人工神经元 ============
fig, ax = new_canvas()
ax.text(4.2, 5.45, "生物神经元", color=MINT, fontsize=16, fontweight="bold", ha="center")
ax.text(11.8, 5.45, "人工神经元", color=AMBER, fontsize=16, fontweight="bold", ha="center")
box(ax, 1.2, 3.4, 2.6, 1.2, "树突 ×N\n接收信号", MINT)
box(ax, 4.6, 3.4, 3.0, 1.2, "细胞体\n累加·超阈值放电", MINT)
box(ax, 1.2, 1.2, 6.4, 1.2, "轴突：把电脉冲传给下一个神经元", MINT)
arrow(ax, (3.8, 4.0), (4.6, 4.0), MINT)
arrow(ax, (6.1, 3.4), (4.5, 2.4), MINT)
box(ax, 9.0, 3.4, 2.6, 1.2, "x₁, x₂, x₃ …\n输入特征", CYAN)
box(ax, 12.2, 3.4, 2.6, 1.2, "z = w·x + b\n加权求和+偏置", CYAN)
box(ax, 9.0, 1.2, 2.6, 1.2, "y = f(z)\n激活函数", CYAN)
box(ax, 12.2, 1.2, 2.6, 1.2, "y 输出", CYAN)
arrow(ax, (11.6, 4.0), (12.2, 4.0), CYAN)
arrow(ax, (13.5, 3.4), (10.2, 2.4), CYAN)
arrow(ax, (11.6, 1.8), (12.2, 1.8), CYAN)
arrow(ax, (8.0, 3.0), (9.0, 3.0), AMBER, lw=3.5, ls="--")
ax.text(8.5, 3.55, "抽象", color=AMBER, fontsize=13, fontweight="bold", ha="center")
fig.savefig(OUT + "neuron-mapping.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：MLP 结构（前向传播） ============
fig, ax = new_canvas()
ax.text(2.2, 5.45, "输入层", color=CYAN, fontsize=15, fontweight="bold", ha="center")
ax.text(8.0, 5.45, "隐藏层（非线性在这里弯折空间）", color=AMBER, fontsize=15, fontweight="bold", ha="center")
ax.text(13.8, 5.45, "输出层", color=CORAL, fontsize=15, fontweight="bold", ha="center")
xs_in = [2.2, 2.2]; ys_in = [4.2, 1.8]
xs_h = [8.0] * 3;   ys_h = [4.7, 3.0, 1.3]
node(ax, xs_in[0], ys_in[0], 0.5, "x₁", CYAN)
node(ax, xs_in[1], ys_in[1], 0.5, "x₂", CYAN)
hs = ["h₁", "h₂", "h₃"]
for x, y, t in zip(xs_h, ys_h, hs):
    node(ax, x, y, 0.5, t, AMBER)
node(ax, 13.8, 3.0, 0.55, "ŷ", CORAL)
for xi, yi in zip(xs_in, ys_in):
    for xh, yh in zip(xs_h, ys_h):
        arrow(ax, (xi + 0.5, yi), (xh - 0.5, yh), CYAN, lw=1.2)
for xh, yh in zip(xs_h, ys_h):
    arrow(ax, (xh + 0.5, yh), (13.8 - 0.55, 3.0), AMBER, lw=1.4)
ax.text(5.2, 0.45, "z₁ = W₁·x + b₁ → h = σ(z₁)", color=TXT, fontsize=13, alpha=0.9)
ax.text(10.6, 0.45, "z₂ = W₂·h + b₂ → ŷ = σ(z₂)", color=TXT, fontsize=13, alpha=0.9)
fig.savefig(OUT + "mlp-structure.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 3：XOR 线性不可分 ============
pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = [0, 1, 1, 0]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), dpi=110, facecolor=NAVY)
def draw_xor(ax, line=None, title=""):
    ax.set_facecolor(NAVY)
    for s in ax.spines.values():
        s.set_color("#3a4a63")
    ax.tick_params(colors=TXT, labelsize=11)
    ax.set_title(title, color=TXT, fontsize=15, pad=14, fontweight="bold")
    ax.grid(True, color="#ffffff", alpha=0.06, linewidth=0.8)
    for (x1, x2), lb in zip(pts, labels):
        c = AMBER if lb == 1 else CYAN
        m = "o" if lb == 1 else "s"
        ax.scatter(x1, x2, s=1500, c=c, marker=m, alpha=0.12, zorder=4)
        ax.scatter(x1, x2, s=560, c=c, marker=m, edgecolors="white", linewidths=1.5, zorder=5)
    if line is not None:
        xs = np.linspace(-0.3, 1.3, 100)
        ax.plot(xs, line(xs), color=CORAL, lw=2.5, ls="--", zorder=3)
    ax.set_xlim(-0.35, 1.35); ax.set_ylim(-0.35, 1.35)
    ax.set_xlabel("$x_1$", color=TXT, fontsize=13); ax.set_ylabel("$x_2$", color=TXT, fontsize=13)
draw_xor(axes[0], title="XOR：同类点对角分布")
draw_xor(axes[1], line=lambda x: 0.5 + 0 * x, title="横着切？总有点站错队")
draw_xor(axes[2], line=lambda x: -x + 0.9, title="斜着切？依然切不开")
fig.suptitle("单层感知机的致命局限：非线性可分问题", color=TXT, fontsize=17, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT + "xor-plot.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 4：学习率的三种命运 ============
rng = np.random.default_rng(7)
steps = np.arange(0, 500)
def curve(k, noise, floor):
    return 0.69 * np.exp(-k * steps) + floor + rng.normal(0, noise, len(steps))
slow = curve(0.004, 0.002, 0.28)
good = curve(0.03, 0.004, 0.005)
big = 0.69 + np.abs(np.sin(steps / 3.5)) * 0.55 + rng.normal(0, 0.01, len(steps))
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), dpi=110, facecolor=NAVY, sharey=True)
for ax, data, c, title, note in [
    (axes[0], slow, CYAN,  "η 太小：龟速爬行", "500 步才挪了一小半"),
    (axes[1], big,  CORAL, "η 太大：震荡发散", "在山谷两壁来回横跳"),
    (axes[2], good, MINT,  "η 合适：平滑下降", "快速且稳定地触底"),
]:
    ax.set_facecolor(NAVY)
    for s in ax.spines.values():
        s.set_color("#3a4a63")
    ax.tick_params(colors=TXT, labelsize=11)
    ax.set_title(title, color=TXT, fontsize=15, pad=14, fontweight="bold")
    ax.grid(True, color="#ffffff", alpha=0.06, linewidth=0.8)
    ax.plot(steps, data, color=c, lw=8, alpha=0.10)
    ax.plot(steps, data, color=c, lw=2.6, alpha=0.95)
    ax.annotate(note, xy=(0.5, 0.08), xycoords="axes fraction", ha="center", color=TXT, fontsize=12, alpha=0.85)
    ax.set_xlabel("训练步数", color=TXT, fontsize=12)
axes[0].set_ylabel("损失 loss", color=TXT, fontsize=12)
axes[0].set_ylim(0, 1.35)
fig.suptitle("同一个网络，三种命运：学习率是训练的第一旋钮", color=TXT, fontsize=17, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT + "lr-curves.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)
print("OK: 4 figures rendered")
