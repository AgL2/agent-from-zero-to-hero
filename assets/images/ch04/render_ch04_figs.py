"""第 4 章三张正式配图（与第 1 章视觉系统一致：深蓝渐变底、五色体系）
1. 文字进模型的完整流水线（替换开篇 ASCII 示意图）
2. Word2Vec 两种模式：CBOW vs Skip-gram（替换 4.2.2 ASCII 示意图）
3. BPE 五轮合并演化（替换 4.1.2 ASCII 手工推演，数值与正文一致）
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
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch04/"


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
            color=TXT, fontsize=fs, zorder=5, linespacing=1.5)


def arrow(ax, p1, p2, color=CYAN, lw=2.0, ls="-", ms=16):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


# ============ 图 1：从一段文字到 Transformer 的流水线 ============
fig, ax = new_canvas(16, 6.2)
steps = [
    ("一段文字", "“猫追狗”", CYAN),
    ("Tokenization", "切词 + 查词表", AMBER),
    ("Token id 序列", "[812, 47, 905]", MINT),
    ("词嵌入查表", "3×d 稠密矩阵", VIOLET),
    ("+ 位置编码", "注入“先后顺序”", CORAL),
    ("Transformer", "（第 5 章）", AMBER),
]
bw, bh, gap, x0, y0 = 2.25, 1.5, 0.3, 0.5, 3.4
for i, (t1, t2, c) in enumerate(steps):
    x = x0 + i * (bw + gap)
    box(ax, x, y0, bw, bh, f"{t1}\n{t2}", c, fs=11.5)
    if i > 0:
        arrow(ax, (x - gap + 0.03, y0 + bh / 2), (x - 0.03, y0 + bh / 2), AMBER, ms=13)
ax.text(8.0, 5.6, "一段文字在进入模型前的全部预处理：本章亲手实现前四步",
        color=TXT, fontsize=15, fontweight="bold", ha="center", zorder=5)
ax.text(8.0, 2.4, "Tokenizer 是文字与数字之间的翻译官：翻译质量决定模型对世界的“分辨率”，也决定你的账单",
        color=AMBER, fontsize=13, ha="center", zorder=5)
ax.text(8.0, 1.6, "token 是 LLM 世界的计量单位：API 按 token 计费，上下文窗口也以 token 为上限",
        color=TXT, fontsize=12.5, alpha=0.85, ha="center", zorder=5)
ax.text(8.0, 0.8, "词嵌入 = 一张 |V|×d 的可训练查找表，按 token id 取行；位置编码 = 给每个向量盖上“我在第几位”的邮戳",
        color=TXT, fontsize=12.5, alpha=0.85, ha="center", zorder=5)
fig.savefig(OUT + "pipeline.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：CBOW 与 Skip-gram ============
fig, ax = new_canvas(16, 6.0)
# 左：CBOW（上下文 → 中心词）
ctx = [("猫", 2.0), ("在", 3.6), ("睡觉", 5.2)]
for t, x in ctx:
    box(ax, x - 0.65, 4.3, 1.3, 0.95, t, CYAN, fs=13)
    arrow(ax, (x, 4.25), (3.9, 3.05), CYAN, lw=1.8)
box(ax, 2.95, 1.85, 1.9, 1.05, "中心词 ?", AMBER, fs=13)
ax.text(3.9, 5.75, "CBOW", color=CYAN, fontsize=16, fontweight="bold", ha="center", zorder=5)
ax.text(3.9, 1.15, "多选一：上下文 → 猜中心词", color=TXT, fontsize=12.5, ha="center", alpha=0.85, zorder=5)
# 右：Skip-gram（中心词 → 邻居）
box(ax, 9.55, 3.1, 1.9, 1.05, "中心词 猫", AMBER, fs=13)
nb = [("在", 12.6, 5.0), ("追", 13.7, 3.85), ("睡觉", 13.7, 2.6), ("狗", 12.6, 1.45)]
for t, x, y in nb:
    box(ax, x - 0.62, y - 0.45, 1.24, 0.9, t, MINT, fs=12.5)
    arrow(ax, (11.5, 3.62), (x - 0.66, y), MINT, lw=1.8)
ax.text(12.4, 5.75, "Skip-gram", color=MINT, fontsize=16, fontweight="bold", ha="center", zorder=5)
ax.text(12.4, 0.55, "1 跳预测多个邻居：中心词 → 上下文", color=TXT, fontsize=12.5, ha="center", alpha=0.85, zorder=5)
# 中缝
ax.text(8.0, 3.62, "VS", color=CORAL, fontsize=18, fontweight="bold", ha="center", va="center", zorder=5)
ax.text(0.7, 0.35, "同一个分布式假设的两种训练姿势：一个词的含义，由它的上下文决定",
        color=TXT, fontsize=12.5, ha="left", alpha=0.7, zorder=5)
fig.savefig(OUT + "word2vec-modes.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 3：BPE 的五轮合并演化 ============
fig, ax = new_canvas(16, 6.4)
rounds = [
    ("第 1 轮", "(l, o) 出现 3 次", "lo"),
    ("第 2 轮", "(lo, w) 出现 3 次", "low"),
    ("第 3 轮", "(e, s) 出现 3 次", "es"),
    ("第 4 轮", "(es, t) 出现 3 次", "est"),
    ("第 5 轮", "(est, </w>) 出现 3 次", "est</w>"),
]
y = 5.45
for i, (rd, pair, sym) in enumerate(rounds):
    c = [CYAN, AMBER, MINT, VIOLET, CORAL][i]
    box(ax, 1.0, y - 0.42, 1.7, 0.84, rd, c, fs=12)
    ax.text(4.6, y, pair, color=TXT, fontsize=13, ha="center", va="center", zorder=5)
    arrow(ax, (6.6, y), (7.9, y), c, lw=2.0)
    box(ax, 8.0, y - 0.42, 2.4, 0.84, f"合并为 “{sym}”", c, fs=12.5)
    y -= 0.95
ax.text(8.0, 6.1, "BPE 合并进化：微型语料 [low, lower, newest, widest, lowest]，每轮合并频率最高的相邻符号对",
        color=TXT, fontsize=14, fontweight="bold", ha="center", zorder=5)
ax.text(12.3, 4.7, "新积木由旧积木拼成：\n第 2 轮的 low = 第 1 轮的 lo + w\n第 5 轮的 est</w> = es + t + </w>",
        color=AMBER, fontsize=11.5, ha="left", va="top", zorder=5, linespacing=1.6)
ax.text(1.0, 0.85, "结果：low → [low, </w>]；lowest → [low, est</w>]", color=MINT,
        fontsize=13.5, ha="left", zorder=5)
ax.text(1.0, 0.25, "lowest 自动共享了 low（前缀）和 est</w>（后缀）两块积木——形态学结构被频率统计自然挖了出来",
        color=TXT, fontsize=12, ha="left", alpha=0.85, zorder=5)
fig.savefig(OUT + "bpe-merges.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 3 figures rendered")
