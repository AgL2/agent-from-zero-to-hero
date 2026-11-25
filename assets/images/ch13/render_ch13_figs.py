"""第 13 章两张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系）
图 1：RAG 全链路（索引期 + 查询期，两期共用同一个 embedding"翻译官"）
图 2：双编码器 vs 交叉编码器（检索用便宜，重排用准）
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
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch13/"


def new_canvas(w=16, h=6):
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


def box(ax, x, y, w, h, text, color, fs=12, glow=True):
    if glow:
        ax.add_patch(FancyBboxPatch((x - 0.06, y - 0.06), w + 0.12, h + 0.12,
                                    boxstyle="round,pad=0.06", fc=color, ec="none", alpha=0.15, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=PANEL, ec=color, lw=2, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=TXT, fontsize=fs, zorder=5, linespacing=1.5)


def arrow(ax, p1, p2, color=CYAN, lw=2.2, ls="-", ms=15):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


# ============ 图 1：RAG 全链路 ============
fig, ax = new_canvas(16, 8.6)
ax.text(8.0, 8.2, "RAG 全链路：索引期（离线）与查询期（在线）", color=TXT,
        fontsize=16, fontweight="bold", ha="center")
# ---- 索引期 ----
ax.text(0.5, 7.55, "索引期（离线，资料入库时做一次）", color=CYAN, fontsize=13,
        fontweight="bold", ha="left")
idx_nodes = [
    (0.5, "文档集合\n（PDF/网页/…）", "段落→小块"),
    (4.55, "切片\nChunking", "每块→向量"),
    (8.6, "Embedding", "亿级向量的增删查"),
    (12.65, "向量数据库", None),
]
for x, name, sub in idx_nodes:
    box(ax, x, 5.85, 2.9, 1.15, name, CYAN, fs=12.5)
    if sub:
        ax.text(x + 1.45, 5.5, sub, color=TXT, fontsize=11, ha="center", alpha=0.85)
for i in range(len(idx_nodes) - 1):
    arrow(ax, (idx_nodes[i][0] + 2.9, 6.42), (idx_nodes[i + 1][0], 6.42), CYAN, lw=2.0)
# ---- 查询期 ----
ax.text(7.75, 4.65, "查询期（在线，用户每次提问都走一遍）", color=AMBER, fontsize=13,
        fontweight="bold", ha="center")
q_nodes = [
    (0.5, "用户问题\n「报销上限？」", None),
    (2.92, "Embedding", "查询向量"),
    (5.34, "检索 top-k", "余弦相似度"),
    (7.76, "Rerank 重排\n（可选）", "精排出前 3"),
    (10.18, "拼上下文", "问题+资料塞进 Prompt"),
    (12.60, "LLM 生成", "「根据资料，报销上限是…」"),
]
for x, name, sub in q_nodes:
    box(ax, x, 2.35, 2.2, 1.15, name, AMBER, fs=11.5)
    if sub:
        ax.text(x + 1.1, 2.0, sub, color=TXT, fontsize=10, ha="center", alpha=0.85)
for i in range(len(q_nodes) - 1):
    arrow(ax, (q_nodes[i][0] + 2.2, 2.92), (q_nodes[i + 1][0], 2.92), AMBER, lw=2.0)
# 共用同一个 embedding"翻译官"：连接两期
ax.add_patch(FancyArrowPatch((4.02, 5.85), (4.02, 3.5), arrowstyle="-", color=MINT,
                             lw=2.0, linestyle=(0, (6, 5)), zorder=3))
ax.add_patch(FancyArrowPatch((11.12, 5.85), (11.12, 3.5), arrowstyle="-", color=MINT,
                             lw=2.0, linestyle=(0, (6, 5)), zorder=3))
ax.text(8.0, 0.9, "两期必须共用同一个 embedding「翻译官」，否则查询向量与文档向量不在同一坐标系",
        color=MINT, fontsize=11.5, ha="center", va="center")
fig.savefig(OUT + "rag-pipeline.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：双编码器 vs 交叉编码器 ============
fig, ax = new_canvas(16, 6.5)
ax.text(4.3, 6.05, "双编码器 Bi-encoder\n（检索用，便宜）", color=CYAN, fontsize=14.5,
        fontweight="bold", ha="center", linespacing=1.4)
ax.text(11.9, 6.05, "交叉编码器 Cross-encoder\n（重排用，准但贵）", color=CORAL, fontsize=14.5,
        fontweight="bold", ha="center", linespacing=1.4)
# 左：Bi-encoder
box(ax, 0.8, 4.3, 2.0, 1.0, "query", CYAN, fs=13, glow=False)
box(ax, 0.8, 1.6, 2.0, 1.0, "doc₁ … docₙ", CYAN, fs=13, glow=False)
box(ax, 3.9, 4.3, 2.3, 1.0, "编码器\n→ 向量 q", CYAN, fs=12)
box(ax, 3.9, 1.6, 2.3, 1.0, "编码器\n→ 向量 dᵢ", CYAN, fs=12)
box(ax, 7.0, 2.95, 2.2, 1.0, "余弦相似度", CYAN, fs=12.5)
arrow(ax, (2.8, 4.8), (3.9, 4.8), CYAN, lw=1.8)
arrow(ax, (2.8, 2.1), (3.9, 2.1), CYAN, lw=1.8)
arrow(ax, (6.2, 4.8), (7.5, 3.95), CYAN, lw=1.8)
arrow(ax, (6.2, 2.1), (7.5, 2.95), CYAN, lw=1.8)
ax.text(5.1, 0.85, "文档向量离线可算，查询时只算一次点积，\n百万文档毫秒级——但两段文本从未「见过面」",
        color=TXT, fontsize=11, ha="center", va="center", linespacing=1.5, alpha=0.9)
# 右：Cross-encoder
box(ax, 9.6, 4.3, 2.0, 1.0, "query", CORAL, fs=13, glow=False)
box(ax, 9.6, 1.6, 2.0, 1.0, "doc", CORAL, fs=13, glow=False)
box(ax, 12.4, 2.95, 2.6, 1.0, "Transformer\n（拼接后一起进）", CORAL, fs=11.5)
arrow(ax, (11.6, 4.8), (12.4, 3.95), CORAL, lw=1.8)
arrow(ax, (11.6, 2.1), (12.4, 2.95), CORAL, lw=1.8)
box(ax, 14.3, 4.3, 1.4, 1.0, "单个\n相关分", CORAL, fs=11.5, glow=False)
arrow(ax, (15.0, 3.95), (15.0, 4.4), CORAL, lw=1.8)
ax.text(12.3, 0.85, "每对 (query, doc) 都要逐对在线算一次，\n只适合精排 50 个候选——注意力在两段间自由流动",
        color=TXT, fontsize=11, ha="center", va="center", linespacing=1.5, alpha=0.9)
ax.text(8.35, 3.45, "vs", color=TXT, fontsize=17, ha="center", va="center", alpha=0.75, fontweight="bold")
fig.savefig(OUT + "bi-vs-cross-encoder.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)
print("OK: ch13 2 figures rendered")
