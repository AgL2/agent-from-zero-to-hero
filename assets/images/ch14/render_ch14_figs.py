"""第 14 章三张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系，对齐 ch01 模板）
图 1 memory-taxonomy.png      记忆分类体系（14.2）
图 2 memgpt-os.png            MemGPT 的操作系统类比（14.5）
图 3 memory-architecture.png  记忆系统三层结构（本章小结收束图）
"""
import os
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
OUT = os.path.dirname(os.path.abspath(__file__)) + "/"


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


def label(ax, x, y, text, color=TXT, fs=12, ha="center", bold=False, alpha=1.0):
    ax.text(x, y, text, ha=ha, va="center", color=color, fontsize=fs,
            fontweight="bold" if bold else "normal", alpha=alpha, zorder=5, linespacing=1.5)


# ============ 图 1：记忆分类体系（14.2） ============
fig, ax = new_canvas(16, 6.4)
label(ax, 3.9, 5.95, "按时间轴切（存多久）", MINT, 16, bold=True)
label(ax, 12.0, 5.95, "按内容性质切（存什么）", AMBER, 16, bold=True)

box(ax, 0.8, 3.4, 6.2, 1.45, "短期记忆：上下文窗口内\n（几轮对话，会话结束即消失）", MINT, 13)
box(ax, 0.8, 1.25, 6.2, 1.45, "长期记忆：外部存储\n（跨会话，需要显式存取）", VIOLET, 13)
label(ax, 3.9, 0.5, "四个象限互相独立：先选存多久，再选存什么", TXT, 12, alpha=0.85)

box(ax, 8.6, 4.4, 6.8, 1.1, "声明性记忆（能说出来的）", CYAN, 13)
box(ax, 8.5, 2.7, 3.3, 1.1, "语义记忆：事实\n“用户住杭州”", CYAN, 12)
box(ax, 12.1, 2.7, 3.3, 1.1, "情节记忆：事件\n“上周二抱怨过退款慢”", CYAN, 12)
box(ax, 8.6, 1.25, 6.8, 1.1, "程序性记忆（会做但不假思索的）", AMBER, 13)
label(ax, 12.0, 0.5, "技能 / SOP —— “处理退款工单的固定流程”，第 15 章的 Skill 就是它的工程化身", AMBER, 11.5)
arrow(ax, (10.9, 4.4), (10.2, 3.8), CYAN, 1.4)
arrow(ax, (13.1, 4.4), (13.7, 3.8), CYAN, 1.4)
arrow(ax, (12.0, 1.25), (12.0, 0.82), AMBER, 1.4)
fig.savefig(OUT + "memory-taxonomy.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：MemGPT = 给 LLM 装上“操作系统”（14.5） ============
fig, ax = new_canvas(16, 6.8)
label(ax, 8.0, 6.4, "MemGPT = 给 LLM 装上“操作系统”", TXT, 17, bold=True)

# RAM 区
ax.add_patch(FancyBboxPatch((1.2, 3.7), 13.6, 2.2, boxstyle="round,pad=0.08",
                            fc=INDIGO, ec=CYAN, lw=2.4, zorder=3))
label(ax, 3.4, 5.55, "主上下文 = RAM（几万 token，贵）", CYAN, 14, bold=True)
box(ax, 1.8, 4.0, 4.6, 1.2, "系统指令区（System）\n—— 永不被换出", MINT, 12.5)
box(ax, 7.0, 4.0, 7.2, 1.2, "消息队列区（最近对话）\nFIFO 滚动，超限触发换页", AMBER, 12.5)

# 磁盘区
ax.add_patch(FancyBboxPatch((1.2, 0.6), 13.6, 1.9, boxstyle="round,pad=0.08",
                            fc=INDIGO, ec=VIOLET, lw=2.4, zorder=3))
label(ax, 3.3, 2.15, "外部存储 = 磁盘（无限，便宜）", VIOLET, 14, bold=True)
box(ax, 7.4, 0.95, 7.0, 0.95, "历史对话库 / 事实库 / 检索区", VIOLET, 12.5)

# 换页双向箭头
arrow(ax, (9.2, 3.65), (9.2, 2.6), CORAL, 2.6)                       # swap out
arrow(ax, (11.8, 2.6), (11.8, 3.65), MINT, 2.6)                      # swap in
label(ax, 6.2, 3.1, "page fault：读写工具 = “换页”系统调用\nswap out 归档旧消息 / swap in 检索久远事实",
      CORAL, 11.5)
label(ax, 8.0, 0.22, "OS 的角色由 LLM 自己扮演：何时换页，靠 System Prompt 里的“内存管理守则”裁量",
      TXT, 12, alpha=0.85)
fig.savefig(OUT + "memgpt-os.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 3：记忆系统三层结构（本章小结收束图） ============
fig, ax = new_canvas(16, 7.0)

# 第一层：上下文窗口（RAM）
ax.add_patch(FancyBboxPatch((1.0, 4.5), 14.0, 2.15, boxstyle="round,pad=0.08",
                            fc=INDIGO, ec=CYAN, lw=2.4, zorder=3))
label(ax, 2.6, 6.2, "上下文窗口（RAM）", CYAN, 14.5, bold=True)
box(ax, 1.6, 4.75, 4.4, 1.1, "系统指令\n（常驻，永不被换出）", MINT, 11.5)
box(ax, 6.4, 4.75, 4.2, 1.1, "滚动摘要区\n远期历史的压缩表示 ← 14.3", AMBER, 11.5)
box(ax, 11.0, 4.75, 3.6, 1.1, "最近 K 轮原文\n保真的工作区 ← 14.3", AMBER, 11.5)

# 第二层：长期记忆库（磁盘）
ax.add_patch(FancyBboxPatch((1.0, 2.4), 14.0, 1.75, boxstyle="round,pad=0.08",
                            fc=INDIGO, ec=VIOLET, lw=2.4, zorder=3))
label(ax, 2.6, 3.75, "长期记忆库（磁盘）", VIOLET, 14.5, bold=True)
box(ax, 5.0, 2.65, 5.2, 1.0, "语义记忆：用户画像短句，存-取-更新 ← 14.4", VIOLET, 11.5)
box(ax, 10.6, 2.65, 4.0, 1.0, "情节记忆：带时间戳的事件片段 ← 14.2", VIOLET, 11.5)

# 第三层：管理机制（OS）
ax.add_patch(FancyBboxPatch((1.0, 0.3), 14.0, 1.75, boxstyle="round,pad=0.08",
                            fc=INDIGO, ec=CORAL, lw=2.4, zorder=3))
label(ax, 2.6, 1.65, "管理机制（OS）", CORAL, 14.5, bold=True)
box(ax, 5.0, 0.55, 4.0, 1.0, "写入策略：何时记/记什么 ← 14.7", CORAL, 11)
box(ax, 9.3, 0.55, 3.4, 1.0, "遗忘：TTL × 衰减 × 访问强化 ← 14.7", CORAL, 10.5)
box(ax, 13.0, 0.55, 1.7, 1.0, "换页\nMemGPT ← 14.5", CORAL, 9.5)

arrow(ax, (8.0, 4.5), (8.0, 4.15), CYAN, 2.0, style="<|-|>")
arrow(ax, (8.0, 2.4), (8.0, 2.05), VIOLET, 2.0, style="<|-|>")
fig.savefig(OUT + "memory-architecture.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 3 figures rendered ->", OUT)
