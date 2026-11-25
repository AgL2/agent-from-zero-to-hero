"""第 6 章两张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系，风格对齐 ch01 模板）"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["font.family"] = ["Hiragino Sans GB", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

NAVY, INDIGO = "#0d1b2a", "#1b263b"
PANEL = "#16324f"
CYAN, AMBER, CORAL, MINT, VIOLET = "#22d3ee", "#fbbf24", "#fb7185", "#34d399", "#a78bfa"
TXT = "#e8f1f8"
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch06/"


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


def arrow(ax, p1, p2, color=CYAN, lw=2.2, style="-|>", ls="-",
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                                 color=color, lw=lw, linestyle=ls, zorder=3,
                                 connectionstyle=connectionstyle))


def node(ax, x, y, r, text, color, fs=12):
    ax.add_patch(Circle((x, y), r * 1.5, fc=color, ec="none", alpha=0.12, zorder=3))
    ax.add_patch(Circle((x, y), r, fc=PANEL, ec=color, lw=2, zorder=4))
    ax.text(x, y, text, ha="center", va="center", color=TXT, fontsize=fs, zorder=5)


# ============ 图 1：三段旅程 预训练 → SFT → 对齐 ============
fig, ax = new_canvas(16, 9.4)
ax.text(8.0, 9.0, "一个模型的诞生：预训练 → SFT → 对齐",
        color=TXT, fontsize=18, fontweight="bold", ha="center")

box(ax, 4.6, 7.9, 4.6, 0.85, "随机初始化的 Transformer", VIOLET, fs=14)

stages = [
    (5.6, CYAN,  "① 预训练（Pre-training）\nBase Model：有知识，但不懂“对话”",
     "数据：万亿 token 的网页 / 书籍 / 代码\n目标：预测下一个 token（监督信号免费）\n成本：数千张 GPU × 数月，上亿元"),
    (3.3, AMBER, "② 指令微调（SFT）\n会听指令、会聊天，但不知道哪个回答更“好”",
     "数据：数万 ~ 百万条“指令-回答”对\n目标：学会“按指令办事”的行为格式\n成本：几张 ~ 几十张 GPU × 几天"),
    (1.0, CORAL, "③ 对齐（RLHF / DPO）\n有用、诚实、无害（Aligned Model）",
     "数据：人类标注“回答 A 优于回答 B”\n目标：让偏好对齐人类价值观\n成本：RLHF 昂贵，DPO 只需一次微调"),
]
for y, color, main, note in stages:
    box(ax, 1.6, y, 6.2, 1.5, main, color, fs=13.5)
    ax.text(8.55, y + 0.75, note, ha="left", va="center", color=TXT,
            fontsize=12, alpha=0.92, linespacing=1.8, zorder=5)
# 自上而下的流程箭头：起点 → ① → ② → ③
arrow(ax, (4.7, 7.9), (4.7, 7.18), VIOLET, lw=2.6)
arrow(ax, (4.7, 5.6), (4.7, 4.88), CYAN, lw=2.6)
arrow(ax, (4.7, 3.3), (4.7, 2.58), AMBER, lw=2.6)
fig.savefig(OUT + "three-stages.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：RLHF 三步管线 ============
fig, ax = new_canvas(16, 6.2)
ax.text(8.0, 5.75, "RLHF：让人类偏好进入训练信号",
        color=TXT, fontsize=18, fontweight="bold", ha="center")

steps = [
    (0.6, CYAN, "第 1 步：SFT", "用指令数据微调\n得到初始策略 π\n\n产出：会回答的起点模型"),
    (5.9, AMBER, "第 2 步：训练奖励模型 RM", "同一 prompt 采样多个回答\n人工两两排序 (y_w, y_l)\n用 Bradley-Terry 损失训练 RM\n产出：人类裁判的打分函数"),
    (11.2, CORAL, "第 3 步：PPO 强化学习", "把 LLM 当策略网络：\n生成回答 → RM 打分 → PPO 更新 π\nKL 散度拴住它别偏离太远\n产出：对齐后的模型"),
]
for x, color, title, body in steps:
    box(ax, x, 1.55, 4.2, 3.3, body, color, fs=12.5)
    ax.text(x + 2.1, 5.05, title, ha="center", color=color,
            fontsize=15, fontweight="bold", zorder=5)
arrow(ax, (4.85, 3.2), (5.85, 3.2), AMBER, lw=3)
arrow(ax, (10.15, 3.2), (11.15, 3.2), CORAL, lw=3)
ax.text(8.0, 0.75, "偏好数据（“这个回答比那个好”）是第 2、3 步共同的燃料——它把“人类觉得好”翻译成了可优化的数字",
        ha="center", color=TXT, fontsize=12.5, alpha=0.9)
fig.savefig(OUT + "rlhf-pipeline.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 2 figures rendered")
