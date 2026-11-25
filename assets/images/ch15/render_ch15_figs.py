"""第 15 章两张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系，对齐 ch01 模板）
图 1 skill-anatomy.png            Skill 的目录结构解剖（15.2）
图 2 progressive-disclosure.png   渐进式披露三层加载（15.3）
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


# ============ 图 1：Skill 的目录结构解剖（15.2） ============
fig, ax = new_canvas(16, 6.6)
label(ax, 8.0, 6.2, "目录即能力：一个 Skill 的标准解剖（以 weather 为例）", TXT, 17, bold=True)

# 外层目录 skills/
ax.add_patch(FancyBboxPatch((0.9, 0.55), 9.2, 5.0, boxstyle="round,pad=0.08",
                            fc=INDIGO, ec=TXT, lw=1.6, zorder=3, alpha=0.9))
label(ax, 1.7, 5.15, "skills/", TXT, 14, bold=True, ha="left")

# 内层目录 weather/
ax.add_patch(FancyBboxPatch((1.4, 0.95), 8.2, 3.8, boxstyle="round,pad=0.08",
                            fc=INDIGO, ec=CYAN, lw=2.2, zorder=3))
label(ax, 2.2, 4.4, "weather/   # 一个 Skill = 一个目录", CYAN, 13.5, bold=True, ha="left")

box(ax, 2.0, 2.9, 7.0, 1.15, "SKILL.md\n① 元数据（name / description 封面）+ ② 指令文档（工作流与约束）", AMBER, 11.5)
box(ax, 2.0, 1.75, 3.35, 0.85, "weather.py\n③ 脚本：确定性代码", MINT, 11.5)
box(ax, 5.65, 1.75, 3.35, 0.85, "reference.md\n④ 资源：大参考（可选）", VIOLET, 11.5)

# 右侧注解
label(ax, 10.6, 4.6, "封面：Agent 靠它判断\n“这个 Skill 管什么用”", AMBER, 11.5, ha="left")
arrow(ax, (9.1, 3.45), (10.5, 4.35), AMBER, 1.4)
label(ax, 10.6, 3.3, "知识：何时做、按什么步骤、\n踩过哪些坑（给 LLM 看的程序）", AMBER, 11.5, ha="left")
arrow(ax, (9.1, 3.2), (10.5, 3.2), AMBER, 1.4)
label(ax, 10.6, 2.1, "手：精确计算/调用交给代码，\n不让 LLM“口算”", MINT, 11.5, ha="left")
arrow(ax, (5.5, 2.15), (10.5, 2.1), MINT, 1.2)
label(ax, 10.6, 1.15, "书架深处的资料：用到才读，\n避免撑爆上下文", VIOLET, 11.5, ha="left")
arrow(ax, (9.1, 1.85), (10.5, 1.35), VIOLET, 1.2)

label(ax, 5.5, 0.28, "指令是“知识”不是代码——这是 Skill 与 Plugin 最本质的区别", CORAL, 12.5, bold=True)
fig.savefig(OUT + "skill-anatomy.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：渐进式披露三层加载（15.3） ============
fig, ax = new_canvas(16, 7.0)
label(ax, 8.0, 6.6, "渐进式披露：上下文是稀缺资源，按需分三层加载", TXT, 17, bold=True)

# 第一层：常驻层
box(ax, 3.2, 4.9, 9.6, 1.15, "常驻层（每次对话都在上下文里）   约 50 token / skill\nname + description（封面）", CYAN, 12.5)
label(ax, 13.3, 5.5, "启动时 scan()\n只读封面", CYAN, 11.5, ha="left")

# 第二层：按需层
box(ax, 3.2, 3.0, 9.6, 1.15, "按需层（命中后才加载）   数千 token\nSKILL.md 正文（指令工作流）", AMBER, 12.5)
label(ax, 13.3, 3.6, "match 命中后\nload()", AMBER, 11.5, ha="left")

# 第三层：惰性层
box(ax, 3.2, 1.1, 9.6, 1.15, "惰性层（指令引用时才读）   任意大\nreference.md / 数据文件 / 脚本", VIOLET, 12.5)
label(ax, 13.3, 1.7, "指令说“详见 xxx”\n才读", VIOLET, 11.5, ha="left")

arrow(ax, (8.0, 4.9), (8.0, 4.15), AMBER, 2.2)
arrow(ax, (8.0, 3.0), (8.0, 2.25), VIOLET, 2.2)

label(ax, 4.2, 0.35, "n = 100 个 Skill 时，常驻成本从 205,000 token 降到 5,000 token（-97.6%）",
      CORAL, 13, bold=True, ha="left")
label(ax, 1.2, 5.5, "越往下\n越便宜\n越少发生", TXT, 11, ha="center", alpha=0.7)
fig.savefig(OUT + "progressive-disclosure.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 2 figures rendered ->", OUT)
