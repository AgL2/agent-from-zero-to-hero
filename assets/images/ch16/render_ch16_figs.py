"""第 16 章七张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系，对齐 ch01 模板）
图 1 complexity-tax.png     复杂度税账单（16.1）
图 2 pattern-pipeline.png   链式：流水线作业（16.2.1）
图 3 pattern-supervisor.png 主管-工人：项目经理制（16.2.2）
图 4 pattern-debate.png     辩论：红蓝对抗（16.2.3）
图 5 pattern-groupchat.png  群聊：圆桌会议（16.2.4）
图 6 pattern-blackboard.png 黑板：公告板协作（16.2.5）
图 7 decision-tree.png      多 Agent 决策图（本章小结收束图）
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


# ============ 图 1：复杂度税账单（16.1） ============
fig, ax = new_canvas(16, 4.8)
label(ax, 8.0, 4.35, "复杂度税账单：每新增 1 个 Agent，四笔固定成本", TXT, 17, bold=True)
items = [
    ("上下文税", "系统提示 + 每轮收发的消息\n全部按 token 计费", CYAN),
    ("延迟税", "一跳消息传递 = 一次完整的\nLLM 往返（几百 ms 到数秒）", AMBER),
    ("失败点税", "Agent 数量翻倍，系统故障\n组合翻指数倍", CORAL),
    ("调试税", "bug 复现需要“对齐三个 Agent\n的随机性”，难度骤增", VIOLET),
]
for i, (t, d, c) in enumerate(items):
    x = 0.9 + i * 3.85
    box(ax, x, 1.0, 3.35, 2.6, f"{t}\n\n{d}", c, 12.5)
fig.savefig(OUT + "complexity-tax.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：链式 Pipeline（16.2.1） ============
fig, ax = new_canvas(16, 3.4)
label(ax, 8.0, 2.95, "链式（Pipeline）：流水线作业", TXT, 16, bold=True)
roles = [("Agent A", "翻译", CYAN), ("Agent B", "润色", AMBER), ("Agent C", "校对", MINT)]
for i, (t, d, c) in enumerate(roles):
    x = 1.6 + i * 4.6
    box(ax, x, 0.7, 3.2, 1.4, f"{t}\n{d}", c, 13.5)
    if i < 2:
        arrow(ax, (x + 3.3, 1.4), (x + 4.5, 1.4), CORAL, 2.6)
label(ax, 8.0, 0.3, "固定顺序、单向流动：每环节的输出是下一环节的输入，无反馈回路", TXT, 12, alpha=0.85)
fig.savefig(OUT + "pattern-pipeline.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 3：主管-工人 Supervisor（16.2.2） ============
fig, ax = new_canvas(16, 6.2)
label(ax, 8.0, 5.8, "主管-工人（Supervisor）：项目经理制", TXT, 16, bold=True)
arrow(ax, (1.2, 4.35), (5.3, 4.35), CORAL, 2.6)
label(ax, 2.6, 4.7, "任务", CORAL, 13, bold=True)
box(ax, 5.5, 3.7, 5.0, 1.3, "Supervisor\n拆解任务 · 分派 · 汇总裁决", AMBER, 13.5)
label(ax, 12.4, 4.35, "◄ 汇总 / 裁决", AMBER, 12.5, ha="left")
workers = [("Researcher", CYAN), ("Writer", MINT), ("Critic", VIOLET)]
for i, (t, c) in enumerate(workers):
    x = 1.4 + i * 4.7
    box(ax, x, 0.8, 3.6, 1.2, t, c, 13.5)
    arrow(ax, (8.0, 3.7), (x + 1.8, 2.1), c, 1.8)
label(ax, 8.0, 0.25, "工人之间不直接对话，所有信息经主管中转；工人也可以是子团队（层级式）", TXT, 12, alpha=0.85)
fig.savefig(OUT + "pattern-supervisor.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 4：辩论 Debate（16.2.3） ============
fig, ax = new_canvas(16, 5.2)
label(ax, 8.0, 4.8, "辩论（Debate）：红蓝对抗", TXT, 16, bold=True)
box(ax, 2.2, 3.0, 4.0, 1.3, "Agent 正方", CYAN, 14)
box(ax, 9.8, 3.0, 4.0, 1.3, "Agent 反方", CORAL, 14)
arrow(ax, (6.4, 3.85), (9.6, 3.85), AMBER, 2.2, style="<|-|>")
label(ax, 8.0, 4.25, "论点交锋（多轮）", AMBER, 12.5, bold=True)
box(ax, 5.9, 0.7, 4.2, 1.2, "裁判 Agent（或人）", AMBER, 13.5)
arrow(ax, (3.6, 3.0), (6.6, 1.95), CYAN, 1.8)
arrow(ax, (12.4, 3.0), (9.4, 1.95), CORAL, 1.8)
label(ax, 2.2, 2.35, "结论", CYAN, 11.5)
label(ax, 13.8, 2.35, "结论", CORAL, 11.5)
label(ax, 8.0, 0.2, "同一问题、对立立场、多轮交锋后裁决；token 成本约为单 Agent 的 3-5 倍", TXT, 12, alpha=0.85)
fig.savefig(OUT + "pattern-debate.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 5：群聊 Group Chat（16.2.4） ============
fig, ax = new_canvas(16, 6.2)
label(ax, 8.0, 5.8, "群聊（Group Chat）：圆桌会议", TXT, 16, bold=True)
box(ax, 5.5, 2.5, 5.0, 1.3, "群聊总线\n（共享会话 + 发言管理器）", AMBER, 13)
agents = [("[Agent 1]", 1.2, 4.2, CYAN), ("[Agent 2]", 11.4, 4.2, MINT),
          ("[Agent 3]", 1.2, 0.7, CORAL), ("[Agent 4]", 11.4, 0.7, VIOLET)]
for t, x, y, c in agents:
    box(ax, x, y, 3.4, 1.0, t, c, 13)
    if x < 8:
        arrow(ax, (x + 3.5, y + 0.5), (5.4, 3.15 if y > 2 else 2.85), c, 1.6, style="<|-|>")
    else:
        arrow(ax, (x - 0.1, y + 0.5), (10.6, 3.15 if y > 2 else 2.85), c, 1.6, style="<|-|>")
label(ax, 8.0, 0.2, "所有消息进共享会话，由发言管理器（或下一发言人 LLM）决定谁接话；消息数随人数平方级膨胀",
      TXT, 12, alpha=0.85)
fig.savefig(OUT + "pattern-groupchat.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 6：黑板 Blackboard（16.2.5） ============
fig, ax = new_canvas(16, 6.2)
label(ax, 8.0, 5.8, "黑板（Blackboard）：公告板协作", TXT, 16, bold=True)

ax.add_patch(FancyBboxPatch((1.2, 3.4), 13.6, 1.9, boxstyle="round,pad=0.08",
                            fc=INDIGO, ec=AMBER, lw=2.4, zorder=3))
label(ax, 2.0, 4.95, "黑板（共享状态区）", AMBER, 13.5, bold=True, ha="left")
label(ax, 8.0, 4.15, "待办: [t1, t2, t3]    已完成: [t0]    中间结果: {...}    锁: t1 → Agent B",
      TXT, 12.5)

workers = [("Agent A", CYAN), ("Agent B", MINT), ("Agent C", VIOLET)]
for i, (t, c) in enumerate(workers):
    x = 1.8 + i * 4.6
    box(ax, x, 0.7, 3.4, 1.1, t, c, 13.5)
    arrow(ax, (x + 1.7, 1.9), (x + 1.7, 3.35), c, 1.8, style="<|-|>")
label(ax, 8.0, 3.75, "每个 Agent 盯黑板：谁有空谁认领任务，认领 / 写入双向流动", TXT, 11.5, alpha=0.9)
label(ax, 8.0, 0.2, "Agent 之间不直接通信，都读写一块共享黑板；适合可并行的批量任务，天然支持动态加入/退出",
      TXT, 12, alpha=0.85)
fig.savefig(OUT + "pattern-blackboard.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 7：决策图（本章小结收束图） ============
fig, ax = new_canvas(16, 7.2)
label(ax, 8.0, 6.8, "任务来了，该用几个 Agent？", TXT, 17, bold=True)

# 三个问题框（左列）
qs = [
    ("一次 LLM 调用能做完吗？", 5.35),
    ("子任务可并行吗？", 3.85),
    ("需要跨工人的信息碰撞吗？", 2.35),
]
for t, y in qs:
    box(ax, 2.2, y, 6.0, 1.05, t, AMBER, 13.5)
arrow(ax, (5.2, 6.35), (5.2, 6.55), TXT, 1.2, style="-")   # 入口
arrow(ax, (5.2, 5.35), (5.2, 4.9), TXT, 1.6)               # Q1 -> Q2
arrow(ax, (5.2, 3.85), (5.2, 3.4), TXT, 1.6)               # Q2 -> Q3
label(ax, 5.75, 5.13, "否", TXT, 12, bold=True)
label(ax, 5.75, 3.63, "是", TXT, 12, bold=True)

# 四个出口（右列）：箭头起点取问题框右缘
outs = [
    ("单 Agent + 好提示词\n（80% 的任务停在这里）", MINT, 5.2, 5.87, "是"),
    ("链式（流程稳定）/\n主管-工人（需裁决反馈）", CYAN, 3.7, 4.37, "否"),
    ("黑板（批量并行，成本线性）★本章实验", VIOLET, 2.2, 2.87, "否"),
    ("群聊（认下 n(n-1) 的贵账）/\n辩论（红蓝对抗）", CORAL, 0.7, 2.87, "是"),
]
for t, c, y, ysrc, branch in outs:
    box(ax, 9.4, y, 5.8, 1.1, t, c, 12)
    arrow(ax, (8.3, ysrc), (9.3, y + 0.55), c, 1.8)
    ly = ysrc + 0.32 if c is not CORAL else ysrc - 0.42
    label(ax, 8.78, ly, branch, c, 12, bold=True)
label(ax, 8.0, 0.12, "无论选哪种，上线前问同一个问题：每个循环，什么条件让它停？", CORAL, 13, bold=True)
fig.savefig(OUT + "decision-tree.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 7 figures rendered ->", OUT)
