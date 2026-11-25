"""第 9 章四张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系，风格对齐 ch01 模板）"""
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
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch09/"


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


def title(ax, text, x=8.0, y=None, h=6):
    ax.text(x, y if y is not None else h - 0.5, text, color=TXT,
            fontsize=18, fontweight="bold", ha="center")


# ============ 图 1：控制流光谱 ============
fig, ax = new_canvas(16, 5.6)
title(ax, "控制流的归属：谁决定“下一步做什么”？", y=5.05)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "spectrum", ["#22d3ee", "#fbbf24", "#fb7185"])
grad = np.linspace(0, 1, 512).reshape(1, -1)
ax.imshow(grad, extent=[1, 15, 2.05, 2.85], aspect="auto", origin="lower",
          cmap=cmap, zorder=3, alpha=0.85)

marks = [
    (2.6, "Chatbot", "用户逐句驱动\n（没有控制流）"),
    (7.5, "Workflow", "开发者预先编排\n（LLM 只负责节点内容）"),
    (12.4, "Agent", "LLM 运行时自主决定\n（代码只提供循环）"),
]
for x, name, desc in marks:
    ax.plot([x, x], [2.85, 3.35], color=TXT, lw=1.6, alpha=0.7, zorder=4)
    ax.text(x, 4.6, name, color=TXT, fontsize=16, fontweight="bold", ha="center", zorder=5)
    ax.text(x, 4.28, desc, color=TXT, fontsize=11.5, ha="center", va="top",
            alpha=0.9, linespacing=1.5, zorder=5)
ax.text(1.0, 1.6, "代码 100% 决定", color=CYAN, fontsize=13, fontweight="bold", ha="left")
ax.text(15.0, 1.6, "LLM 100% 决定", color=CORAL, fontsize=13, fontweight="bold", ha="right")
ax.text(8.0, 0.7, "判据只有一条：能调用工具 ≠ Agent——“下一步做什么”由谁拍板才是",
        ha="center", color=TXT, fontsize=12.5, alpha=0.9)
fig.savefig(OUT + "control-flow-spectrum.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：Agent 核心循环 ============
fig, ax = new_canvas(16, 8.5)
title(ax, "Agent 核心循环：感知 → 规划 → 行动 → 观察", y=8.0)

ax.text(1.7, 7.05, "环境信息\n任务 / 工具结果\n外部反馈", color=MINT, fontsize=11.5,
        ha="center", va="center", linespacing=1.6, zorder=5)
arrow(ax, (3.1, 6.95), (5.45, 6.95), MINT, lw=2.2)

loops = [
    (6.35, "① 感知 Perception\n把环境 / 任务状态装进上下文", CYAN),
    (4.75, "② 规划 Planning\nLLM 决定下一步做什么", AMBER),
    (3.15, "③ 行动 Action\n调用工具 / 执行代码，作用于环境", CORAL),
    (1.55, "④ 观察 Observation\n把结果回填进记忆", VIOLET),
]
for y, text, color in loops:
    box(ax, 5.6, y, 4.8, 1.15, text, color, fs=12.5)
arrow(ax, (8.0, 6.35), (8.0, 5.98), CYAN, lw=2.4)
arrow(ax, (8.0, 4.75), (8.0, 4.38), AMBER, lw=2.4)
arrow(ax, (8.0, 3.15), (8.0, 2.78), CORAL, lw=2.4)

# 循环回路：④ 观察 →（未完成）→ ② 规划
arrow(ax, (5.54, 2.1), (5.54, 5.32), VIOLET, lw=2.4, ls="--", connectionstyle="arc3,rad=-0.3")
ax.text(3.35, 3.7, "未完成：回到 ②（下一轮）", color=VIOLET, fontsize=12,
        rotation=90, ha="center", va="center", zorder=5)
# 终止出口：② 规划 → 输出
arrow(ax, (10.46, 5.32), (12.35, 5.32), MINT, lw=2.4)
ax.text(11.4, 5.62, "完成任务", color=MINT, fontsize=12, fontweight="bold", ha="center", zorder=5)
box(ax, 12.4, 4.9, 3.1, 0.85, "输出最终结果", MINT, fs=13)

ax.text(8.0, 0.6, "while 循环由代码提供，llm_decide 由 LLM 扮演——Agent 的本质，是把“改剧本的权力”移交给运行时的决策函数",
        ha="center", color=TXT, fontsize=12.5, alpha=0.9)
fig.savefig(OUT + "agent-loop.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 3：五大组件架构 ============
fig, ax = new_canvas(16, 8.5)
title(ax, "Agent 的五大组件", y=8.0)

box(ax, 0.5, 6.7, 1.9, 0.9, "任务", MINT, fs=14)
box(ax, 5.0, 6.55, 5.0, 1.2, "LLM 大脑\n理解 · 推理 · 决策", AMBER, fs=13.5)
box(ax, 12.2, 6.55, 3.3, 1.2, "记忆\n本轮轨迹 /\n长期记忆库", VIOLET, fs=12.5)
box(ax, 5.0, 4.1, 5.0, 1.2, "工具\nsearch / calculator\n代码执行 / API", CYAN, fs=12.5)
box(ax, 5.0, 1.65, 5.0, 1.2, "执行环境\n文件系统 / 沙箱 /\n浏览器 / 生产系统", MINT, fs=12.5)

arrow(ax, (2.45, 7.15), (4.92, 7.15), MINT, lw=2.4)
arrow(ax, (10.05, 7.35), (12.14, 7.35), VIOLET, lw=2.2)
arrow(ax, (12.14, 6.95), (10.05, 6.95), AMBER, lw=2.2)
arrow(ax, (7.5, 6.55), (7.5, 5.38), AMBER, lw=2.4)
ax.text(7.75, 5.95, "规划：下一步做什么？", color=TXT, fontsize=12, ha="left", zorder=5)
arrow(ax, (7.5, 4.1), (7.5, 2.93), CYAN, lw=2.4)
ax.text(7.75, 3.5, "行动", color=TXT, fontsize=12, ha="left", zorder=5)
arrow(ax, (10.15, 2.25), (10.0, 6.5), CORAL, lw=2.4, ls="--", connectionstyle="arc3,rad=0.32")
ax.text(12.6, 4.4, "观察结果回传大脑（循环往复）", color=CORAL, fontsize=12,
        rotation=90, ha="center", va="center", zorder=5)

ax.text(8.0, 0.6, "五大组件没有一个是在 2023 年发明的——新的只是把它们焊在一起的那根总线：“循环 + 上下文”",
        ha="center", color=TXT, fontsize=12.5, alpha=0.9)
fig.savefig(OUT + "agent-architecture.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 4：五分钟 Agent 简史时间线 ============
fig, ax = new_canvas(16, 7.2)
title(ax, "五分钟 Agent 简史：我们正处在哪个坐标", y=6.7)

ax.add_patch(FancyArrowPatch((0.8, 3.5), (15.5, 3.5), arrowstyle="-|>",
                             mutation_scale=22, color="#8fa3bd", lw=2.5, zorder=3))
events = [
    (2.0, "2022.11", "ChatGPT 发布\n对话即接口，但它是 L0", CYAN, True),
    (4.2, "2023.03", "AutoGPT / BabyAGI 爆红\n自主决策被推上聚光灯\n失败模式也暴露无遗", AMBER, False),
    (6.4, "2023.06", "OpenAI Function Calling\n工具调用从提示词黑魔法\n变成原生协议（第 11 章）", CORAL, True),
    (8.6, "2024.12", "Anthropic《Building Effective\nAgents》：Workflow 与 Agent 分道\n“最简方案优先”成为共识", MINT, False),
    (10.9, "2025.01", "Operator / Claude Code /\nDeep Research 落地\nL2–L3 级 Agent 进入生产", VIOLET, True),
    (13.5, "现在", "多智能体、MCP / A2A 协议\n评测与安全成为主战场\n（第 12、16、17 章）", CYAN, False),
]
for x, date, desc, color, above in events:
    ax.add_patch(Circle((x, 3.5), 0.11, fc=color, ec="none", zorder=5))
    ax.add_patch(Circle((x, 3.5), 0.06, fc=NAVY, ec=color, lw=1.6, zorder=6))
    if above:
        ax.plot([x, x], [3.5, 3.95], color=color, lw=1.6, alpha=0.7, zorder=3)
        ax.text(x, 5.9, date, color=color, fontsize=14, fontweight="bold",
                ha="center", va="bottom", zorder=5)
        ax.text(x, 5.55, desc, color=TXT, fontsize=11, ha="center", va="top",
                linespacing=1.5, alpha=0.92, zorder=5)
    else:
        ax.plot([x, x], [3.05, 3.5], color=color, lw=1.6, alpha=0.7, zorder=3)
        ax.text(x, 3.0, date, color=color, fontsize=14, fontweight="bold",
                ha="center", va="top", zorder=5)
        ax.text(x, 2.6, desc, color=TXT, fontsize=11, ha="center", va="top",
                linespacing=1.5, alpha=0.92, zorder=5)

ax.text(8.0, 0.45, "三条主线贯穿始终：控制权的渐进移交 · 协议的标准化 · 工程理性的回归",
        ha="center", color=TXT, fontsize=12.5, alpha=0.9)
fig.savefig(OUT + "agent-timeline.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

print("OK: 4 figures rendered")
