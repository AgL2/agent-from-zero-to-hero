"""第 12 章三张正式配图（与全书视觉系统一致：深蓝渐变底、五色体系）
图 1：无协议 N×M 专线 vs 有协议一个标准插座
图 2：MCP 三层架构（Host / Client / Server）
图 3：A2A 协作消息流转（Agent Card → Task 生命周期）
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
OUT = "/Users/lianggeng/Desktop/agent-from-zero-to-hero/assets/images/ch12/"


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


def box(ax, x, y, w, h, text, color, fs=13, glow=True, lw=2):
    if glow:
        ax.add_patch(FancyBboxPatch((x - 0.06, y - 0.06), w + 0.12, h + 0.12,
                                    boxstyle="round,pad=0.06", fc=color, ec="none", alpha=0.15, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                fc=PANEL, ec=color, lw=lw, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=TXT, fontsize=fs, zorder=5, linespacing=1.5)


def arrow(ax, p1, p2, color=CYAN, lw=2.2, ls="-", ms=16):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=3))


# ============ 图 1：N×M 问题 vs 协议 ============
fig, ax = new_canvas(16, 6)
ax.text(4.1, 5.6, "无协议：N × M 根专线", color=CORAL, fontsize=15, fontweight="bold", ha="center")
ax.text(12.2, 5.6, "有协议：一个标准插座", color=MINT, fontsize=15, fontweight="bold", ha="center")
hosts_l = [(1.4, 4.0), (1.4, 1.3)]
tools_l = [(6.4, 4.0), (6.4, 1.3)]
for i, (x, y) in enumerate(hosts_l, 1):
    box(ax, x, y, 1.7, 1.0, f"Host{i}", CYAN, fs=13)
for j, (x, y) in enumerate(tools_l):
    box(ax, x, y, 1.7, 1.0, f"工具{chr(65 + j)}", AMBER, fs=13)
for x1, y1 in hosts_l:
    for x2, y2 in tools_l:
        ax.add_patch(FancyArrowPatch((x1 + 1.7, y1 + 0.5), (x2, y2 + 0.5),
                                     arrowstyle="-", color=CORAL, lw=1.8,
                                     connectionstyle="arc3,rad=0.12", zorder=2))
ax.text(4.1, 0.45, "N × M 条专用线，宿主改版全部重改", color=CORAL, fontsize=12, ha="center")
# 右侧：协议插座
hosts_r = [(10.0, 4.0), (10.0, 1.3)]
tools_r = [(13.6, 4.0), (13.6, 1.3)]
for i, (x, y) in enumerate(hosts_r, 1):
    box(ax, x, y, 1.7, 1.0, f"Host{i}", CYAN, fs=13)
for j, (x, y) in enumerate(tools_r):
    box(ax, x, y, 1.7, 1.0, f"工具{chr(65 + j)}", AMBER, fs=13)
box(ax, 12.35, 2.35, 0.75, 1.4, "协\n议", MINT, fs=12, glow=False)
for x, y in hosts_r:
    arrow(ax, (x + 1.7, y + 0.5), (12.35, 2.9 if y > 2 else 3.0), MINT, lw=2.0)
for x, y in tools_r:
    arrow(ax, (13.1, 3.0 if y > 2 else 2.9), (x, y + 0.5), MINT, lw=2.0)
ax.text(12.2, 0.45, "每人接一次，共 (N + M) 条", color=MINT, fontsize=12, ha="center")
ax.text(8.0, 3.05, "→", color=TXT, fontsize=26, ha="center", va="center", alpha=0.8)
fig.savefig(OUT + "nxm-vs-protocol.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 2：MCP 三层架构 ============
fig, ax = new_canvas(16, 7)
ax.text(8.0, 6.6, "MCP 三层架构：Host 为每个连接派生一个 Client，一对一对接 Server",
        color=TXT, fontsize=15.5, fontweight="bold", ha="center")
# Host 大框
ax.add_patch(FancyBboxPatch((1.2, 4.35), 13.6, 1.75, boxstyle="round,pad=0.08",
                            fc="#132a44", ec=VIOLET, lw=2.2, zorder=3))
ax.text(1.55, 5.95, "Host（宿主应用）：Claude Desktop / Cursor / 你的 Agent 框架",
        color=VIOLET, fontsize=12.5, fontweight="bold", ha="left", zorder=5)
ax.text(1.55, 5.62, "持有 LLM 会话与上下文，负责把工具结果喂给模型",
        color=TXT, fontsize=11.5, alpha=0.9, ha="left", zorder=5)
clients = [(2.0, 4.45), (6.6, 4.45), (11.2, 4.45)]
for i, (x, y) in enumerate(clients, 1):
    box(ax, x, y, 2.8, 0.8, f"MCP Client {i}\n一个连接管一个 Server", CYAN, fs=11, glow=False)
servers = [
    (2.0, "MCP Server\nGitHub 服务\n（远程 HTTP）", AMBER),
    (6.6, "MCP Server\n数据库服务\n（本地进程）", MINT),
    (11.2, "MCP Server\n你的计算器\n（本地进程）", CORAL),
]
for (cx, _), (sx, name, color) in zip(clients, servers):
    x0 = cx + 1.4
    arrow(ax, (x0, 4.45), (sx + 1.4, 2.95), TXT, lw=1.9)
    ax.text((x0 + sx + 1.4) / 2, 3.85, "JSON-RPC", color=TXT, fontsize=10.5,
            ha="center", alpha=0.9)
for sx, name, color in servers:
    box(ax, sx, 1.55, 2.8, 1.4, name, color, fs=11.5)
ax.text(8.0, 0.75, "Client 与 Server 一对一连接：凭证、权限与故障域天然隔离，"
                  "一个 Server 崩溃或作恶，影响面锁在它自己的连接里",
        color=TXT, fontsize=11.5, ha="center", alpha=0.9)
fig.savefig(OUT + "mcp-architecture.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)

# ============ 图 3：A2A 消息流转 ============
fig, ax = new_canvas(16, 7.5)
ax.text(8.0, 7.1, "A2A 协作消息流转：发现（Agent Card）→ 委托（Task）→ 异步推进 → 补充输入 → 交付（Artifact）",
        color=TXT, fontsize=14.5, fontweight="bold", ha="center")
box(ax, 2.2, 5.9, 3.0, 1.0, "旅行 Agent\n（客户端）", CYAN, fs=13)
box(ax, 10.8, 5.9, 3.0, 1.0, "航司 Agent\n（服务端）", AMBER, fs=13)
ax.plot([3.7, 3.7], [0.6, 5.9], color=CYAN, lw=1.5, ls=(0, (4, 5)), alpha=0.55, zorder=2)
ax.plot([12.3, 12.3], [0.6, 5.9], color=AMBER, lw=1.5, ls=(0, (4, 5)), alpha=0.55, zorder=2)
steps = [
    (5.45, 0, "① GET /.well-known/agentcard.json", CYAN, "发现"),
    (4.80, 1, "Agent Card（skills：出票 / 改签），名片上写明能力", AMBER, None),
    (4.15, 0, "② tasks/send：「帮我订 6/12 北京→成都往返」", CYAN, None),
    (3.50, 1, "③ Task 状态：working…（Agent 自主查库存、比价），tasks/get 轮询 / 订阅推送", AMBER, None),
    (2.85, 0, "④ 补充输入 tasks/send：「往返想分开订吗？」", CYAN, None),
    (2.20, 1, "Task 状态：input-required —— 把球踢回给调用方", CORAL, None),
    (1.35, 1, "⑤ Task: completed，Artifact：电子客票 PDF + 订单号", MINT, None),
]
for y, direction, label, color, badge in steps:
    if direction == 0:
        x1, x2 = 3.7, 12.3
        arrow(ax, (x1, y), (x2, y), color, lw=2.0)
    else:
        x1, x2 = 12.3, 3.7
        arrow(ax, (x1, y), (x2, y), color, lw=2.0)
    ax.text(8.0, y + 0.14, label, color=TXT, fontsize=11.5, ha="center", va="bottom", zorder=5)
fig.savefig(OUT + "a2a-flow.png", facecolor=NAVY, bbox_inches="tight")
plt.close(fig)
print("OK: ch12 3 figures rendered")
