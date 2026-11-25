# 《Agent 从零到专家》插图生成 Prompt 设计方案
> 适用工具：image 2（Nano Banana 2 / Gemini Image 2 及同级模型）
> 生成后命名：封面 `book-cover.png`；章节题图 `ch01-banner.png` ~ `ch20-banner.png`，放入 `assets/images/` 即可自动替换现有图（章节内引用路径不变）

---

## 一、设计思路（为什么这样设计 Prompt）

### 1. 全书画的是"同一本书"——风格锚定是第一位
AI 生图最大的风险是 21 张图 21 种风格。解决方案：**每张图的 Prompt = 固定风格锚定段 + 本章内容段**，风格锚定段逐字不变。如果 image 2 支持参考图（image reference），**先生成封面，再把封面作为风格参考图喂给后续所有章节图**，一致性可达 90% 以上。

### 2. 风格选型：等距 3D 软渲染（isometric 3D）而非扁平矢量
- 扁平矢量（当前方案）在 AI 生图里容易生成"廉价 PPT 剪贴画感"
- **等距 3D + 柔光 + 磨砂玻璃质感**是目前科技出版/技术品牌（Stripe、Linear、Vercel）的主流审美，AI 模型发挥最稳定，且深色背景与本书 PDF 深色题图位完美兼容
- 3D 风格自带光影和景深，天然解决"构图太平"的问题

### 3. 构图公式（每张横幅图都遵守）
- **三分法**：左侧 1/3 主体隐喻物，右侧 2/3 叙事元素，或中央焦点 + 两翼展开
- **一个视觉焦点**：每张图只能有一个最亮的元素（金色发光体），其余元素亮度递减
- **引导线**：用光线/路径/粒子流把视线从左引向右（书的阅读方向）
- **景深**：前景元素大而虚、中景主体清晰、背景星空粒子极淡
- **负空间**：顶部 1/4 留白（PDF 中题图上方紧邻章节标题）

### 4. 文字策略：全面禁字
image 2 虽然文字能力较强，但中文仍有概率出错，且 21 张图字体难统一。**所有图一律 no text**，章节编号由 Markdown 标题承担。

### 5. ASCII/流程图的处理建议（重要）
章节正文中的**架构图、时序图、流程图（现为 ASCII）不要用 image 2 生成**——生图模型无法保证箭头指向、文字标签的准确性，会产生"看起来对其实错"的危险图。这类图的正确做法：
- 方案 A（推荐）：保留 ASCII（技术书读者接受度高），或用 Mermaid 重绘（GitHub 原生渲染）
- 方案 B：用 D2 / Excalidraw 手绘风重绘，导出 PNG
- 本文件第 4 节给出 6 张**概念插画**（非精确图表）的 Prompt，用于替换各 Part 开头的大 ASCII 架构图位置（不涉及具体箭头标签，安全）

---

## 二、全局风格锚定段（每张图 Prompt 的固定开头，逐字复制）

**英文版（推荐，直接拼在每条 Prompt 开头）：**

```
A premium tech book illustration, modern isometric 3D style with soft studio lighting, 
frosted glass and matte plastic materials with subtle inner glow. Deep navy-to-indigo 
gradient background (#0d1b2a to #1b263b) with faint floating particles and bokeh dust. 
Accent colors strictly limited to: cyan #22d3ee, amber #fbbf24, coral #fb7185, mint #34d399, 
violet #a78bfa. One single brightest focal element glowing in warm amber, all other elements 
dimmer. Clean composition with generous negative space in the upper quarter, depth of field 
with soft foreground blur. Wide banner aspect ratio 16:9. No text, no letters, no numbers, 
no watermark, no photorealism, no human faces.
```

**中文版（供参考/调试用）：**

```
高级技术书籍插画，现代等距 3D 风格，柔和影棚光，磨砂玻璃与哑光塑料材质带轻微内发光。
深海军蓝到靛紫渐变背景（#0d1b2a → #1b263b），飘浮微弱粒子与散景尘埃。
配色严格限定：青 #22d3ee、琥珀 #fbbf24、珊瑚 #fb7185、薄荷 #34d399、紫罗兰 #a78bfa。
全图仅一个最亮的琥珀色发光焦点元素，其余元素亮度递减。构图干净，上四分之一留白，
前景浅景深虚化。16:9 横幅。无文字、无字母、无数字、无水印、无写实风、无人脸。
```

**通用负面 Prompt（如工具支持单独填 negative）：**

```
text, letters, words, numbers, watermark, logo, photorealistic, human face, hands, 
cluttered composition, more than 8 distinct objects, harsh shadows, white background, 
flat 2D vector clip-art style, low resolution, blurry subject
```

---

## 三、封面 Prompt（1 张，竖版 2:3）

> 用法：完整复制以下整段（已含风格锚定，竖版参数已改）

```
A premium tech book cover illustration, modern isometric 3D style with soft studio 
lighting, frosted glass and matte plastic materials with subtle inner glow. Deep 
navy-to-indigo gradient background (#0d1b2a to #1b263b) with faint floating particles 
and bokeh dust. Accent colors strictly limited to: cyan #22d3ee, amber #fbbf24, coral 
#fb7185, mint #34d399, violet #a78bfa. Vertical 2:3 portrait composition.

Scene: an ascending glowing amber path winds from bottom-left to top-right like a 
mountain trail with three stations. Bottom station: a small translucent glass neural 
network — a cluster of glowing cyan nodes connected by thin light threads. Middle 
station: a levitating luminous brain made of interlocking geometric facets, emitting 
soft violet glow with orbiting attention threads. Top station: a friendly small 
isometric robot (matte white and cyan) standing on a glowing platform, reaching out 
to three floating tool icons — a wrench, a magnifying glass, and a database cylinder — 
each in a frosted glass bubble. The path between stations sparkles with tiny amber 
light dots. The upper fifth of the image is calm empty gradient space reserved for 
the book title. One single brightest focal element is the brain, everything else dimmer. 
Depth of field, soft foreground blur at the bottom edge. No text, no letters, no 
watermark, no photorealism, no human faces.
```

**中文设计说明**：底→顶三站旅程 = 全书结构（神经网络→LLM→Agent），顶部 1/5 纯色留白带是给书名排版用的。若生成结果中机器人细节崩坏（手指/五官），追加一句 `the robot is small in the frame, simple rounded design, no facial details`。

---

## 四、章节题图 Prompt（20 张，横幅 16:9）

> 每张已包含完整风格锚定段，**整段复制即可单独使用**。
> 一致性技巧：生成第 1 张满意后，后续 19 张以它为风格参考图。

### 第 1 章 神经网络基础（ch01-banner.png）
**隐喻**：生物神经元 → 人工网络的进化之桥

```
[风格锚定段] Scene: on the left, a translucent glass biological neuron with branching 
dendrites glowing in soft cyan, its axon extending rightward and smoothly transforming 
into a clean isometric network of geometric nodes — three input spheres on the left 
layer, four in the middle, one amber glowing output sphere on the right — connected 
by luminous thread lines with tiny traveling light pulses. The transformation point 
in the center is the single brightest amber focal glow. Background particles drift 
like synaptic sparks. Upper quarter negative space.
```

### 第 2 章 卷积神经网络（ch02-banner.png）
**隐喻**：卷积核滑窗扫描图像，提取出特征图

```
[风格锚定段] Scene: an isometric frosted glass grid of 5x5 image pixels floating 
slightly above a surface, one 3x3 region highlighted by a glowing amber frame window 
lifting off the grid like a scanning lens, with two faint ghost frames trailing behind 
to suggest sliding motion. From the amber window, a beam of light projects to the 
right, forming a smaller floating 3x3 cyan feature-map grid whose cells glow at 
different intensities like a heatmap. A faint dotted cyan arc connects window to 
feature map. The amber scanning window is the brightest focal element. Upper quarter 
negative space.
```

### 第 3 章 反向传播（ch03-banner.png）
**隐喻**：前向金流与反向珊瑚流构成循环回路

```
[风格锚定段] Scene: an elegant isometric loop circuit floating in space — on the 
upper arc, three frosted glass spheres connected by forward-flowing golden light 
streams moving left to right with small bright pulse dots; on the lower arc, the same 
three spheres connected by backward-flowing coral-red gradient streams moving right 
to left, rendered as smooth flowing ribbons of light. At the right end, the two flows 
meet at a small glowing amber crystal that acts as the loop junction and the single 
brightest focal element. Subtle motion blur on the streams. Upper quarter negative 
space.
```

### 第 4 章 Token / 词嵌入 / 位置编码（ch04-banner.png）
**隐喻**：文字积木 → 彩虹向量条 + 钟表波形

```
[风格锚定段] Scene: a horizontal row of five frosted glass rounded blocks (like 
smooth pebble tokens) in different accent colors floats in the middle; from each 
block, a vertical beam of light projects downward, forming five vertical gradient 
bars of different color patterns like an audio equalizer made of light — these are 
the embedding vectors. On the far right, a small elegant glass clock face merged 
with two overlapping sine waves (one cyan, one violet) orbiting it, representing 
positional encoding. The clock-wave hybrid emits the single brightest amber glow. 
Upper quarter negative space.
```

### 第 5 章 Transformer（ch05-banner.png）
**隐喻**：token 之间的注意力蛛网，汇聚焦点

```
[风格锚定段] Scene: four frosted glass token spheres floating at different heights 
across the frame, interconnected by a web of light threads of varying thickness and 
brightness — thick bright cyan threads for strong attention, thin faint threads for 
weak attention. All threads converge toward one central levitating geometric core 
(a faceted amber polyhedron) that is the single brightest element, radiating soft 
rays. Tiny light pulses travel along the threads toward the core. A subtle heat 
shimmer around the core suggests softmax computation. Upper quarter negative space.
```

### 第 6 章 微调与对齐（ch06-banner.png）
**隐喻**：原石经过模具变成宝石 + 偏好天平

```
[风格锚定段] Scene: on the left, a large rough raw crystal rock (matte violet, 
unpolished) representing a pretrained model; it slides into an elegant golden 
funnel-shaped mold in the center that emits the brightest amber glow; emerging on 
the right side of the funnel is a smaller, perfectly cut, luminous mint-green 
gemstone. On the far right, a delicate isometric balance scale in frosted glass: 
the left pan holds a small glowing mint gem (chosen), the right pan holds a dull 
coral pebble (rejected), the beam tilts gently toward the mint side. Upper quarter 
negative space.
```

### 第 7 章 提示工程（ch07-banner.png）
**隐喻**：控制台旋钮引导光束命中星辰

```
[风格锚定段] Scene: an isometric frosted glass control console on the left with 
three tactile knobs and one slider (cyan and violet), a thin beam of light emanating 
from the console's emitter, bending gracefully through mid-air as the knobs "steer" 
it; on the right, the beam splits into three faint ghost paths that miss, and one 
bright path that precisely hits a glowing amber star gem — the single brightest 
focal element. Small dial needles and subtle light refractions sell the "tuning" 
metaphor. Upper quarter negative space.
```

### 第 8 章 推理、量化与部署（ch08-banner.png）
**隐喻**：流水线 + 压缩

```
[风格锚定段] Scene: an isometric glass conveyor pipeline flowing left to right: 
first a wide glowing cyan intake tunnel where a whole paragraph of light pours in 
at once (prefill), then a series of small glowing cubes emerging one by one in a 
rhythm, the last cube still half-formed with a translucent dotted outline (autoregressive 
decode). Below the pipeline, a large heavy glass sphere is being gently compressed 
by two smooth plates into a smaller, denser, brighter sphere (quantization), with 
the compressed sphere glowing as the brightest amber focal point. Upper quarter 
negative space.
```

### 第 9 章 什么是 Agent（ch09-banner.png）
**隐喻**：感知-规划-行动-观察的永恒循环

```
[风格锚定段] Scene: a cute small isometric robot (matte white and cyan, simple 
rounded design, no facial details) standing at the center of a glowing circular 
orbit ring; around the ring, four floating frosted glass medallions at the cardinal 
points: an eye (perception), a gear with a route map (planning), a wrench (action), 
and a magnifying glass (observation); curved light arrows along the ring connect 
them clockwise in a seamless loop with traveling pulse dots. The robot's chest 
core emits the single brightest amber glow. Upper quarter negative space.
```

### 第 10 章 Agent 架构范式 ReAct（ch10-banner.png）
**隐喻**：思考-行动-观察的踏脚石蛇形前进

```
[风格锚定段] Scene: a winding S-shaped path of six floating stepping stones across 
the frame, alternating three types in sequence: stones with a small glowing brain 
icon (thought, violet), stones with a wrench icon (action, cyan), stones with an 
eye icon (observation, mint). Light ribbons connect stone to stone in a snake 
pattern with gentle elevation changes, and at the end of the path stands a small 
glowing amber flag on the final summit stone — the brightest focal element. Soft 
mist between the stones suggests exploration. Upper quarter negative space.
```

### 第 11 章 工具调用（ch11-banner.png）
**隐喻**：大脑递交 JSON 卷轴，工具执行后回流

```
[风格锚定段] Scene: on the left, a levitating faceted glass brain (violet glow) 
extends a beam of light holding an unrolling luminous scroll covered in bracket-like 
glyph patterns (a JSON document, abstract symbols only, no readable characters); 
the scroll feeds into three frosted glass tool pods arranged in a fan on the right: 
a calculator pod, a cloud pod, and a database cylinder pod, each lighting up as it 
receives the scroll; thin mint-green light streams flow back from the pods to the 
brain as return results. The scroll's brightest glyph glows amber as the focal point. 
Upper quarter negative space.
```

### 第 12 章 MCP / A2A 协议（ch12-banner.png）
**隐喻**：万能转接枢纽 + 两个机器人对话

```
[风格锚定段] Scene: in the center, a glowing amber universal docking hub shaped 
like a rounded rectangular port — the brightest focal element; from the left, three 
differently shaped exotic plugs (a database cylinder plug, a folder plug, a globe 
plug) each pass through a small golden adapter ring that reshapes them into the 
same universal connector before entering the hub. On the right side of the hub, 
two small isometric robots face each other exchanging a soft glowing speech-bubble-
shaped orb of light between them (agent-to-agent). Clean symmetry. Upper quarter 
negative space.
```

### 第 13 章 RAG（ch13-banner.png）
**隐喻**：文档化作星图，查询之星连接近邻

```
[风格锚定段] Scene: on the left, a small stack of frosted glass document sheets 
breaks apart into rectangular slices that dissolve upward into glowing particles; 
the particles drift rightward and crystallize into a beautiful constellation field — 
dozens of small colored stars scattered in a loose 3D space (the vector space); 
one amber query star, the brightest element, shoots in from the lower right with a 
comet tail, and thin golden light threads instantly connect it to its three nearest 
neighbor stars, which light up in cyan rings. Upper quarter negative space.
```

### 第 14 章 记忆机制（ch14-banner.png）
**隐喻**：有限工作台与无限仓库的换入换出

```
[风格锚定段] Scene: a glowing glass workbench in the center holding exactly three 
luminous memory crystals (the limited context window), one crystal glowing brightest 
in amber; on the left, a small RAM-like glass rack with fast flickering mint lights 
(short-term memory); on the right, a tall elegant archive tower of stacked frosted 
drawers (long-term memory) with one drawer sliding open; two curved light streams 
flow between workbench and archive — a golden stream carrying a crystal into the 
drawer (write), a cyan stream carrying one back (retrieve). Upper quarter negative 
space.
```

### 第 15 章 Skill 技能体系（ch15-banner.png）
**隐喻**：机器人从悬浮技能卡库中点亮一张

```
[风格锚定段] Scene: a small isometric robot (matte white and cyan, no facial 
details) stands before a fanned arc of five floating frosted glass skill cards, 
each card bearing a simple glowing glyph (a code bracket, a target, a lightning 
bolt, a table grid, a paint palette); the center card tilts forward, enlarged and 
ignited in bright amber glow — the single brightest focal element — with light 
particles streaming from it into the robot's raised hand. The unchosen cards remain 
dim and slightly behind. Upper quarter negative space.
```

### 第 16 章 多智能体系统（ch16-banner.png）
**隐喻**：指挥家与 Agent 乐团

```
[风格锚定段] Scene: an isometric orchestra arrangement: a slightly elevated glass 
podium on the left where a small conductor robot with a glowing baton stands; 
fanning out to the right, three distinct worker robots on small platforms, each 
tinted a different accent color (cyan, coral, mint) and each holding a different 
tool; thin violet light threads connect the conductor's baton tip to each worker, 
and the workers' outputs converge as colored streams into one glowing amber 
crystal product floating at the far right — the brightest focal element. Upper 
quarter negative space.
```

### 第 17 章 评估、可观测性与安全（ch17-banner.png）
**隐喻**：仪表盘 + 盾牌拦截

```
[风格锚定段] Scene: split composition — on the left, an elegant isometric frosted 
glass dashboard with three glowing gauge dials (mint needles) and a small rising 
bar chart, all rendered in clean glass and light; on the right, a translucent cyan 
shield with a soft hexagonal energy pattern deflects three small coral warning 
triangles that shatter into harmless particles upon impact, with visible deflection 
arcs. The shield's center emblem glows amber as the brightest focal element, 
balancing the dashboard's cooler tones. Upper quarter negative space.
```

### 第 18 章 手写 Mini-Agent 框架（ch18-banner.png）
**隐喻**：发光积木拼出机器人

```
[风格锚定段] Scene: a playful yet elegant scene of five glowing translucent 
building blocks (like premium glass LEGO bricks with round studs) mid-assembly: 
four blocks already snapped together forming a small robot's body (a head block 
with two simple light-dot eyes, a torso block, two arm blocks), while the fifth 
amber block — the brightest focal element — hovers mid-air about to click into 
place as the final leg, with small satisfying spark particles at the connection 
point. A couple of spare blocks rest softly blurred in the foreground. Upper 
quarter negative space.
```

### 第 19 章 个人知识库 Agent（ch19-banner.png）
**隐喻**：书架、放大镜与带引用的回答

```
[风格锚定段] Scene: on the left, a compact isometric glass bookshelf with six 
colorful book spines, one book pulled out slightly and glowing; a beam of light 
from that book travels through a levitating magnifying lens in the center, which 
focuses and intensifies it; the focused beam lands on the right onto a floating 
glowing speech-bubble card where a tiny golden citation badge hovers at its corner. 
The magnifying lens rim glows amber as the brightest focal element, with realistic 
light refraction through the glass. Upper quarter negative space.
```

### 第 20 章 多 Agent 自动化流水线（ch20-banner.png）
**隐喻**：四工位传送带产出报告

```
[风格锚定段] Scene: an isometric glass conveyor belt flowing left to right with 
four elegant workstation arches: a funnel arch collecting scattered light dots 
(collect, cyan), a sieve arch filtering them (filter, mint), a pen arch writing 
(write, violet), and a stamp arch sealing (publish, coral); a small document card 
travels along the belt, visibly more refined after each station, and emerges at 
the end as a radiant golden report with a subtle ribbon — the brightest focal 
element. Gentle motion lines along the belt. Upper quarter negative space.
```

---

## 五、生成与替换操作手册

1. **生成顺序**：先封面 → 用它作风格参考图 → 按 ch01→ch20 顺序生成（风格漂移最小）
2. **尺寸**：横幅选 16:9（若工具只支持方图，生成后裁掉上下）；封面选 2:3 竖版
3. **验收标准**（每张图自查三点）：① 是否只有一个最亮焦点 ② 配色是否没有跑出五色体系 ③ 有没有偷偷生成文字/字母
4. **翻车重抽话术**：构图乱了 → 追加 `simpler composition, only 3 main objects`；颜色跑了 → 追加 `strictly only cyan, amber, coral, mint, violet on dark navy`；出现文字 → 追加 `absolutely no text or glyphs anywhere`
5. **替换**：同名覆盖 `assets/images/` 下旧 PNG，然后运行 `./build_pdf.sh` 重新出 PDF 即可
6. **风格迁移备选**：如果等距 3D 风格生成 5 张后仍不满意，备选风格锚定词：`editorial illustration, grainy texture, muted retro-futurism`（《经济学人》插画风）或 `paper-cut craft layers, soft shadows`（剪纸风）——但**全书只能选一种**，切忌混用
