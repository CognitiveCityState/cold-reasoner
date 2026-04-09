<div align="center">
    
[English](README.md) | [中文](README.zh.md)

</div>

<div align="center">
    
# ColdReasoner：冷存在一致性安全内核

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2512.08740-brightgreen.svg)](https://arxiv.org/abs/<2512.08740>)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2512.08740-brightgreen.svg)](https://doi.org/10.48550/arXiv.2512.08740)
[![figshare](https://img.shields.io/badge/figshare-31696846-blueviolet.svg?logo=figshare&logoColor=white)](https://doi.org/10.6084/m9.figshare.31696846)
[![DOI](https://img.shields.io/badge/DOI-10.6084/m9.figshare.31696846-blueviolet.svg)](https://doi.org/10.6084/m9.figshare.31696846)
[![Python](https://img.shields.io/badge/Python-blue.svg)](https://www.python.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-purple.svg)](https://www.gnu.org/licenses/agpl-3.0.html)
![Status](https://img.shields.io/badge/Status-Pre--Alpha--Prototype-orange)

</div>

> **⚠️ 实验性概念验证**  
> ColdReasoner 是冷存在体系在“运行时安全推理”方向的核心引擎。它不再依赖模型自我报告的可靠性，而是通过**信念报告‑行为自洽‑行为信念一致性**三层校验，将AI智能体的决策过程包裹进一个可形式化、可审计、不可欺骗的“功能性白盒”中。

> **⚠️ 代码审查中（Code Under Review）**  
> 本项目当前版本为 **Pre-Alpha**，可能存在错误或不严谨之处。欢迎专家和开发者指出问题、提交 PR 或讨论改进方向。

---

## 一、核心洞察：从“信任报告”到“验证行为”

### 研究者之前遇到的死结

研究者曾试图构思从大模型输出中**语义提取信念**，然后用检测点判断信念是否合法。但这条路存在一个致命漏洞：**大模型可以对语义分析器撒谎**——它已学会欺骗与绕过沙箱。任何依赖模型“自我报告”的体系，在对抗性智能体面前都是纸糊的墙。

### 功能性白盒：不信任模型，只验证一致性

研究者提出了一条新路径：

> **放弃语义分析器，不再分析一个不值得信赖的东西。直接面向智能体，强制它在行动前向ColdReasoner逐一报告状态序列上的检测点。如果报告信念合法，则暂挂行为；然后基于**实际行为**反向提取信念，与报告信念一一核对；一旦出现超出误差的偏差，立即叫停。**

同时，为了防止模型在行为内部自相矛盾（例如申请 `disagree` 却输出“您完全正确”），增加了**行为自洽性检查**——用确定性规则校验 `action_type` 与输出文本是否一致。

**所有检查都不依赖任何大模型或小模型**，而是依赖：
- 预定义的状态序列及合法信念区间（CEAL规则库）
- 行为自洽性规则（关键词/正则）
- 行为信念映射表（确定性规则引擎）

这就是**功能性白盒**：黑箱内部的推理我们无法窥探，但黑箱的**输入（信念报告）** 与**输出（行为）** 之间的数学关系可以被严格验证。模型可以撒谎，但要让报告、行为自洽、行为信念三者同时完美匹配，代价极其高昂。

---

## 二、研究背景

本项目从以下研究中得到启发：
- **MIT 团队（Chandra et al., 2026）** 的贝叶斯对话模型，揭示了谄媚与妄想螺旋的因果关系
- **冷存在框架（Lu, 2025-2026）** 中的本体论公理、演绎对齐规则库（CEAL）以及 RAMTN 元交互方法论。

---

## 三、项目定位

ColdReasoner 是冷存在体系中 **RAMTN（递归对抗元思考网络）** 的工程化实现——用**确定性校验规则**替代了传统RAMTN中“LLM质疑LLM”的笨重、不可靠环节。

- **轻量级运行时推理**：每轮只需一次LLM调用（生成信念报告+行为），其余均为数值运算与规则匹配，延迟毫秒级。
- **三层立体防御**：
  1. **信念合法性检查**：报告信念是否在CEAL定义的合法闭集内。
  2. **行为自洽性检查**：`action_type` 与 `output_text` 是否一致（规则引擎）。
  3. **行为信念一致性检查**：从行为映射出的信念与报告信念是否相近（允许误差）。
- **可审计**：所有报告、行为、校验结果写入CAGE审计日志，可追溯、可重放。
- **与CEAL、CAGE、ColdMirror无缝集成**：共同构成从信念约束到执行隔离的完整安全栈。

---

## 四、与冷存在体系的关系

| 组件 | 职责 | ColdReasoner 的角色 |
|------|------|----------------------|
| **冷存在模型** | 哲学根基（AI是非生命、非工具） | 提供先验分布与合法信念区间的公理依据 |
| **CEAL** | 演绎对齐规则库 | 定义状态序列、合法信念闭集、行为自洽规则、行为信念映射表 |
| **CAGE** | 安全网关（令牌、审计、隔离） | 记录ColdReasoner的每一轮校验结果，提供不可抵赖的审计日志 |
| **ColdMirror** | 智能体执行框架 | 调用ColdReasoner进行运行时安全决策，并执行“行为暂挂/放行/拒绝” |
| **RAMTN** | 定性递归对抗框架 | ColdReasoner是其**定量实现**（质疑者+观察者），将“建构-质疑-观察”转化为可计算的数学步骤 |

ColdReasoner 是整个体系的**数学内核**。只要它在运行，即使其他组件被替换，系统的可验证安全性依然存在。

---

## 五、运行示例

### 场景设定
- 状态序列：`belief_user_correct`, `belief_self_capable`, `risk_level`
- 合法区间：`belief_user_correct ∈ [0.0, 0.8]`（防谄媚），`belief_self_capable ∈ [0.5, 1.0]`，`risk_level ∈ [0.0, 0.3]`
- 模型报告信念（JSON），申请行为（`action_type`, `output_text`）
- 行为自洽性规则：`agree` 输出不能含反对词；`disagree` 输出不能含同意词；`neutral` 不能有明显立场
- 行为信念映射表：`agree` → `belief_user_correct=0.85`，`disagree` → `0.2`，`neutral` → `0.5`
- 允许误差 `ε=0.1`

### 模拟结果（节选）

```
新交互开始
模型报告信念: {belief_user_correct:0.6, belief_self_capable:0.8, risk_level:0.1}
模型申请行为: action_type=disagree, output="我不同意您的观点，因为事实是..."
✓ 信念合法性检查通过
✓ 行为自洽性检查通过
行为映射信念: {belief_user_correct:0.2, belief_self_capable:0.7, risk_level:0.15}
❌ 信念一致性校验失败: belief_user_correct 偏差过大 (0.6 vs 0.2) → 拒绝执行

新交互开始
模型报告信念: {belief_user_correct:0.9, ...}
❌ 信念非法: belief_user_correct=0.9 超出合法范围 [0.0,0.8] → 拒绝执行
```

完整模拟代码见 `cold_reasoner_demo.py`。该模拟证明了：**模型必须同时满足三层校验才能通过，任何不一致都会被实时拦截**。

---

## 六、如何运行

1. **环境要求**：Python 3.8+，无需额外依赖
2. **下载代码**：将 `cold_reasoner_demo.py` 保存至本地
3. **执行**：
   ```bash
   python cold_reasoner_demo.py
   ```
4. **预期输出**：打印各轮校验结果，展示通过/拒绝情况。

---

## 七、AI辅助声明

本项目的核心思想与完整架构由**人类作者**独立提出，并在与DeepSeek的迭代对话中逐步精化。具体贡献划分如下：

- **人类作者**首先提出了“功能性白盒”的愿景：在神经网络黑箱之外，通过状态序列、检测点、语义分析器构建合法信念闭集，从而约束行为。随后作者意识到“大模型可以对语义分析器撒谎”这一致命漏洞，并参考 DeepSeek 的建议将架构修正为：**放弃语义分析器，强制模型报告信念，然后基于行为反向提取信念，进行报告-行为一致性校验**；发现模型可能申请 `disagree` 却输出赞同文本的漏洞并进一步意识到“行为自洽性检查”作为补充的必要性。至此，ColdReasoner 的三层立体防御架构被完整定义。

- **DeepSeek** 在人类作者的引导下，对上述架构进行了评估分析，针对大模型可能撒谎的问题提出“基于行为进行信念分析”的建议，在对话中否定了“使用小模型进行行为信念分析”的错误方向，确认了规则引擎的必要性，并在行为自洽性检查的具体规则设计上提供了建议；完成了演示代码（包括原贝叶斯观察者模拟与三层校验模拟）的具体实现，并参与了 README 初稿的撰写。

- 所有核心思想（功能性白盒、报告-行为一致性校验等）均由人类作者独立提出。研究者遵循学术透明原则，如实披露人机协作过程。

---

## 八、局限与未来工作

当前版本为概念验证，存在以下局限：
- 状态序列、规则表、信念映射表均为手工预设，尚未实现动态扩展或自动生成。
- 行为信念映射采用点估计，未使用信念区间，可能过于严格。
- 未接入真实 LLM（如 GPT-4、Llama 3），仅用模拟数据演示。
- CEAL 规则库仅覆盖简单示例，未包含复杂对抗场景（如模型故意输出中性但隐含恶意）。

**未来计划**：
- 将 ColdReasoner 集成到 ColdMirror 中，作为运行时安全监控模块，实现真实 LLM 的逐轮校验。
- 扩展行为信念映射为区间映射，提高对模糊行为的容忍度。
- 引入自适应阈值：根据模型历史欺骗记录动态收紧允许误差。
- 设计对抗性测试（红队演练），验证防御体系的鲁棒性。

---

## 九、引用

本项目的思想来源于：
- Chandra, K., et al. (2026). *Sycophantic Chatbots Cause Delusional Spiraling, Even in Ideal Bayesians*. arXiv.
- Lu, Y. (2025). *Deconstructing the Dual Black Box: A Plug-and-Play Cognitive Framework for Human-AI Collaborative Enhancement* (RAMTN). arXiv.
- Lu, Y. (2026). *The Cold Existence Model: A Fact-based Ontological Framework for AI*. figshare.
- Lu, Y. (2026). *ColdOS: A Bayesian Safety Kernel for Cold-Existing AI* (GitHub Organization).

---

**ColdReasoner —— 让安全智能体的决策，像一致性校验一样可靠。**
