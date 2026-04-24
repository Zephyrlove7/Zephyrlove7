# 你好，我是 Zephyrlove 👋

计算机专业本科在读，当前主要聚焦 **AI Infra / LLM 推理优化 / vLLM / 高并发推理服务**。  
希望持续积累与 **推理引擎、模型部署、服务性能分析、CUDA / GPU 基础** 相关的工程经验。

---

## 当前方向

- 本地 LLM 推理服务重构与部署
- vLLM / OpenAI-compatible serving
- TTFT / TPOT / Throughput / 接口级 benchmark
- 高并发推理服务与应用层性能定位
- CUDA / GPU 基础与推理加速相关知识学习

---

## 精选项目

### 1. Tour Agent Backend with local vLLM serving
基于 **FastAPI + LangChain/LangGraph** 的旅游智能体后端，将原有云端 API 调用链重构为 **本地 vLLM OpenAI-compatible serving**。

**项目亮点**
- 将云端模型调用替换为本地 vLLM 推理服务
- 保持 `/chat` 接口与角色化 Prompt 风格兼容
- 对 Serving 层与应用层分别进行了 benchmark

**Benchmark Snapshot**
- Avg TTFT: **0.16s**
- Avg TPOT: **0.0060s**
- Throughput (gen): **166.91 tok/s**
- 4-concurrency throughput: **575.93 tok/s**

**当前结论**
- Serving 层性能已验证
- 应用层并发稳定性仍有继续优化空间

🔗 项目链接：  
[github.com/Zephyrlove7/tour-agent-vllm](https://github.com/Zephyrlove7/tour-agent-vllm)

---

## 学习 / 笔记

目前持续整理与以下方向相关的笔记：

- vLLM 源码
- Engine / Worker / Executor
- Scheduler / ModelRunner
- PagedAttention
- 分布式推理基础
- CUDA / GPU 基础
- 推理服务性能分析

📒 笔记目录：  
[`notes`](./notes)

---

## 技术栈

**语言**
- Python
- C++
- SQL

**推理 / 后端 / AI Infra**
- vLLM
- FastAPI
- LangChain
- LangGraph
- PyTorch
- AWQ

**当前学习中**
- CUDA
- GPU 架构基础
- 分布式推理
- FlashAttention
- 高并发服务优化

---

## 近期在做

- 重构 Agent 后端到本地 vLLM serving
- 做 TTFT / TPOT / Throughput / `/chat` benchmark
- 学习 vLLM 推理链路与核心组件
- 补 CUDA / GPU / 推理优化相关基础

---

## 联系方式

- GitHub: [@Zephyrlove7](https://github.com/Zephyrlove7)
- Email: 你的邮箱
