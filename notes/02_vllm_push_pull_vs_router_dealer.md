# 为什么 vLLM 在输出阶段使用 PUSH/PULL，而不是直接复用 ROUTER/DEALER？

## 摘要

在读 vLLM V1 的通信链路时，产生了一个很自然的问题：

> 既然请求接入阶段已经使用了 `ROUTER/DEALER`，而且 `ROUTER` 天然支持基于 identity 的路由，为什么生成结果时不直接沿原通道返回，而要额外引入 `PUSH/PULL`？

我个人一开始是很暴力地认为：vLLM 给 PUSH 加上了 identify 功能.  
但继续往源码和文档里看，会发现更准确的说法是：

> `PUSH/PULL` 本身并没有 `ROUTER` 那样的 identity 路由能力；  
> vLLM 的做法是把请求所属前端的信息提前写进 `client_index`，然后在 Engine 的输出线程里通过 `sockets[client_index]` 把结果发送到对应的前端.[^1]

换句话说，**"identify"这件事不是由 PUSH 协议层完成的，而是由 vLLM 自己维护的一层索引路由完成的.**[^1]

---

## 目录

- [1. 结论先行](#1-结论先行)
- [2. vLLM 的前后端通信结构](#2-vllm-的前后端通信结构)
- [3. PUSH 为什么能“定向”发回正确前端](#3-push-为什么能定向发回正确前端)
- [4. 为什么不是直接沿 ROUTER/DEALER 原路返回](#4-为什么不是直接沿-routerdealer-原路返回)
- [5. 一个可能更准确的理解方式](#5-一个更准确的理解方式)
- [6. 总结](#6-总结)

---

## 1. 结论先行

我目前对这部分的理解可以先压缩成一句话(个人看法)：

> vLLM 把“请求接入”和“结果输出”拆成了两类不同的通信问题：  
> 输入阶段更适合 `ROUTER/DEALER` 这种异步、可寻址的 request-reply 风格；  
> 输出阶段则更像高频、单向、持续的流式下发，因此 vLLM 选择用 `PUSH/PULL` 承担这部分工作.[^2]

但需要先强调一点：

> **不能说 vLLM 让 PUSH 拥有了 identify 功能.(前面只是我第一时间脑海里的想法)**  
> 更准确的说法是：vLLM 自己保存了“这个请求属于哪个前端”的 `client_index`，然后在输出线程中通过 `sockets[client_index]` 选择正确的 PUSH socket 发送.[^1]

---

## 2. vLLM 的前后端通信结构

从 vLLM 官方架构文档来看，API server 负责接收 HTTP 请求、做输入处理，并把结果流式返回给外部客户端；而 Engine Core 负责调度请求、管理 KV Cache，并协调 GPU worker 执行模型计算。两者之间通过 ZMQ 通信.[^2]

继续看 V1 的实现，会发现 vLLM 在分配前后端通信地址时，并不是只创建一条通道，而是会同时创建：

- `inputs`
- `outputs`

而且这两组地址都是按 `num_api_servers` 成批分配的。也就是说，有几个 API server，就会对应生成几组输入/输出地址.[^3]

客户端侧初始化时，也确实不是只开一个 socket：

- 输入侧绑定的是 `zmq.ROUTER`
- 输出侧创建的是 `zmq.PULL`[^1]

而 Engine 侧：

- 输入线程会创建多个 `zmq.DEALER` 连接 input 地址
- 输出线程会创建一个 `PUSH socket` 列表，对应所有 output 地址[^4]

所以，从结构上看，vLLM 一开始就不是“单个双工 socket 既负责接收请求、又负责回传输出”的设计，而是**把入站和出站拆成了两套通道**.[^3]

---

## 3. PUSH 为什么能“定向”发回正确前端

这部分是我觉得最容易误解的地方.

### 3.1 PUSH/PULL 本身并不提供 ROUTER 那种 identity 路由

ZeroMQ 官方文档里，`ROUTER/DEALER` 属于异步 request-reply 模式；而 `PUSH/PULL` 属于单向 pipeline 模式.`ROUTER` 的优势在于它具备显式路由语义，而 `PUSH/PULL` 更像匿名的单向流水线.[^5]

所以，**vLLM 的“定向返回”并不是因为 PUSH 自身知道该发给谁.**

### 3.2 真正负责“定向”的是 client_index

在 `core_client` 里，客户端在提交请求时会显式执行：

`request.client_index = self.client_index`。[^1]

这就说明：  
请求一进入系统，就已经带上了“它来自哪个客户端”的元数据。

接着在 Engine 的输出线程里，`process_output_sockets()` 会从输出队列里取出：

`client_index, outputs`。[^4]

然后它会维护一个：

`sockets = [ ... zmq.PUSH ... for output_path in output_paths ]`。[^4]

也就是说，Engine 并不是把所有输出都塞进同一个 PUSH socket 里，而是先建立一个与前端一一对应的 PUSH 列表。拿到 `client_index` 之后，输出线程就能把结果发送到对应的 socket。[^4]

所以这里更准确的理解应该是：

> PUSH 不负责识别目标前端；  
> vLLM 先通过 `client_index` 完成应用层路由，再通过对应的 PUSH socket 执行发送.[^1]

---

## 4. 为什么不是直接沿 ROUTER/DEALER 原路返回

这里我想分成“源码里可以直接确认的事实”和“我基于流量特征做出的工程理解”两层来说.

### 4.1 可以直接确认的事实

从官方架构文档和源码可以直接确认：

1. API server 和 Engine Core 之间本来就是通过 ZMQ 通信.[^2]  
2. vLLM 专门为 engine-client 通信分配了独立的 `inputs` 和 `outputs` 地址集合.[^3]  
3. 客户端侧输入口是 `ROUTER`，输出口是 `PULL`.[^1]  
4. Engine 侧输入线程使用 `DEALER`，输出线程使用 `PUSH`.[^4]  

这说明 vLLM 的设计不是“懒得复用原通道”，而是**明确地把输入链路和输出链路分开了**。[^3]

### 4.2 我个人的浅薄理解：输入和输出的流量特征极不对称


在大模型服务里，输入和输出的通信形态通常差异非常大：

- **输入（入站）**：频率相对低，但单条消息可能较大，因为要携带 Prompt、采样参数，甚至多模态输入。
- **输出（出站）**：频率非常高，但单条消息通常很小，尤其在流式生成时，Engine 会持续不断地下发增量 token。API server 本身也明确承担“streams results back to clients”的职责。[^2]

如果把这两种流量都压在同一条 `ROUTER/DEALER` 双工通道上，那么一个很自然的工程风险就是：

> 当 GPU 持续高速地产生 token 并不断向外发送时，这条双工链路既要承担高频出站流，又要承担新请求入站流，输入和输出会争抢同一组 socket、缓冲区和调度路径。

这一点不是我在官方文档里看到的原话，而是我根据 vLLM 当前通信拆分方式做出的设计推断。  
也正因为我觉得这个问题真实存在，vLLM 才没有选择“让 ROUTER/DEALER 一把梭”，而是把：

- **复杂的异步接入与 identity 路由** 交给输入侧的 `ROUTER/DEALER`
- **高频的单向流式下发** 交给输出侧的 `PUSH/PULL`[^5]

### 4.3 从通信语义上看，PUSH/PULL 更像“纯下行流水线”

ZeroMQ 官方文档把 `PUSH/PULL` 定义为 pipeline 模式，本质上就是一种单向消息流。相比之下，`ROUTER/DEALER` 更偏向异步 request-reply。[^5]

所以我会把这部分理解为：

> 对于“请求进来”这件事，vLLM 需要的是异步接入、可寻址路由、多 engine 分发能力；  
> 对于“token 返回”这件事，vLLM 更需要的是一条清晰、独立、持续的单向下行通路。

从这个角度看，输出链路使用 `PUSH/PULL` 的原因柳暗花明，符合它要解决的问题。[^5]

---

## 5. 一个可能更准确的理解方式


> vLLM 并不是因为 `PUSH/PULL` 自带 identity 路由能力才使用它；  
> 实际上，`PUSH/PULL` 本身是匿名的单向流水线模式。  
> vLLM 的做法是：在请求进入系统时记录 `client_index`，然后在 Engine 输出线程中通过 `sockets[client_index]` 把结果发送到目标前端。这样一来，输入侧和输出侧就可以分别采用最适合各自流量特征的通信模式。[^5]

我觉得这比“vLLM 让 PUSH 拥有了 identify 功能”更准确，也更符合源码实际。

---

## 6. 总结

这部分如果只保留最核心的几点，我总结成下面三条：

1. **vLLM 的输出不是复用输入通道，而是单独走一套 `PUSH/PULL`。**[^1]  
2. **PUSH 本身不提供 ROUTER 那种 identity 路由能力，真正负责“定向返回”的是 `client_index + sockets[]` 这层应用侧索引。**[^5]  
3. **我个人的理解是：这是一种针对大模型服务中“低频大包入站 + 高频小包出站”流量不对称问题的拆分。** 其中前两点可以直接从源码和文档确认，第三点更像是基于实现方式做出的设计理解。[^2]

---

## 参考资料

[^1]: vLLM Documentation, `core_client` API: `request.client_index = self.client_index` 以及输出返回逻辑  
      <https://docs.vllm.ai/en/stable/api/vllm/v1/engine/core_client/>

[^2]: vLLM Architecture Overview  
      <https://docs.vllm.ai/en/latest/design/arch_overview/>

[^3]: vLLM Documentation, `engine.utils` 中关于 `inputs` / `outputs` 路径分配  
      <https://docs.vllm.ai/en/latest/api/vllm/v1/engine/utils/>

[^4]: vLLM Documentation, `core` 中输入/输出 socket 处理逻辑  
      <https://docs.vllm.ai/en/stable/api/vllm/v1/engine/core/>

[^5]: ZeroMQ Socket API，关于 `ROUTER/DEALER` 与 `PUSH/PULL` 的模式语义  
      <https://zeromq.org/socket-api/>
