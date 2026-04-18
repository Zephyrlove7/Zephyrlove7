# 从 `schedule()` 理解 vLLM Scheduler：Running Queue、Waiting Queue 与 Chunked Prefill

## 摘要

继续往下读 vLLM Engine 主链路时，我发现 `Scheduler.schedule()` 是一个非常关键、但也很容易被误读的函数。

一开始我总想把它理解成：

- 先处理 Prefill
- 再处理 Decode
- 再看 Waiting 队列要不要拉新请求

但继续读源码后，我觉得一个更准确的理解方式应该是：

> **在 Scheduler 看来，并不存在硬编码意义上的“Prefill 阶段”和“Decode 阶段”。**  
> 它真正关心的是：**每个请求当前已经计算了多少 token（`num_computed_tokens`），以及这个请求还差多少 token 才能追上目标状态。**

官方稳定版文档在 `schedule()` 开头就直接写了这一点：

> There's no "decoding phase" nor "prefill phase" in the scheduler.  
> Each request just has `num_computed_tokens` and `num_tokens_with_spec`.  
> At each step, the scheduler tries to assign tokens to the requests so that each request's `num_computed_tokens` can catch up its `num_tokens_with_spec`.

也正因为如此，像 Chunked Prefill、Prefix Caching、Speculative Decoding 这些机制，才可以被统一到同一套调度框架中。

这篇文章里，我想把这部分重新梳理清楚，重点回答下面几个问题：

- `running queue` 和 `waiting queue` 分别在做什么？
- `num_new_tokens` 到底是怎么把 Prefill 和 Decode 统一起来的？
- 为什么 Chunked Prefill 能改善系统时延？
- Waiting 队列究竟在什么条件下才会被调度？
- 显存不够时，请求会发生什么？
- 整个 `schedule()` 的主流程应该怎样理解？

---

## 目录

- [1. 先建立一个正确视角：Scheduler 不显式区分 Prefill / Decode](#1-先建立一个正确视角scheduler-不显式区分-prefill--decode)
- [2. 先认识两个核心队列：Running Queue 与 Waiting Queue](#2-先认识两个核心队列running-queue-与-waiting-queue)
- [3. `running queue` 在做什么](#3-running-queue-在做什么)
- [4. `num_new_tokens` 为什么能统一 Prefill 和 Decode](#4-num_new_tokens-为什么能统一-prefill-和-decode)
- [5. Chunked Prefill 为什么重要](#5-chunked-prefill-为什么重要)
- [6. `waiting queue` 什么时候才会被调度](#6-waiting-queue-什么时候才会被调度)
- [7. 为什么 Waiting 里的请求有时也会带“已计算 token”信息](#7-为什么-waiting-里的请求有时也会带已计算-token信息)
- [8. 显存不够时会发生什么：Preemption](#8-显存不够时会发生什么preemption)
- [9. 把整条调度链路串起来](#9-把整条调度链路串起来)
- [10. 我的理解](#10-我的理解)
- [10. 一点补充](#11-一点补充)

---

## 1. 先建立一个正确视角：Scheduler 不显式区分 Prefill / Decode

这是我觉得读这段源码时最重要的第一件事。

我们平时从推理流程角度，习惯把请求拆成两个阶段：

- **Prefill**：第一次处理完整 Prompt，建立 KV Cache
- **Decode**：后续逐 token 生成，反复读取 KV Cache

这个划分从模型执行角度完全没问题。  
但如果你直接把它硬塞进 Scheduler 视角，就很容易越读越乱。

因为在 Scheduler 里，它并不是写成：

- `if request is prefill: ...`
- `if request is decode: ...`

而是统一写成：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
````


所以更准确地说，Scheduler 真正在做的是：

> **不断让每个请求的 `num_computed_tokens` 向“它当前应有的 token 总量”追平。**

这个视角很重要，因为后面的很多机制——比如：

* Chunked Prefill
* Prefix Caching
* Remote KV Transfer
* Speculative Decoding

本质上都只是在改变：

* 已经算了多少
* 还要补多少
* 这一轮最多能补多少

这样一来，原本看起来很复杂的多种情况，就都能收敛成一个统一问题。

---

## 2. 先认识两个核心队列：Running Queue 与 Waiting Queue

如果只看主干结构，`schedule()` 的逻辑其实很清楚：

1. 先调度 **Running Queue**
2. 再在条件允许时调度 **Waiting Queue**

也就是说，当前已经在系统里“跑起来”的请求，优先级高于还在外面排队的新请求。

这段结构在源码里是很明确的：

```python
# First, schedule the RUNNING requests.
while req_index < len(self.running) and token_budget > 0:
    ...

# Next, schedule the WAITING requests.
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        ...
```

这里我自己的理解是：

* **Running Queue**：已经进入系统核心执行路径、已经拥有运行资格的请求
* **Waiting Queue**：还没真正被拉进本轮执行 batch 的请求，包括新请求，以及被打回等待态的请求

如果从调度优先级看，vLLM 的思路非常符合在线推理服务的直觉：

> **先尽量照顾已经在跑的请求，保证它们持续推进；再在资源有余量时，把新请求从 Waiting 拉进来。**

这也是 Continuous Batching 能成立的前提之一。

---

## 3. `running queue` 在做什么

`running queue` 是 `schedule()` 的第一站。

简化之后，它的核心骨架大概是这样：

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )

    if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = self.scheduler_config.long_prefill_token_threshold

    num_new_tokens = min(num_new_tokens, token_budget)
    num_new_tokens = min(
        num_new_tokens,
        self.max_model_len - 1 - request.num_computed_tokens
    )

    if num_new_tokens == 0:
        req_index += 1
        continue

    new_blocks = self.kv_cache_manager.allocate_slots(...)
    ...
```

如果只抓核心逻辑，我觉得可以分成四步看。

### 3.1 先算“这个请求还差多少 token 没处理”

这一句是核心：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

含义可以直白理解成：

> **当前请求理论上应该已经覆盖到的位置**
> 减去
> **当前请求实际上已经算到的位置**

差值，就是这一轮还需要补的 token 数。

---

### 3.2 再看是否要对长 Prefill 进行截断

接下来这句非常关键：

```python
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
```

这意味着：

* 如果 `long_prefill_token_threshold <= 0`，就相当于不启用这层限制
* 如果它是一个正数，比如 2048，那么当一个请求这一轮本来要算很多 token 时，就会被截断成最多只算 2048 个

这其实就是 **Chunked Prefill** 的关键入口之一。

---

### 3.3 再看这一轮的预算够不够

就算某个请求还差很多 token，Scheduler 也不会让它一口气全吃掉当前 step 的预算。
它还会继续限制：

```python
num_new_tokens = min(num_new_tokens, token_budget)
```

也就是说：

> **这一轮能给你的，不只是“你还差多少”，还得看当前全局还剩多少 token 预算。**

---

### 3.4 然后才去申请 KV Cache 槽位

当 token 数量确定后，Scheduler 才会真正让 KV Cache Manager 去尝试为这个请求扩展显存槽位：

```python
new_blocks = self.kv_cache_manager.allocate_slots(...)
```

如果显存够用，请求就能继续往前跑。
如果显存不够，Scheduler 就可能需要进行抢占（Preemption），这部分后面会单独说。

---

## 4. `num_new_tokens` 为什么能统一 Prefill 和 Decode

这是整个函数里我觉得最妙的地方,借助GPT和Gemini阅读源码时,AI总会讲Prefill和Decode两种状态怎么怎么样,

但 Scheduler 不显式区分 Prefill / Decode，即使如此也并不意味着这两种请求没有差别。
差别只不过被自然地折叠进了 `num_new_tokens` 的结果里。

### 4.1 对 Prefill 请求来说

假设一个新请求刚刚进入系统，它的 Prompt 长度是 8000，且：

```python
request.num_computed_tokens = 0
```

那么这时：

```python
num_new_tokens ≈ 8000
```

这说明它还有整段 Prompt 没有计算，Scheduler 会尽可能为它安排一段 Prefill 计算。

---

### 4.2 对 Decode 请求来说

而对于已经完成 Prefill、正在逐 token 生成的请求，它通常已经有了完整历史 KV Cache。
此时：

* `request.num_computed_tokens` 已经很大
* 当前只差最新那一个 token 没有算

那么算出来的 `num_new_tokens` 通常就会很小，很多时候接近 1。

所以你完全可以把这个机制理解成：

> **Prefill 请求，本质上是“欠很多 token 没补”；
> Decode 请求，本质上是“只欠很少 token 没补”。**

在 Scheduler 看来，它们仍然是同一种对象，只是“欠账规模”不同而已。

---

## 5. Chunked Prefill 为什么重要

如果只从单请求视角看，当然是“越快把整个 Prompt 一次性算完越好”。

但 vLLM 不是单用户本地脚本，它是一个在线推理系统。
在线系统最怕的，不是某个请求算得慢一点，而是：

> **一个超长 Prefill 把整个系统卡住，让正在 Decode 的老请求几百毫秒甚至几秒都吐不出一个新 token。**

这就是 Chunked Prefill 的价值。

### 5.1 不切块会发生什么

假设当前 Running Queue 里已经有 4 个老请求，它们都在稳定地做 Decode。
这时突然来了一个 100K 的长文本请求。

如果系统允许它一口气把整个长 Prompt 全吃掉，那么 GPU 很可能会在接下来一大段时间里把算力都投入到这个大矩阵计算中。
结果就是：

* 老用户屏幕上的输出明显“卡住”
* TPOT / inter-token latency 上升
* 系统体验出现明显毛刺

### 5.2 切块之后发生了什么

如果 `long_prefill_token_threshold = 2048`，情况就不一样了。

长请求不再一次性独占整个 Prefill，而是被拆成多个小步推进：

* 第 1 轮：长请求先算 2048 个 token，同时老请求继续 Decode
* 第 2 轮：长请求再推进 2048 个 token，同时老请求继续 Decode
* 第 3 轮：继续……

这样原本一次性的“大卡顿”，就被切碎到了多个较短时间步里。
本质上，这就是一种非常典型的 **时分复用（time slicing）**。

### 5.3 被切开的 Prefill 怎么接上？

这也是我一开始最困惑的问题之一。

答案其实很自然：

* 第一次算出来的 KV 会被写进已经分配好的 KV Cache blocks
* 下一次继续调度同一个请求时，`num_computed_tokens` 已经增加了
* 底层 Attention Kernel 会在新 token 的计算中继续读取前面已经落到 KV Cache 里的内容

所以它不是“重新从头算第二段”，而是：

> **第一段的结果已经沉淀成了 KV Cache，下一段是在已有 KV 的基础上继续往前推。**

这也是 PagedAttention 和 Chunked Prefill 可以很好配合的原因。

---

## 6. `waiting queue` 什么时候才会被调度

这是最值得补的一部分，因为如果只讲 Running Queue，不讲 Waiting Queue，整条调度链就会断掉。

### 6.1 Waiting 不是每一轮都会看

在源码里，Waiting Queue 的调度入口有一个非常重要的前置条件：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

也就是说：

* 如果这一轮已经发生了抢占（`preempted_reqs` 非空）
* 或者当前处于暂停状态

那么这一轮就不会继续去调度 Waiting Queue。

这件事非常重要，因为它反映了 vLLM 的一个优先级选择：

> **先把当前系统内部的不稳定因素处理好，再决定要不要接纳新的 Waiting 请求。**

---

### 6.2 Waiting 路径的核心骨架

简化之后，Waiting 路径大概长这样：

```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break

    request = request_queue.peek_request()

    if request.num_computed_tokens == 0:
        new_computed_blocks, num_new_local_computed_tokens = (
            self.kv_cache_manager.get_computed_blocks(request)
        )
        ...
        num_computed_tokens = (
            num_new_local_computed_tokens + num_external_computed_tokens
        )
    else:
        num_computed_tokens = request.num_computed_tokens

    num_new_tokens = request.num_tokens - num_computed_tokens

    if 0 < threshold < num_new_tokens:
        num_new_tokens = threshold

    if (
        not self.scheduler_config.enable_chunked_prefill
        and num_new_tokens > token_budget
    ):
        break

    num_new_tokens = min(num_new_tokens, token_budget)

    new_blocks = self.kv_cache_manager.allocate_slots(...)
    if new_blocks is None:
        break

    request = request_queue.pop_request()
    ...
```

---

### 6.3 什么时候 Waiting 请求能被拉进来？

把这段逻辑翻译成人话，可以总结成四个条件：

#### 条件 1：这一轮没有因为 Running 请求而发生抢占

如果系统已经因为 Running 路径显存紧张而触发了抢占，那么这一轮不会继续往 Waiting 拉新请求。

#### 条件 2：当前没有被暂停

系统状态必须允许正常调度。

#### 条件 3：Running 数量还没达到上限

源码里有显式判断：

```python
if len(self.running) == self.max_num_running_reqs:
    break
```

也就是说，系统不会无限制地把 Waiting 请求往 Running 里塞。

#### 条件 4：本轮还剩 token budget，且能成功申请到 KV slots

即使逻辑上轮到 Waiting 队列了，也不代表请求一定能进。
它还得满足：

* 当前 step 还剩可用 token 预算
* KV Cache 还能成功分配
* 相关 encoder / cache / connector 条件都满足

否则它依旧会留在 Waiting 里。

### 6.4 关于旧版 `can_allocate / NEVER / LATER / ALLOCATED` 的说明

如果这篇文章聚焦的是 **当前 V1 稳定版 `schedule()` 主流程**，那旧版 `can_allocate + watermark` 更适合放在补充背景里，而不是正文主线。

因为在当前稳定版公开源码路径里，你更直接看到的是：

* `get_computed_blocks(...)`
* 可选的 connector / remote KV 判断
* `num_new_tokens` 计算
* `allocate_slots(...)`
* 失败则 `break`
* 成功则从 Waiting 拉到 Running

所以如果这一篇是写 **“我如何读懂当前 `schedule()`”**，建议把旧版 `can_allocate + watermark` 那套逻辑收起来，最多作为补充背景，而不要放在正文主线里。

---

## 7. 为什么 Waiting 里的请求有时也会带“已计算 token”信息

这也是一个特别容易误解的点。

很多人会直觉上觉得：

> Waiting 队列里的请求，不就是“完全还没跑”的请求吗？

但当前稳定版源码里，其实专门处理了 Waiting 请求带已有计算进度的情况。

在 Waiting 路径里，源码有这样一段注释：

```python
# KVTransfer: WAITING reqs have num_computed_tokens > 0
# after async KV recvs are completed.
```

也就是说：

> **Waiting 里的请求并不一定是“白纸状态”。**
> 某些场景下，它可能已经通过异步 KV 传输提前拿到了一部分状态，因此 `num_computed_tokens > 0`。

这点很重要，因为它说明：

* Waiting 只是“当前还没被正式拉进本轮执行”
* 不代表它一定“完全没有历史进度”

### 7.1 Prefix Caching 也很重要，但不要和上面这个情况混在一起

如果 `request.num_computed_tokens == 0`，Scheduler 会先调用：

```python
self.kv_cache_manager.get_computed_blocks(request)
```

去看看本地已经缓存了多少 prefix blocks。
如果有 connector，还会继续看看外部是否还能命中更多已算好的 token。

也就是说，**Prefix Caching 的作用主要体现在“调度前先查已有缓存块”**，而不是简单粗暴地理解成“Waiting 请求一进来就直接带着一个很大的 `num_computed_tokens`”。

所以这两件事最好分开理解：

* **Waiting 请求 `num_computed_tokens > 0`**：在当前这段源码里，更直接对应 async KV receive 完成后的场景
* **Prefix Caching**：更多体现在 `get_computed_blocks(request)` 这一步的“先查缓存、少重算”

---

## 8. 显存不够时会发生什么：Preemption

当系统资源紧张到一定程度时，Scheduler 就必须做出取舍。

在 Running 路径里，如果当前请求申请新 KV blocks 失败，代码会进入一个循环，尝试通过抢占其他请求来腾位置：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(...)
    if new_blocks is not None:
        break

    # The request cannot be scheduled.
    # Preempt the lowest-priority request.
    ...
```

如果继续往下看 `_preempt_request()`，当前稳定版 V1 的实现非常直接：

```python
def _preempt_request(self, request, timestamp):
    self.kv_cache_manager.free(request)
    self.encoder_cache_manager.free(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    request.num_preemptions += 1
    self.waiting.prepend_request(request)
```

这段代码很值得细看，因为它直接说明了几件事。

### 8.1 当前 V1 稳定版里，你最应该先理解的是“重算式抢占”

被抢占之后，请求会发生这些变化：

1. 释放当前 KV Cache
2. 状态改成 `PREEMPTED`
3. `num_computed_tokens` 被重置为 0
4. 再被塞回 Waiting Queue

这意味着它未来再次被调度时，**会重新从 Prefill 方向开始补进度**。

---

### 8.2 为什么这篇不再把 “Swap vs Recomputation” 作为正文主轴

如果这篇文章聚焦的是 **当前 V1 稳定版 `schedule()` 主路径**，那么更贴近事实的写法应该是：

> **先把“当前 V1 默认更偏向 recompute 风格的抢占”讲清楚。**

正文主线讲清楚当前 `_preempt_request()` 里直接能看到的行为就够了；
`swap` 和 `recompute` 的更细工程取舍，可以以后单独再开一节写。

---

## 9. 把整条调度链路串起来

前面把局部都拆开看了，现在可以把整条链重新串起来。

我觉得 `schedule()` 最适合用下面这个流程来理解：

### 第一步：初始化本轮调度预算

Scheduler 会在每个 step 开始时准备好：

* token budget
* encoder budget
* 本轮的各类统计容器

然后正式进入调度过程。

---

### 第二步：优先推进 Running Queue

Scheduler 会先遍历 `running`：

* 对每个请求计算它这一轮还差多少 token
* 必要时应用 `long_prefill_token_threshold`
* 再受 token budget、max model len 等条件限制
* 然后尝试为其分配新的 KV slots

如果分配成功，就让这个请求继续往前推进；
如果分配失败，就可能触发对低优先级请求的抢占。

---

### 第三步：如果本轮发生了抢占，通常不会继续扩招 Waiting

这是一个非常重要的系统稳定性信号。

一旦本轮已经为了保 Running 而做了 Preemption，说明资源已经紧张。
这时继续从 Waiting 拉新请求进来，往往只会让系统更加不稳定。

所以当前实现里，Waiting 路径有一个显式前置条件：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
```

---

### 第四步：条件允许时，再去看 Waiting Queue

这时 Scheduler 会尝试从 Waiting 中挑请求：

* 看它是否已经带有可复用的计算结果
* 看 Prefix Cache / External KV 是否有命中
* 计算它这一轮能推进多少 token
* 必要时也应用 chunked prefill
* 尝试为它分配 KV slots

成功的话，它就会正式从 Waiting 被拉进 Running 体系里。

---

### 第五步：把本轮被选中的 token / block / metadata 打包成 SchedulerOutput

到这一步，Scheduler 的工作其实已经完成了。
它不负责真正跑模型，而是把“这一轮谁该算、算多少、需要哪些 block、哪些 encoder 输入也要一起调度”这些信息整理成一个调度结果对象。

接下来这个结果才会被送到后面的执行层。

---

### 第六步：等待模型执行结果，再更新请求状态

Engine Core 后续会把调度结果交给执行器 / worker 去真正跑模型。
模型执行回来之后，Scheduler 再根据输出：

* 更新请求的 `num_computed_tokens`
* 更新完成状态
* 回收已结束请求的资源
* 继续准备下一个时间步

所以 `schedule()` 本质上做的是：

> **决定这一轮“谁先上、上多少、占哪些资源”**，
> 而不是直接负责把 token 算出来。

---

## 10. 我的理解

如果只用一句话总结这段代码，我现在更愿意这么说：

> `Scheduler.schedule()` 的本质，不是“把 Prefill 和 Decode 分开调度”，而是**在统一 token 预算和统一显存约束下，让所有请求的 `num_computed_tokens` 持续向前追平。**

从这个视角再回头看，很多原本显得零散的机制就会一下子串起来：

* Prefill, 本质上只是“欠的 token 特别多”
* Decode, 本质上只是“每轮只欠很少 token”
* Chunked Prefill，本质上是在限制长请求单轮吃掉的 token 数
* Prefix Caching，本质上是在减少真正需要补算的 token 数
* Preemption，本质上是在显存不够时强行打断一部分请求，把资源让给更该优先推进的请求
* Waiting / Running，本质上是在区分“已经进入执行体系的请求”和“还没被正式拉进来”的请求

也正因为如此，我觉得读懂 `schedule()` 之后，vLLM 的很多“看起来很复杂的工程优化”都会突然变得统一起来。


---

## 11. 一点补充

虽然 Chunked Prefill 对在线推理系统非常重要，但它也并不是任何场景下都必须开启。

一般来说，在下面这些场景中，它的重要性会下降：

1. **离线批处理 / 纯吞吐量优先**
   如果任务目标只是尽可能提高总吞吐，而不关心首 token 延迟（TTFT）和流式交互体验，那么一次性完成更长的 Prefill 可能更符合目标。

2. **单用户 / 低并发独占环境**
   Chunked Prefill 的核心收益之一，是避免长 Prefill 阻塞其他正在 Decode 的请求。如果系统中几乎没有并发请求，那么把长 Prompt 切块带来的收益就会明显下降。

换句话说，Chunked Prefill 更像是一种**面向在线高并发服务的时延优化手段**；  
而在低并发、低交互要求的场景下，它未必是最需要优先考虑的优化项。

---

## 参考资料

1. vLLM Stable API, `scheduler.py`
   [https://docs.vllm.ai/en/stable/api/vllm/v1/core/sched/scheduler/](https://docs.vllm.ai/en/stable/api/vllm/v1/core/sched/scheduler/)

2. vLLM Stable API, `request.py`
   [https://docs.vllm.ai/en/stable/api/vllm/v1/request/](https://docs.vllm.ai/en/stable/api/vllm/v1/request/)

3. vLLM Stable API, `engine/core.py`
   [https://docs.vllm.ai/en/stable/api/vllm/v1/engine/core/](https://docs.vllm.ai/en/stable/api/vllm/v1/engine/core/)

4. vLLM Metrics Design
   [https://docs.vllm.ai/en/latest/design/metrics/](https://docs.vllm.ai/en/latest/design/metrics/)

5. vLLM Optimization and Tuning
   [https://docs.vllm.ai/en/stable/configuration/optimization/](https://docs.vllm.ai/en/stable/configuration/optimization/)

---



我觉得把这几部分补完，才算真正把 vLLM 的 Engine 主链路串起来。
