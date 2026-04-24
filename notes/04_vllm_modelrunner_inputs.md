这是一份为您重新排版、校正并精修后的 Markdown 文档。我补充了缺失的代码块，优化了列表结构的层级，并使用引用块（`>`）和加粗语法突出了核心结论，使其完全对齐您往期的高质量技术长文风格。

-----

# 从 `_update_states` 到 `compute_slot_mapping()`：理解 vLLM ModelRunner V1 的输入准备

## 摘要

这一节我想先把 ModelRunner 的“输入准备”主链路梳理清楚。

如果直接从 `prepare_input_tensors()` 或 `_prepare_inputs()` 开始看，很容易有一种错觉：这些 token、位置、块表、slot mapping 好像都是凭空冒出来的。

但更自然的顺序其实是：

1.  先看 `_update_states()` 如何把 Scheduler 的结果同步成 ModelRunner 内部状态。
2.  再看 `requests` 和 `input_batch` 这两层数据结构分别保存了什么。
3.  接着看 `prepare_input_tensors()` 如何把这些状态翻译成真正送往模型的输入。
4.  最后再解释 `compute_slot_mapping()`：一个 token 到底会被写入哪个物理 slot。

也就是说，这一节真正要回答的问题不是单纯的“怎么把 token 展平”，而是：

> **Scheduler 已经决定了这一轮谁该跑、给了哪些块；ModelRunner 到底是怎样把这些调度结果，翻译成 GPU 真正可以执行的输入与地址映射的？**

-----

## 目录
## 目录

- [1. 为什么在讲 ModelRunner 输入前，要先看 `_update_states`](#1-为什么在讲-modelrunner-输入前要先看-_update_states)
- [2. `requests`：ModelRunner 持有的请求状态总表](#2-requestsmodelrunner-持有的请求状态总表)
- [3. `input_batch`：持续存在的 persistent batch](#3-input_batch持续存在的-persistent-batch)
- [4. `scheduled_req_ids` 与 `cached_req_ids`：一轮 step 的集合运算入口](#4-scheduled_req_ids-与-cached_req_ids一轮-step-的集合运算入口)
- [5. `_update_states` 如何增删改请求状态](#5-_update_states-如何增删改请求状态)
- [6. `_prepare_inputs / prepare_input_tensors`：真正翻译成模型输入](#6-_prepare_inputs--prepare_input_tensors真正翻译成模型输入)
- [7. `prepare_input_tensors` 的第一层价值：Padding-free](#7-prepare_input_tensors-的第一层价值padding-free)
- [8. 为什么 1D / 2D 输入仍然能被线性层正确处理](#8-为什么-1d--2d-输入仍然能被线性层正确处理)
- [9. 除了展平 token，ModelRunner 还要补齐什么信息](#9-除了展平-tokenmodelrunner-还要补齐什么信息)
- [10. `compute_slot_mapping()`：把 token 翻译成“绝对物理槽位”](#10-compute_slot_mapping把-token-翻译成绝对物理槽位)
- [11. 用一个具体例子跑通 `compute_slot_mapping()`](#11-用一个具体例子跑通-compute_slot_mapping)
- [12. 小结](#12-小结)

-----

## 1\. 为什么在讲 ModelRunner 输入前，要先看 `_update_states`

我第一次顺 ModelRunner 这段流程时，其实把 `_update_states()` 这个函数漏掉了。

当时我是直接从 `prepare_input_tensors()` / `_prepare_inputs()` 开始看的，结果越看越别扭，总觉得这些 token、metadata、block table 好像是凭空冒出来的。  
后来回头查源码，才发现 ModelRunner 在真正准备输入之前，先调用了 `self._update_states()`。

我当时对这个函数最直接的理解其实很朴素：**它就是一个“抄表”功能。**

因为 CPU 侧的调度器已经在上一个阶段决定好了：

- 这一轮哪些请求要继续执行
- 每个请求已经算到哪里
- 哪些新 block 被分给了哪些请求

但这些结果还只是调度器视角下的安排。  
ModelRunner 在执行 Executor 发来的 `ExecuteModelRequest` 时，必须先把这份“显存分配方案”和“请求状态变化”同步到自己这一侧的内部结构里，不然后面的 `_prepare_inputs()` 根本不知道该基于哪一份 batch 状态去构造输入。

也正因为如此，ModelRunner 在真正准备输入前，先调用了 `self._update_states()`。  
从这个角度看，这个函数更像是一个**从 Scheduler 到 ModelRunner 的状态映射层**：先把表抄对，后面的 token 展平、位置生成、slot mapping 计算才有意义。


从执行链路上看，ModelRunner 这里更自然的顺序其实不是“先展平 token，再回头解释这些 token 从哪来”，而应该是：

  - 先让 `_update_states()` 把 Scheduler 的结果“落地”为 runner 内部状态。
  - 再由 `_prepare_inputs()` 把这些状态翻译成真正送往模型的张量。

换句话说，`_update_states()` 更像一个**状态同步层**。它的位置在 Scheduler 和 ModelRunner 之间，把“这一轮谁该跑、跑多少”的调度结果，转成“当前 batch 里到底有哪些请求、每个请求走到哪一步、persistent batch 里应该保留哪些行”的 runner 侧现实状态。

所以，想看懂 ModelRunner 的输入准备，最自然的入口不是 `prepare_input_tensors()`，而是 `_update_states()`。


-----

## 2\. `requests`：ModelRunner 持有的请求状态总表

在当前实现里，ModelRunner 自己维护了一张 `self.requests`。可以把它理解成：

```python
req_id -> CachedRequestState
```

也就是说，`requests` 是一张按请求 ID 组织的状态总表。它存的不是“历史上来过哪些请求”，而是**当前还活着、后续还可能继续参与执行的请求状态**。

对新请求来说，ModelRunner 会显式构造一个 `CachedRequestState`。它保存的并不只是 token 本身，而是一整份请求在 runner 侧继续往前执行所需的运行态，例如：

  - `req_id`
  - `prompt_token_ids`
  - `prompt_embeds`
  - `mm_features`
  - `sampling_params` / `pooling_params`
  - `generator`
  - `block_ids`
  - `num_computed_tokens` / `output_token_ids`
  - `lora_request`

如果要给 `requests` 下一个最简洁的定义：

> `requests` 是 ModelRunner 视角下的请求状态总表；它保存的是“每个请求当前已经演化成什么样”，而不是“这一轮临时要算什么”。

-----

## 3\. `input_batch`：持续存在的 persistent batch

如果说 `requests` 更像“请求档案库”，那么 `input_batch` 更像**当前挂在执行链上的 persistent batch（持久批处理）**。

它是一套以 batch 行索引为中心组织起来的持久状态容器。它内部维护的内容通常包括：

  - `req_id_to_index`：每一行对应的 `req_ids`
  - `token_ids_cpu`
  - `num_prompt_tokens`
  - `req_output_token_ids`
  - `spec_token_ids`
  - `sampling_metadata`

以及一整套和采样、LoRA、pooling、logits processor 相关的 batch 级状态。

所以这两层结构一定要分清：

  - **`requests`**：按 `req_id` 保存**请求级**运行态。
  - **`input_batch`**：按 batch row 保存 **batch 级**持久状态。

这个区别如果没讲清，后面看到 `req_id_to_index`、`remove_request()`、`condense()`、`refresh_metadata()` 这些操作时就很容易混掉。

-----

## 4\. `scheduled_req_ids` 与 `cached_req_ids`：一轮 step 的集合运算入口

我一开始看 `_update_states()` 时，最先抓住的是两个集合：

```python
scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
cached_req_ids = self.input_batch.req_id_to_index.keys()
```

它们分别代表两层完全不同的语义：

  - **`scheduled_req_ids`**：这一轮被 Scheduler 安排了 token 计算的请求。
                              
  - **`cached_req_ids`**：   当前 persistent batch 里还占着 batch 行位置的请求

这一层的作用，不是区分 prefill、decode 或 prefix cache，而是先回答一个更基础的问题：

当前 batch 里挂着的请求，哪些这轮还要继续保留，哪些应该先从 persistent batch 中移出去？

所以从这个角度看，ModelRunner 并不是“每一轮都从零开始建 batch”，而是在维护一个不断被增删改的 persistent batch。

这里先只停留在集合层面。
真正到了后面处理请求状态时，代码还会继续区分新请求和已缓存请求，也就是 scheduled_new_reqs 和 scheduled_cached_reqs。

### 4.1 scheduled_new_reqs
这一组表示本轮新加入的请求。  
但这里的“新”指的是**新进入 ModelRunner 的请求**，不等于“完全从 0 开始的请求”。

因为在 scheduler 侧，一个 waiting request 即使此前还没真正进入 running，也可能已经通过本地 prefix cache 或外部 KV connector 命中了一部分已计算 token。也正因为如此，`NewRequestData` 本身就携带了 `num_computed_tokens` 和 `block_ids`，说明新请求进入 ModelRunner 时，并不一定是完全空白状态。:contentReference[oaicite:4]{index=4}

所以更准确地说，`scheduled_new_reqs` 代表的是：

> 这一轮需要新建 runner-side request state，并加入 persistent batch 的请求。

### 4.2 scheduled_cached_reqs
这一组表示本轮继续推进的、已经有 cached state 的请求。  
在当前实现里，这一层更接近“running / resumed requests”，而不是狭义上的“命中了 prefix cache 的请求”。源码注释本身也是按 running/resumed requests 来描述它们的。

对这类请求来说，ModelRunner 不需要重新创建整份 `CachedRequestState`，而是要基于已有状态继续更新：

- 新的 `num_computed_tokens`
- 这一轮 newly appended 的 `block_ids`
- 新的输出 token 和相关 batch 状态

所以如果说 `scheduled_new_reqs` 回答的是“谁要新建请求状态”，  
那 `scheduled_cached_reqs` 回答的就是“谁要在旧状态上继续往前推进”。

### 4.3 为什么要区分这两种请求

我一开始也想过：反正最终都要送给 GPU 去算，为什么不干脆合成一个列表？

后来回头看代码，我觉得这两类请求被分开，首先不是因为“它们会触发两种完全不同的 Transformer”，而是因为它们在 **runner 侧状态维护** 上就是两条不同路径：

- 对 `scheduled_new_reqs`，ModelRunner 要新建 `CachedRequestState`，把 `block_ids`、`num_computed_tokens` 等初始状态注册进去。
- 对 `scheduled_cached_reqs`，ModelRunner 则是在已有请求状态上继续追加和更新。

从 attention 视角看，这两类请求后面当然也会有差异：  
新请求更像是“从当前已有前缀状态出发，开始构造这一轮 query”；而 cached requests 则是在已有 KV cache 和已有 batch 状态上继续推进。  
但这里我觉得先不要把它硬讲成 “Self-Attention vs Cross-Attention”，因为对 decoder-only LLM 来说，本质上仍然是同一套 causal self-attention，只是实现上会复用历史 KV cache。
(我个人一开始确实是想Self-Attention vs Cross-Attention这个角度)

-----

## 5\. `_update_states` 如何增删改请求状态

如果把 `_update_states()` 展开来看，它做的事情其实不像“简单同步一下状态”，而更像一次面向 persistent batch 的增删改重组。

### 5.1 删除已经结束的请求

第一步要处理的是已经真正结束的请求。它们会：

  - 从 `self.requests` 中移除
  - 从 `input_batch` 中移除

这些请求不仅这一轮不该再参与执行，后面也不应该继续占用 runner 侧状态和 batch 行。

### 5.2 删除“这轮已经过时”的 persistent batch 行

删完 finished 还不够，`_update_states()` 还会继续处理一类更容易被忽略的对象：**当前还活着，但这一轮没有被 schedule 到的请求行**。

这些请求并没有被系统彻底遗忘，它们只是不该继续留在这一轮的 persistent batch 里。也就是说：

  - 从 `input_batch` 里删掉
  - 但在 `self.requests` 里保留 cached state（以便未来再次被调度）

这也解释了为什么 `requests` 和 `input_batch` 必须拆成两层：因为“请求还活着”和“请求这一轮还应该留在 batch 里”不是一回事。

### 5.3 处理新增请求：构造 CachedRequestState

对新请求来说，ModelRunner 不是只塞进几个 token，而是先构造一份完整的 `CachedRequestState`，然后再把它们收集到待加入列表中。

新请求进入 ModelRunner，首先进入的是 `requests` 这层被包装成可持续维护的请求状态，而不是直接跳进最终的模型输入。

### 5.4 处理恢复请求和继续推进的请求

除了全新 request，还需要处理**恢复执行的 request**和**已经缓存着、这一轮继续推进的 request**。
这两类请求不需要重新从零构造完整状态，但它们的 runner state 和 persistent batch 里的内容都必须往前推进。所以 `_update_states()` 是一个真正的增删改混合函数。

### 5.5 增加块表：这是从调度走向模型执行的一步硬连接

当这一轮调度为某个请求分配了新的 KV block 时，ModelRunner 会把这些新块真正记进 batch 里的块表。

> Scheduler 已经决定“这个请求这一轮要往前推”，`_update_states()` 则把“新分到的物理块”真正记进了 ModelRunner 侧的 `block_table` 里。

`block_table` 先记住“每个请求有哪些物理块”，后续的 `slot_mapping` 再进一步计算“当前这一个 token 应该写到哪一个绝对物理槽位”。

### 5.6 最后压缩、重排并刷新 metadata

等删除、补入、更新都完成后，`_update_states()` 还会：

  - 压缩空洞
  - 必要时重排 batch
  - 刷新 metadata

这一步确保前面做的增删改，最终整理成一份自洽、稳定、可供后续执行使用的 batch 状态。

-----

## 6\. `_prepare_inputs / prepare_input_tensors`：把更新后的 batch 真正翻译成模型输入

到这里，再去讲 `prepare_input_tensors()` 就自然多了。因为此时我们已经明确知道：

  - 哪些请求还活着，保存在 `self.requests`
  - 哪些请求这一轮被 scheduler 点名挂在 persistent batch 里
  - 哪些行已经被移除、补入、重排、刷新完成

此时 `prepare_input_tensors()` 面对的是一份已经整理好的稳定 batch 状态。接下来的 token 展平、位置生成、块表组织等操作，本质上都是在把这份状态翻译成模型真正要消费的输入表示。

-----

## 7\. `prepare_input_tensors` 的第一层价值：Padding-free

`prepare_input_tensors()` 的核心价值之一就是 **Padding-free**：

  - 不再强行把所有请求 pad 成一个规整的二维矩阵
  - 只保留当前真正有效的 token，最后拼成一个紧凑的一维 token 流

### 展平 Token IDs

ModelRunner 会遍历本轮选中的请求：

  - 如果处理多个 token，就把这一段 token 全部 append 进去。
  - 如果只需要处理一个 token，就只 append 一个。

最后得到的不是传统的 `[batch_size, seq_len]` 形式，而是一个一维 token 流。原本的 `batch_size` 和 `seq_len` 被“压平”了。
它的直接收益是：**GPU 不再需要在 `<PAD>` 上白白做矩阵乘法，只对真正有效的 token 做计算。**

-----

## 8\. 为什么 1D / 2D 输入仍然能被线性层正确处理

传统印象里模型输入是 `[batch_size, seq_len, hidden_dim]`，展平后变成了 `[total_tokens, hidden_dim]`，这能正常进线性层吗？
答案是：**可以**。

关键点在于，对线性层来说，它真正关心的是最后一个维度是否是特征维：

  - 每一个 token 的 4096 维向量，会独立地去乘权重矩阵。
  - 至于这个 token 属于第几个 batch、排在第几个位置，线性层本身并不需要知道。

所以 vLLM 把所有有效 token 直接整理成 `[total_tokens, hidden_dim]`，在计算上完全合法且更紧凑。从硬件角度看，没有填充孔洞的连续内存访问，比维护带大量无效 `<PAD>` 的大矩阵高效得多。

-----

## 9\. 除了展平 token，ModelRunner 还要补齐什么信息

如果只把 token 展平还远远不够，模型还需要知道位置、KV block 分布以及物理 slot。所以 ModelRunner 还要补齐三类信息：

### 9.1 `position_ids`

虽然 token 变成了 1D 流，但 RoPE（旋转位置编码）仍然需要知道当前 token 在自己所属序列里排第几。
ModelRunner 会同步构造一个等长的 `position_ids`：

  - Prefill 请求的 3 个 token 可能是 `[0, 1, 2]`。
  - Decode 请求的 1 个 token（如果是第 100 个词）可能是 `[99]`。

### 9.2 `block_tables`

底层 attention 系统需要知道当前请求的历史 KV Cache 在哪些物理块里。ModelRunner 会组织出 `block_table` 相当于“物理块寻址地图”。它只是查目录用的，不参与大规模矩阵乘法，因此 2D 张量形式也不会有性能问题。

### 9.3 `slot_mapping`

如果 `block_table` 回答的是“这个请求有哪些物理块？”，那么 `slot_mapping` 回答的就是更细的问题：“这一个具体 token，最终应该写入哪一个绝对物理槽位？”。这是 PagedAttention 的关键一跳。

-----

## 10\. `compute_slot_mapping()`：把 token 翻译成“绝对物理槽位”

核心计算逻辑源码如下：

```python
def compute_slot_mapping(self, req_indices: np.ndarray,
                         positions: np.ndarray) -> None:
    
    block_table_indices = (
        req_indices * self.max_num_blocks_per_req
        + positions // self.block_size
    )
    block_numbers = self.block_table_np.ravel()[block_table_indices]
    block_offsets = positions % self.block_size
    np.add(
        block_numbers * self.block_size,
        block_offsets,
        out=self.slot_mapping_np[:req_indices.shape[0]]
    )
```

这段代码做的事情是：**把“某个 token 属于哪个请求、位于该请求的第几个位置”，翻译成“这个 token 最终应该写到哪个绝对物理 slot”。**

两级映射过程为：先从**请求内位置**找到所在的**物理块号**，再从**物理块号 + 块内偏移**算出最终**绝对槽位**。

### 为什么不能直接用展平后的全局 token 序号去算？

读到这里时，我注意到了原作者的一句注释：
# NOTE(woosuk): We can't simply use `token_indices // block_size`
# here because M (max_model_len) is not necessarily divisible by
# block_size.

compute_slot_mapping() 要找的，不是“这个 token 在当前大数组里排第几个”，而是：

这个 token 在它自己所属请求中排第几个位置。

不同请求虽然在输入阶段被拼成了一条扁平 token 流，但它们各自的 KV Cache 仍然是按“请求内位置”去映射到 block table 的。也正因为如此，代码必须同时使用：

req_indices：说明这个 token 属于哪个请求
positions：说明这个 token 在该请求内部的 position 是多少

然后再通过：

req_indices * max_num_blocks_per_req + positions // block_size

去定位 block table 中真正该查的那一格。
如果直接拿 flatten 后的全局 token 下标去做 // block_size，得到的只会是“这条大向量流里的块位置”，而不是“该 token 在自己请求中的逻辑块位置”，这两者不是一回事。

-----

## 11\. 用一个具体例子跑通 `compute_slot_mapping()`

### 11.1 已知条件

假设 `block_size = 4`，`max_num_blocks_per_req = 8`。
我们要计算 3 个 token 的物理落盘地址：

  - 词 A：Req 0，第 6 个词，`pos = 5`
  - 词 B：Req 1，第 3 个词，`pos = 2`
  - 词 C：Req 1，第 7 个词，`pos = 6`

输入数组即 `req_indices = [0, 1, 1]`, `positions = [5, 2, 6]`。
后台展平后的 `block_table_np.ravel()` 长这样（基址区分）：
`[10, 11, -1, -1, -1, -1, -1, -1, 42, 88, -1, -1, -1, -1, -1, -1]`

### 11.2 算出查表索引

公式：`req_indices * 8 + positions // 4`

  - 基址：`[0, 1, 1] * 8 = [0, 8, 8]`
  - 逻辑块位置：`[5//4, 2//4, 6//4] = [1, 0, 1]`
  - 最终查表索引：`[1, 8, 9]`

### 11.3 拿到对应的物理块号

拿着 `[1, 8, 9]` 查账本得到 `block_numbers = [11, 42, 88]`。

### 11.4 算块内偏移

公式：`positions % 4`
偏移量：`[5%4, 2%4, 6%4] = [1, 2, 2]`。

### 11.5 算最终绝对物理槽位

公式：`block_numbers * 4 + block_offsets`
`[11, 42, 88] * 4 + [1, 2, 2] = [45, 170, 354]`。

最终得出：词 A 写到槽位 45，词 B 写到槽位 170，词 C 写到槽位 354。

-----

## 12\. 小结

> `compute_slot_mapping()` 的本质，就是先通过 `(请求索引 + 请求内位置)` 找到该 token 应落入的物理块号，再通过 `(物理块号 × block_size + 块内偏移)` 算出最终的绝对物理槽位。

到此为止，ModelRunner 输入准备的主线已经全部串联：

1.  **Scheduler** 决定跑谁、给哪些块。
2.  **`_update_states()`** 把结果同步成 `requests` 和 `input_batch`。
3.  **`prepare_input_tensors()`** 展平请求为 padding-free 输入。
4.  **`block_table`** 保存请求的物理块分布。
5.  **`compute_slot_mapping()`** 把当前 token 翻译成最终的绝对物理 slot。

