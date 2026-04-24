# 从 KV Cache 到 PagedAttention：vLLM 如何管理显存、复用前缀并处理被切开的请求

> 这篇文章主要想回答三个问题：
>
> 1. 为什么 vLLM 要把 KV Cache 单独做成分块管理，而不是沿用传统连续内存思路？
> 2. Prefix Caching、BlockPool、引用计数这些机制，具体是怎么配合起来工作的？
> 3. 一个请求如果被切成不同时间段去计算，它的 Q / K / V 到底是怎么接上的？

---

## 目录

* [1. 传统推理框架在 KV Cache 上的三个痛点](#1-传统推理框架在-kv-cache-上的三个痛点)
* [2. 为什么只有 KV Cache 适合做成“块池”](#2-为什么只有-kv-cache-适合做成块池)
* [3. vLLM 如何先算出“我到底能拿多少显存做 KV Cache”](#3-vllm-如何先算出我到底能拿多少显存做-kv-cache)
* [4. `block` 和 `page` 到底是什么关系](#4-block-和-page-到底是什么关系)
* [5. BlockPool：vLLM 如何管理空闲块、缓存块和引用计数](#5-blockpoolvllm-如何管理空闲块缓存块和引用计数)
* [6. Prefix Caching：为什么只有满块才能参与缓存复用](#6-prefix-caching为什么只有满块才能参与缓存复用)
* [7. 一个例子看懂：共享前缀、未满块与请求结束后的块命运](#7-一个例子看懂共享前缀未满块与请求结束后的块命运)
* [8. 一个请求被分不同时间算完，它的 Q / K / V 怎么接上](#8-一个请求被分不同时间算完它的-q--k--v-怎么接上)
* [9. 总结](#9-总结)

---

## 1. 传统推理框架在 KV Cache 上的三个痛点

我一开始在看 vLLM 这部分设计时，先问了自己一个问题：

> 它到底是在优化什么？

如果只从大模型在线推理的角度看，传统方案在 KV Cache 管理上主要有三个痛点。

### 1.1 KV Cache 长度动态增长，连续内存分配很容易产生碎片

KV Cache 的大小不是固定的，它会随着生成长度不断膨胀。
用户刚开始生成时，Cache 很小；用户生成到第 4000 个词时，Cache 又会变得很大。

如果底层坚持“每个请求必须对应一段连续的显存空间”，就会很麻烦：

* **内部浪费**：为了保险，系统往往会预留比当前实际需要更大的空间，很多位置最后根本没被用到。
* **外部碎片**：总空闲显存也许够，但找不到一段足够长的连续空间，于是新请求仍然进不来。

### 1.2 相同前缀被重复缓存，显存被白白浪费

多个请求带着相同的系统提示词、相同的文档前缀甚至相同的多轮对话上下文，这是很常见的。

如果每个请求都各自维护一份独立 KV Cache，那么这部分前缀会被重复计算、重复存储，显存浪费非常严重。

### 1.3 最终表现为吞吐量受限

前面两个问题最后都会落到同一个结果上：

> 同一张 GPU 上能同时容纳的有效请求数下降，batch 做不大，吞吐量也就上不去。

所以 vLLM 这一套设计，表面上是在讲内存块、哈希、引用计数，实际上是在解决：

> 如何让 KV Cache 这部分显存，既不容易碎掉，又能跨请求复用。

---

## 2. 为什么只有 KV Cache 适合做成“块池”

读到这里时，我当时又产生了一个很自然的问题：

> 为什么是把 **KV Cache** 划成物理块池，而不是直接把 GPU 的全部显存都做成块池？

想清楚这个问题之后，PagedAttention 的设计动机就会非常清楚。

### 2.1 模型权重不适合分页化管理

模型权重通常占显存的大头，而且它在模型加载进 GPU 之后基本就是静态的：

* 大小不变
* 位置基本不变
* 计算时强依赖连续访存

如果把模型权重也切成很多离散的小块，GPU 在做矩阵乘法时反而会不断跳指针，严重破坏访存效率。

所以模型权重最需要的是**稳定、连续、适合大规模矩阵计算的显存布局**，而不是灵活的分页管理。

### 2.2 中间激活值也不适合分页化管理

中间激活值的特点是：

* 生命周期短
* 大小通常可预测
* 一层算完下一层，前一层很多中间结果就可以被复用或覆盖

这种数据更适合用一块连续临时空间反复写，不需要复杂的块级管理。

### 2.3 只有 KV Cache 同时具备三个特征

真正适合做成分页式块池的，恰恰是 KV Cache，因为它同时具有：

1. **动态增长**：会随着生成不断膨胀
2. **长度不可预知**：不同请求最终会生成多长，事先并不知道
3. **可能跨请求共享前缀**：相同前缀的历史 K / V 完全有复用空间

所以更准确地说，PagedAttention 不是“把显存块化”这么简单，而是：

> 只把**最需要弹性管理**的那部分显存，也就是 KV Cache，做成类似虚拟内存的块式管理。

---

## 3. vLLM 如何先算出“我到底能拿多少显存做 KV Cache”

PagedAttention 不是凭空开始分块的。
在真正分块之前，vLLM 先要回答一个更基础的问题：

> 扣掉模型权重和运行过程中的其他开销之后，GPU 还剩多少显存可以分给 KV Cache？

这一步在 `determine_available_memory()` 这类逻辑中完成，本质上是一次 **profile run**。

```python
def determine_available_memory(self) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with memory_profiling(...) as profile_result:
        self.model_runner.profile_run()

    free_gpu_memory = profile_result.after_profile.free_memory
    available_kv_cache_memory = (
        self.requested_memory - profile_result.non_kv_cache_memory
    )
    return int(available_kv_cache_memory)
```

这段逻辑的关键点在于：

* 系统先做一次模拟前向，看看模型真实会吃掉多少显存
* 然后把 **非 KV Cache 部分的显存占用** 扣掉
* 最后得到还能留给 KV Cache 的预算

也就是说，vLLM 不是“先拍脑袋定一个 num_blocks”，而是：

> 先 profile，再反推出这张卡最多能切出多少个 KV blocks。

---

### 3.1 `page_size` 和 `num_blocks` 是怎么来的

接下来系统需要根据 attention 层的规格去算：

* 一个块到底有多大
* 一层最多能分到多少块

相关逻辑可以压缩理解成：

```python
def get_kv_cache_config_from_groups(...):
    group_size = ...
    page_size = get_uniform_page_size(kv_cache_specs)
    num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)
```

这里最关键的两个量是：

* `page_size`
* `num_blocks`

### 3.2 一个容易疑惑的点：为什么有的公式要乘 2，有的不用

我一开始读这里时也有个疑问：

> 一个 token 的 KV Cache 大小，为什么有时公式里不乘 2？K 和 V 不是两份吗？

后来我才想明白，这里其实是两个不同层次的问题。

#### 当你在算“完整 KV page 的字节数”时，要乘 2

因为一个 block 里需要同时存：

* Key
* Value

所以完整的 page size 通常可以近似理解为：

```text
page_size ≈ block_size × num_kv_heads × head_size × dtype_bytes × 2
```

这里的 `× 2` 就是把 K 和 V 都算进去。

#### 当你在单独盯某一路缓存形状时，不一定显式看到乘 2

因为很多实现里：

* K Cache 和 V Cache 是分开管理 / 分开传指针的
* 某些公式只是在计算单路张量的体积或偏移

所以你会看到有的地方像是在算单个 K 或单个 V 的体积，没有显式写 `× 2`。
但如果你问的是“一个完整 KV page 一共占多少字节”，那最终一定要把 K 和 V 两部分都算进去。

---

## 4. `block` 和 `page` 到底是什么关系

我觉得这里也很容易混。

我们平时在讲 `req1`、`req2` 的时候，总是说：

* 这个请求占了几个 block
* 那个请求又复用了哪几个 block

但到了显存规划时，代码里又开始讲 `page_size`。
这两个词到底什么关系？

### 4.1 在调度和管理层，我们更习惯说 `block`

因为调度器真正关心的是：

* 一个请求需要多少块
* 哪些块被谁占用了
* 哪些块可以共享
* 哪些块可以回收

所以在上层逻辑里，`block` 更像是调度器的“通用货币”。

### 4.2 在底层显存规划里，我们更习惯说 `page`

因为从物理显存的角度看，系统需要先知道：

* 一个块在字节层面到底有多大
* 用可用显存预算最多能切出多少份

这里讨论的其实是：

> 一块物理连续显存片段的字节大小

所以会更自然地写成 `page_size`。

### 4.3 两者其实是一体两面

你可以把它简单理解成：

* **block**：逻辑管理单位
* **page**：这个逻辑单位在物理显存中的字节大小描述

所以平时我们说“一个请求占了 2 个 block”，
翻译到底层硬件规划里，其实就是：

> 这个请求最终占用了 2 份 page_size 大小的 KV Cache 显存页。

---

## 5. BlockPool：vLLM 如何管理空闲块、缓存块和引用计数

当可用显存预算和块大小都确定后，vLLM 就会开始真正构建块池。

在这一步里，BlockPool 是核心角色。

### 5.1 BlockPool 里最重要的两个数据结构

它内部可以先抓住两个核心结构：

1. **`free_block_queue`**
   记录当前系统里可分配的空闲块

2. **`cached_block_hash_to_block`**
   从块哈希值到块对象的映射，用于支持 Prefix Caching

除此之外，每个块本身还带有几个关键字段：

* `block_id`
* `ref_cnt`
* `block_hash`

其中最关键的是 `ref_cnt`。

### 5.2 引用计数代表“这个块现在还有没有人用”

这一点非常重要。

假设一个 block 同时被两个请求共享，那么它的：

```text
ref_cnt = 2
```

如果其中一个请求结束了，这个块的 `ref_cnt` 会减 1；
只有当所有依赖它的请求都结束时，`ref_cnt` 才会降到 0。

所以块是否能被真正释放，不取决于“某一个请求是不是结束了”，而取决于：

> 这个块是不是已经不再被任何请求引用。

### 5.3 申请新块时到底发生了什么

当系统需要新块时，会从 `free_block_queue` 里取块。逻辑可以简化理解成：

```python
def get_new_blocks(self, num_blocks: int):
    ret = self.free_block_queue.popleft_n(num_blocks)
    for block in ret:
        self._maybe_evict_cached_block(block)
        block.ref_cnt += 1
    return ret
```

这里有一个特别关键的细节：

> 如果这个块以前存过别的满块数据，并且还挂着旧的 hash 映射，那么在它被重新分配之前，必须先把这条旧映射清掉。

否则以后别的请求拿着旧 hash 来查，就会命中一块已经被覆写成新内容的脏块。

---

## 6. Prefix Caching：为什么只有满块才能参与缓存复用

Prefix Caching 的目标很简单：

> 如果不同请求有相同前缀，就不要重复计算这部分 KV。

但 vLLM 并不是按“任意前缀长度”去做复用，而是按 **块粒度** 来做。

### 6.1 只有 full block 才能被 hash 并进入缓存

这点很关键。

一个块只有在：

* token 数量达到 `block_size`
* 内容稳定，不会再被改写

时，才会被视为 **full block**，此时才能计算 hash，进入全局缓存表。

未满块不参与缓存复用。
原因也很简单：它的内容还不完整，不具备稳定的缓存语义。

### 6.2 block hash 不是只看当前块 token，还会串上父块 hash

这也是 Prefix Caching 正确性的关键。

块哈希并不是简单地对当前 block 内的 token 做 hash，
而是会把：

* 父块 hash
* 当前块 token ids
* 额外上下文键（如有）

一起送进 hash 过程。

逻辑可以抽象成：

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    return BlockHash(
        hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys)),
        tuple(curr_block_token_ids),
        extra_keys
    )
```

这意味着：

> 即使两个 block 内部 token 一样，只要它们前面的前缀路径不同，最终的 block hash 也不会一样。

这就避免了“中间某一块 token 相同，但整个前缀链不同”时被错误复用。

### 6.3 最长前缀命中是怎么找的

系统会从请求的第一个块开始，一块一块往后查：

* 当前块 hash 是否能在缓存表里命中
* 如果能命中，就继续看下一块
* 一旦某一块命不中，就立刻停止

所以本质上，Prefix Caching 找的是：

> **按块连续匹配的最长前缀**

而不是“零散地命中若干相同块”。

---

## 7. 一个例子看懂：共享前缀、未满块与请求结束后的块命运

这一段我觉得最适合用具体例子来理解。

假设：

* 每个 block 最多存 4 个 token
* 请求 C 最先来
* 请求 A、B 后来
* 三者前缀其实完全一样，总长度 14 个 token

### 7.1 第一阶段：C 先把前缀算出来

C 到来后，系统为它分配块，并计算前缀 KV Cache。

14 个 token 会被分成：

* block 10：满
* block 11：满
* block 12：满
* block 13：只装了最后 2 个 token，未满

于是：

* `block 10 / 11 / 12`：是 full block，可以计算 hash，进入全局缓存表
* `block 13`：未满，不算 hash，也不进入全局缓存表

### 7.2 第二阶段：A 和 B 带着相同前缀进来

系统会按块查 hash：

* 命中 block 10
* 命中 block 11
* 命中 block 12
* 到第 4 块时，发现前面的 `block 13` 根本没 hash，因为它未满，于是匹配在这里中断

这就意味着：

* A 和 B 只能共享前 3 个 full block
* 尾部那 2 个 token 仍然要各自重新算

所以系统会：

* 给 A 新分配一个块（比如 88）
* 给 B 新分配一个块（比如 99）

然后分别把那 2 个没命中的尾部 token 重算进去。

### 7.3 第三阶段：如果 C 先结束了，会发生什么

现在假设 C 结束了。

C 原本占的块是：

```text
[10, 11, 12, 13]
```

系统会把这些块的引用计数都减 1。

#### 共享块 10、11、12

因为 A 和 B 还在用，所以：

* `ref_cnt` 从 3 变成 2
* 仍然大于 0

这三个块还活着，不能回收。

#### 私有未满块 13

因为 block 13 只属于 C 自己，而且它本来就不是 full block，没有进入 hash cache，
所以当 C 结束后：

* `ref_cnt` 变成 0
* 它没有缓存复用价值
* 可以直接回到空闲池，等待被新请求覆盖

### 7.4 如果 A 和 B 也都结束了，又会怎样

此时：

* A 的尾块 88：未满，`ref_cnt -> 0`，直接回空闲池
* B 的尾块 99：未满，`ref_cnt -> 0`，直接回空闲池

而 block 10、11、12 则不一样：

* 它们是满块
* 曾经进入过 hash 表
* 现在虽然 `ref_cnt = 0`，但仍然具备前缀复用价值

所以它们不会被立刻清空，而是进入一种“可再次命中、但当前没人使用”的候补状态。
只有当未来系统真的需要拿它们的物理空间去写入全新内容时，才会先清理旧 hash，再执行覆写。

### 7.5 一个容易误解的坑：`free_block` 不等于“里面的数据已经被清空”

这点我一开始也容易理解错。

看到 `free_block` 时，很容易下意识以为：

> 既然 free 了，那里面的数据肯定已经没了。

但在这里更准确的含义其实是：

> **引用计数归零了，不再被任何请求占用**
> 而不一定是“物理内容已经被立刻擦除”

这正是 Prefix Caching 能跨请求复用历史块的基础。

---

## 8. 一个请求被分不同时间算完，它的 Q / K / V 怎么接上

到这里自然会产生一个更核心的问题：

> 一个请求如果被切成不同时间段去计算，比如 chunked prefill 或者分页读取历史 KV，它的 Q / K / V 到底怎么接上？

我一开始直觉上会担心：

> Attention 不是要算
> `Softmax(QK^T / sqrt(d)) V`
> 吗？
> 如果 K 和 V 都碎在不同 block 里，难道每次还得先拼成连续大矩阵？

如果真这样做，那前面费尽心机做的分页管理就几乎白费了。
所以 vLLM 的关键点就在于：

> 它不是先把 K / V 全部重新拼成连续张量，再去算 attention；
> 而是直接在 **分页化的 KV Cache** 上做 attention。

### 8.1 新一轮计算时，哪些东西是“重新算”的，哪些东西是“直接读”的

这点最容易混。

假设系统现在要为某个请求生成一个新 token，或者计算 chunked prefill 的下一段。

此时会发生两类事情：

#### 新增部分的 Q / K / V 需要重新计算

对于当前这轮真正新进入模型的 token：

* 它们会过 embedding
* 过线性层
* 算出当前这轮自己的 Q、K、V

也就是说，**新增 token 的 Q / K / V 一定是这一轮现算的**。

#### 历史部分的 K / V 不需要重算

历史 token 早就已经在之前的 step 中把 K / V 写进 KV Cache 了。
这一轮只需要通过 block table 去找到它们，把它们读出来参与 attention 即可。

所以更准确地说：

> 新 token 算新的 Q / K / V
> 老 token 不重算，只复用它们已经缓存好的历史 K / V

### 8.2 PagedAttention Kernel 怎么在“不连续显存”上做 attention

vLLM 的关键做法是：
让底层注意力算子直接接受分页化的 KV Cache，而不是要求它们必须是一整块连续显存。

具体可以把它理解成四步。

#### 第一步：先拿到当前 token 的 Q

当前这轮的 token 过完前面的线性层之后，会先得到属于自己的 Q。

#### 第二步：根据 block table 去按块读取历史 K

算子不会先去拼接一个完整的连续 K 矩阵，而是会按 block table 给出的块号，逐块读取历史 K。

所以逻辑上更像：

* 去块 10 读一段 K
* 去块 11 再读一段 K
* 去块 12 再读一段 K
* 去当前新块再读一段 K

#### 第三步：用 Online Softmax 逐块累计

这里最关键的问题是：

> Softmax 不是要看到所有 logits 才能算吗？

vLLM 这里借助的思想和 FlashAttention 一脉相承：
不必把所有分数全部存下来再统一算，而是可以在遍历各个块时维护在线的中间量，逐步更新最大值和归一化系数。

这样就可以做到：

* K / V 分块读取
* logits 分块计算
* Softmax 仍然得到全局正确结果

#### 第四步：再逐块读取 V，累计最终输出

有了各块对应的 attention 权重之后，算子再逐块读取 V，做加权求和，最后得到当前 token 的 attention 输出。

所以这一整套流程的关键不是“把碎掉的 K / V 重新拼起来”，而是：

> **让 attention kernel 自己学会在分页化的 KV Cache 上逐块计算。**

### 8.3 所以“请求被切成不同时间算完”到底是怎么接上的

现在可以把这个问题说得更明确一点了。

一个请求被切开后，真正把前后两段接起来的，不是“重新把旧 token 塞回输入里再从头算”，而是这三样东西：

1. **历史 K / V 已经保存在 paged KV cache 中**
2. **block table 记录了这些历史 KV 分别落在哪些物理块里**
3. **下一轮的新 token 会现算自己的 Q / K / V，再通过 block table 去读取历史 K / V**

所以接上的方式是：

* 旧段：不重算，只提供历史 K / V
* 新段：现算当前 Q / K / V
* attention kernel：跨块读取，逐块完成 `QK^T`、Softmax 和乘 `V`

这样，哪怕一个请求不是一次性算完，而是被分在不同时间段推进，整个 attention 仍然可以自然接上。

---

## 9. 总结

如果只用一句话概括这一整篇：

> vLLM 的核心，不是“把 KV Cache 简单切块”，而是围绕块这个单位，重新设计了 **显存规划、块分配、前缀复用、块生命周期管理，以及分页化 attention 计算方式**。

具体来说，我觉得这篇最值得记住的是下面几件事。

### 9.1 为什么要有 PagedAttention

因为传统连续内存思路在 KV Cache 上会遇到：

* 动态增长带来的碎片问题
* 重复前缀带来的冗余缓存
* 最终 batch 做不大、吞吐上不去的问题

### 9.2 为什么只有 KV Cache 被做成块池

因为模型权重是静态的，中间激活值是短生命周期的，
只有 KV Cache 同时具备：

* 动态增长
* 长度不可预知
* 可跨请求共享前缀

这三个特征。

### 9.3 BlockPool 和 Prefix Caching 解决的是“怎么存”和“怎么复用”

* `free_block_queue` 负责管理空闲块
* `cached_block_hash_to_block` 负责支持前缀命中
* `ref_cnt` 决定一个块能不能被真正释放

而 Prefix Caching 的关键约束是：

> 只有 **full block** 才能算 hash、进入缓存复用链路。

### 9.4 请求结束时，不是所有块都会立刻被擦掉

* 共享块只要还有请求引用，就不能删
* 未满私有块在 `ref_cnt = 0` 后可以直接回空闲池
* 满块即使 `ref_cnt = 0`，也可能因为仍有 hash 映射而保留复用价值，直到真正被新内容覆写前才清理旧 hash

### 9.5 被切成不同时间算完的请求，靠的是“缓存的历史 KV + 分页 attention”接上

新的 token 会现算自己的 Q / K / V，
历史 token 的 K / V 则直接从 paged KV cache 中按块读取。
底层 attention kernel 不要求 K / V 必须连续，而是直接在分块显存上逐块完成 attention 计算。


