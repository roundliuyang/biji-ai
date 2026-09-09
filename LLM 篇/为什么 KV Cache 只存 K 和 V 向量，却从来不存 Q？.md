# 为什么 KV Cache 只存 K 和 V 向量，却从来不存 Q？

LLM 是**自回归（autoregressive）**的，因此每个 token 都是基于它之前的所有 token，一个接一个地生成。

这种自回归特性会直接影响模型内部的计算方式。

对于包含 **n 个 token** 的输入，模型进行一次前向传播会产生 **n 个 hidden state（隐藏状态）**，但最终只有**最后一个 hidden state** 会被投影到 logits，并用于生成下一个 token。

所以，要理解为什么 KV Cache 只缓存 K 和 V，而不缓存 Q，我们需要先回过头看看：最后一个 hidden state 到底是怎么产生的。

下面以一个 **10-token 的 prompt** 为例。

## 1.Prefill：预填充阶段

10 个 token 会在一次前向传播中**并行**经过模型。

因为整个 prompt 已经是已知的，所以可以并行计算，只需要通过 **causal mask（因果掩码）**保证每个 token 不能看到它后面的 token。

在每一层中，10 个位置都会分别产生：

- Query（Q）
- Key（K）
- Value（V）

然后，每个位置的 attention 都会与它之前的所有位置进行计算。

这个阶段计算量比较大，因此也是为什么**第一个 token 通常生成得明显更慢**。

通常来说：

> **TTFT（Time To First Token，首 token 延迟）主要来自 prefill 阶段。**



## 2.生成第一个输出 token

现在我们要生成第 **11 个 token**。

实际上，我们只需要**第 10 个 token 的 hidden state**。

因此，把第 10 个 token 的 hidden state 从：

> hidden dimension → vocabulary dimension

进行投影，得到整个词表上的 **logits**。

然后 logits 经过：

> softmax → sampling（采样）

最终生成 **token 11**。



## 3.回溯 hidden state 是怎么来的

我们继续往回看。

最后一个 hidden state，是 **Feed Forward Network（FFN，前馈网络）输出的最后一行**。

而 FFN 是一个**逐位置（position-wise）**操作：

> 每一行独立进行计算，不会和其他位置直接交互。

所以，第 10 个位置的 FFN 输出，只来自于第 10 个位置的 attention 输出。

因此，我们现在只需要搞清楚：

> **第 10 行的 attention output 是怎么计算出来的？**



## 4.Attention 矩阵

对于一个 10-token 的 prompt：

**QKᵀ** 会得到一个 **10 × 10 的矩阵**。

其中第 i 行表示：

> Queryᵢ 与所有 Key 的点积。

因此，第 10 行就是：

> Q₁₀ · K₁
>  Q₁₀ · K₂
>  ...
>  Q₁₀ · K₁₀

注意：

**这里面只出现了 Q₁₀。**

Q₁ 到 Q₉ 分别只属于 attention 矩阵的第 1～9 行。

而这些位置对应的 hidden state 已经被我们丢弃了，因为：

> 它们并不需要用来生成下一个 token。

接下来，第 10 行的 attention score 经过 softmax，然后与完整的 Value 向量集合：

> V₁、V₂、……、V₁₀

进行加权求和，从而得到第 10 个位置的 attention output。

所以，最后的 hidden state 实际上依赖于：

> **Q₁₀ + 所有 K + 所有 V**

也就是：

- 当前 token 的 Q
- 所有历史 token 的 K
- 所有历史 token 的 V



## 5.生成 Token 12

现在，token 11 被添加进来了。

这一次，我们需要计算第 11 个位置的 hidden state，用它来生成 token 12。

从数学上看，attention 计算变成：

> **Q₁₁ 与 K₁～K₁₁ 进行计算**

然后再与：

> **V₁～V₁₁**

进行加权计算。

关键点来了：

**K₁～K₁₁ 和 V₁～V₁₁ 中，之前已经计算过的 K₁～K₁₀、V₁～V₁₀ 可以直接复用。**

为什么？

因为在 causal masking 下，一个 token 的 K 和 V 只依赖：

> 当前 token 以及它前面的 token

而**不会依赖后面的 token**。

所以，当我们在后面追加 token 11 时：

> token 3 的 K₃ 和 V₃ 不会发生任何变化。

因此，之前计算好的 K 和 V 可以直接保存下来，下一次继续使用。



## 6.KV Cache 到底缓存了什么？

综合起来就很清楚了：

在每一次 decoding（解码）过程中，我们只需要：

> **保留之前所有位置的 Key 和 Value**

然后，对于新来的 token，只计算它自己的：

> **Q、K、V**

也就是说：

### Q 不需要缓存

因为每个 decode step 只需要一个新的 Query：

> Q₁₁ → 用于计算当前 token 的 attention

计算完之后：

> **Q₁₁ 再也不会被后续 token 使用。**

下一步需要的是：

> Q₁₂

而不是 Q₁₁。

所以没有必要把 Q₁₁ 保存下来。

### 一句话理解

可以把 KV Cache 理解成：

> **历史 token 的 K、V 要反复被未来 token 查询，所以需要缓存；当前 token 的 Q 只用一次，用完就丢，所以不需要缓存。**

可以画成：

```
历史 Token：
K₁ V₁
K₂ V₂
K₃ V₃
...
K₁₀ V₁₀
   ↓
   缓存起来
   ↓
KV Cache


当前 Token：
Q₁₁ ─────────┐
             ↓
        Attention
             ↑
     K₁...K₁₁ / V₁...V₁₁
```

因此，每生成一个新 token：

```
Q_new ──────────────┐
                    ↓
              Attention
                    ↑
KV Cache ───────────┘
                    ↓
              新的 hidden state
                    ↓
                  logits
                    ↓
              下一个 token
```



## KV Cache 只是 LLM 系统中的四种缓存之一

KV Cache 只是 LLM serving（推理服务）中的一种缓存机制。

另外还有三种常见的缓存：

1. **Prefix Cache（前缀缓存）**
    服务端缓存已经计算过的 prompt 前缀。
2. **Prompt Cache（提示词缓存）**
    一些模型服务商会对重复使用的 prompt 进行缓存，并可能针对缓存命中进行计费。
3. **Semantic Cache（语义缓存）**
    如果发现用户的问题和之前的问题语义上高度相似，甚至可以**直接返回之前的结果，完全跳过模型推理**。

所以，LLM Serving 中可以从四个层面理解缓存：

> **KV Cache → 缓存 Attention 的 K/V**
> **Prefix Cache → 缓存重复的 Prompt 前缀计算**
> **Prompt Cache → 服务商层面的 Prompt 缓存**
> **Semantic Cache → 直接跳过模型推理**

其中你这段文章重点讲的是第一种：**KV Cache**。



## Q、K、V 本质含义

**从 Transformer 的数学/张量层面看，Q、K、V 本质上都是由 token 的 hidden state 经过不同线性投影得到的向量。**

假设某一层输入：
$$
X\in\mathbb{R}^{n\times d_{\text{model}}}
$$
其中 `n` 是 token 数量，$d_{\text{model}}$ 是 hidden size。

经过三个不同的线性层：
$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

所以：

- **Q（Query）**：由当前 token 的 hidden state 经过 $W_Q$ 投影得到的向量
- **K（Key）**：由 token 的 hidden state 经过 $W_K$ 投影得到的向量
- **V（Value）**：由 token 的 hidden state 经过 $W_V$ 投影得到的向量

它们**没有什么独立的物理实体**，就是三组不同的向量。



### 那三个向量分别参与什么计算？

Attention：
$$
A=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$
然后：
$$
O=AV
$$
所以从计算过程来看：

### Q

Q 参与：
$$
QK^T
$$
它决定**当前 token 与其他 token 的相关性分数**。

### K

K 也参与：
$$
QK^T
$$
它和 Q 一起计算相关性。

对于第 $i$ 个 token：
$$
\text{score}_{ij}=\frac{Q_iK_j^T}{\sqrt{d_k}}
$$
表示：

> 第 $i$ 个位置对第 $j$ 个位置的 Attention 分数。

### V

V 不参与 Q-K 的匹配。

它参与：
$$
O_i=\sum_j A_{ij}V_j
$$
展开就是：
$$
O_i = A_{i1}V_1 + A_{i2}V_2 + A_{i3}V_3 + \cdots + A_{in}V_n
$$
也就是根据 Attention 权重，把各个位置的 V 加权求和。

因此最物理、最准确的理解就是：

```
Hidden State X
      │
      ├── × W_Q ──→ Q
      │
      ├── × W_K ──→ K
      │
      └── × W_V ──→ V
```

然后：

```
Q ─────┐
       ├── QKᵀ ──→ Attention 权重
K ─────┘                 │
                         ↓
V ───────────────────→ 加权求和
                         │
                         ↓
                    Attention 输出
```

**一句话：**

> **Q、K、V 都只是 hidden state 的三种不同线性投影；Q 和 K 用来计算注意力权重，V 被这些权重加权后形成 Attention 输出。**

