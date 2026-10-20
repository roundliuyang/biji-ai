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

### 详细解释（手把手的教你）

下面以 **prefill 阶段、某一层的单个注意力头**为例，完整展开。

**1. 输入矩阵有 10 行，每行对应一个 token**

将 10 个位置的输入向量排列成矩阵：

$$
X=
\begin{bmatrix}
X_1\\
X_2\\
\vdots\\
X_{10}
\end{bmatrix}
\in\mathbb{R}^{10\times d_{\text{model}}}
$$

通过线性投影，一次得到全部位置的 Q、K、V：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

其中：

$$
Q,K\in\mathbb{R}^{10\times d_k},
\qquad
V\in\mathbb{R}^{10\times d_v}
$$

**2. 计算全部位置之间的 attention 分数**

$$
S=\frac{QK^\top}{\sqrt{d_k}}
\in\mathbb{R}^{10\times10}
$$

展开为：

$$
S=\frac{1}{\sqrt{d_k}}
\begin{bmatrix}
Q_1K_1^\top & Q_1K_2^\top & \cdots & Q_1K_{10}^\top\\
Q_2K_1^\top & Q_2K_2^\top & \cdots & Q_2K_{10}^\top\\
\vdots & \vdots & \ddots & \vdots\\
Q_{10}K_1^\top & Q_{10}K_2^\top & \cdots & Q_{10}K_{10}^\top
\end{bmatrix}
$$

第 $i$ 行表示：**位置 $i$ 的 Query 与所有位置的 Key 的匹配分数。**

所以，矩阵运算同时算出了全部 10 行，并不是只算第 10 行。

**3. 加入因果掩码，再逐行计算 softmax**

因为位置 $i$ 不能关注它后面的位置，所以定义：

$$
M_{i,j}=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

加入掩码后的分数矩阵为：

$$
S+M=
\begin{bmatrix}
S_{1,1} & -\infty & \cdots & -\infty\\
S_{2,1} & S_{2,2} & \cdots & -\infty\\
\vdots & \vdots & \ddots & \vdots\\
S_{10,1} & S_{10,2} & \cdots & S_{10,10}
\end{bmatrix}
$$

然后对**每一行分别计算 softmax**：

$$
A=\operatorname{softmax}(S+M)
$$

其中 $A_{i,j}$ 表示位置 $i$ 分配给位置 $j$ 的注意力权重。由于 $e^{-\infty}=0$：

$$
A_{i,j}=0\qquad(j>i)
$$

因此，权重矩阵是：

$$
A=
\begin{bmatrix}
A_{1,1} & 0 & \cdots & 0\\
A_{2,1} & A_{2,2} & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
A_{10,1} & A_{10,2} & \cdots & A_{10,10}
\end{bmatrix}
$$

每一行的权重之和为 1。

**4. 用矩阵乘法，一次得到全部位置的输出**

注意力权重与 V 相乘：

$$
O=AV
$$

先按照完整的矩阵乘法展开，**不省略任何项**：

$$
O=
\begin{bmatrix}
A_{1,1}V_1+A_{1,2}V_2+\cdots+A_{1,10}V_{10}\\
A_{2,1}V_1+A_{2,2}V_2+\cdots+A_{2,10}V_{10}\\
\vdots\\
A_{10,1}V_1+A_{10,2}V_2+\cdots+A_{10,10}V_{10}
\end{bmatrix}
=
\begin{bmatrix}
O_1\\
O_2\\
\vdots\\
O_{10}
\end{bmatrix}
$$

每一行都遵循同一个公式：

$$
O_i=\sum_{j=1}^{10}A_{i,j}V_j
$$

再考虑因果掩码：当 $j>i$ 时，$A_{i,j}=0$。所以，上面的完整写法**等价于省略零项后的写法**：

$$
O=
\begin{bmatrix}
A_{1,1}V_1\\
A_{2,1}V_1+A_{2,2}V_2\\
\vdots\\
A_{10,1}V_1+A_{10,2}V_2+\cdots+A_{10,10}V_{10}
\end{bmatrix}
$$

也就是：

$$
O_i=\sum_{j=1}^{i}A_{i,j}V_j
$$

这里，每个 $A_{i,j}$ 是一个标量，每个 $V_j$ 是一个向量，因此每个 $O_i$ 也是一个向量。最终：

$$
O\in\mathbb{R}^{10\times d_v}
$$

**得到的是 10 行输出，每个 token 对应一行，而不是将 10 个 token 合并成一个输出。**

**5. 这与原文中“只看第 10 行”是什么关系？**

原文单独写出：

$$
O_{10}=\sum_{j=1}^{10}A_{10,j}V_j
$$

只是为了说明：**第 10 个位置在这一层的 attention 输出，直接使用 Q₁₀ 和全部 K、V。**

它并不代表 prefill 只计算第 10 行。Prefill 会通过上述矩阵运算计算全部位置；中间层的这些输出还需要继续传给下一层。到了最终层，为生成 token 11，通常只取最后一个位置的最终 hidden state 来计算 logits。

而 **KV Cache 是把每一层已经算出的 K、V 保存下来，供后续 decode 复用**。下一步处理 token 11 时，只需计算该位置的新 Q、K、V，再用 Q₁₁ 关注历史与当前的 K、V。旧 Q₁～Q₁₀ 不再被读取，所以不需要缓存。

#### 更进一步

采用原始 Transformer 基础配置：`d_model = 512`，8 个注意力头，每个头的 `d_k = d_v = 64`。图中的 Q、K、V 投影按**单个头**表示，每层有 8 组。

```text
初始输入 X：10 × 512
（10 行，每行对应一个 token 的 512 维向量）
    │
    ▼
第 1 层：
    每个注意力头分别计算：

    Q¹ = X × W_Q¹
       = (10 × 512) × (512 × 64)
       → 10 × 64

    K¹ = X × W_K¹
       = (10 × 512) × (512 × 64)
       → 10 × 64

    V¹ = X × W_V¹
       = (10 × 512) × (512 × 64)
       → 10 × 64
    │
    ▼
    每个头的 Attention：

    Q¹(K¹)ᵀ：
    (10 × 64) × (64 × 10) → 10 × 10

    缩放、softmax 后的权重 A¹：10 × 10

    O¹ = A¹ × V¹：
    (10 × 10) × (10 × 64) → 10 × 64
    │
    ▼
    拼接 8 个头的输出：
    10 × (8 × 64) → 10 × 512
    │
    ▼
    输出投影，乘以 W_O¹：
    (10 × 512) × (512 × 512) → 10 × 512
    │
    ▼
    与本层输入 X 做残差相加，再归一化：
    10 × 512
    │
    ▼
    FFN：
    10 × 512 → 10 × 2048 → 10 × 512
    │
    ▼
    残差相加、归一化
    │
    ▼
    输出 H¹：10 × 512
    │
    │  这整个 H¹ 矩阵，就是第 2 层的输入
    ▼
第 2 层：
    每个注意力头分别计算：

    Q² = H¹ × W_Q²
       = (10 × 512) × (512 × 64)
       → 10 × 64

    K² = H¹ × W_K²
       = (10 × 512) × (512 × 64)
       → 10 × 64

    V² = H¹ × W_V²
       = (10 × 512) × (512 × 64)
       → 10 × 64
    │
    ▼
    每个头的 Attention：

    Q²(K²)ᵀ：
    (10 × 64) × (64 × 10) → 10 × 10

    缩放、softmax 后的权重 A²：10 × 10

    O² = A² × V²：
    (10 × 10) × (10 × 64) → 10 × 64
    │
    ▼
    拼接 8 个头的输出：
    10 × (8 × 64) → 10 × 512
    │
    ▼
    输出投影，乘以 W_O²：
    (10 × 512) × (512 × 512) → 10 × 512
    │
    ▼
    与本层输入 H¹ 做残差相加，再归一化：
    10 × 512
    │
    ▼
    FFN：
    10 × 512 → 10 × 2048 → 10 × 512
    │
    ▼
    残差相加、归一化
    │
    ▼
    输出 H²：10 × 512
    │
    │  这整个 H² 矩阵，就是第 3 层的输入
    ▼
第 3 层：
    每个注意力头分别计算：

    Q³ = H² × W_Q³
       = (10 × 512) × (512 × 64)
       → 10 × 64

    K³ = H² × W_K³
       = (10 × 512) × (512 × 64)
       → 10 × 64

    V³ = H² × W_V³
       = (10 × 512) × (512 × 64)
       → 10 × 64
    │
    ▼
    与前面相同的处理流程
    │
    ▼
    输出 H³：10 × 512
    │
    ▼
    ……
```

这里沿用前图的 **Encoder 层结构**，省略 dropout；如果是因果自注意力，需要在 softmax 前加入因果掩码，矩阵维度不变。

**上标表示层数，各层的 W 参数独立。H¹、H² 的形状相同，但内容不同；第 2 层必须使用第 1 层算出的 H¹。**

#### 为啥 dk ,dv 都是64

因为原始 Transformer 的基础配置选了：

- 隐藏维度 `d_model = 512`
- 注意力头数 `h = 8`
- 每个头的维度设为 `d_model / h`

所以：

```text
d_k = d_v = 512 / 8 = 64
```

这样，8 个头的输出拼接后，维度正好回到 512：

```text
每个头的输出：10 × 64
          ↓
8 个头按列拼接：10 × (8 × 64)
          ↓
             10 × 512
```

**64 是这个模型的设计选择，不是 Attention 的数学规定。** 需要区分：

| 维度关系                       | 是否必须 | 原因                           |
| ------------------------------ | -------- | ------------------------------ |
| 同一个头的 Q 和 K 维度相同     | 是       | 要计算 Q 与 K 的点积           |
| V 的维度与 Q、K 相同           | 否       | V 参与的是注意力权重的加权求和 |
| 每个头的维度等于 `d_model / h` | 否       | 这是标准配置中采用的设计       |

例如，即使设 `d_k = 64`、`d_v = 32`，计算也成立：

```text
QKᵀ：
(10 × 64) × (64 × 10) → 10 × 10

AV：
(10 × 10) × (10 × 32) → 10 × 32

8 个头拼接：
10 × (8 × 32) → 10 × 256

再用 W_O 投影：
(10 × 256) × (256 × 512) → 10 × 512
```

所以，**之前图中的 `d_k = d_v = 64`，是因为我们采用了原始 Transformer 的基础配置；并不是它们只能等于 64。**

#### 通常情况下 d_k  d_v  需要相等吗

通常会设成相等，但数学上不要求相等。

- **Q 和 K 的维度必须相同**：它们要计算点积，通常都记为 `d_k`。
- **V 的维度 `d_v` 可以不同**：它只参与加权求和，不与 Q、K 做点积。

看矩阵维度就很清楚：

```text
Q：n × d_k
K：n × d_k
V：n × d_v

注意力权重：
A = softmax(QKᵀ / √d_k)     → n × n

输出：
O = AV
  = (n × n) × (n × d_v)    → n × d_v
```

这里没有任何一步要求 d_k = d_v。

常见的标准多头注意力会选择 d_k = d_v = d_model / h，方便维度安排和实现；这是常用配置，不是必要条件。

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

每一层的 K、V 只由当前 token 在该层的输入向量计算。第一层的输入不包含前文信息；后续层的输入通过之前的 attention 融入了前文信息。由于 causal masking，它们始终不依赖后面的 token。

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

