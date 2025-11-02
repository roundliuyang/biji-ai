# PyTorch自带的 Embedding

PyTorch的`Embedding`模块是一个简单的**查找表**，用于存储固定字典和大小的嵌入。这通常用于存储**词嵌入**，并通过**索引**检索它们。`Embedding`的参数包括：

- 词汇表大小(`num_embeddings`)
- 每个嵌入向量的大小(`embedding_dim`)
- 以及可选参数如`padding_idx`（用于指定哪些索引不应该贡献梯度，从而在训练中保持不变）

通俗点来说，PyTorch中的Embedding技术，**就像是一本巨大的字典，其中每个单词都对应一个数字列表（向量）。**这种技术帮助计算机理解单词之间的关系，就像我们通过单词的使用上下文来理解它们的意义一样。在处理文本或语言数据时，Embedding可以将简单的单词转换成计算机能够处理的数值形式，让计算机能够更好地学习和理解自然语言。

### nn.Embedding 作用：

1. **表示离散数据**：`nn.Embedding`主要用于将离散的数据（如单词、标签、任何类型的类别数据）转换为**连续的向量表示**。这种表示使得模型能够更好地处理和学习离散数据的特征。
2. **捕捉语义信息**：在自然语言处理（NLP）任务中，**嵌入向量能够捕捉单词之间的语义关系**。例如，语义上相近的单词（如“king”和“queen”）在嵌入空间中的向量表示也会相近。
3. **降维**：文本数据通常以高维的稀疏形式表示（如one-hot编码），其中大部分元素都是0。`nn.Embedding`可以**将这些高维稀疏表示转换为低维稠密向量**，从而减少模型的计算负担和提高效率。
4. **参数共享**：使用嵌入层可以实现**参数共享**，即相同的单词在不同的上下文中会使用相同的嵌入表示。这有助于模型在训练过程中学习到更泛化的特征。

### nn.Embedding 的原理：

`nn.Embedding`层本质上是**一个权重矩阵**，其中每一行代表词汇表中每个单词的向量表示。这个**权重矩阵的大小是`[num_embeddings, embedding_dim]`**，其中`num_embeddings`是词汇表的大小，`embedding_dim`是嵌入向量的维度。

- 当我们向嵌入层传递一个单词的**索引**时，它实际上是在**查找这个索引对应的权重矩阵中的行**。**这一行就是该单词的嵌入向量。**
- **这个权重矩阵是可以学习的，意味着在模型的训练过程中，通过反向传播算法，嵌入向量会根据损失函数不断更新，以更好地表示数据中的语义关系。**

## 例一：PyTorch 自带 Embedding 的简单实用

接下来，我们可以通过一个简单的例子来展示如何使用 PyTorch 进行 Embedding 的过程。这个过程大致可以分为以下几个步骤：

### 第一步：导入并探索数据

为了简单起见，假设我们有以下句子作为我们的数据集：

- "I love machine learning"
- "PyTorch is great for deep learning"

我们首先需要对这些句子进行预处理，包括分词和构建词汇表。

```python
import torch
import torch.nn as nn

# 假定的数据集
sentences = ["I love machine learning", "PyTorch is great for deep learning"]

# 分词并构建词汇表
word_set = set(word for sentence in sentences for word in sentence.lower().split())
word_to_ix = {word: ix for ix, word in enumerate(word_set)}

# 定义 Embedding 的维度
embeds = nn.Embedding(len(word_to_ix), 5)  # 假设嵌入维度是 5

# 查看一下词汇表
word_to_ix

# 结果如下：

{'pytorch': 0,
 'learning': 1,
 'love': 2,
 'is': 3,
 'i': 4,
 'for': 5,
 'deep': 6,
 'machine': 7,
 'great': 8}
```

### 第二步：使用 PyTorch 完成 Embedding

在这一步中，我们将使用 PyTorch 的 `torch.nn.Embedding` 模块来创建一个 Embedding 层。这个层将每个唯一的单词映射到一个高维空间中的向量。

### 第三步：展示 Embedding 前后的数据对比

在完成 Embedding 后，我们将展示原始文本和经过 Embedding 转换后的向量表示，以便观察到转换前后的区别。

```python
## 我们把上面的两个步骤代码放在一起

# 示例：获取单词 "love" 和 "pytorch" 的嵌入向量
words_to_embed = ['love', 'pytorch']

word_indices = torch.tensor([word_to_ix[word] for word in words_to_embed], dtype=torch.long)

embedded_words = embeds(word_indices)

# 查看嵌入向量
embedded_words.data

## 结果如下：
tensor([[ 0.9290, -0.9860, -2.3795, -0.0714, -1.7284],
        [-0.8182,  0.3285,  0.9673,  0.8071, -1.0288]])
```

从上面可以看出，我们通过语料建立词汇表、然后将两个单词通过PyTorch自带的神经网络模型，将两个单词，转换成为了2个词向量。

### 为什么在 Embedding 的时候，要知道词汇表的长度？（len(word_to_ix)，Embedding 的第二个参数）

-- 传入词汇表大小的参数是为了定义嵌入矩阵的大小，这个矩阵有多少行就代表可以嵌入多少不同的单词。即使在转换过程中只考虑了两个单词，**整个嵌入层仍然需要为词汇表中的每一个单词准备好一个嵌入向量**。这是因为嵌入层设计成可适用于整个词汇表的，以便在训练过程中对任何单词进行嵌入转换，而不仅仅是当前正在处理的那两个单词。

既然权重矩阵是可以学习的，那平时我们直接调用 nn.embedding 的时候，我们需要重新学习这个权重矩阵吗？

当你在PyTorch中直接使用nn.Embedding时，权重矩阵确实是可学习的，意味着它会在训练过程中根据反向传播算法自动更新。但是，是否需要从头开始学习这个权重矩阵，还是使用预训练的嵌入向量，取决于具体的应用场景：

1. **从头开始学习**：如果你的任务是特定的，且你有足够的标记数据来训练模型，那么你可以从随机初始化的嵌入向量开始，让模型在训练过程中学习这些向量。这种方法的优点是能够学习到针对特定任务的嵌入表示，但缺点是需要大量的训练数据。
2. **使用预训练的嵌入向量**：在很多情况下，尤其是当训练数据较少时，使用预训练的嵌入向量（如[Word2Vec](https://zhida.zhihu.com/search?content_id=239754351&content_type=Article&match_order=1&q=Word2Vec&zhida_source=entity)、GloVe或[FastText](https://zhida.zhihu.com/search?content_id=239754351&content_type=Article&match_order=1&q=FastText&zhida_source=entity)）可以显著提高模型的性能。这些预训练向量通常是在非常大的文本语料库上训练得到的，能够捕捉到丰富的语义信息。在PyTorch中，你可以通过初始化`nn.Embedding`层的权重为这些预训练向量来使用它们。即使使用预训练向量，你也可以选择在训练过程中进一步微调（更新）这些向量，或者保持它们不变。

## 例二: 基于Embedding 的建模

下面，我们通过一个简化的例子来展示如何使用PyTorch实现文本分类。关键步骤包括：

1. **准备数据**：将文本数据转换成整数索引，构建一个词汇表。
2. **定义模型**：创建一个简单的神经网络模型，该模型包含一个嵌入层和一个线性层。
3. **训练模型**：使用简化的文本数据和对应的标签来训练模型。
4. **预测与评估**：使用训练好的模型对新句子进行分类，并计算准确率。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 假定的简化文本数据集和对应的标签
sentences = ["I love machine learning", "PyTorch is great for deep learning"]
labels = torch.tensor([0, 1], dtype=torch.long)  # 0 和 1 代表两个不同的类别

# 分词并构建词汇表：每个唯一单词映射到一个唯一的整数
word_to_ix = {word: i for i, word in enumerate(set(" ".join(sentences).lower().split()))}

# 定义一个简单的文本分类模型
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleClassifier, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # 嵌入层
        self.linear = nn.Linear(embedding_dim, 2)  # 线性层，假设有两个输出类别

    def forward(self, inputs):
        embeds = self.embeddings(inputs)  # 将输入的整数索引转换为嵌入向量
        return self.linear(torch.mean(embeds, dim=0))  # 对嵌入向量取平均并通过线性层

# 初始化模型、损失函数和优化器
model = SimpleClassifier(len(word_to_ix), 5)  # vocab_size为词汇表大小，embedding_dim为嵌入向量的维度
loss_function = nn.CrossEntropyLoss()  # 使用交叉熵损失函数，适用于分类问题
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降作为优化器

# 准备训练数据：将文本转换为整数索引的形式
data = [[word_to_ix[word] for word in sentence.lower().split()] for sentence in sentences]
data = torch.tensor([sum(x)//len(x) for x in data], dtype=torch.long)  # 简化处理：将句子表示为平均索引

# 训练模型
for epoch in range(100):  # 迭代100次
    for instance, label in zip(data, labels):
        model.zero_grad()  # 清除梯度
        log_probs = model(instance.view(1, -1))  # 前向传播
        loss = loss_function(log_probs, label.view(-1))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

# 使用模型进行预测
test_sentence = "PyTorch is great for deep learning"  # 测试句子
test_label = torch.tensor([1], dtype=torch.long)  # 测试句子的真实标签

# 准备测试数据
test_data = [word_to_ix[word] for word in test_sentence.lower().split()]
test_data = torch.tensor(sum(test_data)//len(test_data), dtype=torch.long)  # 简化处理：将测试句子表示为平均索引

# 不计算梯度，进行预测
with torch.no_grad():
    log_probs = model(test_data.view(1, -1))

# 获取预测结果
_, predicted_label = torch.max(log_probs, 1)

# 计算准确率
accuracy = (predicted_label == test_label).float().mean().item()

accuracy

## 结果如下
1.0
```

(文章结束)

