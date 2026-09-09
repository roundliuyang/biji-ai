# Qwen-VL系列解析(一)——Qwen2-VL



### 前言

本文将从Motivation的角度来理解`Qwen2-VL`的设计。

### Architecture

总体来看`Qwen2-VL`是一个decode-only的多模态架构。图片、视频通过`ViT`变为序列特征（后面统称为visual features或visual token）。随后visual features和text features在输入层面进行拼接，作为input-embeddings，再输入到causal transformers中。监督目标任为language loss（visual token的位置不参与loss计算）。从计算层面，与常规decoder-only语言模型的差异在于输入多了visual token。

![notion image](Qwen-VL系列解析(一)——Qwen2-VL.assets/attachment%3A83d5d349-ce96-4d21-9923-56c15b9eed54%3Aimage.png)

Q：如何把视觉信息嵌入到一个纯 decoder-only 的语言建模框架中

我们将从3个层面理解`Qwen2-VL`的设计。

- Visual Tokenization

- Multimodal Sequence Construction

- Multimodal Position Encoding

### Visual Tokenization

`Qwen2-VL`通过vision encoder将视觉输入（图片、视频）变为visual features。架构层面vision encoder包含两个部分：

- Vision Transformer $(\text{Image/Video} \xrightarrow{\mathrm{ViT}} \text{Visual Features})
  $

- PatchMerger （4x压缩visual token， merge spatial）。不做这个压缩视觉token太长了，对decode-only架构负担太大。

区别于一些早年的多模态模型，`Qwen2-VL`的`ViT`有以下两点区别：

- 支持原生分辨率的输入(Native Resolution Input，参考论文[NaViT](https://arxiv.org/abs/2307.06304))，

- `ViT`的参数参与训练。

ViT本身是支持动态分辨率推理的，但需要考虑到：

1. 位置编码需要支持多分辨率。

1. 任意分辨率输入的batch化问题。

针对第一点，`Qwen2-VL`采用[2D-RoPE](https://arxiv.org/abs/2104.09864)。

针对第二点，作者精心设计了image_processing，以便ViT用packing的方式对多分辨率的输入进行并行推理。下面来看具体是怎么做的。









