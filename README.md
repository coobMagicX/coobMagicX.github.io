# coobMagicX.github.io
## 主要的语言模型分类

无监督表征学习首先在大规模未标记的文本语料库上预训练神经网络，然后在下游任务中，对这些模型或表示进行finetune。在这种共享的思想中，存在不同的无监督预训练目标。其中，自回归（Autoregressive，简称AR）语言建模和自编码（Autoencoder，简称AE）一直是最成功的两个预训练目标。我们将此与上面所说的Transformer架构联系起来，**Transformer encoder**是一个**AE模型**，**Transformer decoder**则是一个**AR模型**。

如下图所示（一些主要的基于transformer架构模型），==蓝色==表示Transformer encoder（AE模型），==红色==表示Transformer decoder（AR模型），==灰色==表示Transformer Encoder-Decoder（seq2seq模型）。

![image-20230712214545932](G:\typora_workspace\typora-user-images\image-20230712214545932.png)

![image-20230712214427926](G:\typora_workspace\typora-user-images\image-20230712214427926.png)

## Autoregressive自回归模型

### 介绍

**AR模型，代表作GPT，从左往右学习的模型。**AR模型从一系列time steps中学习，并将上一步的结果作为回归模型的输入，以预测下一个time step的值。**AR模型通常用于生成式任务，在长文本的生成能力很强，比如自然语言生成（NLG）领域的任务：摘要、翻译或抽象问答。**

AR模型利用上/下文词，通过估计文本语料库的概率分布，预测下一个词。

给定一个文本序列，$$x=(x_1,...x_T)$$ 。AR模型可以将似然因式分解为

前向连乘： $$p(x)=∏^T_{t=1}p(x_t|x_{<t})$$

或者后向连乘： $$p(x)=∏^T_{t=T}p(x_t|x_{>t})$$。

![img](https://pic4.zhimg.com/v2-eec5311e9b34be722e195f28d384d1df_r.jpg)

我们知道，训练参数模型（比如神经网络），是用来拟合条件概率分布的。**AR语言模型仅仅是单向编码的（前向或后向），因此它在建模双向上下文时，效果不佳。**下图清晰解释了AR模型的前向/后向性。

### **模型优缺点**

我们总结AR语言模型的优缺点如下：

- 优点：AR模型擅长生成式NLP任务。AR模型使用注意力机制，预测下一个token，因此自然适用于文本生成。此外，AR模型可以简单地将训练目标设置为预测语料库中的下一个token，因此生成数据相对容易。
- 缺点：AR模型只能用于前向或者后向建模，不能同时使用双向的上下文信息，不能完全捕捉token的内在联系。

## Autoencoder自编码模型

### 介绍

**AE模型，代表作BERT**，它不会进行精确的估计，但却具有从被mask的输入中，重建原始数据的能力，即***fill in the blanks\***（填空）。**AE模型通常用于内容理解任务，比如自然语言理解（NLU）中的分类任务：情感分析、提取式问答。**

BERT一直都是很先进的预训练方法，它可以利用双向上下文信息，对原始输入进行重建（恢复）。这个就是相比于AR模型来说的直接优势：缩小了双向信息gap，从而可提高模型性能。然而，BERT在预训练期间使用的[MASK]符号，在微调阶段的真实数据中并不存在，这就导致了预训练-微调的差异。此外，由于预测的token在输入中被mask，导致BERT无法像AR语言模型那样，使用乘积方式对联合概率进行建模。换言之，BERT假设，在给定unmask的token时，待预测的token彼此之间相互独立，这个假设过于简单化了，在自然语言中，high-order和long-range依赖是非常普遍的。

![img](https://pic2.zhimg.com/v2-6d0d201718e86630328cc3eab9e4df09_r.jpg)

### 模型原理 & 优缺点

BERT（及其所有变体，如RoBERTa、DistilBERT、ALBERT等），XLM都是AE模型。

双向的transformer作为编码器，在语言理解相关的文本表示效果很好。缺点是不能直接用于文本生成。自编码模型是通过某个降噪目标（如掩码语言模型）训练的语言编码器，如BERT、ALBERT、DeBERTa。自编码模型擅长自然语言理解任务（natural language understanding tasks），常被用来生成句子的上下文表示。

我们总结AE模型的优缺点如下：

- 优点：上下文依赖：**AR的表示 $$ℎ_(�1:�−1)$$ 仅仅到位置t之前（比如左边的所有token），BERT的表示 ��(�)� 可以涵盖前后向两边的上下文。**BERT使用双向transformer，在语言理解相关的任务中表现很好。

- 缺点：

- - 输入噪声：BERT在预训练过程中使用【mask】符号对输入进行处理，这些符号在下游的finetune任务中永远不会出现，这会导致**预训练-微调差异**。而AR模型不会依赖于任何被mask的输入，因此不会遇到这类问题。
  - BERT在对联合条件概率 �(�¯|�^) 进行因式分解时，基于一个独立假设：在给定了unmasked tokens时，所有待预测（masked）的tokens是相互独立的。

