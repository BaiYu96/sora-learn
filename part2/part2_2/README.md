## Attention & LLM思考和练习

### Attention

- 你怎么理解Attention？

> Attention，注意力机制，是让模型能够知道数据集里面哪些数据是重要。注意力机制有三要素，Query, Key, Value. 
>
> - Value 承载着被分为多个部分的目标信息
> - Key 则是谢谢目标信息的索引
> - Query 代表着注意力的顺序
>
> 注意力的运算过程就是通过 Query 序列去检索 Key 值，从而获得适合的 Value 信息

- 乘性Attention和加性Attention有什么不同？

> 在乘性 Attention  中，注意力权重是通过两个向量之间的乘积来计算得到的
>
> 加性注意力中，注意力权重是通过将查询向量和键向量连接起来，然后经过一个神经网络层来计算得到的

- Self-Attention为什么采用 Dot-Product Attention？

> 在实际应用中，乘性Attention （Dot-Product Attention）更快、更节省空间，因为它可以使用高度优化的矩阵乘法代码来实现。

- Self-Attention中的Scaled因子有什么作用？必须是 `sqrt(d_k)` 吗？

> 如果 d<sub>k</sub> 很小， additive attention 和 dot-product attention 相差不大。
> 但是如果 d<sub>k</sub> 很大，点乘的值很大，如果不做 scaling，结果就没有 additive attention 好。
> 另外，点乘结果过大，使得经过 softmax 之后的梯度很小，不利于反向传播，所以对结果进行 scaling。
>
> 必须是sqrt(d_k)

- Multi-Head Self-Attention，Multi越多越好吗，为什么？

> 不是越多越好。在 Multi-Head Self-Attention 中，通过并行地应用多个注意力头（attention head），可以增加模型对不同关注点的表达能力，从而提高模型对序列中复杂关系的建模能力。每个注意力头可以学习关注序列中不同方面的信息，然后将这些信息整合在一起。增加注意力头的数量会增加模型的参数量和计算量，从而增加训练和推理的时间和资源消耗。

- Multi-Head Self-Attention，固定`hidden_dim`，你认为增加 `head_dim` （需要缩小 `num_heads`）和减少 `head_dim` 会对结果有什么影响？

> head_dim表示每个注意力头所能表示的信息维度，对应模型的表达能力。增加head_dim,意味着减少num_heads的数量，会使得整体的计算量减少。增加head_dim在一定程度上会提高计算效率。

- 为什么我们一般需要对 Attention weights 应用Dropout？哪些地方一般需要Dropout？Dropout在推理时是怎么执行的？你怎么理解Dropout？

> 在注意力机制中应用 Dropout 的主要目的是为了增加模型的泛化能力和减少过拟合

- Self-Attention的qkv初始化时，bias怎么设置，为什么？

> 在 Self-Attention 中，通常不需要显式地设置 qkv 的偏置（bias）。qkv 的值是通过输入序列的特征经过线性变换（通常是全连接层）得到的，这个线性变换包括了偏置项（bias）。

- 你还知道哪些变种的Attention？它们针对Vanilla实现做了哪些优化和改进？
- 你认为Attention的缺点和不足是什么？

> 注意力机制需要计算序列中所有位置之间的关联性，因此在处理长序列数据时会导致计算复杂度的增加，特别是当注意力头的数量较多时，计算量会更大。

- 你怎么理解Deep Learning的Deep？现在代码里只有一个Attention，多叠加几个效果会好吗？

> "Deep" 指的是模型的深度，也就是指模型中的层数较多。在深度学习中，通过增加模型的深度，可以使模型更好地学习到数据中的复杂特征和抽象表示，从而提高模型的性能和泛化能力。

- DeepLearning中Deep和Wide分别有什么作用，设计模型架构时应怎么考虑？

> "Wide" 指的是模型的宽度，即模型中的参数数量较多，但层数相对较少。Wide部分的主要作用是让模型具有较强的“记忆能力”；
>
> "Deep" 指的是模型的深度，即模型中的层数较多。Deep部分的主要作用是让模型具有“泛化能力”

### LLM

- 你怎么理解Tokenize？你知道几种Tokenize方式，它们有什么区别？

> "Tokenize" 是指将文本或语言数据分割成一个个单独的单位，通常称为 token。Tokenize 在自然语言处理中是一个重要的预处理步骤，用于将原始文本转换成模型可以理解和处理的输入形式。

- 你觉得一个理想的Tokenizer模型应该具备哪些特点？

> 一个理想的 Tokenizer 模型应该具备准确性、通用性、灵活性、效率、可扩展性、鲁棒性和可解释性等特点，以满足不同应用场景和需求下的文本处理需求。
>
> Tokenizer 应该能够准确地将输入文本划分为合适的 token，保留文本的语义和结构信息。具有灵活的配置选项，以适应不同的应用场景和需求

- Tokenizer中有一些特殊Token，比如开始和结束标记，你觉得它们的作用是什么？我们为什么不能通过模型自动学习到开始和结束标记？

> 特殊标记在 Tokenizer 中起着指示序列边界、填充序列、分割序列和注意力掩码等作用，有助于模型更好地处理序列数据和执行不同的序列任务

- 为什么LLM都是Decoder-Only的？

> 1. Decoder-only架构计算高效：相对于Encoder-Decoder架构，Decoder-only架构不需要编码器先编码整个输入序列，所以训练推理速度更快。
> 2. Decoder-only架构内存占用少：Encoder-Decoder架构由于编码器的特点，每个patch的sentence都需要用pad来补齐，Decoder only架构不需要，因此可以减少内存占用。
> 3. Decoder-only架构良好的泛化能力：Decoder only架构通常使用自回归模型,即每个单词的预测都是基于之前生成的单词。这种方法可以更好地处理复杂的语言结构，并提高模型的泛化能力。

- RMSNorm的作用是什么，和LayerNorm有什么不同？为什么不用LayerNorm？

> RMSNorm(Root Mean Square Normalization) 是一种归一化技术，类似于 Batch Normalization（Batch Norm）和 Layer Normalization（Layer Norm）。RMSNorm 的主要作用是在深度学习模型中帮助提高训练稳定性和泛化性能。
>
> RMSNorm 与 LayerNorm 的不同之处在于 LayerNorm是对样本内特征维度的均值和方差进行归一化，而RMSNorm是使用样本内通道维度的方差进行归一化

- LLM中的残差连接体现在哪里？为什么用残差连接？

> 体现在模型的层之间，特别是在Transformer等架构中的每个子层。
>
> 残差连接在深度学习模型中的使用有助于缓解梯度消失问题、简化网络训练、加速网络收敛

- PreNormalization和PostNormalization会对模型有什么影响？为什么现在LLM都用PreNormalization？

> PreNormalization 和 PostNormalization 都是模型归一化的方式。相比PostLN，使用PreLN的深层transforme的训练更稳定，但是性能有一定损伤。为了提升训练稳定性，很多大模型都采用了PreLN

- FFN为什么先扩大后缩小，它们的作用分别是什么？

> FFN(Feed Forward Network) 前馈神经网络。扩展层主要用于增加模型的表示能力，使得模型可以学习到更加复杂和抽象的特征表示；而压缩层则用于减少模型的计算成本和参数数量，并提高模型的泛化能力和鲁棒性。通过这种扩大后缩小的设计，模型可以充分利用扩展层学到的丰富特征表示，同时通过压缩层将这些特征表示压缩为更简洁的形式，从而提高模型的效率和性能。

- 为什么LLM需要位置编码？你了解几种位置编码方案？

> 在LLM中位置编码的作用主要是**处理序列信息**和**克服位置信息丢失问题**

- 为什么RoPE能从众多位置编码中脱颖而出？它主要做了哪些改进？

> RoPE（Rotary Positional Embedding）是一种用于表示序列位置信息的新型方法，特别适用于在自注意力模型（如Transformer）中使用。RoPE的提出源于对位置编码的改进，旨在提高模型的性能和效率。

- 如果让你设计一种位置编码方案，你会考虑哪些因素？

> 1. **序列长度**：位置编码方案应该能够适应不同长度的输入序列，从而保持序列中的位置关系。
> 2. **维度规模**：位置编码方案应该具有较低的维度表示，以减少模型的参数数量和计算复杂度。
> 3. **可解释性**：位置编码方案应该具有良好的可解释性，使得模型能够理解和利用输入序列中的位置信息。
> 4. **泛化能力**：位置编码方案应该能够适用于不同任务和数据集，并且具有良好的泛化能力。
> 5. **不依赖额外参数**：位置编码方案应该尽可能地减少对额外参数的依赖，以提高模型的效率和稳定性。

- 请你将《LLM部分》中的一些设计（如RMSNorm）加入到《Self-Attention部分》的模型设计中，看看能否提升效果？