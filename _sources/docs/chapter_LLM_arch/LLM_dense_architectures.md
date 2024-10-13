(chapter_LLM_arch_sec_LLM_arch_fundamentals)=
# LLM Architectures Fundamentals

## Overview



```{figure} ../img/chapter_LLM_arch//large_langage_models_release_timeline.png
---
scale: 50%
name: chapter_foundation_fig_pretrained_LM_transformer_arch
---
A timeline of existing large language models (having a size larger than 10B) in recent years. We mark the open-source LLMs in yellow color. Image from {cite:p}`zhao2023survey`.
```


* **Scale**: LLMs are trained on enormous datasets, often containing hundreds of billions of words or tokens. This massive scale allows them to capture intricate patterns and nuances in language. For example, GPT-3 was trained on about 500 billion tokens, while some more recent models have used even larger datasets.

* **Transformer architecture**: Most modern LLMs use transformer architectures, which were introduced in the "Attention is All You Need" paper. These models can have billions of parameters - GPT-3 has 175 billion, for instance. The transformer architecture allows for efficient parallel processing and captures long-range dependencies in text.

* **Few-shot learning via prompting**: LLMs can often perform new tasks with just a few examples provided in the prompt. This "in-context learning" allows them to adapt to new tasks without changing their weights, demonstrating a form of meta-learning.

* **Multi-task capability**: A single LLM can perform various language tasks such as translation, summarization, question-answering, and text generation without needing separate models for each task. This versatility makes them powerful tools for a wide range of applications.

* **Self-supervised learning**: LLMs are typically pre-trained using self-supervised learning techniques. The most common approach is next-token prediction, where the model learns to predict the next word in a sequence given the previous words. This allows the model to learn from vast amounts of unlabeled text data.

* **Emergent reasoning abilities**: As LLMs grow in size and complexity, they often develop capabilities that weren't explicitly trained for. For example, a) Arithmetic: Some LLMs can perform basic math operations despite not being trained specifically on mathematics. b) Logical reasoning: Models may show ability to follow simple logical arguments or solve puzzles.

* **Transfer learning**: The pre-trained LLM can be fine-tuned on specific tasks or domains with much smaller datasets. This transfer learning approach is powerful because the model can leverage its general language understanding for specialized tasks, often outperforming models trained from scratch on those tasks.

*  **Hallucination**: LLMs can sometimes generate text that sounds plausible but is factually incorrect or nonsensical. This "hallucination" occurs because the models are optimizing for plausible text generation rather than strict factual accuracy. It's a significant challenge in deploying LLMs for applications requiring high reliability.

## Position Embeddings

### Absolute Position


### Rotary Postion Embedding

#### The mechanism
The key idea of Rotary position embedding (Rope) is to multiply query vector $\boldsymbol{q}$ (of a token) and key vector $\boldsymbol{k}$ (of another token) by a rotational matrix $R(\theta_q)$ and $R(\theta_k)$, where $\theta_q$ and $\theta_k$ are taking values to indicate the positions of query vector token and key vector token, respectively.

Specifically, let $\theta_q = m\theta$ and $\theta_k = n\theta$, where $m$ and $n$ are integer positions of query vector token and key vector token. 
Now we are showing that the rotated query-key inner product is a function of the relative position $(m - n)$

$$
\begin{aligned}
\left\langle R\left(\theta_q\right) \boldsymbol{q}, R\left(\theta_k\right) \boldsymbol{k}\right\rangle & =\boldsymbol{q}^{\top} R\left(\theta_q\right)^{\top} R\left(\theta_k\right) \boldsymbol{k} \\
& =\boldsymbol{q}^{\top} R\left(\theta_k-\theta_q\right) \boldsymbol{k} \\
& =\left\langle R\left(\theta_q-\theta_k\right) \boldsymbol{q}, \boldsymbol{k}\right\rangle \\
& =\left\langle R\left(\theta (m-n)\right) \boldsymbol{q}, \boldsymbol{k}\right\rangle 
\end{aligned}
$$

We have used the following important properties of rotational matrix:

1. The transpose of a rotation matrix is equal to its inverse: $R(\theta)^{\top}=R(-\theta)$. 
   
2. The matrix multiplication of rotational matrices satisfies: $R(\theta_x)\cdot R(\theta_y) = R(\theta_x + \theta_y)$

In other words, the inner product of two rotated vectors is equal to the inner product of one vector rotated by their angle difference and the other original vector.


#### Practival Implementations

First,The rotation matrix is as follows:

$$
R(\theta)=\left[\begin{array}{cc}
\cos (\theta) & -\sin (\theta) \\
\sin (\theta) & \cos (\theta)
\end{array}\right]
$$


## Layer normalization

### Layer normalization basics

The LayerNorm was originally proposed to overcome the  in combating the internal covariate shift issue {cite:p}`ioffe2015batchnormalizationacceleratingdeep`, where a layer’s input distribution changes as previous layers are updated, causing the difficulty of traning deep models.

The key idea in LayerNorm is to normalize the input to the neural network layer via
* re-centering by subtracting the mean
* re-scaling by dividing the standard deviation.
  
The calculation formula for an input vector $x$ with $H$ feature dimension is given by

$$
\operatorname{LayerNorm}(x) =\frac{x-\mu}{\sqrt{\sigma+\epsilon}} \cdot \gamma+\beta
$$(chapter_LLM_arch_layer_nomalization_formula)

where
* $\mu$ is the mean across feature dimensions, i.e., $\mu = \frac{1}{H} \sum_{i=1}^H x_i $.
* $\sigma$ is the standard deviation across feature dimensions, i.e.,
$\sigma =\sqrt{\frac{1}{H} \sum_{i=1}^H\left(x_i-\mu\right)^2+\epsilon}$.
* $\epsilon$ is a small number acting as regularizer for division.
* $\gamma$ and $\beta$ are learnable scaling and shifting parameters



### RMS Norm (Root Mean Square Norm)

A common hypothesis on why layer normalization can help stalize training and boost model convergence is the capability in  handling re-centering and re-scaling of both inputs and weight matrix. RMSNorm {cite:p}`zhang2019rootmeansquarelayer` is a technique aiming to achieve similar model training stablizing benefit with a reduced computational overhead compared to LayerNorm. RMSNorm hypothesizes that only the re-scaling component is necessary and proposes the following simplified normalization formula

$$
\operatorname{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{H} \sum_{i=1}^H x_i^2}} \cdot \gamma
$$

where $\gamma$ is learnable parameter. Experiments show that RMSNorm can achieve on-par performance with LayerNorm with much reduced training cost.


In the following, we summarize the differences between RMSNorm and LayerNorm

::::{grid}
:gutter: 2

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #e4ac94">**Computational complexity**</span></p>
**LayerNorm** involves both mean and variance calculation for each normalization layer, which brings sizable computational cost for high-dimensional inputs in LLM (e.g., GPT-3 $d_model = 12288$). 
**RMSNorm**, on the other hand, only keeps the variance calculation, reducing the normalization cost by half and increses efficiency
:::

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #b4c9da">**Gradient propogation**</span></p>
**LayerNorm** stablizes the input distribution between layers through normalization and benefits deep networks training by alleviating the problem of vanishing or exploding gradients. However, LayerNorm can also be affected by noise and input shifts when calculating the mean, potentially leading to unstable gradient propagation.
**RMSNorm**, by using only RMS for normalization, can provide a more robust, smoother gradient flow, especially in deeper networks. It reduces the impact of mean on gradient fluctuations, thereby improving the stability and speed of training.
+++
See {cite:p}`zhang2019rootmeansquarelayer` for math derivation
:::
::::



### Layer normalization position



{cite:p}`xiong2020layer`

The residual connection $x+F(x)$ in the Transformer layer will modify the variance of input $x$. To see this, let the variance of $x$ be $\sigma_1^2$ and the variance of $F(x)$ be $\sigma_2^2$. Then the variance of $x + F(x)$ will be given by 

$$ 
Var[x + F(x)] = \sigma_1^2 + \sigma_2^2 + \rho \sigma_1\sigma_2
$$

The Post-Norm thus can stablize the variance of the output by applying the LayerNorm after the residual connection, which is given by

$$\operatorname{PostNorm Output} = \operatorname{LayerNorm}(X + \operatorname{SubLayer}(X))$$

Here the SubLayer could be the FeedForward Layer or the Attention Layer.

Clearly, the normalization will reduce the effect of identity mapping $I(x) = x$ and therefore the gradient flow via the residual connection (aka high-way connection). 
As a result, using Post-Norm in general will require carefully designed learning rate warm-up stage (the optimization starts with
a small/tiny learning rate, and then gradually increases it to a pre-defined maximum value within certain number of steps.) to speed up model training and covergence, including learning rate warm-up and other hyper-parameter tuning.

When it comes to training very deep models, Post-norm can lead to more unstable gradients during training, especially in very deep networks. This can lead to slower convergence and increased likelihood of training failure.



Pre-LN Transformers without the warm-up stage can reach comparable results
with baselines while requiring significantly less training time and hyper-parameter tuning on a
wide range of applications.

$$\operatorname{PreNorm Output} = X + \operatorname{SubLayer}(\operatorname{LayerNorm}(X))$$

Intuitively, as the residual connection route $X$ is not being normalized, the gradient flow via the idenity mapping is therefore not compromised. This help the training of very deep neural networks by mitigating the vanishing gradient issue. 

```{figure} ../img/chapter_foundation/pretrainedLM/Transformer_arch/layer_normalization_position.png
---
scale: 30%
name: chapter_LLM_arch_fig_pretrained_LM_transformer_layernormalizationposition
---
Post-layer normalization and pre-layer normalization in an encoder layer.
```

::::{grid}
:gutter: 2

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #e4ac94">**Pre-Norm**</span></p>

- In the Pre-Norm architecture, the normalization operation (RMS Norm or Layer Norm) is performed before the self-attention or feed-forward neural network (FFN) calculations. In other words, the input to each layer is first normalized before being passed to the attention or feed-forward layers.
- Pre-Norm ensures that the magnitude of inputs remains within a stable range in deep networks, which is particularly beneficial for models with long-range dependencies. By performing normalization operations early, the model can learn from more stable inputs, thus helping to address the problem unstable gradients in deep models.
- LLMs like GPT, LLama are using Pre-Norm design
:::

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #b4c9da">**Post-Norm**</span></p>
- In the Post-Norm architecture, the normalization operation is performed after the self-attention or FFN calculations. The model first goes through unnormalized operations, and finally, the results are normalized to ensure balanced model outputs.
- Post-Norm can achieve good convergence effects in the early stages of training, performing particularly well in shallow models. However, in deep networks, the drawback of Post-Norm is that it may lead to gradient instability during the training process, especially as the network depth increases, gradients may become increasingly unstable during propagation.
:::
::::

### Layer normalization example choices


The core advantages of RMS Pre-Norm lie in its computational simplicity and gradient stability, making it an effective normalization choice for deep neural networks, especially large language models. This is exampified by the fact that LLaMa series started to use Pre-RMSNorm whereas GPT-3 model used Pre-LayerNorm.  
* Improved computational efficiency: As RMS Norm omits mean calculation, it reduces the computational load for each layer, which is particularly important in deep networks. Compared to traditional Layer Norm, RMS Norm can process high-dimensional inputs more efficiently.
* Enhanced gradient stability: RMS Pre-Norm can reduce instances of vanishing gradients, especially in deep networks. This normalization method improves training efficiency by smoothing gradient flow.
* Suitable for large-scale models: For models like LLaMA, RMS Pre-Norm supports maintaining a relatively small model size while ensuring powerful performance. This allows the model to maintain good generalization capabilities without increasing complexity.



## Self-attention and Variants



Certainly! I'll provide a detailed summary of different attention modules used in Large Language Models (LLMs), including Multi-Head Attention (MHA), Grouped Query Attention (GQA), and others. I'll explain their mechanisms, advantages, and use cases.

### Multi-Head Attention (MHA)

Multi-Head Attention is the foundation of many transformer-based models, including the original transformer architecture.

Computation of an MHA given input $X$ matrix and project matrices $W^Q_i, W^K_i, W^V_i$ 

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

where each head is computed as:

$$
\text{head}_i = \text{Attention}(XW^Q_i, XW^K_i, XW^V_i)
$$

$$
\text{Attention}(Q, K, V) = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

```{figure} ../img/chapter_LLM_arch/attention/MHA.png
---
scale: 50%
name: chapter_LLM_arch_fig_fundamentals_attention_MHA
---
Multi-head attention has $H$ query, key, and value heads for each token.
```

::::{grid}
:gutter: 2

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #e4ac94">**Advantages**</span></p>

- Improves the model's overall learning capacity
- Different heads allow the model to jointly attend to information from different representation subspaces
:::

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #b4c9da">**Drawbacks**</span></p>
- Computational complexity scales quadratically with sequence length (i.e., huge cost for long context applications)
- During inference stage, each head has its own key and value to cache, bring additional memory burden to inference process.
:::
::::


### Multi Query Attention (MQA)

{cite:p}`shazeer2019fasttransformerdecodingwritehead`


MQA reduces $H$ key and value heads in MHA to a single key and value head, reducing the size of the key-value cache by a factor of $H$. However,
larger models generally scale the number of heads (e.g., GPT-2 has 12 heads; GPT-3 has 96 heads), such that multi-query attention represents a more
aggressive cut in both memory bandwidth and capacity.

```{figure} ../img/chapter_LLM_arch/attention/MHA.png
---
scale: 50%
name: chapter_LLM_arch_fig_fundamentals_attention_MQA
---
Multi-head attention has $H$ query, and one shared single key head and single value head for each token.
```

::::{grid}
:gutter: 2

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #e4ac94">**Advantages**</span></p>

- During inference stage, each head has its own key and value to cache, bring additional memory burden to inference process.
:::

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #b4c9da">**Drawbacks**</span></p>
- Computational complexity scales quadratically with sequence length (i.e., huge cost for long context applications)

- Modeling capacity is largely compromised due to the usage of single head, leading to quality degradation.
:::
::::


### Grouped Query Attention (GQA)

{cite:p}`ainslie2023gqatraininggeneralizedmultiquery`
GQA is an optimization of MHA and MQA that reduces computational complexity while maintaining performance.

A generalization of MQA which uses an intermediate (more than one, less than number of query heads) number of key-value heads.
GQA is shown to achieve quality close to MHA with comparable speed to MQA


Formula:

$$
\text{GQA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

where:

$$
\text{head}_i = \text{Attention}(QW^Q_i, KW^K_{g(i)}, VW^V_{g(i)})
$$

$g(i)$ is a function that maps head index to group index.


```{figure} ../img/chapter_LLM_arch/attention/GQA.png
---
scale: 50%
name: chapter_LLM_arch_fig_fundamentals_attention_GQA
---
GQA divides the key and value heads into multiple groups. Within each group, a single shared
key and value heads are attended to by query heads. GQA interpolats between MHA and MQA.

```


Advantages:
- Reduces parameters and computation compared to MHA
- Maintains most of the performance of MHA

Use cases:
- Large-scale language models where efficiency is crucial

### Sliding Window Attention

This method restricts attention to a local window around each token.

Key idea:
- Each token attends only to a fixed number of neighboring tokens

Formula:

$$
\text{Attention}(Q_i, K_{i-w:i+w}, V_{i-w:i+w})
$$

where $w$ is the window size.

Advantages:
- Linear complexity with sequence length
- Useful for tasks requiring local context

Drawbacks:
- Limited in capturing long-range dependencies

### Sparse Attention

Sparse Attention introduces patterns of sparsity in the attention matrix.

Key idea:
- Predefined or learned patterns determine which tokens can attend to which other tokens

Advantages:
- Can capture both local and global context
- Reduces computational complexity

Examples:
- Longformer: combines sliding window attention with global attention
- Big Bird: uses random, window, and global attention patterns


## Activation



Certainly! I'll expand on each of these activation functions and their formulas, providing more context and explanations.

1. FFN (Feed-Forward Network) Block:

$$
FFN(x)=f(xW_1+b_1)W_2+b_2
$$

The FFN block is a crucial component in many transformer-based architectures. It typically consists of two linear transformations with a non-linear activation function in between.

- $x$ is the input vector
- $W_1$ and $W_2$ are weight matrices
- $b_1$ and $b_2$ are bias vectors
- $f$ is an activation function (often ReLU or GELU)

The FFN block helps in capturing complex patterns and increasing the model's capacity. The first transformation ($xW_1+b_1$) usually projects the input to a higher dimensional space (often 4 times the input dimension), and the second transformation projects it back to the original dimension.

2. GeLU (Gaussian Error Linear Unit):

$$
\operatorname{GeLU}(x) \approx 0.5x\left(1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715x^3\right)\right)\right)
$$

GeLU is a smooth approximation of the ReLU function that incorporates properties of the Gaussian cumulative distribution function.

- It's differentiable everywhere, unlike ReLU
- For positive inputs, it behaves similarly to ReLU
- For negative inputs, it has a small, smooth curve instead of being zero

The formula provided is an approximation of the true GeLU function, which is computationally efficient while maintaining the key properties of GeLU. It's used in models like BERT and GPT-3, often outperforming ReLU in deep networks.

3. Swish:

$$
\operatorname{Swish}_\beta(x)=x \cdot \sigma(\beta x)
$$

Swish is a self-gated activation function introduced by Google Brain.

- $\sigma$ is the sigmoid function: $\sigma(x) = \frac{1}{1+e^{-x}}$
- $\beta$ is a trainable parameter or can be set to 1

Properties of Swish:
- Smooth and non-monotonic
- Unbounded above and bounded below
- Approaches linear function for large positive inputs
- Can outperform ReLU in very deep networks

4. GLU (Gated Linear Unit) in FFN:

$$
\begin{aligned}
& GLU(x)=\sigma(xW+b) \otimes xV \\
& FFN_{GLU}=(f(xW_1) \otimes xV)W_2
\end{aligned}
$$

GLU introduces a gating mechanism to the FFN block.

- $\sigma$ is the sigmoid function
- $\otimes$ represents element-wise multiplication
- $f$ is an activation function (often GeLU)
- $W$, $V$, $W_1$, and $W_2$ are weight matrices

The gating mechanism allows the network to control information flow, potentially capturing more complex dependencies. The sigmoid function acts as a gate, determining how much of the linear transformation should pass through.

5. GeGLU (GeLU-based Gated Linear Unit):

$$
GeGLU(x)=GeLU(xW) \otimes xV
$$

GeGLU combines the GeLU activation with the gating mechanism of GLU.

- GeLU is applied to one branch ($xW$)
- The other branch ($xV$) remains linear
- Element-wise multiplication combines the two branches

This formulation can provide the benefits of both GeLU activation and gated mechanisms, potentially leading to improved performance in some tasks.

6. SwiGLU (Swish-based Gated Linear Unit):

$$
SwiGLU=\text{Swish}_\beta(xW) \otimes xV
$$

SwiGLU replaces the GeLU function in GeGLU with the Swish activation.

- Swish is applied to one branch ($xW$)
- The other branch ($xV$) remains linear
- Element-wise multiplication combines the two branches

The use of Swish can potentially provide different dynamics compared to GeLU, and the trainable parameter $\beta$ in Swish adds an extra degree of flexibility to the model.

These variations on activation functions and gating mechanisms represent ongoing research in improving the performance and capabilities of neural networks, especially in the context of large language models. Each has its own strengths and may be more suitable for different types of tasks or model architectures.


Examples in LLM:

| LLM | Activation Function |
| :---: | :---: |
| GPT3 | GeLU |
| LLaMA | SwiGLU |
| LLaMA2 | SwiGLU |
| baichuan | SwiGLU |
| ChatGLM- <br> 6B | GeLU |
| ChatGLM2- <br> 6B | SwiGLU |
| Bloom | GeLU |
| Falcon | GeLU |

## Tokenziation, vocabulary, and weight tying

### BPE Tokenization
Byte Pair Encoding (BPE) is a commonly used subword tokenization algorithm in NLP. It starts with individual characters and iteratively merges the most frequent pairs to create new subword units, repeating this process N times to build the final subword vocabulary. The following is a summary of the algorithm.


```{prf:algorithm} BPE
:label: BPE-algorithm

**Inputs** Word list $W$, Number of desired merges $N$

**Output** Subword vocabulary $V = \emptyset$

1. Represent each word as a sequence of characters
2. Initialize the subword vocabulary as a set of single characters
3. For i in 1 to N:
	3.1 Calculate the frequency of each consecutive character pair
	3.2 Find the character pair $(x, y)$ with the highest frequency
	3.3 Merge the character pair $c = (x, y)$, update the subword vocabulary $V = V \cup c$.
4. Return the subword vocabulary $V$.
```

````{prf:example}
GPT-2's vocabulary size is 50,257, corresponding to 256 basic byte tokens, a special end-of-text token, and 50,000 tokens obtained through merging process.
````
### From BPE to BBPE

BPE (Byte Pair Encoding) and BBPE (Byte-level BPE) are both subword tokenization following the same idea of merging algorithm {prf:ref}`BPE-algorithm` but operating on different granularities. 

In short, BPE Works on character or unicode level whereas BBPE works on byte level of UTF-8 representation. Their comparison is summarized as the following.

::::{grid}
:gutter: 2

:::{grid-item-card} <span style="background-color: #e4ac94">**BPE**</span>

In BPE, the generation of subwords is more consistent with linguistic rules (e.g., utilizing word roots). The subword choices often better align with common vocabulary.

However, it often requires different treatments for different languages (like English vs Chinese) and it cannot effectively represent emojis and unseen special tokens. 
:::

:::{grid-item-card} <span style="background-color: #b4c9da">**BBPE**</span>
BBPE has following advantages:
- It can process all character sets (including Unicode characters), making it suitable for multilingual scenarios.
- It provides good support for unconventional symbols and emojis.

However, as BBPE is working on smaller granularity level than characters, it might result in larger vocabulary size (i.e., larger embedding layers) and unnatural subword units.
:::
::::

Currently, many mainstream large language models (such as the GPT series, Mistral[^footnote1], etc.) primarily use BBPE instead of BPE. The reasons for the widespread adoption of this method include:

* Ability to process multilingual text: Large models typically need to handle vast amounts of text in different languages. BBPE operates at the byte level, allowing it to process all Unicode characters, performing particularly well for languages with complex character sets (such as Chinese, Korean, Arabic).
* Unified tokenization: The BBPE method does not rely on language-specific character structures. Therefore, it can be uniformly applied to multilingual tasks without adding extra complexity, simplifying the tokenization process in both pre-training and downstream tasks.
* Compatibility with emojis and special characters: Modern large language models need to process large amounts of internet data, which contains many emojis, special characters, and non-standard symbols. BBPE can better support these types of symbols.

[^footnote1]: See https://docs.mistral.ai/guides/tokenization/ for details for tokenizer construction and usage in Mistral LLMs. 





````{prf:remark} What does Byte-level mean?
"Byte-level" in the context of BBPE means that the algorithm operates on individual bytes of data rather than on characters or higher-level text units. Note that characters are typically encoded using schemes like UTF-8, where a single character might be represented by one or more bytes. In other words, BBPE treats the input as a sequence of raw bytes, without interpreting them as characters or considering character boundaries.

Below is more context about UTF-8 encoding.

1. ASCII encoding:
   - In the original ASCII encoding, each character is represented by a single byte (8 bits).
   - This allows for 256 different characters (2^8 = 256).
   - ASCII mainly covers English letters, numbers, and some basic symbols.

2. Unicode and UTF-8:
   - Unicode was developed to represent characters from all writing systems in the world.
   - UTF-8 is a variable-width encoding scheme for Unicode.
   - In UTF-8, characters can be encoded using 1 to 4 bytes:
     - ASCII characters still use 1 byte
     - Many other characters use 2 or 3 bytes
     - Some very rare characters use 4 bytes

3. Examples:
   - The letter 'A' (U+0041 in Unicode) is represented as a single byte: 01000001
   - The Euro symbol '€' (U+20AC) is represented by three bytes: 11100010 10000010 10101100
   - The emoji '😊' (U+1F60A) is represented by four bytes: 11110000 10011111 10011000 10001010

This multi-byte representation for single characters is why text processing algorithms that work at the character level can be more complex than those that work at the byte level, especially when dealing with multilingual text.

````

## Parameter composition in Transformer models

### Input layer

Word embedding: $n_{vocab} \times d_{model}$
Position embedding: $n_{max\_len} \times d_{model}$

### Attention layer

In general $n_{head} \times d_{head} = d_{model}$
QKV transformation matrix for $n_{head}$: $3 \times n_{head} \times d_{model} \times d_{head}$
There is a transformation matrix takes the multi-head attention output $n_{head} \times d_{head}$ as the input and outputs a $d_{model}$ feature vector. This transformation matrix has weight parameters $n_{head} \times d_{head} \times d_{model}$
In total, we have $4n_{head}d_{head}d_{model} = 4d_{model}^2$.

### Feed-forward layer

The feed-forward network after the attention layer is a two-layer, with two weight matrices of the sizes $d_{model} \times d_{ff}$ and $d_{ff} \times d_{model}$ and two bias vectors of the sizes $d_{model}$ and $d_{ff}$. 
In general $d_{ff} = 4d_{model}$, so the total number of parameters are $8d_{model}^2 + 5d_{model}$. 

### Output layer

The weight-matrix in the output Softmax layer is often tied to the embedding layer.

### Total weight

$$n_{vocab} \times d_{model} + n_{max\_len} \times d_{model} + n_{layer}(4d_{model}^2 + 8d_{model}^2 + 5d_{model})$$

$n_{max\_len} = 2048$ in GPT-3



```{table} Parameters in a Transformer
| Module | Computation | Parameter Name | Shape | Parameter Number |
| :--- | :--- | :--- | :--- | :--- |
| Attention | ${Q} / {K} / {V}$ | weight / bias | $[{d}, {d}] /[{d}]$ | $3 d^2+3 d$ |
|  | Output 映射 | weight / bias | $[{d}, {d}] /[{d}]$ | $d^2+d$ |
|  | layernorm | $V$ 和 $\beta$ | $[{d}] /[{d}]$ | $2 d$ |
| FFN | $f_1$ | weight / bias | $[{d}, 4 {~d}] /[{d}]$ | $4 d^2+d$ |
|  | $f_2$ | weight / bias | $[4 {~d}, {~d}] /[4 {~d}]$ | $4 d^2+4 d$ |
|  | Layernorm | $V$ 和 $\beta$ | $[{d}] /[{d}]$ | $2 d$ |
| Embedding | - | - | $[{V}, {d}]$ | $V d$ |
| **Total** |  |  |  | $V d+L\left(12 d^2+13 d\right)$ |
```

The total number of parameters scales linearly with number of layers $L$ and quadratically with model hidden dimensionality $d$.

````{prf:example}
Take the following GPT-3 13B and 175B as an example, 175B model has approximate 2.4 times of $L$ and $d_{model}$. Extrapolating from 13B model, we estimate the 175B model to have model parameters of $13\times 2.4^3 = 179B$, which is very close.

| Model Name | $n_{\text{params}}$ | $n_{\text{layers}}$ | $d_{\text{model}}$ | $n_{\text{heads}}$ | $d_{\text{head}}$ |
|------------|----------|----------|---------|---------|--------|
| GPT-3 13B | 13.0B | 40 | 5140 | 40 | 128 |
| GPT-3 175B or "GPT-3" | 175.0B | 96 | 12288 | 96 | 128 |
````


## Dense Architecture Examples 

## Summary

```{table} Model cards of several selected LLMs with public configuration details. Here, PE denotes position embedding, #L denotes the number of layers, #H denotes the number of attention heads, dmodel denotes the size of hidden states, and MCL denotes the maximum context length during training.
| Model | Size | Normalization | PE | Activation | Bias | #L | #H | $d_{\text {model }}$ | MCL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT3 [55] | 175B | Pre LayerNorm | Learned | GeLU | $\checkmark$ | 96 | 96 | 12288 | 2048 |
| Llama | 207B | Pre RMSNorm | Learned | SwiGLU | $\checkmark$ | 64 | 128 | 16384 | 1024 |
| Qwen 2 {cite:p}`yang2024qwen2technicalreport`| 72B | Pre RMSNorm | RoPe | SwiGLU | $\checkmark$ | 80 | 64 | 8192 | ... |
```

% from A Survey of Large Language Models 


## LLama architectures



(chapter_LLM_arch_sec_LLM_arch_fundamentals_forward_pass_computation)=
## Forward Pass Computation Breadown


b: batch_size
s: seq_len
d: d_model
V: vocabulary_size
L: n_layers


以矩阵乘为例, 输入 $[M, K] \times[K, N]=[M, N]$, 计算时间复杂度为 $2 M N K$ 。也就是说输出矩阵 $M N$ 个元素, 每个元素经过一次乘法和一次加法运算。



```{table} Computation breakdown
| Module | Computation | Matrix Shape Changes | FLOPs |
| :--- | :--- | :--- | :--- |
| Attention | ${Q} / {K} / {V}$ Projection | $[{b}, {s}, {d}] \times [{~d}, {~d}]\to[{b}, {s}, {d}]$ | $3\times 2 b s d^2$ |
|  | $Q K^T$ | $[{~b}, {~s}, {~d}] \times [{~b}, {~d}, {~s}]\to[{b}, {s}, {s}]$ | $2 b s^2 d$ |
|  | score $ \times V$ | $[{~b}, {~s}, {~s}] \times [{~b}, {~s}, {~d}]\to[{b}, {s}, {d}]$ | $2 b s^2 d$ |
|  | Output | $[{b}, {s}, {d}] \times[{~d}, {~d}]\to[{b}, {s}, {d}]$ | $2 b s d^2$ |
| FFN | $f_1$ | $[{~b}, {~s}, {~d}] \times[{~d}, 4 {~d}] \to [{b}, {s}, 4 {~d}]$ | $8 b s d^2$ |
|  | $f_2$ | $[{~b}, {~s}, 4 {~d}] \times[4 {~d}, {~d}]\to[{b}, {s}, {d}]$ | $8 b s d^2$ |
| Embedding |  | $[{b}, {s}, 1] \times[{~V}, {~d}]\to[{b}, {s}, {d}]$ | $2 b s d V$ |
| In total |  |  | $\left(24 b s d^2+4 b d s^2\right) \times L+2 b s d V$ |
```

Some examples and trends

Qwen2 0.5B


Llama 7B


Llama 405B



## Bibliography

Good reviews {cite:p}`zhao2024surveylargelanguagemodels`


```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```