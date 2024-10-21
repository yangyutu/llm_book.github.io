(chapter_LLM_arch_sec_LLM_arch_fundamentals)=
# LLM Architectures Fundamentals

## Overview

LLM have revolutionized natural language processing and artificial intelligence, demonstrating remarkable capabilities in understanding and generating human-like text. Tech companies like Google, Microsoft, Meta, OpenAI are producing LLMs with increasing sizes and capabilities just over the past few years [{numref}`chapter_foundation_fig_pretrained_LLM_timeline`]. 

From a high level, LLMs share the following characteristics:
* **Scale**: LLMs are trained on enormous datasets, often containing hundreds to thousands of billions of words or tokens. This massive scale allows them to capture intricate patterns and nuances in language. For example, GPT-3 {cite:p}`brown2020language` was trained on about 500 billion tokens, and recent Llama3 405B {cite:p}`dubey2024llama3herdmodels` was trained on 15.6T tokens.
* **Transformer architecture**: Most modern LLMs use transformer architectures, which were introduced in the "Attention is All You Need" paper. These models can have billions of parameters - GPT-3 has 175 billion, for instance. The transformer architecture allows for efficient parallel processing and captures long-range dependencies in text.
* **Self-supervised learning**: LLMs are typically pre-trained using self-supervised learning techniques. The most common approach is next-token prediction, where the model learns to predict the next word in a sequence given the previous words. This allows the model to learn from vast amounts of unlabeled text data.
* **Multi-task capability**: A single LLM can perform various language tasks such as translation, summarization, question-answering, reasongin, and text generation without needing separate models for each task. This versatility makes them powerful tools for a wide range of applications.
* **Few-shot learning via prompting**: The multi-task ability of LLMs can often be simply invoked by prompting with a few examples provided in the prompt. This "in-context learning" allows them to adapt to new tasks without model weight update.
* **Emergent reasoning abilities**: As LLMs grow in size and complexity, they often develop capabilities that were hard to acquire among small models, for example, arithmetic reasoning and logical reasoning: Models may show ability to follow simple logical arguments or solve puzzles.
*  **Hallucination**: LLMs can sometimes generate text that sounds plausible but is factually incorrect. This hallucination behavior is a significant challenge in deploying LLMs for applications requiring high reliability.

```{figure} ../img/chapter_LLM_arch//large_langage_models_release_timeline.png
---
scale: 50%
name: chapter_foundation_fig_pretrained_LLM_timeline
---
A timeline of existing large language models (having a size larger than 10B) in recent years. We mark the open-source LLMs in yellow color. Image from {cite:p}`zhao2023survey`.
```





This chapter aims to go over the core components and architectural considerations that form the foundation of modern LLMs. 

We begin by examining **layer normalization**, a crucial technique that stabilizes the learning process and allows for training of very deep networks, enhancing the overall performance of LLMs.

Next, we review the **activation** functions commonly used in LLM age, discussing their properties and impact on model performance and training dynamics.

The **self-attention mechanism**, a cornerstone of transformer models, is explored in depth, along with its variants that have emerged to address specific challenges or improve efficiency in processing and understanding context.

Next we cover **position encoding** techniques that allow transformer models to understand the sequential nature of language, with a focus on methods for handling **long context** – a critical challenge in scaling LLMs to process extensive inputs.

Finally, we examining the intricate details of LLM structure and function. We break down the **distribution of parameters** across different model components, provide a detailed explanation of the **forward pass computation**, and present **examples of dense transformer architectures**.

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
$$(chapter_LLM_arch_RMS_nomalization_formula)

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

## Nonlinearity in FFN

As introduced in {ref}`chapter_foundation_sec_pretrained_LM_transformer_arch_FFN`, FFN block plays a critical role in improving model capacity via nonlinear activations. 

With $x$ is the input vector, $W_1, b_1$ and $W_2, b_2$ are weight matrices and biases for the two layers, the FFN block is given by

$$
\operatorname{FFN}(x)=f(xW_1+b_1)W_2+b_2
$$

where $f$ is the activation function.

While in the original Transformer paper ReLU is used, many other different activations are explored. In the latest LLMs, GLU activations {cite:p}`shazeer2020gluvariantsimprovetransformer` are widely adopted and its variations SwiGLU are also widely used to achieve better performance in practice. 


**Gated Linear Units (GLU)** is a neural network layer defined as the componentwise product of two linear transformations of the input, one of which is sigmoid-activated.

$$\operatorname{GLU}(x; W, V, b)=\sigma(xW+b) \otimes xV
$$(chapter_LLM_arch_eq_GLU)

where $W, V$ are weight matrices and $b$ is the bias. Note that intuitively GLU introduces a gating mechanism on the product $xV$ via the sigmoid function $\sigma(xW+b)$. Such gating mechanism allows the model to learn when to emphasize or de-emphasize certain features.

Apply GLU in the FFN block, we yield

$$
\operatorname{FFN}_{GLU}(x; W_1,W_2,V, b_1, b_2)=(\sigma(xW_1 + b) \otimes xV)W_2 + b_2
$$(chapter_LLM_arch_eq_FFN_GLU)

where $W_1, W_2, V$ are weight matrices. Note that the FFN layer with GLU have three weight matrices, as opposed to two for the original FFN.


One variant of GLU is Swish {cite:p}`ramachandran2017searchingactivationfunctions`, which is given by

$$
\operatorname{Swish}_\beta(x)=x \cdot \sigma(\beta x)
$$(chapter_LLM_arch_eq_Swish)

where $\beta$ is a hyperparameter for Swish. Compared to GLU, Swish is a self-gated activation function. As showed in {numref}`chapter_foundation_fig_pretrained_LLM_activation_swish`, Swish has the following appealing properites:
- Smooth derivative leading to better gradient flow, while ReLU is nonsmooth at $x=0$
- Non-monotonicity: The non-monotonic nature of Swish allows it to capture more complex relationships in the data
- Unbounded above and bounded below, where as GLU is bounded above and below
- Non-zero gradient for negative inputs: For very negative inputs, Swish has a small but non-zero gradient, unlike ReLU which has a zero gradient. This can help mitigate the "dying ReLU" problem.
- Self-gating property allows the network to learn when to emphasize or de-emphasize certain features.


```{figure} ../img/chapter_LLM_arch/activations/swish.png
---
scale: 50%
name: chapter_foundation_fig_pretrained_LLM_activation_swish
---
(Left) The Swish activation function. (Right)  First derivatives of Swish. Image from {cite:p}`ramachandran2017searchingactivationfunctions`.
```

If we use Swish function in the GLU, we can obtain the following variations:
$$
SwiGLU=\text{Swish}_1(xW) \otimes xV
\operatorname{FFN}_{SwiGLU} = (Swish_1{xW_1}\otimes xV) W_2
$$(chapter_LLM_arch_eq_FFN_SwiGLU)


Example activation in recent LLMs:

| LLM | Activation Function |
| :---: | :---: |
| Mistral | SwiGLU |
| LLaMA | SwiGLU |
| Qwen | SwiGLU |

## Self-attention Variants


### Multi-Head Attention (MHA)

Multi-Head Attention [detailed in {ref}`chapter_foundation_sec_pretrained_LM_transformer_arch_MHA`] is the foundation of many transformer-based models, including the original transformer architecture.

The computation of an $H$-headed MHA given input $X\in \mathbb{R}^{n\times d_{model}}$ matrix and $H$ projection matrices $W^Q_i, W^K_i, W^V_i \in\mathbb{R}^{d_{model}}\times d_{head}$, $i\in \{1,...,H\}$ is given by

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_H)W^O
$$

where each head is computed as:

$$
\text{head}_i = \text{Attention}(XW^Q_i, XW^K_i, XW^V_i)
$$

with the attention given by

$$
\text{Attention}(Q, K, V) = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V.
$$

```{figure} ../img/chapter_LLM_arch/attention/MHA.png
---
scale: 40%
name: chapter_LLM_arch_fig_fundamentals_attention_MHA
---
Multi-head attention has $H$ query, key, and value heads for each token.
```

In the following, we summarize the advantages and drawbacks of MHA.
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



To reduce the inference cost from MHA, {cite:p}`shazeer2019fasttransformerdecodingwritehead` proposed **MQA**, which reduces $H$ key and value heads in MHA to a single key and value head. 
During inference, MQA reducing the size of the key-value cache by a factor of $H$ (see {ref}`chapter_inference_sec_inference_acceleration_KV_cache`). However, larger models generally scale the number of heads (e.g., GPT-2 has 12 heads; GPT-3 has 96 heads), such that multi-query attention represents a more
aggressive cut in both memory bandwidth and capacity footprint.

In MQA, the single head attnetion is computed as

$$
\text{head}_i = \text{Attention}(XW^Q_i, XW^K, XW^V).
$$

Note that we only have one group of $W^K, W^V$ matrices.


```{figure} ../img/chapter_LLM_arch/attention/MHA.png
---
scale: 40%
name: chapter_LLM_arch_fig_fundamentals_attention_MQA
---
Multi-head attention has $H$ query, and one shared single key head and single value head for each token.
```

MQA often comes at the cost of quality degradation. In the following, we summarize the MHA advantages and drawbacks.
::::{grid}
:gutter: 2

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #e4ac94">**Advantages**</span></p>

- During inference stage, each head has its own key and value to cache, bring additional memory burden to inference process.
:::

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #b4c9da">**Drawbacks**</span></p>
- Computational complexity scales quadratically with sequence length (i.e., huge cost for long context applications)

- Modeling capacity is largely compromised due to the reduction of multiple heads to single head, leading to quality degradation.
:::
::::

(chapter_LLM_arch_sec_self_attention_variant_GQA)=
### Grouped Query Attention (GQA)


GQA {cite:p}`ainslie2023gqatraininggeneralizedmultiquery` is an optimization of MHA and MQA that reduces computational complexity while maintaining performance.

Unlike MQA, GQA uses an intermediate (more than one, less than number of query heads) number of key-value heads.
GQA is shown to achieve quality close to MHA with comparable speed to MQA

In GQA, the single head attnetion is computed as

$$
\text{head}_i = \text{Attention}(QW^Q_i, KW^K_{g(i)}, VW^V_{g(i)})
$$

Here $g(i)$ is a function that maps head index to group index (e.g., $g(1): \{1, 2\} \to \{1\}$) and we have $G$ groups of $W^K,W^V$ matrices. 


Studies [{numref}`chapter_LLM_arch_fig_fundamentals_attention_GQA`] show that GQA (with $G <=  8$) can improve latency by reduces parameters and computation compared to MHA and at the same time maintain most of the performance of MHA.

```{figure} ../img/chapter_LLM_arch/attention/GQA_performance.png
---
scale: 35%
name: chapter_LLM_arch_fig_fundamentals_attention_GQA
---
(Top) GQA divides the key and value heads into multiple groups. Within each group, a single shared
key and value heads are attended to by query heads. GQA interpolats between MHA and MQA. (Bottom) GQA-8 performance and latency compared with MHA and MQA. Image from {cite:p}`ainslie2023gqatraininggeneralizedmultiquery`

```

GQA is widely adopted in the latest LLM. Following shows example configurations of Qwen2 and Mistral LLM series {cite:p}`yang2024qwen2technicalreport,jiang2023mistral7b`. 

```{table} Model configuration of Qwen2 and Mistral, which uses GQA (# KV heads is number of groups )
| Configuration | Hidden Size | # Layers | # Query Heads | # KV Heads |
| :--- | :---: | :---: | :---: | :---: |
| Qwen2 0.5B | 896 | 24 | 14 | 2 |
| Qwen2 1.5B | 1,536 | 28 | 12 | 2 |
| Qwen2 7B | 3,584 | 28 | 28 | 4 |
| Qwen2 72B | 8,192 | 80 | 64 | 8 |
|Mistral 7B| 4096 | 32 | 32 | 8|
```

### Sliding Window Attention

The computational complexity for MHA, MQA, GQA are scaling quadratically with the sequence length. This constrains the context length that LLM can effectively process, impacting their ability to handle long documents or maintain coherence over extended generations. 

To address this challenge, recent LLMs (e.g., Mistral {cite:p}`jiang2023mistral7b`) adopts **sliding window attention**, which reduces the computational complexity by restricting each token's attention to a fixed-size window $W$ of preceding tokens, rather than attending to the entire sequence. The computational complexity is reduced from quadratic $O(s^2)$ to linear $O(\min(W, s)\times s)$, where $s$ is the sequence length. Although the token can only capture local context within its fixed window, with multiple layers stacked upon each other, a token at layer $L$ can effectively attend to previous $L\times W$ tokens. 

In Mistral 7B with $L = 32$, and $W$ set to 4096, the effective attention length is about $131K$ tokens.


```{figure} ../img/chapter_LLM_arch/attention/sliding_window_attention.png
---
scale: 70%
name: chapter_LLM_arch_fig_fundamentals_attention_sliding_window_attention
---
Illustration of sliding window attention (Middle), which restrict each token to attend at most $W$ preceding tokens. As a comparison, MHA (Left) attends to all the preceding tokens. While at each layer the information flow is limited by window size, after $L$ layers, information can flow forward by up to $L\times W$ tokens.  Image from {cite:p}`jiang2023mistral7b`

```


<!-- ### Sparse Attention

Sparse Attention introduces patterns of sparsity in the attention matrix.

Key idea:
- Predefined or learned patterns determine which tokens can attend to which other tokens

Advantages:
- Can capture both local and global context
- Reduces computational complexity

Examples:
- Longformer: combines sliding window attention with global attention
- Big Bird: uses random, window, and global attention patterns -->





## Position Encoding and Long Context

### Absolute Position Encoding

In {ref}`chapter_foundation_sec_pretrained_LM_transformer_arch_absolute_PE`, we discuss **absolute position encoding**, which maps an integer $i$ (used to represent the position of the token) to a $d_{model}$ sinusoidal vector. Specifically, let $PE(i)_j$ represent the $j$th dimention position encoding, we have

$$
\operatorname{PE}(i)_j = \left\{\begin{array}{l}
\sin \left(w_j i\right), \quad \text { if } j \text{is even} \\
\cos \left(w_j i\right), \quad \text { if } j \text{is odd}
\end{array}\right.
$$

where $w_j=1/10000^{j / d_{model}}$ if $j$ is even and $w_j=1/10000^{j-1 / d_{model}}$ if $j$ is odd.

While absolute position encoding has achieved success in BERT, it has several key issues when is applied in LLMs:
* Lack of extrapolation due to limited sequence length: Models are restricted to a maximum sequence length during training (e.g., BERT 512), limiting their ability to  generalize to positions beyond the maximum length at inference time.
* Position insensitivity: The position encoding is added on top of token embedding and go through linear projection before interacting with other tokens, instead of directly interacting with other tokens during attention score computation. 
* Lack of invariance to shift: For two tokens with fixed relative position disance, their interaction at attention score computation layer is dependent on their absolute position. For relative position encodings, this property is by construction.

### ALiBi

{cite:p}`press2022trainshorttestlong` is a simple approach that suprisinly addresses all drawbacks in the sinusoidal abolute position encoding above. The key idea is to 1) simply add a static, relative position dependent bias into the Softmax computation step[{numref}`chapter_LLM_arch_fig_fundamentals_position_encoding_Alibi`]. Specifically, for the attention weight between query token $i$ to all the key vectors, we have

$$\operatorname{AttentionWeight} = \operatorname{Softmax}(\underbrace{Q_iK^T/d_{Head}}_{\text{scaled query-key doc product}} + \underbrace{m \cdot[−(i − 1), ..., −2, −1, 0]}_{\text{ALiBi bias vector}})
$$

where scalar $m$ is a head-specific slope hyperparameter fixed before training (e.g., for a model with 8 heads, $1/2, 1/2^2,...,1/2^8$).

Note that AliBi has the following nice property by construction:
* It is a relative position encoding
* Long distance decay, tokens with larger distance have smaller impact.

```{figure} ../img/chapter_LLM_arch/position_encoding/Alibi.png
---
scale: 40%
name: chapter_LLM_arch_fig_fundamentals_position_encoding_Alibi
---
When computing attention weights for each head, ALiBi adds a constant bias (Right) to each attention score ($Q_iK^T), with scaling factor omitted (Left).  Image from {cite:p}`press2022trainshorttestlong`.
```

Compare with sinusoidal absolute position encoding baseline [{numref}`chapter_LLM_arch_fig_fundamentals_position_encoding_Alibi_comparison`], there are several advantages of Alibi:
* When train and validate on the same input token length $L$, Alibi shows advantages over baseline.
* When train on shorter L (e.g., 512), but validate on longer (e.g., 1024,...,3072), Alibi method extropolates well.


```{figure} ../img/chapter_LLM_arch/position_encoding/Alibi_vs_absolution_PE_performance.png
---
scale: 70%
name: chapter_LLM_arch_fig_fundamentals_position_encoding_Alibi_comparison
---
Comparision between the ALiBi models trained and evaluated on varying sequence lengths on the WikiText-103
validation set and the sinusoidal absolute position encoding baseline. Image from {cite:p}`press2022trainshorttestlong`.
```

### Rotary Postion Embedding

#### The mechanism

Rotary Position Encoding (RoPE) {cite:p}`su2023roformerenhancedtransformerrotary` is a widely adopted and proved-effective position encoding method in latest LLM (e.g., Llama, Qwen, etc.). RoPE has ideas similar to ALiBi and sinusoid position encoding:
* Like ALiBi, relative positional information is directly used in attention score computation.
* Sinusoid functions are used in construction for their nice mathematical properties.

Specifically, the key idea of RoPE is to multiply query vector $Q_m$ (of a token at position $m$) and key vector $K_n$ (of another token at position $n$) by a rotational matrix $\boldsymbol{R}(m; \Theta)$ and $\boldsymbol{R}(n; \Theta)$ before taking the scaled doc product. Here rotational matrix $\boldsymbol{R}(\cdot; \Theta)$ is constructed a group of 2D rotational matrices, whose wave-length are specified by $\Theta$. 

The $d_{model}\times d_{model}$ rotational matrix is given by

$$
\boldsymbol{R}_{\Theta, m}^d=\left(\begin{array}{ccccccc}
\cos m \theta_1 & -\sin m \theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m \theta_1 & \cos m \theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m \theta_2 & -\sin m \theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m \theta_2 & \cos m \theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m \theta_{d / 2} & -\sin m \theta_{d / 2} \\
0 & 0 & 0 & 0 & \cdots & \sin m \theta_{d / 2} & \cos m \theta_{d / 2}
\end{array}\right)
$$

is the rotary matrix with pre-defined parameters $\Theta=\left\{\theta_i=10000^{-2(i-1) / d}, i \in[1,2, \ldots, d_{model} / 2]\right\}$. 

Pre-SoftMax input (omitting scaling) for query token at position $m$ and key token at position $n$ is given by 

$$
\operatorname{PreSoftmax}(Q_m, K_n) =\left(\boldsymbol{R}_{m;\Theta m} Q_m\right) \cdot \left(\boldsymbol{R}{n;\Theta} K_n \right)
$$

````{prf:example}
For $d_{model} == 2$, the rotation matrix for position $m $ is:

$$
\boldsymbol{R}(m; \Theta)=\left[\begin{array}{cc}
\cos (m\theta_1) & -\sin (m\theta_1) \\
\sin (m\theta_1) & \cos (m\theta_1)
\end{array}\right]
$$

Where $\theta_1 = 1$.

For $d_{model} == 4$, the rotation matrix for position $m $ is:

$$
\boldsymbol{R}(m; \Theta)=\left[\begin{array}{cc}
\cos (m\theta_1) & -\sin (m\theta_1) & 0 & 0 \\
\sin (m\theta_1) & \cos (m\theta_1) & 0 & 0 \\
0 & 0 & \cos (m\theta_2) & -\sin (m\theta_2) \\
0 & 0 & \sin (m\theta_2) & \cos (m\theta_2)  
\end{array}\right]
$$

Where $\theta_1 = 1, \theta_2 = 10000^{-2/4}$.

````

#### Properties of RoPE 

**Relative position encoding**:
Now we are showing that the rotated query-key inner product is a function of the relative position in 2D cases (the conclusion can be generalized to high-dimensional rotational matrix). Specifically, let $\theta_q = m\theta$ and $\theta_k = n\theta$, where $m$ and $n$ are integer positions of query vector token and key vector token. 

$$
\begin{aligned}
\left\langle R\left(\theta_q\right) Q_m, R\left(\theta_k\right) K_n\right\rangle & =Q_m^{\top} R\left(\theta_q\right)^{\top} R\left(\theta_k\right) K_n \\
& =Q_m^{\top} R\left(\theta_k-\theta_q\right) K_n \\
& =\left\langle R\left(\theta_q-\theta_k\right) Q_m, K_n\right\rangle \\
& =\left\langle R\left(\theta (m-n)\right) Q_m, K_n\right\rangle 
\end{aligned}
$$

That is, the Pre-Softmax input of $Q_m, K_n$ is a funciton of $m - n$.

We have used the following important properties of rotational matrix:
1. The transpose of a rotation matrix is equal to its inverse: $R(\theta)^{\top}=R(-\theta)$. 
2. The matrix multiplication of rotational matrices satisfies: $R(\theta_x)\cdot R(\theta_y) = R(\theta_x + \theta_y)$

In other words, the inner product of two rotated vectors is equal to the inner product of one vector rotated by their angle difference and the other original vector.

**Long-term decay**: In {cite:p}`su2023roformerenhancedtransformerrotary`, it is shown that the inner-product will decay when the relative position increase. This property aligns with desired property that a pair of tokens will have gradually descreasing semantic impact on each other when they are far apart. 

### Understanding RoPE with Visualization

```{figure} ../img/chapter_LLM_arch/position_encoding/RoPE/RoPE_visualization.png
---
scale: 70%
name: chapter_LLM_arch_fig_fundamentals_position_encoding_Alibi_comparison
---
Visualization of 2D RoPE and its mechanism in encoding context. Image from [Blog](https://mp.weixin.qq.com/s/dn8Pb80iRF9UkRn4vPOhHA).
```


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
(chapter_LLM_arch_sec_parameter_composition)=
## Parameter composition in Transformer models

In this section, we do an accounting exercise by estimating the number of parameters in a Transformer model. This will give us some insight on 
* which component makes up the majority of parameters and
* how the total number of parameters scales when we scale up different components.

Let $V$ be the vocabulary size, $d$ be the model hidden dimensions, $L$ be the number of layer, 

```{table} Parameters in a Transformer
| Module | Computation | Parameter Name | Shape | Parameter Number |
| :--- | :--- | :--- | :--- | :--- |
| Attention | ${Q} / {K} / {V}$ projection | weight / bias | $[{d}, {d}] /[{d}]$ | $3 d^2+3 d$ |
|  | Attention output projection | weight / bias | $[{d}, {d}] /[{d}]$ | $d^2+d$ |
|  | Layernorm | $\gamma, \beta$ | $[{d}] /[{d}]$ | $2 d$ |
| FFN | First layer up-projection | weight / bias | $[{d}, 4 {~d}] /[{d}]$ | $4 d^2+d$ |
|  | Second layer down-projection | weight / bias | $[4 {~d}, {~d}] /[4 {~d}]$ | $4 d^2+4 d$ |
|  | Layernorm | $\gamma, \beta$ | $[{d}] /[{d}]$ | $2 d$ |
| Embedding (tied) | - | - | $[{V}, {d}]$ | $V d$ |
| **Total** |  |  |  | $V d+L\left(12 d^2+13 d\right)$ |
```
The **key scaling properties** from this table are:
* The total number of parameters scales linearly with number of layers $L$
* The total number of parameters scales quadratically with model hidden dimensionality $d$.

````{prf:remark}
We have simplification in the above computation for MHA but the results are the same. Suppose we have $H$ heads, head dimension $d_{head}$ and $H \times d_{head} = d$. QKV transformation matrices have weight parameters $3 \times H \times d \times d_{head} = 3d^2$. 

With GQA that has $G$ key-value shared heads, the total parameters are $d^2 + 2Gd_{head}d$.
````

````{prf:example}
Take the following GPT-3 13B and 175B as an example, 175B model has approximate 2.4 times of $L$ and $d_{model}$. Extrapolating from 13B model, we estimate the 175B model to have model parameters of $13\times 2.4^3 = 179B$, which is very close.

| Model Name | $n_{\text{params}}$ | $L$ | $d$ | $H$ | $d_{head}$ |
|------------|----------|----------|---------|---------|--------|
| GPT-3 13B | 13.0B | 40 | 5140 | 40 | 128 |
| GPT-3 175B or "GPT-3" | 175.0B | 96 | 12288 | 96 | 128 |
````

(chapter_LLM_arch_sec_LLM_arch_fundamentals_forward_pass_computation)=
## Forward Pass Computation Breadown

In this section, we estimate the computational cost (in term of FLOPS) for a forward pass.

````{prf:remark} FLOPs estimation
If $A \in R^{m \times k}, B \in R^{k \times n}$ then, to compute $A B$ the number of floating-point arithmetic required is $2 m n k$.

For example, for 

$$A = \begin{bmatrix}a_{11} & a_{12} \\ a_{21} & a_{22}\end{bmatrix}, B = \begin{bmatrix}b_{11} & b_{12} \\ b_{21} & b_{22}\end{bmatrix},$$

The resulting $C = AB$ has $k$ terms, which are given by

$$c_{ij} = \sum_{t=1}^k a_{ik}b_{kj}.$$

It is clear that for each $c_{ij}$ there are $k$ multiplications and $k$ additions (technically $k-1$ additions among $k$ terms).
````

Let $V$ be the vocabulary size, $b$ be the batch size, $s$ be sequence length, $d$ be the model hidden dimensions, $L$ be the number of layer, we have summarized the computation breakdown in the following.


```{table} Computation breakdown
| Module | Computation | Matrix Shape Changes | FLOPs |
| :--- | :--- | :--- | :--- |
| Attention | ${Q} / {K} / {V}$ Projection | $[{b}, {s}, {d}] \times [{~d}, {~d}]\to[{b}, {s}, {d}]$ | $3\times 2 b s d^2$ |
|  | $Q K^T$ dot product | $[{~b}, {~s}, {~d}] \times [{~b}, {~d}, {~s}]\to[{b}, {s}, {s}]$ | $2 b s^2 d$ |
|  | Score Matrix $ \dot V$ | $[{~b}, {~s}, {~s}] \times [{~b}, {~s}, {~d}]\to[{b}, {s}, {d}]$ | $2 b s^2 d$ |
|  | Output (with $W_o$) | $[{b}, {s}, {d}] \times[{~d}, {~d}]\to[{b}, {s}, {d}]$ | $2 b s d^2$ |
| FFN | First layer up-projection | $[{~b}, {~s}, {~d}] \times[{~d}, 4 {~d}] \to [{b}, {s}, 4 {~d}]$ | $8 b s d^2$ |
|  | Second layer down-projection | $[{~b}, {~s}, 4 {~d}] \times[4 {~d}, {~d}]\to[{b}, {s}, {d}]$ | $8 b s d^2$ |
| Embedding |  | $[{b}, {s}, 1] \times[{~V}, {~d}]\to[{b}, {s}, {d}]$ | $2 b s d V$ |
| In total |  |  | $\left(24 b s d^2+4 b d s^2\right) \times L+2 b s d V$ |
```

The **key scaling properties** from this table are:
* The total compute scales linearly with number of layers $L$, and number of batch size $b$
* The total compute scales quadratically with model hidden dimensionality $d$ and $s$.



## Dense Architecture Examples 


<!-- 
```{table} Model cards of several selected LLMs with public configuration details. Here, PE denotes position embedding, #L denotes the number of layers, #H denotes the number of attention heads, dmodel denotes the size of hidden states, and MCL denotes the maximum context length during training.
| Model | Size | Normalization | PE | Activation | Bias | #L | #H | $d_{\text {model }}$ | MCL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT3 [55] | 175B | Pre LayerNorm | Learned | GeLU | $\checkmark$ | 96 | 96 | 12288 | 2048 |
| Llama | 207B | Pre RMSNorm | Learned | SwiGLU | $\checkmark$ | 64 | 128 | 16384 | 1024 |
| Qwen 2 {cite:p}`yang2024qwen2technicalreport`| 72B | Pre RMSNorm | RoPe | SwiGLU | $\checkmark$ | 80 | 64 | 8192 | ... |
```

% from A Survey of Large Language Models  -->



## Bibliography

Good reviews {cite:p}`zhao2024surveylargelanguagemodels`


```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```