# LLM Dense Architectures Fundamentals

## Tokenization

## Position Embeddings

### Absolute Position


### Rotary Postion Embedding

#### The mechanism
The key idea of Rotary position embedding (Rope) is to multiply query vector $\boldsymbol{q}$ (of a token) and key vector $\boldsymbol{k}$ (of another token) by a rotational matrix $R(\theta_q)$ and $R(\theta_k)$, where $\theta_q$ and $\theta_k$ are taking values to indicate the positions of query vector token and key vector token, respectively.

Specifically, let $\theta_q = m\theta$ and $\theta_k = n\theta$, where $m$ and $n$ are integer positions of query vector token and key vector token. 
Now we are showing that the rotated query-key inner product is a function of the relative position $(m - n)$

$$
\begin{aligned}
\left\langle R\left(\theta_q\right) \boldsymbol{q}, R\left(\theta_k\right) \boldsymbol{k}\right\rangle & =\left(R\left(\theta_q\right) \boldsymbol{q}\right)^{\top}\left(R\left(\theta_k\right) \boldsymbol{k}\right) \\
& =\boldsymbol{q}^{\top} R\left(\theta_q\right)^{\top} R\left(\theta_k\right) \boldsymbol{k} \\
& =\boldsymbol{q}^{\top} R\left(\theta_k-\theta_q\right) \boldsymbol{k} \\
& =\left(R\left(\theta_q-\theta_k\right) \boldsymbol{q}\right)^{\top} \boldsymbol{k} \\
& =\left\langle R\left(\theta_q-\theta_k\right) \boldsymbol{q}, \boldsymbol{k}\right\rangle \\
& =\left\langle R\left(\theta (m-n)\right) \boldsymbol{q}, \boldsymbol{k}\right\rangle 
\end{aligned}
$$

We have used the following important properties of rotational matrix:

1. The transpose of a rotation matrix is equal to its inverse: $R(\theta)^{\top}=R(-\theta)$. 
   
2. The matrix multiplication of rotational matrices satisfies:
   $$R(\theta_x)\cdot R(\theta_y) = R(\theta_x + \theta_y)$$

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

The LayerNorm was proposed to overcome the  in combating the internal covariate shift issue {cite:p}`ioffe2015batchnormalizationacceleratingdeep`, where a layer’s input distribution changes as previous layers are updated, causing the difficulty of traning deep models.

The key idea in LayerNorm is 
* re-centering by subtracting the mean
* re-scaling by dividing the standard deviation.
  
The calculation formula is given by
$$
\begin{aligned}
\mu &= \frac{1}{H} \sum_{i=1}^H x_i \\
\sigma &=\sqrt{\frac{1}{H} \sum_{i=1}^H\left(x_i-\mu\right)^2+\epsilon} \\
LayerNorm(x) &=\frac{x-\mu}{\sqrt{\sigma+\epsilon}} \cdot \gamma+\beta
\end{aligned}
$$ (chapter_LLM_arch_layer_nomalization_formula)

where $\gamma$ is a trainable rescaling parameter and $\beta$ is a trainable re-shifting parameter.




### RMS Norm (Root Mean Square Norm)

A common hypothesis on why layer normalization can help stalize training and boost model convergence is the capability in  handling re-centering and re-scaling of both inputs and weight matrix. RMSNorm {cite:p}`zhang2019rootmeansquarelayer` is a technique aiming to achieve similar model training stablizing benefit with a reduced computational overhead compared to LayerNorm. RMSNorm hypothesizes that only the re-scaling component is needed and proposes the following normalization formula

$$
\begin{aligned}
RMSNorm(x)&=\sqrt{\frac{1}{H} \sum_{i=1}^H x_i^2} \\
x&=\frac{x}{RMS(x)} \cdot \gamma
\end{aligned}
$$

The advantages RMS Norm simplifies Layer Norm from the two following aspects and the performance is basically not impacted.

1. RMS Norm simplifies Layer Norm by removing the part that calculates the mean for shifting.
2. RMS Norm calculates faster. The effect is basically equivalent, and even slightly improved.

### Layer normalization position



### Layer normalization example choices

| LLM model | normalization |
| :---: | :---: |
| GPT3 | Pre layer Norm |
| LLaMA | Pre RMS Norm |
| baichuan | Pre RMS Norm |
| ChatGLM-6B | Post Deep Norm |
| ChatGLM2-6B | Post RMS Norm |
| Bloom | Pre layer Norm |
| Falcon | Pre layer Norm |

## Self-attention

Certainly! I'll provide a detailed summary of different attention modules used in Large Language Models (LLMs), including Multi-Head Attention (MHA), Grouped Query Attention (GQA), and others. I'll explain their mechanisms, advantages, and use cases.

1. Multi-Head Attention (MHA)

Multi-Head Attention is the foundation of many transformer-based models, including the original transformer architecture.

Key components:
- Query (Q), Key (K), and Value (V) matrices
- Multiple attention heads

Formula:
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$
where each head is computed as:
$$
\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$
$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

Advantages:
- Allows the model to jointly attend to information from different representation subspaces
- Improves the model's ability to capture various aspects of the input

Drawbacks:
- Computational complexity scales quadratically with sequence length

2. Grouped Query Attention (GQA)

GQA is an optimization of MHA that reduces computational complexity while maintaining performance.

Key idea:
- Group key and value projections while keeping separate query projections

Formula:
$$
\text{GQA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$
where:
$$
\text{head}_i = \text{Attention}(QW^Q_i, KW^K_{g(i)}, VW^V_{g(i)})
$$
$g(i)$ is a function that maps head index to group index.

Advantages:
- Reduces parameters and computation compared to MHA
- Maintains most of the performance of MHA

Use cases:
- Large-scale language models where efficiency is crucial

3. Sliding Window Attention

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

4. Sparse Attention

Sparse Attention introduces patterns of sparsity in the attention matrix.

Key idea:
- Predefined or learned patterns determine which tokens can attend to which other tokens

Advantages:
- Can capture both local and global context
- Reduces computational complexity

Examples:
- Longformer: combines sliding window attention with global attention
- Big Bird: uses random, window, and global attention patterns

5. Linformer

Linformer reduces the complexity of self-attention from O(n^2) to O(n) by projecting the keys and values to a lower-dimensional space.

Key idea:
- Project keys and values to a fixed lower dimension

Formula:
$$
\text{Attention}(Q, EK, EV)
$$
where E is a projection matrix.

Advantages:
- Linear complexity in sequence length
- Maintains performance for many tasks

Drawbacks:
- May lose some fine-grained information in the projection

6. Performer (FAVOR+)

Performer uses Fast Attention Via Orthogonal Random features (FAVOR+) to approximate the attention mechanism.

Key idea:
- Approximate the softmax function using random orthogonal features

Advantages:
- Linear time and space complexity
- Unbiased estimation of standard attention

Drawbacks:
- Approximation may not be exact for all cases

7. Rotary Position Embedding (RoPE)

While not strictly an attention module, RoPE is an important technique used in conjunction with attention mechanisms.

Key idea:
- Encode position information directly into the attention computation using complex rotations

Formula:
$$
q' = q(\cos(\theta) + i\sin(\theta)), k' = k(\cos(\theta) + i\sin(\theta))
$$

Advantages:
- Allows for extrapolation to longer sequences
- Preserves relative positional information effectively

These attention modules and related techniques represent ongoing research in improving the efficiency and effectiveness of LLMs. Each has its own trade-offs between computational complexity, memory usage, and model performance. The choice of attention mechanism often depends on the specific requirements of the task, available computational resources, and desired model capabilities.

Would you like me to elaborate on any specific attention module or aspect of their implementation in LLMs?

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

## Optimization


## Number of model parameters

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

The weight-matrix in the output Softmax layer is tied to the embedding layer.

### Total weight

$$n_{vocab} \times d_{model} + n_{max\_len} \times d_{model} + n_{layer}(4d_{model}^2 + 8d_{model}^2 + 5d_{model})$$

$n_{max\_len} = 2048$ in GPT-3

| Model Name | $n_{\text{params}}$ | $n_{\text{layers}}$ | $d_{\text{model}}$ | $n_{\text{heads}}$ | $d_{\text{head}}$ |
|------------|----------|----------|---------|---------|--------|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 |
| GPT-3 13B | 13.0B | 40 | 5140 | 40 | 128 |
| GPT-3 175B or "GPT-3" | 175.0B | 96 | 12288 | 96 | 128 |

# Dense Architecture Examples 

## LLama architectures



