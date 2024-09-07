# LLM Training Acceleration

## The memory requirement for training LLM
<!-- % https://medium.com/@maxshapp/understanding-and-estimating-gpu-memory-demands-for-training-llms-in-practise-c5ef20a4baff -->

We will discuss the following in this section
* How much GPU memory do you need to train $X$ billion Transformer based LLM per each GPU device.
* What is the formula to estimate memory requirements.
* What would you do in practise to reduce the memory needs if the model does not fit.



### Model states
Consider the case that we train a LLM using Adam optimizer, we need to have enough GPU memory to store
* Copy of model parameter
* Copy of model parameter gradients
* Copy of optimizer states, include copy of the model parameters, momentum, and variance.

Assume that 
* Model parameters and graidents are stored in FP16 (2 bytes), 
* Optimizer states are stored in FP32 (4 bytes) for stable training
then training a $X$ billion model requires following GPU memory amount just to store the model and training states

$$(2 + 2 + 12) X ~\text{(GB)}.$$

The following table gives the example for the memory requirement for the common 7B and 70B models. 
| Model Size   | GPU Memory (GB)    |
| :--- | ---: |
| 7B    | 112 B    |
| 70B    | 1120 B    |


### Activations

First, let's have the following notations:
* $L$ - number of transformer layers
* $s$ - sequence length
* $b$ - batch size
* $h$ - hidden dimension size
* $a$ - number of attention heads
* $p$ - precision



#### MLP part

* The output of the first linear layer, which is $4psbh$ (as this linear enlarges the output dimension to $4h$).
* The output of the GeLU activation, which is $4psbh$ bytes.
* The output of the second linear layer, which is $psbh$ bytes
* The binary dropout marks specifies which dimensions are droped, which is $sbh$ 
So in total, MLP part will require to store: $9psbh + sbh$ bytes for activations.

#### Self-attention part

Attention block: which includes self attention followed by a linear projection and an attention dropout. 
* Before entering the self-attention block, query, key, and value are passed through a linear projection layer, whose outputs requires in totoal $3psbh$ bytes
* The Softmax output is a $b \times s\times s$ attention matrix for each head, which in total is $pas^2b$ bytes; The Softmax input is the logis, which is also $pas^2b$ bytes.
* The Dropout mask after the Softmax layer needs $as^2b$
* Output from Self-Attention, which will require $psbh$ bytes
* Output from the Linear layer, which will require $psbh$ bytes. 
* Dropout mask after the linear layer, this will require $sbh$ bytes.

To sum up, we need $5psbh + sbh + 2pas²b + as²b$ bytes for Attention part.

Additionally, there are 2 Norm Layers in the Transformer Layer, the output from each such layer will require to store psbh bytes, so in total 2psbh bytes.

total amount of bytes required to store the activations will be approximately:

$$Lpsbh\left(16+\frac{2}{p}+\frac{2 a s}{h}+\frac{a s}{p h}\right)$$

```
def activations_memory(num_layers, seq_len, batch_size, hidden_dim, num_heads, precision=2):
    "Returns amount of GPU VRAM (in GB) required to store intermediate activations for traditional Transformer Encoder block"
    mem_bytes = num_layers * precision * seq_len * batch_size * hidden_dim * (
        16 + 2/precision + 2*num_heads*seq_len/hidden_dim + num_heads*seq_len/(precision*hidden_dim))
    return round(mem_bytes / 10**9, 2)
```

### Activation checkpointing techniques




## Flash Attention


### Online Softmax Motivation

A typical computation of Softmax 
$$a_i = \mathbf{Softmax} (x_i | x_1,...,x_N) = \frac{\exp(x_i)}{\sum_{j}^N \exp(x_j)}$$
within the self-attention module involves the following three steps:

1. Motivated by preventing overflow, we first find the maximum value $m$ of $\{x_1,...,x_N\}$
$$
\begin{aligned}
\text{for }  i & \leftarrow 1, N \text{ do} \\
m & = \max(m, x_i)
\end{aligned}
$$

2. Calculate the denominator of Softmax (with offset to $m$)
$$
\begin{aligned}
\text{for } i & \leftarrow 1, N \text{ do} \\
d_i & = d_{i-1} + \exp(x_i-m)
\end{aligned}
$$

3. Calculate the Softmax for each corresponding position
$$
\begin{aligned}
\text{for } i & \leftarrow 1, N \text{ do} \\
a_i & = \frac{\exp(x_i-m)}{d_N}
\end{aligned}
$$

Without any optimization, at least six communications with the GPU are required (three writes and three reads). If we apply some parallel partitioning to each step of the for loop, we would also need to add the communication costs of operations like reduce_sum and reduce_max. Is it possible to fuse certain operations to reduce communication? 

Based on previous experience with layernorm parallelization, we need to look for an Online Algorithm.

### Online Softmax Algorithm

Nvidia :cite:`milakov2018onlinenormalizercalculationsoftmax`

Since we're looking for an Online algorithm, we need to find a recursive expression.
For the second step, $d_i=d_{i-1}+e^{x_i-m_N}$, we aim to remove its dependency on $m_N$.
Let $d_i^{\prime}=\sum_j^i e^{x_j-m_i}$, note that here we subtract the current maximum instead of the global maximum. This expression has the following property:
$$
\begin{aligned}
d_i^{\prime} & =\sum_j^i e^{x_j-m_i} \\
& =\sum_j^{i-1} e^{x_j-m_i}+e^{x_i-m_i} \\
& =\sum_j^{i-1} e^{x_j-m_{i-1}+m_{i-1}-m_i}+e^{x_i-m_i} \\
& =\left(\sum_j^{i-1} e^{x_j-m_{i-1}}\right) e^{m_{i-1}-m_i}+e^{x_i-m_i} \\
& =d_{i-1}^{\prime} e^{m_{i-1}-m_i}+e^{x_i-m_i}
\end{aligned}
$$
We can see that the calculation of $d_i^{\prime}$ depends on $d_{i-1}^{\prime}, m_i, m_{i-1}$, allowing us to merge the first two steps. The previous three steps can be reduced to two:

Find the maximum value $m$ of $x$, calculate the denominator of softmax.
$$
\begin{aligned}
& \text { for } i \leftarrow 1, N \text { do } \\
& \quad m_i=\max \left(m_i, x_i\right) \\
& \quad d_i^{\prime}=d_{i-1}^{\prime} e^{m_{i-1}-m_i}+e^{x_i-m_i}
\end{aligned}
$$

Calculate the softmax for each position
$$
\begin{aligned}
\text { for } i & \leftarrow 1, N \text { do } \\
a_i & =\frac{e^{x_i-m_N}}{d_N}
\end{aligned}
$$

Can we further fuse operators? Not really, because the denominator in the second step depends on the calculation from the first step.
However, we can use GPU shared memory to store intermediate results and implement the above two steps in a single kernel. This way, we only need to communicate with global memory twice: once to write data and once to read results.

### From Online Softmax to Flash Attention

Using a method similar to Online Softmax, we can place all Attention operations into a single for loop (implementable in one Kernel).
Let's first look at how it's calculated:
$$
\begin{aligned}
\text { for } i & \leftarrow 1, N \text { do } \\
a_i & =\frac{e^{x_i-m_N}}{d_N^{\prime}} \\
o_i & =\sum_j^i a_j v_j=\sum_j^i \frac{e^{x_j-m_N}}{d_N^{\prime}} v_j
\end{aligned}
$$
We can see that $o_i$ contains $m_N$ and $d_N^{\prime}$. We want to eliminate these dependencies. Similar to Online Softmax, we define:
$$
o_i^{\prime}=\sum_j^i \frac{e^{x_j-m_i}}{d_i^{\prime}} v_j
$$

Let's find the recursive expression. The key is to identify $o_{i-1}^{\prime}$:
$$
\begin{aligned}
\sigma_i^{\prime} & =\sum_j^i \frac{e^{x_j-m_i}}{d_i^{\prime}} v_j \\
& =\sum_j^{i-1} \frac{e^{x_j-m_i}}{d_i^{\prime}} v_j+\frac{e^{x_i-m_i}}{d_i^{\prime}} v_i \\
& =\sum_j^{i-1} \frac{e^{x_j-m_{i-1}}}{d_{i-1}^{\prime}} \frac{e^{x_j-m_i}}{e^{x_j-m_{i-1}}} \frac{d_{i-1}^{\prime}}{d_i^{\prime}} v_j+\frac{e^{x_i-m_i}}{d_i^{\prime}} v_i \\
& =\left(\sum_j^{i-1} \frac{e^{x_j-m_{i-1}}}{d_{i-1}^{\prime}} v_j\right) \frac{e^{x_j-m_i}}{e^{x_j-m_{i-1}}} \frac{d_{i-1}^{\prime}}{d_i^{\prime}}+\frac{e^{x_i-m_i}}{d_i^{\prime}} v_i \\
& =o_{i-1}^{\prime}\left(e^{m_{i-1}-m_i}\right) \frac{d_{i-1}^{\prime}}{d_i^{\prime}}+\frac{e^{x_i-m_i}}{d_i^{\prime}} v_i
\end{aligned}
$$
We can see that the calculation of $o_i$ depends on $o_{i-1}, m_i, m_{i-1}, d_i^{\prime}, d_{i-1}^{\prime}$. This allows us to place the entire Attention calculation into a single for loop:
$$
\begin{aligned}
\text { for } i & \leftarrow 1, N  \text{ do } \\
m_i & =\max \left(m_i, x_i\right) \\
d_i^{\prime} & =d_{i-1}^{\prime} e^{m_{i-1}-m_i}+e^{x_i-m_i} \\
o_i^{\prime} & =o_{i-1}^{\prime}\left(e^{m_{i-1}-m_i}\right) \frac{d_{i-1}^{\prime}}{d_i^{\prime}}+\frac{e^{x_i-m_i}}{d_i^{\prime}} v_i
\end{aligned}
$$
From the above, we can see that all operations within the for loop satisfy the associative property. This means the for loop can be block-processed, allowing for more efficient parallel computation on GPUs.
This is the mathematical principle behind Flash Attention's parallel acceleration.

:bibliography:`../llm_book.bib`