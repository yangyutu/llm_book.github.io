# LLM Training Acceleration


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