# LLM Training Acceleration

## The Memory Requirement For Training LLM
<!-- % https://medium.com/@maxshapp/understanding-and-estimating-gpu-memory-demands-for-training-llms-in-practise-c5ef20a4baff -->

We will discuss the following in this section
* How much GPU memory do you need to train $X$ billion Transformer based LLM per each GPU device.
* What is the formula to estimate memory requirements.
* What would you do in practise to reduce the memory needs if the model does not fit.



### Model and Optimizer States
Consider the case that we train a LLM using Adam optimizer, we need to have enough GPU memory to store
* Copy of model parameter
* Copy of model parameter gradients
* Copy of optimizer states, include copy of the model parameters, momentum, and variance.

Assume that 
* Model parameters and graidents are stored in FP16 (2 bytes), 
* Optimizer states are stored in FP32 (4 bytes) for stable training
then training a $X$ billion model requires following GPU memory amount just to store the model and training states

$$(2 + 2 + 12) X ~\text{(GB)}.$$

The following table gives the example for the memory requirement for models of different sizes. 
| Model Size   | GPU Memory     |
| :--- | ---: |
| 0.5B    | 8 GB    |
| 3B    | 48 GB    |
| 7B    | 112 GB    |
| 70B    | 1120 GB    |


### Activations

Let's have the following notations:

* $s$ - sequence length
* $b$ - batch size
* $d$ - hidden dimension size
* $H$ - number of attention heads
* $p$ - precision

Based on the **FFN** architecture detailed in {ref}`chapter_foundation_sec_pretrained_LM_transformer_arch_FFN`, we can estimate the memory requireement for FFN activations 

| Component   | Memory    | Note|
| :--- | ---: | ---: | 
| First Layer    | $4bsdp$   | Output dimension is $4h$ |
| Activation    | $4bsdp$    | |
| Second Layer    | $4bsdp$    | |
| Dropout Layer | $sbd$ | |
| Total| $9bsdp + bsd$ | | 


Based on the **MHA** architecture detailed in {ref}`chapter_foundation_sec_pretrained_LM_transformer_arch_MHA`, we can estimate the memory requireement for MHA activations. 

| Component   | Memory    | Note|
| :--- | ---: | ---: | 
| Q/K/V projection    | $3bsdp$   | |
| Softmax input and output    | $2bs^2Hp$    | $s\times s$ attention matrix for each head|
| Dropout after Softmax | $bs^2H$ | |
| Output from $H$ attention head | $bsdp$ | |
| Output layer ($W_O$)| $bsdp$ | |
| Dropout | $bsd$| |
| Total| $5bsdp + 2bs^2Hp + bsd + bs^2H$ | | 


Additionally, there are two **Normalization Layers** in each Transformer Layer, the output from each such layer will require in total $2bsdp$ bytes.

### Total Memory Requirement

Now we arrive at the total amount of bytes required to store the activations for a $L$ layer Transformer:

$$M =  \underbrace{17bsdp}{Linear Layer} + \underbrace{2bs^2Hp}_{Softmax} + \underbrace{2bsd + bs^2H}_{Dropout}$$

If we ignore the small quantity $2bsd$ and take $p = 2$ (which is float16, 2bypte), we have

$$M_{approx} = 34bsd + 5bs^2H$$

The implication on activation memory requirement are
* $M$ scales linearly with batch size
* $M$ scales quadratically with sequence length. During training, we cannot afford large context windows.
* Using technique like GQA [{ref}`chapter_LLM_arch_sec_self_attention_variant_GQA`] can help save training memory.
<!-- ```
def activations_memory(num_layers, seq_len, batch_size, hidden_dim, num_heads, precision=2):
    "Returns amount of GPU VRAM (in GB) required to store intermediate activations for traditional Transformer Encoder block"
    mem_bytes = num_layers * precision * seq_len * batch_size * hidden_dim * (
        16 + 2/precision + 2*num_heads*seq_len/hidden_dim + num_heads*seq_len/(precision*hidden_dim))
    return round(mem_bytes / 10**9, 2)
``` -->
{cite:p}`yang2024qwen2technicalreport`

| Model Size   | $L$ | $d$| $s$ |$H$ | $b$ |GPU Memory  |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 0.5B  | 24 | 896 | 4096 | 14 | 1 | 2.9 GB + 2 GB = 4.9 GB |
| 7B | 28 | 3584 | 4096 | 28 | 1 | 3.9 GB + 4 GB = 7.9GB|
| 72B | 64 | 8192 | 4096 | 64 | 1 | 71 GB + 8 GB = 79GB |
<!-- | 3B    | 48 GB    |
| 7B    | 112 GB    |
| 70B    | 1120 GB    | -->


### Activation Checkpointing Techniques

LLM have an enormous number of parameters. In the typical backpropogation during training, we save all the activation values from the forward pass to compute gradient, which consumes a large amount of GPU memory.

On one extreme, we can completely discard the activation values from the forward pass and recalculate the necessary activation values when computing gradients. While this mitigates the activation memory footprint issue, it increases the computational load and slows down training.

**Gradient Checkpointing** {cite:p}`chen2016trainingdeepnetssublinear` sits in the middle of these two approaches. This method employs a strategy that selects and saves a portion of the activation values from the computational graph, discarding the rest. The discarded activation values need to be recalculated during gradient computation.

Specifically, during the forward pass, activation values of computational nodes are calculated and saved. After computing the next node, the activation values of intermediate nodes are **selectively discarded**. During backpropagation, saved activations for gradient computation are used directly. If not, the actitions of the current node are recalculated using the saved activations from the previous node.

## Mixed Precision Training

### Overview
% https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/?ssp=1&darkschemeovr=1&setlang=en-US&safesearch=moderate

% to change
Training billion-scale LLM requires huge number of memory, which include model loading, optimizer state storage, and gradient storage. The idea of using low-precision for precision-insensitive computation and high-recision for precision sensitive computation leads to Mixed-precision training{cite:p}`micikevicius2017mixed`. Mixed-precision training lowers the required resources by using lower-precision arithmetic, and it therefore widely used in LLM training. It has the following benefits.

**Reduced memory footprint**: Mixed precision training leverages half-precision floating point format (FP16), which uses only 16 bits per number, in contrast to the 32 bits used by single precision (FP32). This significant reduction in memory usage offers two key advantages:
1. Enables training of larger models: With the same memory constraints, developers can design and train models with more parameters or greater complexity.
2. Allows for larger minibatches: Increased batch sizes can lead to more stable gradients and potentially faster convergence in some cases.

**Accelerated training and inference**: The performance gains from mixed precision training stem from two main factors:
1. Reduced memory bandwidth usage: Since FP16 requires half the memory bandwidth of FP32, layers that are memory-bound can see substantial speed improvements.
2. Faster arithmetic operations: Many modern GPUs have specialized hardware for FP16 (and lower precision like int8, FP8) operations , allowing them to perform these calculations much faster than FP32 operations.
These factors combine to potentially shorten both training and inference times, especially for large models or when processing substantial amounts of data.

### Training Process

This section describes three techniques for successful training of DNNs with half precision: accumulation of FP16 products into FP32; loss scaling; and an FP32 master copy of weights. With these techniques NVIDIA and Baidu Research were able to match single-precision result accuracy for all networks that were trained.{cite:p}`micikevicius2017mixed`


As shown in {numref}`chapter_training_fig_efficient_training_mixed_precision_process_demo`, key steps in the mixed-precision training are

* Maintain a master copy of model parameters, optimizer momentums and variances with fp32 precision.
* Before the model forward pass begins, allocate new storage to save model parameters in the fp16 format.
* Perform forward pass, the produced activations will be saved as fp16.
* Perform backward pass, the produced gradients will be saved as fp16.
* Use fp16 gradients to update model parameters that are saved as fp32.



```{figure} ../img/chapter_training/efficient_training/mixed_precision/mixed_precision_process_demo.png
---
scale: 35%
name: chapter_training_fig_efficient_training_mixed_precision_process_demo
---
Model training step with mixed precision using classifical Adam algorithm [{prf:ref}`Adam_stochastic_gradient_descent_algorithm`].
```


We can estimate the memory storage consumption according to the following table. Denote the model parameter size by $\Phi$. Let the storage unit be byte. We need $16\Phi$ memory storage in total. 

```{table} Storage requirement for different components during LLM training using Adam.
| Type | Storage Size |
| :--- | :--- | 
| Parameter (fp32) | $4 \Phi$ |
| Momentum(fp32) | $4 \Phi$ | 
| Variance (fp32) | $4 \Phi$ |
| Parameter (fp16) | $2 \Phi$ | 
| Gradients (fp16) | $2 \Phi$ | 
|  **Total**: | $16 \Phi$ |
```



<!-- 
\begin{remark}[the size of activations and model parameters]
The number of activations is typically much smaller than the size of the model parameters. 
Take an MLP with input dim $D_0$, hidden dim $D_1$, and output dim $D_2$ as an example. The total number of activations, including initial inputs, are $D_0 + D_1 + D_2$. 

The total number of model parameters are $D_0\times D_1 + D_1\times D_2$.
\end{remark}

\begin{remark}[Precision options fp16, bf16, and fp8]
Both BF16 and FP16 are 16-bit floating-point formats used in deep learning models. The difference between them is how the bits are divided between the exponent and fraction. BF16 has the same exponent range as FP32 but with fewer bits for the fraction1. Each number has one sign bit.

Although having similar theoretical performance benefits, BF16 and FP16 can have different speeds in practice. It’s recommended to try both formats and use the one with best speed while maintaining the desired numeric behavior.	

% https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
Out-of-the-box mixed precision training with either float16 or bfloat16 is effective at speeding up the convergence of many deep learning models, but some models may require more careful numerical accuracy management. Here are Best practice 
Figure out by experimentation if your network is sensitive to range and/or precision of a format. For example fine-tuning bfloat16-pretrained models in float16 can easily run into range issues in float16 because of the potentially large range from training in bfloat16, so users should stick with bfloat16 fine-tuning if the model was trained in bfloat16.

\end{remark} -->


## Distributed Parallel Training

### Overview of parallel training techniques



(chapter_training_sec_distributed_parallel_training_model_parallelism)=
### Model parallelism (tensor parallelism)


## ZeRO Via DeepSpeed

Data parallelism is the most widely used technique because it is simple and easy to implement. However, tt is typically challenging to apply the vanilla flavor data parallelism since it requires each GPU to store the parameters of the whole model. As a result, the size of GPU memory becomes the ceiling of the model scale we can train. 

Model parallelism (e.g, Megatron) [{ref}`chapter_training_sec_distributed_parallel_training_model_parallelism`], desipte its success in T5 (11B) and Megatron-LM (8.3B) is hard to scale beyond model sizes that cannot fit into a single GPU node. This is because model parallelism typically partitions the model weights or layers across GPU devices, incurring a significant communication between devices. 

ZeRO (Zero Redundancy Optimizer) {cite:p}`rajbhandari2020zeromemoryoptimizationstraining` adopt the data parallism paradigm, and optimize memory efficieny and commnication efficiency.

### GPU Memory Allocation

GPU memory is mainly allocated into two parts: **model states** and **residual states**.


```{figure} ../img/chapter_training/efficient_training/ZeRO/gpu_memory_allocation.png
---
scale: 40%
name: chapter_training_fig_efficient_training_ZeRO_gpu_memory_allocation
---
Model training step with mixed precision.
```


Model states refer to the content that is closely related to the model itself and must be stored. Specifically, they include:
\begin{itemize}
	\item Optimizer states: momentum and variance in the Adam optimization algorithm.
	\item Gradients: model gradients
	\item Parameters: model parameters 
\end{itemize}

Residual States refer to the content that is not necessary for the model, but is generated during the training process. Specifically, they include:
\begin{itemize}
	\item Activation: activation values. We have discussed this in detail in pipeline parallelism. It is used when calculating gradients using the chain rule in the backward process. It can speed up gradient calculation, but it is not necessary to store because it can be calculated by redoing the Forward process.
	\item Temporary buffers: temporary storage. For example, storage generated when aggregating gradients sent to a GPU for summation.
	\item Unusable fragment memory: fragmented storage space. Although the total storage space is sufficient, if contiguous storage space cannot be obtained, related requests will also fail. Memory defragmentation can solve this type of space waste.
\end{itemize}




### ZeRO-Stage-One
Here's the English translation of the provided text:

(1) $P_{os}$ (Optimizer State Partitioning)
ZeRO-Stage-One reduces the required memory on each device by partitioning the optimizer state across $N_d$ data-parallel processes. Each process only stores and updates its corresponding partition of the optimizer state, which is $\frac{1}{N_d}$ of the total optimizer state. At the end of each training step, results from each process are collected to obtain the overall updated state parameters.

(2) The result after ZeRO-Stage1 memory optimization, mainly targeting the optimizer state :

$$
(2+2) \Psi+\frac{K \times \Psi}{N_d}
$$

As can be seen, the optimizer state memory has a divisor of $N_d$ compared to the original.


````{prf:example}
For a 7.5B parameter model, the standard case requires 120 GB of memory, but using $P_{os}$ with $N_d=64$ only requires 31.4 GB of memory.
When $N_d$ is very large, the memory consumption approaches:

$$
(2+2) \Psi+\frac{K \times \Psi}{N_d} \approx 4 \Psi
$$

The ratio compared to the original:

$$
\frac{4}{4+K}
$$

When $K=12$, this becomes $\frac{1}{4}$, meaning the memory usage is $\frac{1}{4}$ of the original.

````

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

### From Online Softmax To Flash Attention

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

## Appendix

### Floating Data Types

Float32 (FP32) stands for the standardized IEEE 32-bit floating point representation. With this data type it is possible to represent a wide range of floating numbers. In FP32, 8 bits are reserved for the "exponent", 23 bits for the "mantissa" and 1 bit for the sign of the number. In addition to that, most of the hardware supports FP32 operations and instructions.

In the float16 (FP16) data type, 5 bits are reserved for the exponent and 10 bits are reserved for the mantissa. This makes the representable range of FP16 numbers much lower than FP32. This exposes FP16 numbers to the risk of overflowing (trying to represent a number that is very large) and underflowing (representing a number that is very small).

For example, if you do 10k * 10k you end up with 100M which is not possible to represent in FP16, as the largest number possible is 64k. And thus you'd end up with NaN (Not a Number) result and if you have sequential computation like in neural networks, all the prior work is destroyed. Usually, loss scaling is used to overcome this issue, but it doesn't always work well.

A new format, bfloat16 (BF16), was created to avoid these constraints. In BF16, 8 bits are reserved for the exponent (which is the same as in FP32) and 7 bits are reserved for the fraction.

This means that in BF16 we can retain the same dynamic range as FP32. But we lose 3 bits of precision with respect to FP16. Now there is absolutely no problem with huge numbers, but the precision is worse than FP16 here.

In the Ampere architecture, NVIDIA also introduced TensorFloat-32 (TF32) precision format, combining the dynamic range of BF16 and precision of FP16 to only use 19 bits. It's currently only used internally during certain operations.

In the machine learning jargon FP32 is called full precision (4 bytes), while BF16 and FP16 are referred to as half-precision (2 bytes). On top of that, the int8 (INT8) data type consists of an 8-bit representation that can store $2^8$ different values (between [0, 255] or [-128, 127] for signed integers).

While, ideally the training and inference should be done in FP32, it is two times slower than FP16/BF16 and therefore a mixed precision approach is used where the weights are held in FP32 as a precise "main weights" reference, while computation in a forward and backward pass are done for FP16/BF16 to enhance training speed. The FP16/BF16 gradients are then used to update the FP32 main weights.

During training, the main weights are always stored in FP32, but in practice, the half-precision weights often provide similar quality during inference as their FP32 counterpart -- a precise reference of the model is only needed when it receives multiple gradient updates. This means we can use the half-precision weights and use half the GPUs to accomplish the same outcome.

```{figure} ../img/chapter_training/efficient_training/mixed_precision/different_precision_demo.png
---
scale: 90%
name: chapter_training_fig_efficient_training_different_precision_demo
---
Comparison of different float number types.
```

### GPU Parallel Operations


References https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#:~:text=The%20ReduceScatter%20operation%20performs%20the%20same%20operation%20as,mapping%20since%20the%20ranks%20determine%20the%20data%20layout.


```{figure} ../img/chapter_training/efficient_training/gpu_parallel_operation/gpu_parallel_operation_broadcast.png
---
scale: 25%
name: chapter_training_fig_efficient_training_gpu_parallel_operation_broadcast
---
Broadcast operation: data in one device is sent to all other devices.
```

```{figure} ../img/chapter_training/efficient_training/gpu_parallel_operation/gpu_parallel_operation_scatter.png
---
scale: 25%
name: chapter_training_fig_efficient_training_gpu_parallel_operation_scatter
---
Scatter operation.
```

```{figure} ../img/chapter_training/efficient_training/gpu_parallel_operation/gpu_parallel_operation_gather.png
---
scale: 25%
name: chapter_training_fig_efficient_training_gpu_parallel_operation_gather
---
Gather operation: every device broadcasts their data patition to a designated devices. Eventually, this desigated device has the complete data.
```

```{figure} ../img/chapter_training/efficient_training/gpu_parallel_operation/gpu_parallel_operation_reduce.png
---
scale: 25%
name: chapter_training_fig_efficient_training_gpu_parallel_operation_reduce
---
Reduce operation.
```

```{figure} ../img/chapter_training/efficient_training/gpu_parallel_operation/gpu_parallel_operation_allgather.png
---
scale: 25%
name: chapter_training_fig_efficient_training_gpu_parallel_operation_allgather
---
AllGather operation: every device broadcasts their chuck of data to all other devices. Eventually, every device has a complete data copy.
```

```{figure} ../img/chapter_training/efficient_training/gpu_parallel_operation/gpu_parallel_operation_allgather_ring.png
---
scale: 25%
name: chapter_training_fig_efficient_training_gpu_parallel_operation_allgather_ring
---
Communication efficient implementation for AllGather via ring style. Every device sends its chuck of data to the next device in the ring. 
```



```{figure} ../img/chapter_training/efficient_training/gpu_parallel_operation/gpu_parallel_operation_reducescatter.png
---
scale: 25%
name: chapter_training_fig_efficient_training_gpu_parallel_operation_reducescatter
---
ReduceScatter operation performs the same operation as Reduce, except that the result is scattered in equal-sized blocks across devices.
```

```{figure} ../img/chapter_training/efficient_training/gpu_parallel_operation/gpu_parallel_operation_allreduce.png
---
scale: 25%
name: chapter_training_fig_efficient_training_gpu_parallel_operation_allreduce
---
AllReduce operation.
```

A **naive AllReduce** implementation would be two steps: 
1. Reduce: All devices first send their data to Rank0 device, and performing reduce operation on Rank0. 
2. Broadcast: the reduce results are sent to all other devices. 
This amounts to a total $2(N_d-1)\Phi$ communication volume, in which the step 1 has $(N_d-1)\Phi$ and step 2 has $(N_d-1)\Phi$. The naive implementation has the communication load imbalance issue as all data are sent into and sent out from Rank0 device.

The **RingAllReduce** address the load imbalance issue by engaging all devices in data communication and reduction (i.e., more parallelism). The RingAllReduce is equivalent to first RingReduceScatter and then AllGather.


```{table} Communication volumne summary for different operations. Let $\Phi$ be the total data size in one device and $N$ be the total number of devices.
| Type | Storage Size |
| :--- | :--- | 
| Broadcast | $(N_d-1) \Phi$ |
| Scatter | $\frac{N_d-1}{N_d}\Phi$ | 
| Reduce | $(N_d-1)\Phi$ |
| Gather | $\frac{N_d-1}{N_d}\Phi$ |
| AllGather | $\frac{N_d-1}{N_d}\Phi \times N_d = (N_d-1)\Phi$ | 
| ReduceScatter | $\frac{N_d-1}{N_d}\Phi \times N_d = (N_d-1)\Phi$ |
| AllReduce(Ring)| $2(N_d-1) \Phi$|
```


## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
