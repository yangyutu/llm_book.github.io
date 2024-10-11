# LLM Training Acceleration

## The Memory Requirement For Training LLM
<!-- % https://medium.com/@maxshapp/understanding-and-estimating-gpu-memory-demands-for-training-llms-in-practise-c5ef20a4baff -->

We will discuss the following in this section
* How much GPU memory do you need to train $X$ billion Transformer based LLM per each GPU device.
* What is the formula to estimate memory requirements.
* What would you do in practise to reduce the memory needs if the model does not fit.



### Model States
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



#### MLP Part

* The output of the first linear layer, which is $4psbh$ (as this linear enlarges the output dimension to $4h$).
* The output of the GeLU activation, which is $4psbh$ bytes.
* The output of the second linear layer, which is $psbh$ bytes
* The binary dropout marks specifies which dimensions are droped, which is $sbh$ 
So in total, MLP part will require to store: $9psbh + sbh$ bytes for activations.

#### Self-attention Part

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

### Activation Checkpointing Techniques



## Mixed Precision Training

### Overview
% https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/?ssp=1&darkschemeovr=1&setlang=en-US&safesearch=moderate

% to change
Deep Neural Networks (DNNs) have lead to breakthroughs in a number of areas, including image processing and understanding, language modeling, language translation, speech processing, game playing, and many others. DNN complexity has been increasing to achieve these results, which in turn has increased the computational resources required to train these networks. Mixed-precision training{cite:p}`micikevicius2017mixed` lowers the required resources by using lower-precision arithmetic, which has the following benefits.

* Decrease the required amount of memory. Half-precision floating point format (FP16) uses 16 bits, compared to 32 bits for single precision (FP32). Lowering the required memory enables training of larger models or training with larger minibatches.
* Shorten the training or inference time. Execution time can be sensitive to memory or arithmetic bandwidth. Half-precision halves the number of bytes accessed, thus reducing the time spent in memory-limited layers.

Half-precision floating point format consists of 1 sign bit, 5 bits of exponent, and 10 fractional bits.  Supported exponent values fall into the $[-24, 15]$ range, which means the format supports non-zero value magnitudes in the $[2^{-24}, 65,504]$ range. Since this is narrower than the $[2-149, \sim3.4\times1038]$ range supported by single-precision format, training some networks requires extra consideration. 





### Training Process

This section describes three techniques for successful training of DNNs with half precision: accumulation of FP16 products into FP32; loss scaling; and an FP32 master copy of weights. With these techniques NVIDIA and Baidu Research were able to match single-precision result accuracy for all networks that were trained.{cite:p}`micikevicius2017mixed`





```{figure} ../img/chapter_training/efficient_training/mixed_precision/mixed_precision_process_demo.png
---
scale: 35%
name: chapter_training_fig_efficient_training_mixed_precision_process_demo
---
Model training step with mixed precision using classifical Adam algorithm [{prf:ref}`Adam_stochastic_gradient_descent_algorithm`].
```

key elements in the mixed-precision training

* Maintain a master copy of model parameters, optimizer momentums and variances with fp32 precision.
* Before the model forward pass begins, allocate new storage to save model parameters in the fp16 format.
* Perform forward pass, the produced activations will be saved as fp16.
* Perform backward pass, the produced gradients will be saved as fp16.
* Use fp16 gradients to update model parameters that are saved as fp32.

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

\end{remark}


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

# Appendix

## Floating Data Types

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

## GPU Parallel Operations


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
