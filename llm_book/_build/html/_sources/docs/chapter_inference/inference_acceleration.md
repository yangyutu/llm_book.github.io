# Inference acceleration: Overview

## The fundamental challenge of LLM inference

LLM inference is **memory-IO bound, not compute bound**. In other words, it currently takes more time to load 1MB of data to the GPU’s compute cores than it does for those compute cores to perform LLM computations on 1MB of data. 

This is because LLM contains relative usually simple calculations (multiplication, addition, max-pooling, etc) that can be executed by high performing GPU parallel computing units. 



This means that LLM inference throughput is largely determined by how large a batch you can fit into high-bandwidth GPU memory. 

More specific reasons for LLM inference to be memory-bound is because of the following:

* **Model size**: Modern LLMs often have billions of parameters, requiring significant memory just to store the model weights.
* **Attention mechanism**: Transformers, which most LLMs are based on, use attention mechanisms that can require large amounts of memory, especially for long sequences.
* **Activations**: Storing intermediate activations during inference can consume substantial memory.
* **KV-cache**: For autoregressive generation, maintaining the key-value cache for past tokens uses additional memory that grows with sequence length.

While LLM inference is mainly memory-bound, the following factors also contributes to the computation intensity:

Matrix multiplications: The core operations in LLMs are matrix multiplications, which can be computationally intensive.
Nonlinear activations: Operations like softmax and layer normalization require additional computation.
Beam search: If used, beam search for text generation adds computational overhead.



# Inference acceleration: Quantization

## Basic Concepts

**Quantization** is the process of using a finite number of low-precision values (usually int8) to approximate high-precision (usually float32) numbers with relatively low loss in inference precision.

The objective of quantization is to **reduce memory usage** and improve inference speed without significantly compromising performance.

In the development of different quantization methods, there are
* QAT (Quantization-Aware-Training), which involves retraining or fine-tuning by approximating the differential rounding operation. While QAT is popular for small neural models, it is rarely used for LLMs.
* PTQ (Post-Training Quantization), which directly quantizes pre-trained LLM models. It requires a small amount of data for determining quantization parameters. This is the mainstream quantization method for LLMs.


Quantization can be applied to different parts of model, including
* weights
* activations
* KV Cache
 
with different levels of **quantization granularities**, including:
* per-tensor
* per-token/per-channel
* group-wise

## Standard quantization techniques

To introduce standard quantization techniques, we take 16-bit floating-point model weight $W_{f16}$ and its quantization into 8-bit integer as example. 

The **Absmax/Scale quantization** technique scales $W_{f16}$ into the 8-bit representation in the range of $[-127,127]$ via multiplying with 

$$s_{W} = \frac{127}{\max_{ij}|W_{f16}|},$$

which is equivalent to scaling the entire tensor to fit into the range of [0, 127]. Specificially, the value with minimal magnitude is mapped to 0 and the value with maximal magnitude is mapped to 127.

That is the 8-bit integer representation is given by

$$
W_{i8}=\operatorname{Round}\left(\frac{127 \cdot W_{f16}}{\max_{ij}|W_{f16}|}\right)=\operatorname{Round}\left(s_{W} \mathbf{W}_{f 16}\right)
$$ (chapter_inference_acceleration_quantization_eq_absmax_quantization)

where $\operatorname{Round}$ indicates rounding to the nearest integer.

The **de-quantization** of $W_{i8}$ is given by

$$\operatorname{DeQ}(W_{i8}) = \frac{W_{i8}}{s_{W}}.$$


In practice, there are chances where 8-bit integer value after the quantization is outside of the 8 bit representation. To mitigate this, we will add an additional clipping step,

$$W_{i8} = $$



**Affline quantization** shifts the input values such that its min value is mapped to -127 and its max value is mapped to 127. 



 and its into the full range $[-127,127]$ by scaling with the normalized dynamic range $n d_x$ and then shifting by the zeropoint $z p_x$. With this affine transformation, any input tensors will use all bits of the data type, thus reducing the quantization error for asymmetric distributions. 

For example, for ReLU outputs, in absmax quantization all values in $[-127,0)$ go unused, whereas in zeropoint quantization the full $[-127,127]$ range is used. Zeropoint quantization is given by the following equations:

$$
\begin{gathered}
n d_{x_{f 16}}=\frac{2 \cdot 127}{\max _{i j}\left(\mathbf{X}_{f 16}^{i j}\right)-\min _{i j}\left(\mathbf{X}_{f 16}^{i j}\right)} \\
z p_{x_{i 16}}=\left\lfloor\mathbf{X}_{f 16} \cdot \min _{i j}\left(\mathbf{X}_{f 16}^{i j}\right)\right\rceil \\
\mathbf{X}_{i 8}=\left\lfloor n d_{x_{f 16}} \mathbf{X}_{f 16}\right\rceil
\end{gathered}
$$




The **Round-to-Nearest (RTN) quantization** is a basic method used in the process of quantizing neural networks.

For a given numerical value $r$, RTN applies the following quantization formula

$$q = \operatorname{Clip}(\operatorname{Round}(\frac{r}{s}) + z, q_{min}, q_max)$$

where $s$ is scaling parameter, $z$ is the shifting parameter, and $q_{min}, q_{max}$ are the clipping range.

## Quantized matrix multiplication

The modern GPU hardware can significantly speed up the matrix multiplication in the integer representation. As matrix multiplication involves accumulation operations, the typical convension for accumulation data types are

| Low-Precision Data Type   | Accumulation Data Type    |
| :--- | ---: |
| float16    | float16    |
| bfloat16    | float32    |
| int16    | int32    |
| int6    | int32    |

The matrix multiplication between $X_{f16}$ and $W_{f16}$ can be approximated by

$$
\begin{aligned}
X_{f16}W_{f16} &\approx  \operatorname{DeQ}(X_{i8}) \operatorname{DeQ}(W_{i8}) \\
                &=\frac{X_{i8}}{s_{X}} \frac{W_{i8}}{s_{W}} \\
                &=\frac{1}{s_{X}\cdot s_W} X_{i8}W_{i8} 
\end{aligned}
$$

where the summation or accumation operation in matrix multiplication of $X_{i8}W_{i8}$ will be conducted in int-32 data type.


## Quantization granularities

### Overview

These different granularities offer trade-offs between quantization accuracy and computational efficiency. Per-tensor is the fastest but least accurate, per-channel/per-token is the most accurate but computationally expensive, and group-wise provides a balance between the two extremes

### Per-tensor quantization
Per-tensor quantization applies the same scaling factor and zero point to an entire tensor (usually a weight matrix or activation tensor). Each weight tensor and activation tensor in the model will have its own set of quantization parameters. Note that the biggest challenges for per-tensor quantization is that a single outlier can reduce the quantization precision of all other values. 

````{prf:example} Per-tensor quantization
Let's consider a weight matrix $W$, the quantize $W$ into int8, we have the following steps:
1. Find the minimum and maximum values in the entire tensor: min_val = -0.5, max_val = 0.8
2. Calculate the scale and zero point: scale = (max_val - min_val) / (2^8 - 1) = (0.8 - (-0.5)) / 255 ≈ 0.00510
zero_point = round(-min_val / scale) = round(0.5 / 0.00510) ≈ 98
3. Quantize the entire tensor using these parameters:
W_quant = round(W / scale) + zero_point

In this case, all elements use the same scale (0.00510) and zero_point (98) for quantization and dequantization.
````


### Per-channel quantization

Per-channel (from model weights perspective) quantization applies different scaling factors and zero points to each channel in the tensor.
Explanation:
This method computes separate quantization parameters for each channel (in the case of weights) or each token (in the case of activations). This allows for more fine-grained quantization, potentially preserving more information.

````{prf:example} Per-channel quantization
Let's consider the same weight matrix $W$ of shape (1024, 768), but now we'll quantize each output channel separately.

To quantize this to int8 per-channel, for each of the 768 columns (channels):
1. Find the minimum and maximum values in that column
2. Calculate the scale and zero point for that column
3. Quantize the column using its specific parameters

Each column in W_quant uses its own scale and zero_point for quantization and dequantization.
````


### Per-token quantization

Per-token (from activations perspective) quantization applies different scaling factors and zero points to each token in the tensor.

For a given activation tensor $A$ of shape $(B, S, H)$, where:
* $B$ is the batch size
* $S$ is the sequence length
* $H$ is the hidden dimension

There will be $S$ sets of quantization parameters; each set of parameters is computed from the $H$ hidden dimensionality values.

### Groupwise quantization

Group-wise quantization is a middle ground between per-tensor and per-channel quantization. It applies different quantization parameters to groups of channels or elements within a tensor.
Explanation:
In this approach, we divide the tensor into groups and compute separate quantization parameters for each group. This allows for more flexibility than per-tensor quantization while being more computationally efficient than per-channel quantization.
Example:
Let's consider the same weight matrix W of shape (1024, 768), and we'll use a group size of 32.
Original tensor (float32):
W = [[-0.5, 0.1, 0.7, ..., 0.3],
[0.2, -0.4, 0.6, ..., -0.1],
...
[0.8, -0.3, 0.5, ..., 0.4]]
To quantize this to int8 with group-wise quantization:

Divide the 768 channels into 24 groups of 32 channels each.
For each group:
a. Find the minimum and maximum values in that group
b. Calculate the scale and zero point for that group
c. Quantize the group using its specific parameters

For example, for the first group (channels 0-31):
min_val_g1 = -0.5, max_val_g1 = 0.8
scale_g1 = (0.8 - (-0.5)) / 255 ≈ 0.00510
zero_point_g1 = round(0.5 / 0.00510) ≈ 98
For the second group (channels 32-63):
min_val_g2 = -0.6, max_val_g2 = 0.7
scale_g2 = (0.7 - (-0.6)) / 255 ≈ 0.00510
zero_point_g2 = round(0.6 / 0.00510) ≈ 118
And so on for all 24 groups.
Resulting quantized tensor (int8):
W_quant = [[98, 255, 235, ..., 156, | 137, 0, 215, ..., 78, | ...],
[137, 0, 215, ..., 78,  | 255, 51, 196, ..., 176, | ...],
...
[255, 51, 196, ..., 176, | 98, 255, 235, ..., 156, | ...]]
In this case, each group of 32 channels shares the same scale and zero_point for quantization and dequantization.


## Quantization-performance trade-off in language models

Early research [{cite:p}`bondarenko2021understandingovercomingchallengesefficient`] during the BERT era revealed significant challenges in quantizing large language models. {cite:p}`bondarenko2021understandingovercomingchallengesefficient` demonstrated that applying round-to-nearest (RTN) quantization to both weights and activations of BERT models, reducing them to 8-bit precision, resulted in substantial performance deterioration on language understanding benchmarks.

Further ablation shows that quantization on activation is major cause of the performance drop and quantization on the model weights have minimal impact. The reason is that activation values from FFN's input and output can have strong outliers, which can directly cause notable error in the quantization process.

 As summary in the following table [{cite:p}`bondarenko2021understandingovercomingchallengesefficient`], a strategy of quantizing only the model weights to 8-bit precision while maintaining 32-bit precision for activations (referred to as 'W8A32') achieved performance comparable to full-precision models. This finding highlights the importance of selective quantization strategies that preserve critical information in activations while still benefiting from the efficiency gains of weight quantization. 

| Configuration | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE | GLUE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FP32 | 57.27 | 93.12 | 88.36 | 89.09 | 89.72 | 84.91 | 91.58 | 70.40 | 83.06 |
| W8A8 | 54.74 | 92.55 | 88.53 | 81.02 | 83.81 | 50.31 | 52.32 | 64.98 | 71.03 |
| W32A8 | 56.70 | 92.43 | 86.98 | 82.87 | 84.70 | 52.80 | 52.44 | 53.07 | 70.25 |
| W8A32 | 58.63 | 92.55 | 88.74 | 89.05 | 89.72 | 84.58 | 91.43 | 71.12 | 83.23 |


As the model size continues to grow to billions of parameters, outlier features of high magnitude start to emerge in all transformer layers, causing failure of simple low-bit quantization. Dettmers et al. (2022) observed such a phenomenon for OPT models larger than 6.7B parameters. Larger models have more layers with extreme outliers and these outlier features have a significant impact on the model performance. The scale of activation outliers in a few dimensions can be $\sim 100 \times$ larger than most of the other values.


As language models grow to encompass billions of parameters, a significant challenge emerges: the appearance of high-magnitude outlier features across all transformer layers. This phenomenon compromises the effectiveness of simple low-bit quantization techniques. {cite:p}`dettmers2022llmint88bitmatrixmultiplication` identified this issue in OPT models exceeding 6.7 billion parameters.

The problem intensifies with model size; larger models exhibit more layers with extreme outliers. These outlier features disproportionately influence model performance. In some dimensions, the scale of activation outliers can be approximately 100 times larger than the majority of other values.

This disparity poses a significant challenge for quantization, as traditional methods struggle to accurately represent both the outliers and the more typical values within the same low-bit format. Consequently, addressing these outliers has become a critical focus in the development of quantization techniques for large language models.

## Advanced quantization techniques

### LLM.int8()

{cite:p}`dettmers2022llmint88bitmatrixmultiplication`

Motivation

Basic quantization methods often resulted in significant performance degradation, especially for larger models.


maintain high performance while significantly reducing the memory footprint and computational requirements of large language models. This makes it possible to run these models on more modest hardware or to scale them more efficiently in production environments.


The key idea is:
* Group-wise quantization: Instead of quantizing the entire model uniformly, the method divides weight matrices into groups and quantizes each group separately. This allows for more fine-grained representation of the weights.
* Outlier handling: The method identifies and separates outlier values before quantization. These outliers are stored in 16-bit precision, while the rest of the weights are quantized into 8-bit int.


Method details:

For a given input matrix $\mathbf{X}_{f 16} \in \mathbb{R}^{s \times h}$, 
* First identify the subset of hidden dimensions that have at least one outliers based on certain magnitude criterion. We denote these dimensions by $O=\{i \mid i \in \mathbb{Z}, 0 \leq i \leq h\}$.
* For columns of $X$ and rows in $W$ that reside in $O$, we preserve its 16-bit precision; for the remaining columns and rows, we quantize into 8-bit precision. 
* The final resulting matrix can be represented by the adding inner products together, that is
  
$$
\mathbf{C}_{f 16} \approx \sum_{h \in O} \mathbf{X}_{f 16}^h \mathbf{W}_{f 16}^h+\mathbf{S}_{f 16} \cdot \sum_{h \notin O} \mathbf{X}_{i 8}^h \mathbf{W}_{i 8}^h
$$

where $\mathbf{S}_{f 16}$ is the denormalization term for the Int8 inputs and weight matrices $\mathbf{X}_{i 8}$ and $\mathbf{W}_{i 8}$.

It is found that 99.9% values can be represented by 8-bit int.

```{figure} ../img/chapter_inference/quantization/LLM_int8_illustration.png
---
scale: 50%
name: chapter_inference_quantization_fig_LLM_int8_illustration_plot
---
Figure 2: Schematic of LLM.int8(). Given 16-bit floating-point inputs $\mathbf{X}_{f 16}$ and weights $\mathbf{W}_{f 16}$, the features and weights are decomposed into sub-matrices of large magnitude features and other values. The outlier feature matrices are multiplied in 16 -bit. All other values are multiplied in 8 -bit. We perform 8 -bit vector-wise multiplication by scaling by row and column-wise absolute maximum of $\mathbf{C}_x$ and $\mathbf{C}_w$ and then quantizing the outputs to Int8. The Int32 matrix multiplication outputs $\mathrm{Out}_{i 32}$ are dequantization by the outer product of the normalization constants $\mathbf{C}_x \otimes \mathbf{C}_w$. Finally, both outlier and regular outputs are accumulated in 16-bit floating point outputs. Image from {cite:p}`dettmers2022llmint88bitmatrixmultiplication`.
```








```{figure} ../img/chapter_inference/quantization/LLM_int8_performance_plot.png
---
scale: 60%
name: chapter_inference_quantization_fig_LLM_int8_performance_plot
---
OPT model mean zeroshot benchmark accuracy at different quantization settings, including 16-bit baseline, regular 8-bit quantization method, and the LLM.int8() quantization method. Systematic outliers
emerge at a scale of 6.7B parameters, causing regular quantatization methods to have severe performance degradation. Image from {cite:p}`dettmers2022llmint88bitmatrixmultiplication`.
```


### Smooth Quant


### AWQ


### GPTQ


### FP8



## References and software

https://lilianweng.github.io/posts/2023-01-10-inference-optimization/

Quantization: 
https://leimao.github.io/article/Neural-Networks-Quantization/

:bibliography:`../llm_book.bib`