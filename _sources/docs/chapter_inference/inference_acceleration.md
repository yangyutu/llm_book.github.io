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

## RTN quantization
The **Round-to-Nearest (RTN) quantization** is a basic method used in the process of quantizing neural networks.

For a given numerical value $r$, RTN applies the following quantization formula

$$q = \operatorname{Clip}(\operatorname{Round}(\frac{r}{s}) + z, q_{min}, q_max)$$

where $s$ is scaling parameter, $z$ is the shifting parameter, and $q_{min}, q_{max}$ are the clipping range.


## Basic Quantization-performance trade-off in language models

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


### Smooth Quant


### AWQ


### GPTQ


### FP8



## References and software

https://lilianweng.github.io/posts/2023-01-10-inference-optimization/

:bibliography:`../llm_book.bib`