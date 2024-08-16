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

The Round-to-Nearest (RTN) quantization is a basic method used in the process of quantizing neural networks.

For a given numerical value $r$, RTN applies the following quantization formula

$$q = \operatorname{Clip}(\operatorname{Round(\frac{r}{s})} + z, q_{min}, q_max)$$

where $s$ is scaling parameter, $z$ is the shifting parameter, and $q_{min}, q_{max}$ are the clipping range.


## References

https://lilianweng.github.io/posts/2023-01-10-inference-optimization/

:bibliography:`../llm_book.bib`