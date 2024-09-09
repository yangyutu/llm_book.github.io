# Transformers and BERT

## Pretrained Language Models

Pretrained language models are a key technology in modern natural language processing (NLP) that leverages large scale of un-labelled text data and computational power to drastically improve language understanding and generation tasks.

At their core, pretrained language models are large neural networks that have been exposed to enormous amounts of text data – including millions of books, articles, and websites. Through this exposure, they learn to recognize patterns in language, grasp context, and even pick up on subtle nuances in meaning.

The key advantage of pretrained language model lies in the fact that it can be universally adapted (i.e., fine-tuning) to all sorts of specific tasks with a small amount of labeled data. As a comparison, training a task-specific model from scratch would require large amount of labeled data, which can be expensive to obtain.

Pretrained language models typically use neural network architectures designed to process sequential data like text. The most prominent architectures in recent years have been based on the Transformer model, but there have been other important designs as well. Let's explore some of the key architectures:
Transformer-based models:
The Transformer architecture, introduced in 2017, has become the foundation for most modern language models. It uses a mechanism called self-attention to process input sequences in parallel, allowing the model to capture long-range dependencies in text more effectively than previous approaches.

BERT (Bidirectional Encoder Representations from Transformers):
BERT uses the encoder portion of the Transformer. It's bidirectional, meaning it looks at context from both sides of each word when processing text. This makes it particularly good at tasks like sentence classification and named entity recognition.
GPT (Generative Pre-trained Transformer):
GPT models use the decoder portion of the Transformer. They process text from left to right, making them well-suited for text generation tasks. Each version (GPT, GPT-2, GPT-3, etc.) has scaled up in size and capability.


Other architectures:

ELMo (Embeddings from Language Models):
Before Transformers, ELMo used bidirectional LSTMs (Long Short-Term Memory networks) to create contextual word embeddings. While less common now, it was an important step in the evolution of language models.

## Transformers

### Overall architecture

Since 2007, Transformer {cite:p}`vaswani2017attention` has emerged as one of most successful architectures in tackling challenging seq2seq NLP tasks like machine translation, text summarization, etc. 

Traditionally, seq2seq tasks heavily use RNN-based encoder-decoder architectures, plus attention mechanisms, to transform one sequence into another sequence. Transformer, on the other hand, does not rely on any recurrent structure and is able to process all tokens in a sequence at the same time. This enables computation efficiency optimization via parallel optimization and address long-range dependency, both of which mitigate the shortages of RNN-based encoder-decoder architectures. 

On a high level, Transformer falls into the category of encoder-decoder architecture [\autoref{ch:neural-network-and-deep-learning:Advanced:NLP:fig:transformer}], where the encoder encodes an input token sequence into low-dimensional embeddings, and the decoder takes the embeddings as input, plus some additional prompts, outputs an output sequence probabilities. The position information among tokens, originally stored in recurrent network structure, is now provided through position encoding added at the entry point of the encoder and decoder modules.  

Attention mechanisms are the most crucial components in the Transformer architecture to learn contextualized embeddings and overcome the limitation of recurrent neural network in learning long-term dependencies (e.g., seq2seq model with attention). 

The encoder module on the left [{numref}`chapter_foundation_fig_pretrained_LM_transformer_arch`] consists of blocks that are stacked on top of each other to obtain the embeddings that retain rich information in the input. Multi-head self-attentions are used in each block to enable the extraction of contextual information into the final embedding.
Similarly, the decoder module on the right also consists of blocks that are stacked on top of each other to obtain the embeddings. Two different types of multi-head attentions are used in each docder block, one is self-attention to capture contextual information among output sequence and one is to encoder-decoder attention to capture the dynamic information between input and output sequence. 

In the following, we will discuss each component of transformer architecture in detail. 


```{figure} ../img/chapter_foundation/pretrainedLM/Transformer_arch/transformer_arch.png
---
scale: 30%
name: chapter_foundation_fig_pretrained_LM_transformer_arch
---
The transformer architecture, which consists of an Encoder (left) and a Decoder (right).
```

### Input output conventions

To understand the training and inference of transformer architecture, we need to distinguish different types of sequences:
* Input sequence $x = (x_1,x_2,...,x_p,..., x_n), x_i\in \mathbb{N}$ and input position sequence $x^p = (1, 2, ..., n)$
* Output sequence $y = (y_1,y_2,...,y_p,...,y_m), y_i \in \mathbb{N}$ and output position sequence $y^p = (1, 2, ..., m)$. Output sequence is the input to the decoder.
* Target sequence $t = (t_1,t_2,...,t_p,...,t_m), t_i \in \mathbb{N}$. Input sequence and target sequence form a pair in the training examples.

For example, consider a translational task with input and target sequence given by
<span style="color:red">**Ich möchte eine Flasche Wasser**</span> and <span style="color:red">**I want a bottle of water**</span>. In the typical supervised learning training, the input sequence, target sequence, and output sequence are in the following form
* $x = (Ich, möchte, eine, Flasche, Wasser, PAD, PAD)$
* $t = (I, want, a, bottle, of, water, EOS, PAD)$
* $y = (SOS, I, want, a, bottle, of, water, EOS)$

The output sequence is the right-shifted target sequence with a starting token \code{SOS}. The output sequence will be fed into the decoder to predict the next token.

### Position encodings


Consider an input sequence represented by an integer sequence $x = (x_1,...,x_i,...,x_n), x_i\in \mathbb{N}$, $e.g., x = (3, 5, 30, 2, ..., 21)$ that is fed into the Transformer architecture.  For the input sequence $s$, the word embedding vectors of dimensionality $d_{model}$ alone do not encode positional information in the sequence. This can be fixed by utilizing a position encoding $PE$ maps an integer index representing the position of the token in the sequence, $x^p = (1, 2, ..., n)$ to $n$ dense vectors with same dimensionality $d_{model}$, i.e., $\operatorname{PE}(s^p)\in \mathbb{R}^{n\times d_{model}}$. 

The position encoding vectors can be specified in analytical forms and then be fixed during training.  The position encoding mapping can also be learned from data. In theory, the specification of position encoding should preserve position information by mapping nearby positions to nearby high-dimensional vectors. 

Notably, given the token position $i \in \{1, ..., n\}$ $\operatorname{PE}(i) \in \mathbb{R}^{d_{model}}$ is given by

$$
[\operatorname{PE}(i)]_j =\left\{\begin{array}{ll}
	\sin \left(\frac{i}{10000^{j/ d_{model}}}\right) & \text { if } j \text{ is even} \\
	\cos \left(\frac{i}{10000^{(j-1)/ d_{model}}}\right) & \text { if } j \text{ is odd}
\end{array}\right.
$$

where $j = \{1,...,d_{model}\}$. Note that the position encodings have the same dimension $d_{model}$ as the word embeddings such that they can be summed.
Intuitively, each dimension of the positional encoding corresponds to a sine/cosine wave of different wavelengths ranging from $2 \pi$ (when $j=1$) to approx $10000 \cdot 2 \pi$(when $j=d_{model}$). An example position encodings of dimensionality 256 for position index from 1 to 256 is shown in {numref}`chapter_foundation_fig_pretrained_LM_transformer_positionencoding`.




```{figure} ../img/chapter_foundation/pretrainedLM/Transformer_arch/positionEncoding.png
---
scale: 30%
name: chapter_foundation_fig_pretrained_LM_transformer_positionencoding
---
Example position encodings of dimensionality 256 for position index from 1 to 256.
```

### Multihead attention with masks

```{figure} ../img/chapter_foundation/pretrainedLM/Transformer_arch/Multi-head_attention.png
---
scale: 30%
name: chapter_foundation_fig_pretrained_LM_transformer_multiheadattention
---
The multi-head attention architecture.
```

Multi-head attention mechanism with masks plays a central role in both encoder and decoder side. Attention mechanism enables contexualization and masks control the context each token can get access to. Attention of multiple heads, rather than a single head, allows the model to jointly attend to information from different representation subspaces at different positions {cite:p}`li2018multi`.

Multi-head attentions are used in three places: 
* In the encoder module, multi-head attention without using masks are used to construct intermediate contextualized embedding for each token that depends of its context. The query, key, and values are all the same input sequence.
* In the decoder module, multi-head attention with masks are used to construct contextualized embedding for each token by attending to only its \textit{preceding} or \textit{seen} tokens. The query, key, and values are all the same input sequence.
* From the encoder module to the decoder module, multi-head attention without using masks is used to construct embedding for each output token that depends of its \textit{input} context (i.e., attention between input and output sequences).


Given a query matrix $Q\in \mathbb{R}^{n\times d_{model}}$ representing $n$ queries, a key matrix $K\in \mathbb{R}^{m\times d_{model}}$ representing $m$ keys, and a value matrix $V\in \mathbb{R}^{m\times d_{model}}$ representing $m$ values, the multi-head ($h$ heads) attention associated with $(Q, K, V)$ is given by

$$\operatorname{MultiHeadAttention}\left(Q,K,V\right)=Concat\left(head_1,\cdots,head_h\right)W^O$$

where $head_i \in \mathbb{R}^{n\times d_v}$ and is given by

$$head_i=\operatorname{Attention}\left(QW_i^Q,KW_i^K,VW_i^V\right).$$

Here
$W^Q, W^K, W^V\in\mathbb{R}^{d_{model}\times d_k}, W^O\in\mathbb{R}^{h\times d_v\times d_{model}}$ are additional linear transformations applied to query, key, and value matrices, respectively. Note that each head has its own corresponding $W^Q, W^K, W^V$, and we omit the subscript $i$ for simplicity. In general, we require $d_k = d_v = d_{model}/h$ such that the output of $\operatorname{MultiHeadAttention}\left(Q,K,V\right)$ has the dimensionality of $n\times d_{model}$.

The attention output of a single head among $(Q, K, V)$ is given by 

$$\operatorname{Attention}\left(Q W^{Q}, K W^{K}, V W^{V}\right)=\operatorname{Softmax}\left(\frac{Q W^{Q}\left(K W^{K}\right)^{T}}{\sqrt{d_{k}}}\right) V W^{V},$$

where $\sqrt{d_k}$ is the scaling factor preventing the doc product value from saturating the Softmax. This type of attention is also known as ***scaled dot product*** attention.

Note that the Softmax normalize each row such that the $\operatorname{Softmax}(\cdot)$ produces a weight matrix $w^{att}\in \mathbb{R}^{n\times m}$, with each row summing up to unit 1.

The un-normalized weight matrix is given by

$$\tilde{w}_{ij}^{att} = [QW^Q]_i[KW^K]^T_j,$$

where  $[QW^Q]_i$ is the $i$th row vector of query matrix $QW^Q$, and $[KW^K]_j$ is the $j$th row vector of key matrix $KW^K$.

Explicitly, we have the attention output as the weighted sum of transformed value vectors:

$$\begin{bmatrix}
	w^{att}_{11} & w^{att}_{12} & \cdots & w^{att}_{1m}\\ 
	w^{att}_{21} & w^{att}_{22} & \cdots & w^{att}_{2m} \\ 
	\vdots & \vdots & \ddots & \vdots\\ 
	w^{att}_{n1} & w^{att}_{n2} & \cdots & w^{att}_{nm}
\end{bmatrix}\begin{bmatrix}
	[VW^V]_1\\ 
	[VW^V]_2\\ 
	\vdots\\ 
	[VW^V]_m
\end{bmatrix} = \begin{bmatrix}
	\sum_{j=1}^m w^{att}_{1j}[VW^V]_j\\ 
	\sum_{j=1}^m w^{att}_{2j}[VW^V]_j\\
	\vdots\\ 
	\sum_{j=1}^m w^{att}_{nj}[VW^V]_j\\
\end{bmatrix}$$

where $[VW^V]_i$ is the $i$th row vector of value matrix $VW^V$.

We can apply mask to tokens when we want to only allow a subset of keys and values to be queried. Normally, we associate each token with a binary ***mask*** $mask \in \{0, 1\}^{m}$, where $0$ indicates exclusion of its key. With masks applied, we can compute un-normalized attention weights via 

$$
\tilde{w}_{ij}^{att} =\left\{\begin{array}{ll}
	[QW^Q]_i[KW^K]^T_j & \text { if token } j \text{ is not masked} \\
	-\infty & \text { if token } j \text{ is masked}
\end{array}\right.
$$

Here by setting un-normalized attention weight to $-\inf$, we are effectively setting the normalized attention weight to zero. 

At this point, it is clear that the memory and computational complexity required to compute the attention matrix is quadratic in the input sequence length $n$. This could be a bottleneck the overall utility of attention-based models in applications involving the processing of long sequences. 


### Comparison with recurrent layer in sequence modeling

This section compares self-attention layers with recurrent layers, which are commonly used in sequence modeling.


The following table summarize the comparison of these layer types based on the following aspects:
* Total computational complexity per layer
* Potential for parallelization, measured by the minimum number of required sequential operations

First, regarding the computational complexity for processing a sequence with length $n$, 
* As self-attention layers connect all positions with a constant number of operations, so it has $O(1)$ complexity for the number of sequential operations. As each token in the sequence needs to attend to all other tokens, and the attention between two tokens are dot product with complexity $O(d)$, the total computational complexity for self-attention is $O(n^2d_{model})$
* Recurrent layers require $O(n)$ sequential operations. Within each sequential operation, the state vector of dimensionality $d_{model}$ will be updated by multiplying matrices of $d_{model}\times d_{model}$, and input vecotr of dimensionality $d_{model}$. In total, each sequential operation will have $O(d_{model}^2$ computational complexity, and the complexity of sequence of length $n$ becomes $O(n\cdot d^2_{model})$. 

The

To reduce computational complexity for self-attention when it comes to very long sequences, we can restrict the attention window size to be a neighborhood of size r, which will reduce the complexity to $O(r \cdot n \cdot d)$ .

| Layer Type | Complexity per Layer | Sequential <br> Operations |
| :--- | :---: | :---: |
| Self-Attention | $O\left(n^2 \cdot d_{model}\right)$ | $O(1)$ |
| Recurrent | $O\left(n \cdot d^2_{model}\right)$ | $O(n)$ |
| Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ |


### Pointwise FeedForward Layer


After all input embedding vectors go through multi-head self-attention layer, each of the output contextualized embedding vectors is still a linear weighted sum of input vectors, with the weight given by the attention matrix values. 
The motivation of a point-wise two-layer feed-forward network is to enhance the modeling capacity with non-linearity transformations (e.g., with ReLU activations) and interactions between different feature spaces. The pointwise feed-forward layer applies to input $x\in \mathbb{R}^{d_{model}}$ at each position separately and identically (i.e., sharing parameters across positions) [{numref}`chapter_foundation_fig_pretrained_LM_transformer_pointwiseffn`]:

$$
\operatorname{FFN}(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}.
$$

Typically, the first layer first maps the embedding vector  of dimensionality $d_{model}$ to a larger dimensionality $d_{ff}$ (e.g., $d_{model} = 512, d_{ff}=2048$) and then the second layer map the intermediate vector to a vector with same input vector size


```{figure} ../img/chapter_foundation/pretrainedLM/Transformer_arch/pointwise_FFN.png
---
scale: 30%
name: chapter_foundation_fig_pretrained_LM_transformer_pointwiseffn
---
Point-wise feed-forward network to perform nonlinear transformation on the contextualized embedding at each position.
```

There are additional dropouts, residual connections, and layer-wise normalization after the point-wise feed-forward layer. Taken all together, now we can define the following encoder layer.

````{prf:definition} Encoder layer
:label: chapter_foundation_def_pretrained_LM_transformer_encoder_layer

Given $n$ sequential input embeddings represented as $e_{in} = (e_{in,1},...,e_{in,n})$. The Transformer encoder layer performs the following calculation procedures

$$
\begin{align}
    e_{mid} &= \operatorname{LayerNorm} (e_{in} + \operatorname{MultiHeadAttention}(e_{in}, e_{in}, e_{in}, padMask)) \\
    e_{out} &= \operatorname{LayerNorm} (e_{mid} + \operatorname{FFN}(e_{mid}))
\end{align}  
$$

where $e_{mid}, e_{out} \in \mathbb{R}^{n\times d_{model}}, $

$$\operatorname{FFN}(e_{mid}) = \max(0, e_{mid} W_1 + b_1)W_2 + b_2,$$

with $W_1\in \mathbb{R}^{d_{model}\times d_{ff}}$, $W_2\in \mathbb{R}^{d_{ff}\times d_{model}}$, $b_1,b_2 \in \mathbb{R}^{d_{ff}}$, and the $padMask$ excludes padding symbols in the sequence.

````
In a typical setting, we have $d_{\text {model }}=512$, and the inner-layer has dimensionality $d_{ff}=2048$.



The whole computation in the encoder module can be summarized in the following.
````{prf:definition} computation in encoder module
:label: chapter_foundation_def_pretrained_LM_transformer_encoder_computation

Given an input sequence represented by integer sequence $s = (i_1,...,i_p,...,i_n)$ and its position $s^p = (1,..., p, ..., n)$. The encoder module takes $s, s^p$ as inputs and produce $e_N \in \mathbb{R}^{n\times d_{model}}$. 	

$$\begin{align}
    e_{0}&=\operatorname{WE}(s)+ \operatorname{PE}(s^p) \\
    e_1 & = \operatorname{EncoderLayer}(e_0) \\
    e_2 & = \operatorname{EncoderLayer}(e_1) \\
    &\cdots \\
    e_L & = \operatorname{EncoderLayer}(e_{L - 1})
\end{align}
$$

where $e_i \in \mathbb{R}^{n\times d_{model}}$, $\operatorname{EncoderLalyer}: \mathbb{R}^{n\times d_{model}}\to \mathbb{R}^{n\times d_{model}}$ is an encoder sub-unit, $N$ is the number of encoder layers. Specifically, this encoder layer is given by \autoref{ch:neural-network-and-deep-learning:Advanced:NLP:def:BERTencoderLayer}.
Note that Dropout operations are not shown above. Dropouts are applied after initial embeddings $e_0$, every self-attention output, and every point-wise feed-forward network output. 	
````

Also note that there is active research on where to optimally add the layer normalization in the encoder {cite:p}`xiong2020layer`. As shown in {numref}`chapter_foundation_fig_pretrained_LM_transformer_layernormalizationposition`, post-layer normalization (the one in the original paper {cite:p}`vaswani2017attention`) adds normalization layer after multi-head attention output and feed-forward layer output. Pre-layer normalization, on the other hand, adds normalization layer before the inputs entering into the multi-head attention and feed-forward layers.
 

```{figure} ../img/chapter_foundation/pretrainedLM/Transformer_arch/layer_normalization_position.png
---
scale: 30%
name: chapter_foundation_fig_pretrained_LM_transformer_layernormalizationposition
---
Post-layer normalization and pre-layer normalization in an encoder layer.
```

### Decoder anatomy


In the decoder side, we are similarly given an output sequence represented by integer sequence $o = (o_1,...,o_p,...,o_m), o_p\in \mathbb{N}$, $e.g., o = (5, 10, 30, 2, ..., 21)$. The decoder module  aims to converts an output sequence $o$, combining with resulting embedding $e_N$ in the encoder module to its corresponding probabilities over the vocabulary. The output probability can be used to compute categorical loss that drives the learning process of the encoder and the decoder.



Note that there are two type of attention in the decoder module, one is self-attention among the output sequence itself and one is attention between encoder output and decoder output, i.e., ***encoder-decoder attention***. 
The encoder-decoder attention uses a mask that excludes padding symbol in the input sequence. The queries come from the output of previous sub-unit and keys and values come from the final output of the encoder module. This allows decoder sub-units to attend to all positions in the input sequence. 

In each decoder layer, inputs are first contextualized via multi-head self-attention. Because we restrict each input token to only attend to its preceding input tokens, we apply a mask  

The decoder self-attention uses a mask that excludes padding symbol and future symbol, which can be computed via logical $\operatorname{OR}$ between $padMask$ and $seqMask$. $seqMask$ for a symbol at position $i$ is a binary vector $seqMask_i \in \mathbb{R}^m$ whose value is 1 at position equal or greater than $i$. Therefore, for a sequence of length $m$, the complete $seqMask$ would be a upper triangle matrix value 1.

{numref}`chapter_foundation_fig_pretrained_LM_transformer_transformer_arch_module` illustrates the connection between the Encoder component and Decoder component. 


```{figure} ../img/chapter_foundation/pretrainedLM/Transformer_arch/transformer_arch_module.png
---
scale: 30%
name: chapter_foundation_fig_pretrained_LM_transformer_transformer_arch_module
---
Illustration of the interaction between encoder module output and deconder in Transformer.
```


The whole computation in the decoder module can be summarized in the following.

````{prf:definition} computation in decoder module

Given an input sequence represented by integer sequence $o = (o_1,...,o_p,...,o_n)$ and its position $o^p = (1,..., p, ..., m)$. The encoder module takes $o, o^p$ as inputs, combines final contextualized embedding $e_N$ from the encoder, and produce $d_N \in \mathbb{R}^{n\times d_{model}}$ and probabilities over the vocabulary. 	

$$\begin{align}
    d_{0}&=\operatorname{WE}(o)+ \operatorname{PE}(o^p) \\
    d_1 & = \operatorname{DecoderLayer}(d_0, e_N) \\
    d_2 & = \operatorname{DecoderLayer}(d_1, e_N) \\
    &\cdots \\
    d_N & = \operatorname{DecoderLayer}(d_{N - 1}, e_N) \\
    output~prob &= \operatorname{Softmax}(d_N W)
\end{align}
$$

where $d_0 \in \mathbb{R}^{m\times d_{model}}$, $\operatorname{DecoderLayer}: R^{m\times d_{model}}\to R^{m\times d_{model}}$ is a decoder sub-unit, $N$ is the number of decoder sub-units, $W\in\mathbb{R}^{ d_{model} \times |V|^O}$. Specifically, each decoder layer can be decomposed into following calculation procedures

$$\begin{align}
    d_{mid1} &= \operatorname{LayerNorm} (d_{in} + \operatorname{MultiHeadAttention}(d_{in}, d_{in}, d_{in})) \\
    d_{mid2} &= \operatorname{LayerNorm} (d_{mid1} + \operatorname{MaskedMultiHeadAttention}(d_{mid1}, e_{N}, e_{N})) \\
    d_{out} &= \operatorname{LayerNorm} (d_{mid2} + \operatorname{FFN}(d_{mid2}))
\end{align}  
$$

where $d_{mid1}, d_{mid2}, d_{out} \in \mathbb{R}^{m\times d_{model}}, $

$$FFN(d_{out}) = \max(0, d_{mid} W_1 + b_1)W_2 + b_2,$$

with $W_1\in \mathbb{R}^{d_{model}\times d_{ff}}, W_2\in \mathbb{R}^{d_{ff}\times d_{model}}, b_1 \in \mathbb{R}^{d_{ff}}, b_2\in \mathbb{R}^{d_{model}}$.
````

### Computational breakdown analysis

In the following, we analyze the two core components of the Transformer (i.e., the self-attention module and the position-wise FFN). Let the model dimension be $D$, and the input sequence length be $T$. We also assume that the intermediate dimension of FFN is set to $4D$ and the dimension of keys and values are set to $D/H$ in the self-attention module.

The following data summarize the complexity and number of parameters for these two modules.

| Module | Complexity | #Parameters |
| :---: | :---: | :---: |
| self-attention | $O\left(T^2 \cdot D\right)$ | $4 D^2$ |
| position-wise FFN | $O\left(T \cdot D^2\right)$ | $8 D^2$ |

Here are the key observations:
* When the input sequences are short, the hidden dimension $D$ dominates the complexity of self-attention and position-wise FFN. The bottleneck of Transformer thus lies in FFN.
* When the input sequences grow longer, the sequence length $T$ gradually dominates the complexity of these modules, in which case self-attention becomes the bottleneck of Transformer.
* Furthermore, the computation of self-attention requires that a $T \times T$ attention distribution matrix is stored, which makes the computation of Transformer to be memory bounded. 


## Different branches of developments

### Overview


```{figure} ../img/chapter_foundation/pretrainedLM/Transformer_arch/transformer_families.png
---
scale: 30%
name: chapter_foundation_fig_pretrained_LM_transformer_transformer_families
---
Different branches of developments derived from the Transformer architecture: (left) Encoder branch, (middle) Encoder-Decoder branch, and (right) Decoder branch.
```

The Transformer architecture was originally intended to tackle challenges in Seq2Seq tasks such as machine translation or summarization. The simplicity in architecture and effectiveness of attention mechanism draw much interest since its invention. Transformer type of architectures have become the dominant model architecture for most of NLP tasks. Moreover, Transformer architecture has also been widely adopted in computer vision {cite:p}`dosovitskiy2020image` and recommender systems {cite:p}`sun2019bert4rec`, which were previously by other CNN and DNN architectures. 

In the process of adapting Transformer for different applications, there have been efforts that continue the improvement on the original encoder-decoder architecture as well as efforts that use only the encoder part or the decoder part separately. 


The objective, also known as ***denoising objective***, is to fully recover the original input from the corrupted one in a bidirectional fashion, as shown on the left side of Figure 4.1, which you will see shortly. As seen in the ***Bidirectional Encoder Representations from Transformers (BERT)*** architecture, which is a notable example of AE models, they can incorporate the context of both sides of a word. However, the first issue is that the corrupting [MASK] symbols that are used during the pre-training phase are absent from the data during the fine-tuning phase, leading to a pre-training-fine-tuning discrepancy. Secondly, the BERT model arguably assumes that the masked tokens are independent of each other.

On the other hand, AR models keep away from such assumptions regarding independence and do not naturally suffer from the pre-train-fine-tuning discrepancy because they rely on the objective predicting the next token conditioned on the previous tokens without masking them. They merely utilize the decoder part of the transformer with masked selfattention. They prevent the model from accessing words to the right of the current word in a forward direction (or to the left of the current word in a backward direction), which is called ***unidirectionality***. They are also called Causal Language Models (CLMs) due to their unidirectionality.

For different branches of models, we employ different training strategies:
\begin{itemize}
	\item Generative pretrained models like the GPT family are trained using a Causal Language Modeling objective.
	\item Denoising models like the BERT family are trained using a Masked Language Modeling objective.
	\item Encoder-decoder models like the T5, BART or PEGASUS models are trained using heuristics to create pairs of (inputs, labels). These heuristics can be for instance a corpus of pairs of sentences in two languages for a machine translation model, a heuristic way to identify summaries in a large corpus for a summarization model or various ways to corrupt inputs with associated uncorrupted inputs as labels which is a more flexible way to perform denoising than the previous masked language modeling.
\end{itemize}

### The encoder branch

The most influential encoder-based model is ***BERT*** {cite:p}`devlin2018bert`, which stands for Bidirectional Encoder Representations from Transformers.  BERT is pretrained with the two objectives:
* Predicting masked tokens in texts, known as masked language modeling (MLM)
* Determining if two text passages follow each other, which is known as next-sentence-prediction (NSP). 
The MLM helps learning of contextualized word-level representation, and the NSP objective aims to improve the tasks like question answering and natural language inference, which require reasoning over sentence pairs. BERT used the BookCorpus and English Wikipedia for pretraining and the model can then be fine-tuned with supervised data on downstream natural language understanding (NLU) tasks such as text classification, named entity recognition, and question-answering. At the time it was published, it achieved all state-of-the-art results on the popular GLUE benchmark. The success of BERT drew significant attention and up to date BERT like Encoder-only models dominate research and industry on natural language understanding (NLU) tasks {cite:p}`xia2020bert`. We will discuss BERT in the following chapter. 

***RoBERTa*** (Robustly Optimized BERT) is a follow-up study of BERT, which reveals that the performance of BERT can be further improved by modifying the pretraining scheme. RoBERTa uses larger batches with more training data and dropped the NSP task to significantly improve the performance over the original BERT model.

Although BERT model delivers great results, it can be expensive and difficult to deploy in production due to its model size and memory footprint. The ***ALBERT*** model [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:sec:BERT_ALBERT}] introduced three changes to make the encoder architecture more efficient. First, it reduces embedding dimensionality via matrix factorization, which saves parameters especially when the vocabulary gets large. Second, all layers share the parameters which decreases the number of effective parameters even further. Finally, ALBERT enhance the NSP objective with a more challenging sentence-ordering prediction (SOP), which primary focuses on inter-sentence coherence for sentence pairs in the same text segment.

By using model compression techniques like knowledge distillation, we can preserve most of the BERT performance with much smaller model size and memory footprint. Representative models include ***DistilBERT*** and ***TinyBERT***.


## The decoder branch

The decoder component in the Transformer model can be used for auto-regressive language modeling. GPT series are among the most successful auto-regressive pretrained language models, and they form the foundation of LLM.

***GPT-1*** {cite:p}`radford2018improving`: One of the major contributions of the GPT-1 study is the introduction of a two-stage unsupervised pretraining and supervised fine-tuning scheme. They demonstrates that a pre-trained model with fine-tuning can achieve satisfactory results over a range of diverse tasks, not just for a single task.

***GPT-2*** {cite:p}`radford2019language`:  is a larger model trained on much more training data, called WebText, than the original one. It achieved state-of-the-art results on seven out of the eight tasks in a zero-shot setting in which there is no fine-tuning applied. The key contribution of GPT-2 is demonstrating the capability of zero-shot learning with extensively pretrained language model alone (i.e., no finetuning).  

***GPT-3*** {cite:p}`brown2020language`: GPT-3 is up-scaled from GPT-2 by a factor of 100. It demonstrated that lead to significant improvements in performance and capabilities, which also marked the beginning of LLM era. Besides being able to generate impressively realistic text passages, the model also exhibits few-shot learning capabilities: with a few examples of a novel task such as text-to-code examples the model is able to accomplish the task on new examples. 

## The encoder-decoder branch

Although it has become common to build models using a single encoder or decoder stack, there are several encoder-decoder variants of the Transformer that have novel applications across both NLU and NLG domains:


***T5*** The T5 model unifies all NLU and NLG tasks by converting all tasks into a text-to-text paradigm. As such all tasks are framed as sequence-to-sequence tasks where adopting an encoder-decoder architecture is natural. The T5 architecture uses the original Transformer architecture. Using the large crawled C4 dataset, the model is pre-trained with masked language modeling as well as the SuperGLUE tasks by translating all of them to text-to-text tasks. The largest model with 11 billion parameters yielded state-of-the-art results on several benchmarks although being comparably large.

***BART*** BART combines the pretraining procedures of BERT and GPT within the encoder-decoder architecture. The input sequences undergoes one of
several possible transformation from simple masking, sentence permutation, token deletion to document rotation. These inputs are passed through the encoder and the decoder has to reconstruct the original texts. This makes the model more flexible as it is possible to use it for NLU as well as NLG tasks and it achieves state-of-the-artperformance on both.

