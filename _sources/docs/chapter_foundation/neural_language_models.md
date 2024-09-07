# Early Neural Language Models

## Motivation

As we mentioned above, $n$-gram models are count based methods, aimming to learn the joint distribution of word sequences, with the key assumption that the probability of a word depends only on the n-1 words that precede it.

It has challenges in language modeling from the following aspects:
* curse of dimensionality when $n$ becomes large. For example, consider a language with a vocabulary of size $V = 10^6$, a 10-gram model would require model parameters of $V^{10}$.
* difficulty in modeling long sequence dependency due to the sparsity of the long sequencenes data as well as the curse of dimensionaltiy for large $n$. 
* inaccuracy in modeling sequences containing rare words, although we can apply smoothing functions to alleviate the difficulty. 
* poor genearlization to unseen word combination
  
The recent neural language models are good at capturing the semantics of words, and they give good prediction for low frequency sequences.

These limitations of $n$-gram models motivated researchers to explore neural network approaches [{cite:p}`bengio2003neural,Mikolov2010recurrent`] that could capture deeper semantic relationships and longer-range dependencies in language. The resulting neural language models are proved to have the following advantages
* Improved representation: Neural networks can learn distributed representations of words (word embeddings), capturing semantic similarities.
* Better generalization with efficient model parameters: Neural models can generalize to unseen word combinations more effectively.
* Handling longer contexts: Recurrent neural network architectures allow for theoretically unlimited context.

$n$-gram models also fails to capture the semantic meaning of words.  

### Feed-forward neural language model

The core idea of $n$-gram model is nothing but a mechanical counter of co-occurrence of words. In natural language, there are many words that are similar in their meaning as wells as their grammar rules. 
For example, \textit{A cat is walking in the living room} vs. \textit{a dog is running in the bedroom} have similar word pairs (cat, dog), (walking, running), (living room, bedroom) and use similar patterns. 
These similarities or word semantic meaning (i.e., the latent representation of words) can be exploited to construct model with much smaller model parameters. 

{cite:p}`bengio2003neural` proposed neural language models, which predict and generate next word based on its context and operating at a low dimensional dense vector space (i.e., word embedding). 

The core idea is that by projecting words into low dimensional space (via learning and gradient descent), words with similar semantics are automatically clustered together, thus providing opportunities for effecient parameterization and mitigating the the curse of dimensionality in $n$-gram language models. 

In the feed-forward network model, each word, together with its preceding $n - 1$ words as context are projected into low-dimensional space and further predict the next word probability. Note that the context has a fixed length of $n$, which it is limited in the same way as in $n$-gram models [{ref}`chapter_foundation_fig_language_model_feedforward_model`].

Formally, the model is optimized to maximize

$${P}\left(w_{t} \mid w_{t-1}, \cdots w_{t-n+1}\right)=\frac{e^{y_{w_{t}}}}{\sum_{i} e^{y_{i}}}.$$

The $y_{i}$ is the logic for word $i$, given by

$$
y=b+We+U \tanh(d+He)
$$

where $W, H$ are matrices and $b, d$ are biases. Here $W$ can optionally zero, meaning no direct connections. $e = (e_1,...,e_{n-1})$ is the concatenations of word embeddings of each preceding token. 

Feed-forward neural language model brings several improvements over the traditional $n$-gram language model: it provides a compact and efficient parameterization to capture word dependencies among text data. Recall that $n$-gram model would have to store all observed $n$-grams. However, feed-forward neural language model still meet challenges to capture long-distance dependencies in natural language. In the feed-forward neural language model, capturing long-distance dependencies will require an increase of the context window size, which linearly scales with model parameters $W$. Another drawback is that a word that appears at different locations will be multiplied by different weights to get its embedding, which is inconsistent. 

```{figure} ../img/chapter_foundation/languageModeling/FeedForwardModel_v2.png
---
scale:30%
name: chapter_foundation_fig_language_model_feedforward_model
---
Feedforward neural netowk based language model.
```
## Recurrent neural language model

In recurrent network model [{ref}`chapter_foundation_fig_language_model_recurrent_model`], context is extended via recurrent connections and context length is theoretically unlimited.
Specially, let the recurrent network input be $x$, hidden layer output be $h$  and output probabilities be $y$. Input vector $x_t$ is a concatenation of a word vector $w_t$ and the previous hidden layer output $s_{t-1}$, which represents the context. To summarize, we have recurrent computation given by

$$
\begin{align}
	x_t &=\operatorname{Concat}(e_t,h_{t-1}) \\
	h_t &=\operatorname{Sigmoid}\left(W_xx_t + b_x\right) \\
\end{align}
$$

The prediction probability at each $t$ is given by

$$	p(y_{k}(t) =\operatorname{Softmax}\left(W_ys(t) + b_y\right).$$


```{figure} ../img/chapter_foundation/languageModeling/RecurrentModel_v2.png
---
scale: 40%
name: chapter_foundation_fig_language_model_recurrent_model
---
Recurrent neural network based language model.
```

Compared with $n$-gram language model and feed-forward neural language model, RNN language model can in principle process input of any length without increasing the model size. 

RNN language model also has several drawbacks: Computing a conditional probability $p(w_t|w_{1:t-1})$ is expensive. One mitigating strategy is to cache and re-use previous computed results or to pre-compute conditional probabilities for frequent $n$-grams.  some In practical applications, RNN language models from different domains are not easy to merge. 
