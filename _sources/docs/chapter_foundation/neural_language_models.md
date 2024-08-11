# Early Neural Language Models

## Motivation

As we mentioned above, N-gram models are count based methods. For long sequences or sequences containing rare words, we have to apply smoothing functions. The recent neural language models are good at capturing the semantics of words, and they give good prediction for low frequency sequences.


The $n$-gram statistical language model aims to learn the joint distribution of word sequences. This approach suffers from curse of dimensionality when $n$ becomes large. 
For example, consider a language with a vocabulary of size $V = 10^6$, a 10-gram model would require model parameters of $V^{10}$. On the other hand, natural languages tend to have long-range dependency, i.e., the occurrence probability of a word may depend on the existence of an another word multiple steps before. Therefore, a high-quality language model must require large $n$ and, unfortunately, the curse of dimensionality becomes one inherent limitation of $n$-gram models.

$n$-gram models also fails to capture the semantic meaning of words\cite{bengio2003neural}. The core idea of $n$-gram model is nothing but a mechanical counter of co-occurrence of words. In natural language, there are many words that are similar in their meaning as wells as their grammar rules. 
For example, \textit{A cat is walking in the living room} vs. \textit{a dog is running in the bedroom} have similar word pairs (cat, dog), (walking, running), (living room, bedroom) and use similar patterns. 
These similarities or word semantic meaning can be exploited to construct model with smaller parameters.   

### Feed-forward neural language model


To overcome the limitation of $n$-gram models, Bengio et al \cite{bengio2003neural} proposed neural language models, which predict and generate next word based on its context and operating at a low dimensional dense vector space (i.e., word embedding). In the early development of neural language model, there are two important attempts. One is feed-forward neural network \cite{bengio2003neural} and one is recurrent neural network {cite:p}`Mikolov2010recurrent` [\autoref{ch:neural-network-and-deep-learning:ApplicationsNLP:fig:NeuralNetworkLanguageModel}].

The core idea is that by projecting words into low dimensional space (via learning and gradient descent), similar words are automatically clustered together. The curse of dimensionality is naturally addressed because of the efficient parameterization in neural networks. 

In feed-forward network model [\autoref{ch:neural-network-and-deep-learning:ApplicationsNLP:fig:feedforwardmodelv2}], each word, together with its preceding $N - 1$ words as context are projected into low-dimensional space and further predict the next word probability. Note that the context has a fixed length of $n$, which it is limited in the same way as in $n$-gram models.

Formally, the model is optimized to maximize

$${P}\left(w_{t} \mid w_{t-1}, \cdots w_{t-n+1}\right)=\frac{e^{y_{w_{t}}}}{\sum_{i} e^{y_{i}}}.$$

The $y_{i}$ is the logic for word $i$, given by

$$
y=b+We+U \tanh(d+He)
$$

where $W, H$ are matrices and $b, d$ are biases. Here $W$ can optionally zero, meaning no direct connections. $e = (e_1,...,e_{n-1})$ is the concatenations of word embeddings of each preceding token. 

Feed-forward neural language model brings several improvements over the traditional $n$-gram language model: it provides a compact and efficient parameterization to capture word dependencies among text data. Recall that $n$-gram model would have to store all observed $n$-grams. However, feed-forward neural language model still meet challenges to capture long-distance dependencies in natural language. In the feed-forward neural language model, capturing long-distance dependencies will require an increase of the context window size, which linearly scales with model parameters $W$. Another drawback is that a word that appears at different locations will be multiplied by different weights to get its embedding, which is inconsistent. 

\begin{figure}[H]
	\centering
	\includegraphics[width=0.7\linewidth]{../figures/deepLearning/ApplicationsNLP/languageModeling/FeedForwardModel_v2}
	\caption{Feedforward neural netowk based language model.}
	\label{ch:neural-network-and-deep-learning:ApplicationsNLP:fig:feedforwardmodelv2}
\end{figure}


## Recurrent neural language model

In recurrent network model [\autoref{ch:neural-network-and-deep-learning:ApplicationsNLP:fig:recurrentmodelv2}], context is extended via recurrent connections and context length is theoretically unlimited.
Specially, let the recurrent network input be $x$, hidden layer output be $h$  and output probabilities be $y$. Input vector $x_t$ is a concatenation of a word vector $w_t$ and the previous hidden layer output $s_{t-1}$, which represents the context. To summarize, we have recurrent computation given by

$$
\begin{align}
	x_t &=\operatorname{Concat}(e_t,h_{t-1}) \\
	h_t &=\operatorname{Sigmoid}\left(W_xx_t + b_x\right) \\
\end{align}
$$

The prediction probability at each $t$ is given by

$$	p(y_{k}(t) =\operatorname{Softmax}\left(W_ys(t) + b_y\right).$$




\begin{figure}[H]
	\centering
	\includegraphics[width=0.7\linewidth]{../figures/deepLearning/ApplicationsNLP/languageModeling/RecurrentModel_v2}
	\caption{Recurrent neural network based language model.}
	\label{ch:neural-network-and-deep-learning:ApplicationsNLP:fig:recurrentmodelv2}
\end{figure}


Compared with $n$-gram language model and feed-forward neural language model, RNN language model can in principle process input of any length without increasing the model size. 
RNN language model also has several drawbacks: Computing a conditional probability $p(w_t|w_{1:t-1})$ is expensive. One mitigating strategy is to cache and re-use previous computed results or to pre-compute conditional probabilities for frequent $n$-grams.  some In practical applications, RNN language models from different domains are not easy to merge. 
