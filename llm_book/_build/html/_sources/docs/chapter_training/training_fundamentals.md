# LLM Training Fundamentals
## Training Overview

The chapter discuss LLM training, which can be broadly divided into two stages, each with its own purpose and methodologies:
1. **Pretraining**: This initial phase involves using self-supervised learning (e.g., autoregressive learning objective) let the model to learn language structures of a diverse and extensive corpus of text data. The model learns to predict words or tokens based on context, developing a broad understanding of language structure and semantics. We'll discuss the basics of pretraining and explore the concept of continuing pretraining, which allows models to adapt to new domains or languages.

2. **Post-training**: After the initial pretraining, models often undergo additional training phases to enhance their performance on specific tasks or to better follow instructions. This includes:
   - **Finetuning**: Adapting the pretrained model to specific tasks or domains. Particularly, **instruction finetuning** involves teaching the model to follow explicit instructions or prompts by training on large-scale and diverse (instruction, response) pair data. We cover this in {ref}`chapter_training_sec_LLM_finetuning`.
   - **Alignment and Preference Learning**: Ensuring the model's outputs align with human values and preferences. 
we cover this direction in {ref}`chapter_training_sec_LLM_alignment`.

Finally, we cover fundamentals in  **LLM optimization algorithms**. Throughout the LLM training process, various optimization algorithms are employed to adjust the model's parameters efficiently. We'll examine popular techniques such as stochastic gradient descent (SGD), Adam, and their variants, discussing how they contribute to the model's learning process.




## Pretraining Techniques

### Next-Token Prediction
Pretraining has become a cornerstone in the development of LLM, which contributes to
* the general langugae understand and generation ability
* acquire world knowledge
* other emergent abilities like reasoning

The dominant LLM pretraining objective is auto-regressive language modeling, which predict the next words given preceding word sequence. Given an input sequence $\mathbf{x} = (x_1,...,x_T)$, auto-regressive language modeling minimize the negative log likelihood given by

$$L = - \sum_{t=1}^{T} \log p\left(x_{t} \mid \mathbf{x}_{t-k-1:t-1},\theta\right)$$

where $p\left(x_{t} \mid \mathbf{x}_{t-k-1:t-1}\right)$ is the predicted probability distribution for token $x_t$ given preceding token sequence $\mathbf{x}_{t-k-1:t-1}$ with a context window size $k$ ($k$ can range from hundreds to tens of thousands, depending on the model configuration).

There are scaling laws {cite:p}`kaplan2020scaling,henighan2020scaling` discovered on LLM pretraining, which establishes mathematical relationships model performance given model size, dataset size, and the amount of compute. The availability of scaling laws has several benefits:
* It provides to benchmark to enable LLM pretraining to be done in a predictable way.
* It help design better training strategy by optimizing the model size and data size under compute constraint.

### Fill-in-the Middle

### Multiple Token Prediction

Besides training using a next-token prediction loss, there are efforts exploring training language models to predict multiple future tokens at once [{numref}`chapter_training_fig_fundamentals_multiple_token_prediction_demo`].  More specifically, at each position in the training corpus, we ask the model to predict the following $n$ tokens using $n$ independent output heads, operating on top of a shared model trunk.

The advantages are:
* in higher sample efficiency.
* 

{cite:p}`gloeckle2024better`
During inference, we employ only the next-token output head. Optionally, the other three heads may be used to speed-up
inference time.

```{figure} ../img/chapter_training/training_fundamentals/multiple_token_prediction/mtp_demo.png
---
scale: 50%
name: chapter_training_fig_fundamentals_multiple_token_prediction_demo
---
Overview of multi-token prediction. During training, the model predicts 4 future tokens at once, by
means of a shared trunk and 4 dedicated output heads. Image from {cite:p}`gloeckle2024better`.
```

In this work, we generalize the above by implementing a multi-token prediction task, where at each position of the training corpus, the model is instructed to predict $n$ future tokens at once. This translates into the cross-entropy loss

$$
L_n=-\sum_t \log P_\theta\left(x_{t+n: t+1} \mid x_{t: 1}\right)
$$


To make matters tractable, we assume that our large language model $P_\theta$ employs a shared trunk to produce a latent representation $z_{t: 1}$ of the observed context $x_{t: 1}$, then fed into $n$ independent heads to predict in parallel each of the $n$ future tokens (see Figure 1). This leads to the following factorization of the multi-token prediction cross-entropy loss:

$$
\begin{aligned}
L_n & =-\sum_t \log P_\theta\left(x_{t+n: t+1} \mid z_{t: 1}\right) \cdot P_\theta\left(z_{t: 1} \mid x_{t: 1}\right) \\
& =-\sum_t \sum_{i=1}^n \log P_\theta\left(x_{t+i} \mid z_{t: 1}\right) \cdot P_\theta\left(z_{t: 1} \mid x_{t: 1}\right)
\end{aligned}
$$


In practice, our architecture consists of a shared transformer trunk $f_s$ producing the hidden representation $z_{t: 1}$ from the observed context $x_{t: 1}, n$ independent output heads implemented in terms of transformer layers $f_{h_i}$, and a shared unembedding matrix $f_u$. Therefore, to predict $n$ future tokens, we compute:

$$
P_\theta\left(x_{t+i} \mid x_{t: 1}\right)=\operatorname{softmax}\left(f_u\left(f_{h_i}\left(f_s\left(x_{t: 1}\right)\right)\right)\right)
$$

for $i=1, \ldots n$, where, in particular, $P_\theta\left(x_{t+1} \mid x_{t: 1}\right)$ is our next-token prediction head. See Appendix B for other variations of multi-token prediction architectures.

<!-- ## Comparison

| Approach | Training set | Training set size | Implementation <br> Complexity | Total training cost (inc. experimentation) |
| :---: | :---: | :---: | :---: | :---: |
| Prompt engineering | Not needed | 0 | Low | 0 (no training) |
| RAG | Not needed | 0 | Low - Medium | 0 (no training) |
| Supervised-Fine-tuning | labelled | Can be as little as few hundreds examples (e.g. with PEFT approaches) but can increase to several thousands depending on number of tasks | Medium - High <br> depending on use case | $ $100-5 \mathrm{~K}$ |
| Continuous pre-training | unstructured | Can vary - from 10 K <br> tokens to Bitlions | Medium on Bedrock and Jumpstart, Higher with SageMaker Training | $-\$ 2500$ for scanning 18 <br> tokens for a 7B model |
| Full Pretraining | unstructured | 100s of billion/trillion tokens (e.g. 700 billion tokens for BloombergGPT for 50B model) | Very High | $$500K | -->

### Continued Pretraining

Continued pretraining of LLM involves updating pre-trained models with new data (usually in large scale) instead of re-training them from scratch. 
addresses a fundamental challenge in the application of large language models (LLMs): the mismatch between the general knowledge acquired during initial pretraining and the specific knowledge required for domain-specific tasks.
While pretrained LLMs demonstrate impressive general language understanding, they may lack the nuanced knowledge and vocabulary necessary for specialized domains such as medicine, law, or specific scientific fields. Continued pretraining aims to bridge this gap by further training the model on domain-specific corpora, allowing it to adapt its learned representations and knowledge to better suit the target domain or task.
It improve LLM's performance in the target domain by enhance language understanding and acquiring domain knowledge in the target domain.

There are also cost associated with continued pretraining, including
* **Catastrophic forgetting**: The model may degrade its general language understanding when it is heavily continued pretrained on the domain data.
* **Computational cost**: Although more efficient than full pretraining, continued pretraining can still be computationally expensive for very large models.
* **Data requirements**: High-quality, domain-specific data is crucial for effective continued pretraining.


One example of continued pretraining is the **Linly-Chinese-LLaMA-2** project (https://github.com/CVI-SZU/Linly). The motivation behind this project is to improve the cross-lingual capability, particularly in Chinese, of many open Large Language Models (LLMs) such as Llama and Falcon. These models were initially pretrained on text data that is predominantly in English.

Key technical details on the continued pretraining:

**Training data composition**: The continued pretraining used hundreds of millions of high-quality public Chinese text data, including news, community Q&A, encyclopedias, literature, and scientific publications. Besides, the project incorporated 1) a large amount of Chinese-English parallel corpora in the early stages of training to help the model quickly transfer English language capabilities to Chinese and 2) English text corpus like SlimPajama and RefinedWeb to prevent the model from forgetting previously acquired knowledge. 

**Training data schedule**: A curriculum learning strategy was employed. In the early stages of training, more English language materials and parallel corpora were used. As the number of training steps increased, the proportion of Chinese data was gradually increased. This helps the convergence of the model training.



## Pretaining Data Sources and Cleaning

The quality and diversity of training data significantly impact the performance of pretrained models. Common sources include:

1. Web Crawls: Web are avilable in large scale and serve as the primary data source to provide diverse, multilingual data, but web data usually require extensive filtering and cleaning. Example data source include CommonCrawl, C4 (The Colossal Clean Crawled Corpus), RedPajama-Data, RefinedWeb, WebText, etc.
2. Books and Literature: Projects like BookCorpus, the Gutenberg Project, arXiv offer high-quality, long-form text.This is an important source for LLM to learn world knowledge and liguistic information.
3. Wikipedia: A reliable source of factual information across many languages and domains.
4. Social Media and Forums: Platforms like Reddit or X (twitter) provide more informal, conversational language.
5. Code: Github code (as used in Codex {cite:p}`chen2021evaluating`) and code-related question-answering
platforms (e.g., StackOverflow).
6. Domain specific Corpora: Domain-specific datasets (e.g., scientific papers, legal documents) for targeted pretraining.


The following {numref}`chapter_training_fig_fundamentals_pretrain_data_distribution` summarize the data source and ratio for existing LLM pretraining.
   
```{figure} ../img/chapter_training/training_fundamentals/pretrain_data/training_data_distribution_summary.png
---
scale: 50%
name: chapter_training_fig_fundamentals_pretrain_data_distribution
---
Pretrain data source distribution for existing LLMs. Image from {cite:p}`zhao2023survey`.
```

While the scale is one factor impacting resulting model performance (i.e., the scaling law), the quality of data and the ratio of different data types play an equally important role. As dominant pretraining data is from the web, data clearning and quality control is a crucial step for sucessful LLM pretraining [{numref}`chapter_training_fig_fundamentals_pretrain_data_distribution`]. 
The following {numref}`chapter_training_fig_fundamentals_pretrain_data_clean_pipeline` summarize the key steps on cleaning training data.. 

```{figure} ../img/chapter_training/training_fundamentals/pretrain_data/pretraining_data_cleaning_pipeline.png
---
scale: 60%
name: chapter_training_fig_fundamentals_pretrain_data_clean_pipeline
---
Illustration of data cleaning pipeline for curating LLM pretraining data. Image from {cite:p}`zhao2023survey`.
```

Onogoing challenges for constructing LLM pretraining data include:
* Data quality and bias: Ensuring data quality and mitigating biases present in web-scraped data is an ongoing challenge.
* Multilingual representation: Balancing representation across languages, especially for low-resource languages, remains difficult.

### Data mixture and schedule

With cleaned data from different data sources, it is essential to design data feeding strategies to pretrain LLM with target capabilities. Two important aspects of data feeding strategy are 
* the portition of different data sources
* the order of each data source used in pretraining






## Optimization Algorithms

### Minibatch Stochastic Gradient Descent


The classical gradient descent algorithm requires the evaluation of the gradient over the whole set of training data. This is both computational prohibitive and sample inefficient - many samples are similar, making the gradient of the whole data sample is simply the multiplier of the gradient of a much smaller, representative sample data set.
**Minibatch stochastic gradient descent** is much efficient way of gradient descent, which uses a random sample of the training data set to estimate the gradient on each step. 

A typical algorithm is showed as follows.

```{prf:algorithm} Minibatch stochastic gradient descent algorithm
:label: Minibatch_stochastic_gradient_descent_algorithm

**Inputs** Learning rate $\alpha_k$, iniital model parameter $\theta$.

**Output** $\theta_k$

1. Set $k=1$ 
2. Repeat until stopping criteria is met:
    1. Sample a minibatch of training samples of size $m$: $(x^{(i)},y^{(i)}),i=1,2,...,m$.
    2. Compute a gradient estimate over this minibatch samples via

    $$\hat{g}_k = \frac{1}{m}\nabla_{\theta} \sum_{i=1}^{m} L(f(x^{(i)};\theta),y^{(i)}).$$

    3. Apply update $\theta_k = \theta_k - \alpha_k \hat{g}_k$.
    
    4. Set $k=k+1$.
```


```{prf:remark} choice of minibatch size
* The estimation quality of gradient via minibatch gradient descent is strongly affected by the minibatch size. In general, the gradient estimate is unbiased, irrespective of the choice of minibatch size, but its variance will decrease as the minibatch increases.
* For larger minibatch size, we can increase learning rate since the estimated gradient is more certain. There is an empirical ***Linear Scaling Rule***: When the minibatch size is multiplied by $k$, multiply the learning rate by $k$ {cite:p}`goyal2017accurate`.
```

### Adaptive Gradient Method
#### Adaptive Gradient (AdaGrad)

For simple stochastic gradient methods, we need to set the learning rate hyperparameter or even dynamically schedule learning rate, which is usually a difficult task or problem specific. Further, a uniform learning rate is usually not an effective way for high-dimensional gradient descent methods, since one learning rate could be too large for one-dimension but, on the contrary, too small for another dimension. 

The AdaGrad algorithm{cite:p}`duchi2011adaptive` addresses the issue by choosing different learning rates for each dimension. The algorithm adaptively scales the learning rate for each dimension based on the **accumulated gradient magnitude** on that dimension so far.

Let $G_k$ be the **accumulated gradient** up to iteration $k$, given by

$$G_k = G_{k-1} + \hat{g}_k\odot \hat{g}_k, G_0 = 0.$$

The parameter update is given by 

$$\theta_k = \theta_{k-1} - \frac{\alpha_0}{\delta + \sqrt{G_k}} \hat{g}_k$$

where $\alpha_0$ is the initial learning speed, usually set at a small number (say, 1e-9 to 1e-7) and $\delta$ is a small positive constant to avoid division by zero.

As we can see, the learning rate in AdaGrad is monotonically decreasing, which may dramatically slow down the convergence as the learning rate becomes too small. In general, AdaGrad algorithm performs best for **convex optimization** (however, neural network optimization is usually non-convex).

#### RMSProp

As we mentioned before, AdaGrad tend to shrink learning rate too aggressively. This is an advantage when applying AdaGrad to convex function optimization as it enables the algorithm to converge fast and stably. However, non-convex function optimization usually require large, adaptive learning rate to escape bad local minimum and converge stably to better local minimums. 

The first remedy is to prevent the learning rate from shrinking too fast. Let $G_k$ be the accumulated gradient up to iteration $k$, given by

$$G_k = \rho G_{k-1} + (1- \rho) g_k\odot g_k, G_0 = 0.$$

Then we compute update 

$$\theta_k= \theta_{k-1} - \frac{\alpha_0}{\delta + \sqrt{G_k}} \hat{g}_k.$$

which is the core part of the RMSProp algorithm {cite:p}`Hinton2012Neural`.

How this modification can make the $G_k$ smaller than that in the AdaGrad, as can be seen from following expansion. 

```{prf:remark} Expansion of $G_k$
In RMSProp, we have

$$
\begin{align}
G_k &= \rho G_{k-1} + (1- \rho) g_k\odot g_k \\
	&= \rho G_{k-2} + \rho(1- \rho) g_k\odot g_k + \rho(1 - \rho)g_{k-1}\odot g_{k-1} \\
	&\approx g\cdot g (1 - \rho) (1 + \rho + \cdot + \rho^{k-1})\\
	&= (1 - \rho^k) g\cdot g
\end{align}
$$

where we assume $g_k\cdot g_k \approx g\cdot g, \forall k$. Clearly, we have
roughly, 

$$G_k^{RMSProp} \approx  \frac{(1 - \rho^k)}{k} G_k^{AdaGrad}, G_k^{AdaGrad}\approx k g\odot g.$$

```

```{prf:remark} Importance of adaptive learning rate
One example to demonstrate the importance of having adaptive learing rate is learning word embeddings. Embeddings of rare words only get limited chances to update because they have limited presence in the training data. On the other hand, embeddings of common words get update frequently. With adaptive learning rate, embeddings of rare words will have large learning rate whenever it gets update. This help the model learn better embeddings for rare words. 
```


### Momentum Method

Simple SGD with small learning rate can lead to extremely slow learning for functional surfaces with long, narrow valleys {cite:p}`sutskever2013importance`. One intuition inspired by the physics of a heavy ball falling down is to add momentum to the gradient descent steps. Mathematically, adding momentum is equivalent to adding historical weighted averaged gradient to the current gradient. The total gradient will then be hopefully large enough to enable fast movement on relatively flat regions.  


```{figure} ../img/chapter_training/training_fundamentals/SGDMomentum.jpg
---
scale: 30%
name: chapter_training_fig_optimizer_SGD_momentum
---
SGD without momentum and with momentum. SGD with momentum can accumulate gradient/velocity in horizontal direction and move faster towards the minimum located at the center.
```

Consider the gradient 

$$\hat{g}_k = \frac{1}{m}\nabla_{\theta} \sum_{i=1}^{m} L(f(x^{(i);\theta}),y^{(i)}),$$

we can compute a **speed** (an intermedidate parameter) via 

$$v_k = \mu v_{k-1} - \alpha_k \hat{g}_k,$$

where $\mu \in [0, 1]$ is the momentum coefficiency, and $\alpha_k$ is the learning rate. 

The speed (with momentum considered) is then used to update parameter $\theta$ via 

$$\theta_k = \theta_k + v_k.$$

To see that the update velocity is the weighted average gradient, we now show that the velocity is an exponentially decaying moving average (similar to AR(1) process) of the negative gradients, given by

$$
\begin{align}
v_k &= \alpha v_{k-1} - \hat{g}_k\\
&= \alpha (\alpha v_{k-2} - \hat{g}_{k-1}) - \hat{g}_k \\
&= \alpha^2 v_{k-2} - \alpha \hat{g}_{k-1} - \hat{g}_k \\
&= \cdots \\
&= \sum_{i=0}^\infty  \alpha^{i} \hat{g}_{k-i}.
\end{align}	
$$


### Combined Together: Adam and AdamW

#### Adam
By combining the ideas of momentum and adaptive learning rate, we yield Adam, one of most popular gradient descent algorithm in deep learning community{cite:p}`kingma2014adam`. The name Adam is derived from adaptive moment estimation. As its name suggests, Adam will compute velocity via momentum type of averaging and adjust the learning rate using inverse of accumulated gradients. 

Specifically, the velocity is computed via

$$
\begin{align}
M_k &= \rho_1 M_{k-1} + (1-\rho_1)\hat{g}_k \\
\tilde{M}_k &= \frac{M_k}{1-\rho_1^k}
\end{align}
$$

and the accumulated gradient magnitude is compute via

$$
\begin{align}
G_k &= \rho_2 G_{k-1} + (1-\rho_2)\hat{g}_k \odot \hat{g}_k \\
\tilde{G}_k &= \frac{G_k}{1-\rho_2^k}
\end{align}
$$

Note that we correct the $M_k$ and $G_k$ be dividing the factor $1 - \rho_i^k, i= 1, 2$ to get the average estimation. 

The final algorithm is given by the following.


```{prf:algorithm} Adam stochastic gradient descent algorithm
:label: Adam_stochastic_gradient_descent_algorithm
**Inputs** Learning rate $\alpha$(set to 0.001), iniital model parameter $\theta$, decay parameters $\rho_1$(set to 0.9), $\rho_2$(set to 0.999). $\delta = 1e-8$

**Output** $\theta_k$

1. Set $k=1$. 
2. Set $M_k = 0, G_k = 0$.
3. Repeat until stopping criteria is met：
    1. Sample a minibatch of training samples of size $m$ $(x^{(i)},y^{(i)}),i=1,2,...,m$.\\
    2. compute gradient estimate over minibatch $N$ samples via

		$$\hat{g}_k = \frac{1}{m}\nabla_{\theta} \sum_{i=1}^{m} L(f(x^{(i);\theta}),y^{(i)}).$$ 
		
    3. Accumulate $M_k = \rho_1 M_{k-1} + (1-\rho_1)\hat{g}_k$. Accumulate $G_k = \rho_2 G_{k-1} + (1-\rho_2)\hat{g}_k \odot \hat{g}_k$.
	
    4. Correct biases

		$$\tilde{M}_k = \frac{M_k}{1-\rho_1^k}, \tilde{G}_k = \frac{G_k}{1-\rho_2^k}.$$
	
    5. Apply update $$\theta_k = \theta_{k-1} -\frac{\alpha \cdot \tilde{M}_k}{\delta + \sqrt{\tilde{G}_k }}.$$
	6. Set $k=k+1$.
	
```

#### $L_2$ Weight Decay and AdamW

$L_2$ regularization on model parameters often reduce model overfitting and improves the generalization ability of the model. In the SGD optimization framework, the implementation of $L_2$ regularization term is often realized via **weight decay**, resulting in an additional term in the gradient that penalize large weights. That is,

$${g}_{k} \leftarrow \nabla f_{k}\left({\theta}_{k-1}\right)+\lambda {\theta}_{k-1},$$

where $\lambda$ is the decay parameter, corresponding the strength of the regularization.

AdamW {cite:p}`loshchilov2017decoupled` is the algorithm that correctly implements Adam with $L_2$ regularization, which is also called Adam with decoupled weight decay.

The algorithm is given by the following.


```{prf:algorithm} Adam stochastic gradient descent algorithm with weight decay
:label: Adam_stochastic_gradient_descent_algorithm_with_weight_decay
**Inputs** Learning rate $\alpha$(set to 0.001), iniital model parameter $\theta$, decay parameters $\rho_1$(set to 0.9), $\rho_2$(set to 0.999). $\delta = 1e-8$.  <span style="background-color: #e4ac94">Weight decay parameter $\lambda \in \mathbb{R}$</span>.

**Output** $\theta_k$

1. Set $k=1$. 
2. Set $M_k = 0, G_k = 0$.
3. Repeat until stopping criteria is met：
    1. Sample a minibatch of training samples of size $m$ $(x^{(i)},y^{(i)}),i=1,2,...,m$.\\
    2. compute gradient estimate over minibatch $N$ samples via

		$$\hat{g}_k = \frac{1}{m}\nabla_{\theta} \sum_{i=1}^{m} L(f(x^{(i);\theta}),y^{(i)}).$$ 
		
    3. Accumulate $M_k = \rho_1 M_{k-1} + (1-\rho_1)\hat{g}_k$. Accumulate $G_k = \rho_2 G_{k-1} + (1-\rho_2)\hat{g}_k \odot \hat{g}_k$.
	
    4. Correct biases

		$$\tilde{M}_k = \frac{M_k}{1-\rho_1^k}, \tilde{G}_k = \frac{G_k}{1-\rho_2^k}.$$
	
    5. <span style="background-color: #e4ac94"> Apply update
    
    $$\theta_k = \theta_{k-1} -\frac{\alpha \cdot \tilde{M}_k}{\delta + \sqrt{\tilde{G}_k }} - \lambda \theta_{k}.$$
    
    </span>
	6. Set $k=k+1$.
	
```


## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
