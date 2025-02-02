(chapter_text_embedding_sec_text_embedding)=
# Text Embedding


## Introduction

There are many NLP tasks involves determining the relationship of two sentences, including semantic similarity, semantic relation reasoning, questioning answering etc. For example, Quora needs to determine if a question asked by a user has a semantically similar duplicate. The GLUE benchmark as an example [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP:sec:BERTDownstreamTasks}], 6 of them are tasks that require learning sentences Inter-relationship. Specifically,

**MRPC**: The Microsoft Research Paraphrase Corpus {cite:p}`dolan2005automatically` is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.\\

**QQP**: The [Quora Question Pairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) dataset is a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent. As in MRPC, the class distribution in $\mathrm{QQP}$ is unbalanced $(63 \%$ negative), so we report both accuracy and F1 score. We use the standard test set, for which we obtained private labels from the authors. We observe that the test set has a different label distribution than the training set. \\

**STS-B**: The Semantic Textual Similarity Benchmark {cite:p}`cer2017semeval` is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data.

**Natural language inference**: {cite:p}`bowman2015large` Understanding entailment and contradiction is fundamental to understanding natural language, and inference about entailment and contradiction is a valuable testing ground for the development of semantic representations. The semantic concepts of entailment and contradiction are central to all aspects of natural language meaning,
from the lexicon to the content of entire texts.
Natural language inference is the task of determining whether a *hypothesis* is true (entailment), false (contradiction), or undetermined (neutral) given a *premise*.


```{table} *Entailment*, *contradiction*, and *neural* examples in a natural language inference task.
| Premise | Label | Hypothesis |
| :---: | :---: | :---: |
| A man inspects the uniform of a figure in some East Asian country. | Contraction | The man is sleeping. |
| An older and younger man smiling. | Neutral | Two men are smiling and laughing at the cats playing on the floor. |
| A soccer game with multiple males playing. | Entailment | Some men are playing a sport. | 
```
Although BERT model and its variant have achieved new state-of-the-art among many sentence-pair classification and regression tasks. It has many practical challenges in tasks like large-scale semantic similarity comparison, clustering, and information retrieval via semantic search, etc. These tasks require that both sentences are fed into the network, which causes a massive computational overhead for large BERT model. Considering the task of finding the most similar pair among $N$ sentences, then it requires $N^2/2$ forward pass computation of BERT. 

An alternative approach is to derive a semantically meaningful sentence embeddings for each sentence. A sentence embedding is a dense vector representation of a sentence. Sentences with similar semantic meanings are close and sentences with different meanings are apart. With sentence embeddings, similarity search can be realized simply via a distance or similarity metrics, such as cosine similarity. Besides performing similarity comparison between two sentences, the sentence embedding can also be used as generic sentence feature for different NLP tasks.

Early on, there have been efforts directed to derive sentence embedding by aggregating word embeddings. For example, one can average the BERT token embedding output as the sentence embedding. Another common practice is to use the output of the first token (the [CLS] token) as the sentence, which is found to be worse than averaging GloVe embeddings\cite{reimers2019sentence}. Recently, contrastive learning and Siamese type of learning have been successfully applied in deriving sentence embeddings, which will be the focus of our following sections. 

```{prf:remark} Why we need sentence embedding fine tuning
Pre-trained BERT models do not produce efficient and independent sentence embeddings as they always need to be fine-tuned in an end-to-end supervised setting. This is because we can think of a pre-trained BERT model as an indivisible whole and semantics is spread across all layers, not just the final layer. Without fine-tuning, it may be ineffective to use its internal representations independently. It is also hard to handle unsupervised tasks such 
as clustering, topic modeling, information retrieval, or semantic search. Because we have to evaluate many sentence pairs during clustering tasks, for instance, this causes massive computational overhead.
```

## Contextual Text Embedding Fundamentals

### Anisotropy

{cite:p}`ethayarajh2019contextual`

Contextualized representations are **anisotropic** in all non-input layers. If word representations from a particular layer were isotropic (i.e., directionally uniform), then the average cosine similarity between uniformly randomly sampled words would be 0. The closer this average is to 1, the more anisotropic the representations. The geometric interpretation of anisotropy is that the word representations all occupy a narrow cone in the vector space rather than being uniform in all directions; the greater the anisotropy, the narrower this cone. 

As seen in {numref}`chapter_text_embedding_fig_text_embedding_anisotropy`, this implies that in almost all layers of BERT and GPT-2, the representations of all words occupy a narrow cone in the vector space. 

Contextual word embeddings are generally more anisotropic in higher layers.


anisotropy is also viewed as representation degeneration problem {cite:p}`gao2019representation`, where the expressiveness of the embedding model is been compromised. 


```{figure} ../img/chapter_text_embedding/contextual_embedding_fundamentals/self_similarity_across_layers.png
---
scale: 80%
name: chapter_text_embedding_fig_text_embedding_anisotropy
---
In almost all layers of BERT and GPT-2, the word representations are anisotropic (i.e., not directionally uniform): the average cosine similarity between uniformly randomly sampled words is non-zero.
```

### Context-specificity

For example, the word *dog* appears in *A panda dog is running on the road* and *A dog is trying to get bacon off his back*. 
If a model generated the same representation for *dog* in both these sentences, we could infer that there was no contextualization; conversely, if the two representations were different, we could infer that they were contextualized to some extent. 

It is found that:
* Contextualized word embeddings are more context-specific in higher layers. Particularly, representations in GPT-2 are the most context-specific, with those in GPT-2’s last layer being almost maximally context-specific.

In image classification models, lower layers recognize more generic features such as edges while upper layers recognize more class-specific features. 

Therefore, it follows that upper layers of neural language models learn more context-specific representations, so
as to predict the next word for a given context
more accurately.

* Stopwords (e.g., *the*, *of*, *to*) have among the most context-specific representations, meaning that their representation are highly dependent on its context.



### Static vs Contextual Word embeddings

A word's embedding changes when the word resides in different contexts. It is found that, on average, less than 5% of such variance can be explained by the first principal component, which can be viewed as a static embedding. This also indicates that contextually word embedding cannot be easily replaced by a static embedding. 

However, we can still use principal components of contextualized representations in **lower layers** as proxy static word embeddings, which can be used in low-resource scenarios. It is found {cite:p}`ethayarajh2019contextual` that such proxy static embedding can still outperform GloVe and FastText on many benchmarks. 

We can create static embeddings for each word by taking the first principal component (PC) of its contextualized representations in a given layer.

Given that upper layers are much more contextspecific than lower layers, and given that GPT2’s representations are more context-specific than ELMo and BERT’s (see Figure 2), this suggests that the PCs of highly context-specific representations are less effective on traditional benchmarks.

{cite:p}`wang2019improving`


## Contrastive Learning Fundamentals

(chapter_text_embedding_sec_text_embedding_contrastive_learning_coSENTLoss)=
### coSENTLoss

Following the prior study (Su, 2022), we employ the cosine objective function for end-to-end optimization of cosine similarity between representations, as follows:

$$
\mathcal{L}_{\text {cos }}=\log \left[1+\sum_{s\left(\mathbf{X}_i, \mathbf{X}_j\right)>s\left(\mathbf{X}_m, \mathbf{X}_n\right)} \exp{\left(\cos \left(\mathbf{X}_m, \mathbf{X}_n\right)-\cos \left(\mathbf{X}_i, \mathbf{x}_j\right)\right)/\tau}\right]
$$

where $\tau$ is a temperature hyperparameter, $\cos (\cdot)$ is the cosine similarity function, and $s(u, v)$ is the similarity between $u$ and $v$. By optimizing the $\mathcal{L}_{\text {cos }}$, we expect the cosine similarity of the high similarity pair to be greater than that of the low similarity pair.

### Embedding quality intrinsic measurement

Embedding quality, often as a result of contrastive learning, can be intrinsically measured using two properties: **Alignment** and **uniformity** {cite:p}`wang2020understanding`. Given a distribution of positive pairs $p_{\text {pos }}$, alignment calculates expected distance between embeddings of the paired instances (assuming representations are already normalized):

$$
\ell_{\mathrm{align}} \triangleq \underset{\left(x, x^{+}\right) \sim p_{\mathrm{pos}}}{\mathbb{E}}\left\|f(x)-f\left(x^{+}\right)\right\|^2
$$


On the other hand, uniformity measures how well the embeddings are uniformly distributed:

$$
\ell_{\text {uniform }} \triangleq \log \underset{\substack{\text { i.i.d. } \\ x, y \sim p_{\text {data }}}}{\mathbb{E}} \quad e^{-2\|f(x)-f(y)\|^2}
$$

where $p_{\text {data }}$ denotes the data distribution. These two metrics are well aligned with the objective of contrastive learning: positive instances should stay close and embeddings for random instances should scatter on the hypersphere. In the following sections, we will also use the two metrics to justify the inner workings of our approaches.

### Anisotropy problem in embeddings

{cite:p}`ethayarajh2019contextual`

Recent work identifies an anisotropy problem in
language representations (Ethayarajh, 2019; Li
et al., 2020), i.e., the learned embeddings occupy a
narrow cone in the vector space, which severely
limits their expressiveness.

Wang et al. (2020) show that singular values
of the word embedding matrix in a language model
decay drastically: except for a few dominating singular values, all others are close to zero.

The anisotropy problem is naturally connected to
uniformity (Wang and Isola, 2020), both highlighting that embeddings should be evenly distributed
in the space. Intuitively, optimizing the contrastive
learning objective can improve uniformity (or ease
the anisotropy problem), as the objective pushes
negative instances apart.


## BERT based embeddings
### Sentence-BERT

While a pretrained BERT has provided contextulized embeddings for each token, it is found that high-quality, semantically meaning sentence embeddings can not be directly derived from token embedding. 

Sentence-BERT {cite:p}`reimers2019sentence` enables sentence embedding extraction from the BERT model after fine-tuning the BERT model via Siamese type of learning{numref}`chapter_text_embedding_fig_bert_sentencebert`. Specifically, three strategies can be used to derived sentence embedding
-  the output of the CLS-token
-  the mean of all output vectors (**default**)
-  max-over-time of the output vectors 


```{figure} ../img/chapter_text_embedding/classical_bert_embeddings/sentenceEmbedding/sentenceBERT.png
---
scale: 30%
name: chapter_text_embedding_fig_bert_sentencebert
---
(**left**) Sentence-BERT architecture with classification objective function for training. The two BERT networks are pre-trained have tied weights (Siamese network structure). (**right**) Sentence-BERT architecture at inference. The similarity between two input sentences can be computed as the similarity score between two sentence embeddings.
```

```{table}
| Model | Spearman |
| :--- | :---: |
| Avg. GloVe embeddings | 58.02 |
| Avg. BERT embeddings | 46.35 |
| SBERT-NLI-base | 77.03 |
|SBERT-NLI-base + STSb FT| 85.35|
```


```{prf:remark} Training time and inference time inconsistency

In the inference time, the pooled embeddings from the last layer are directly used in cosine similarity scoring. In the training time, the pooled embeddings plus its difference vector are passed through an MLP network first and then make prediction. The pooled embeddings are not optimized directly towards cosine similarity scoring. This issue is discussed in coSENTLoss 

```

<!-- ### Universal Sentence Encoder

\cite{cer2018universal}

### Contrastive learning

#### Contrastive learning via data augmentation

\cite{shen2020simple}

```{figure} images/../deepLearning/ApplicationsNLP/sentence_representation/data_augmented_contrastive_learning/data_cut_off_demo
:name: fig:datacutoffdemo
Figure 1: Schematic illustration of the proposed cutoff augmentation strategies, including token cutoff, feature
		cutoff and span cutoff, respectively. Blue area indicates that the corresponding elements within the sentence’s
		input embedding matrix are removed and converted to 0. Notably, this is distinct from Dropout, which randomly
		transforms elements to 0 (without considering any underlying structure of the matrix).
``` -->




### SimCSE

SimCSE {cite:p}`gao2021simcse` introduces a simple Dropout strategy to identify positive examples and conduct unsupervised learning.
As shown in {numref}`chapter_text_embedding_fig_bert_sim_sce_demo`, 
* Positive example pairs can be constructed by passing one sentence passing through the encoder network with different dropout masking. 
* Negative example pairs are coming from in-batch negatives.



```{figure} ../img/chapter_text_embedding/classical_bert_embeddings/SimCSE/sim_sce_demo.png
---
scale: 55%
name: chapter_text_embedding_fig_bert_sim_sce_demo
---
(Left) Unsupervised SimCSE predicts the input sentence itself from in-batch negatives, with different hidden dropout masks applied. (Right) Supervised SimCSE leverages the NLI datasets and takes the entailment (premisehypothesis) pairs as positives, and contradiction pairs as well as other in-batch instances as negatives.
```

Contrastive learning aims to learn effective representation by pulling semantically close neighbors together and pushing apart non-neighbors (Hadsell et al., 2006). It assumes a set of paired examples $\mathcal{D}=\left\{\left(x_{i}, x_{i}^{+}\right)\right\}_{i=1}^{m}$, where $x_{i}$ and $x_{i}^{+}$are semantically related.  with in-batch negatives 

let $\mathbf{h}_{i}$ and $\mathbf{h}_{i}^{+}$denote the representations of $x_{i}$ and $x_{i}^{+}$, the training objective for $\left(x_{i}, x_{i}^{+}\right)$with a mini-batch of $N$ pairs is:

$$
\ell_{i}=-\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{i}^{+}\right) / \tau}}{\sum_{j=1}^{N} e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{+}\right) / \tau}}
$$

where $\tau$ is a temperature hyperparameter and $\operatorname{sim}\left(\mathbf{h}_{1}, \mathbf{h}_{2}\right)$ is the cosine similarity $\frac{\mathbf{h}_{1}^{\top} \mathbf{h}_{2}}{\left\|\mathbf{h}_{1}\right\| \cdot\left\|\mathbf{h}_{2}\right\|}.$ 



**supervised contrastive learning with hard negatives**

Finally, we further take the advantage of the NLI datasets by using its contradiction pairs as hard negatives. In NLI datasets, given one premise, annotators are required to manually write one sentence that is absolutely true (entailment), one that might be true (neutral), and one that is definitely false (contradiction). Therefore, for each premise and its entailment hypothesis, there is an accompanying contradiction hypothesis.

Formally, we extend $\left(x_{i}, x_{i}^{+}\right)$to $\left(x_{i}, x_{i}^{+}, x_{i}^{-}\right)$, where $x_{i}$ is the premise, $x_{i}^{+}$and $x_{i}^{-}$are entailment and contradiction hypotheses. The training objec-
tive $\ell_{i}$ is then defined by $N$ is mini-batch size 

$$-\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{i}^{+}\right) / \tau}}{\sum_{j=1}^{N}\left(e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{+}\right) / \tau}+e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{-}\right) / \tau}\right)}$$


```{table} Effects of different dropout probabilities $p$ (from 0.0 to 0.5) on the STS-B development set (Spearman's correlation). Fixed 0.1: default 0.1 dropout rate but apply the same dropout mask on both $x_i$ and $x_i^{+}$.
| $p$ | 0.0 | 0.01 | 0.05 | 0.1 |
| :--- | :---: | :---: | :---: | :---: |
| STS-B | 71.1 | 72.6 | 81.1 | $\mathbf{8 2 . 5}$ |
| $p$ | 0.15 | 0.2 | 0.5 | Fixed 0.1 |
| STS-B | 81.4 | 80.5 | 71.0 | 43.6 |
```


#### CERT

\cite{fang2020cert}

CERT: Contrastive self-supervised Encoder Representations
from Transformers, which pretrains language representation models using contrastive selfsupervised learning at the sentence level. CERT creates augmentations of original sentences
using back-translation. Then it finetunes a pretrained language encoder (e.g., BERT) by
predicting whether two augmented sentences originate from the same sentence.

CERT takes a pretrained language representation model (e.g., BERT) and finetunes it using
contrastive self-supervised learning on the input data of the target task.

```{figure} images/../deepLearning/ApplicationsNLP/sentence_representation/CERT/CERT_workflow
:name: fig:certworkflow
The workflow of CERT. Given the large-scale input texts (without labels) from
		source tasks, a BERT model is first pretrained on these texts. Then we continue
		to train this pretrained BERT model using CSSL on the input texts (without
		labels) from the target task. We refer to this model as pretrained CERT model.
		Then we finetune the CERT model using the input texts and their associated
		labels in the target task and get the final model that performs the target task.
```

### Benchmark dataset
#### MultiNLI data

{cite:p}`williams2017broad`

The Multi-Genre Natural Language Inference (MultiNLI) corpus <sup>[^3]</sup> is a dataset designed for use in the development and evaluation of machine learning models for sentence understanding. It has over 433,000 examples and is one of the largest datasets available for natural language inference (a.k.a recognizing textual entailment). The dataset is also designed so that existing machine learning models trained on the Stanford NLI corpus can also be evaluated using MultiNLI. 


#### Standard semantic textual similarity tasks

{cite:p}`agirre2012semeval, agirre2013sem, agirre2014semeval, agirre2015semeval, agirre2016semeval, cer2017semeval`


## Matryoshka Representation Learning

new state-of-the-art (text) embedding models started producing embeddings with increasingly higher output dimensions, i.e., every input text is represented using more values. Although this improves performance, it comes at the cost of efficiency of downstream tasks such as search or classification.

{cite:p}`kusupati2022matryoshka`

These Matryoshka embedding models are trained such that these small truncated embeddings would still be useful. In short, Matryoshka embedding models can produce useful embeddings of various dimensions.

Matryoshka embedding models aim to store more important information in earlier dimensions, and less important information in later dimensions. This characteristic of Matryoshka embedding models allows us to truncate the original (large) embedding produced by the model, while still retaining enough of the information to perform well on downstream tasks.

For Matryoshka Embedding models, the training aims to optimize the quality of your embeddings at various different dimensionalities. For example, output dimensionalities are 768, 512, 256, 128, and 64. The loss values for each dimensionality are added together, resulting in a final loss value. The optimizer will then try and adjust the model weights to lower this loss value.

In practice, this incentivizes the model to frontload the most important information at the start of an embedding, such that it will be retained if the embedding is truncated.

### Loss Function

For MRL, we choose $\mathcal{M}=\{8,16, \ldots, 1024,2048\}$ as the nesting dimensions.
Suppose we are given a labelled dataset $\mathcal{D}=\left\{\left(x_1, y_1\right), \ldots,\left(x_N, y_N\right)\right\}$ where $x_i \in \mathcal{X}$ is an input point and $y_i \in[L]$ is the label of $x_i$ for all $i \in[N]$. MRL optimizes the multi-class classification loss for each of the nested dimension $m \in \mathcal{M}$ using standard empirical risk minimization using a separate linear classifier, parameterized by $\mathbf{W}^{(m)} \in \mathbb{R}^{L \times m}$. All the losses are aggregated after scaling with their relative importance $\left(c_m \geq 0\right)_{m \in \mathcal{M}}$ respectively. That is, we solve

$$
\min_{\left\{\mathbf{W}^{(m)}\right\}_{m \in \mathcal{M}}} \frac{1}{N} \sum_{i \in[N]} \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}\left(\mathbf{W}^{(m)} \cdot h_i^{1: m} ; y_i\right)
$$

where $\mathcal{L}: \mathbb{R}^L \times[L] \rightarrow \mathbb{R}_{+}$is the multi-class softmax cross-entropy loss function. This is a standard optimization problem that can be solved using sub-gradient descent methods. We set all the importance scales, $c_m=1$ for all $m \in \mathcal{M}$; see Section 5 for ablations. Lastly, despite only optimizing for $O(\log (d))$ nested dimensions, MRL results in accurate representations, that interpolate, for dimensions that fall between the chosen granularity of the representations (Section 4.2).

## LLM Embedding Model

[Improving text embeddings with large language models](https://arxiv.org/pdf/2401.00368)
[NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models.](https://arxiv.org/pdf/2405.17428)

### NV-Embed

NV-Embed from Nvidia {cite:p}`lee2024nv` proposed several improvement techniques on LLM-based embedding model, which include:
* Model architecture improvement, which introduces a **latent attention layer** to obtain better pooled embeddings.
* Traing process improvement, which introduces a **two-stage contrastive instruction-tuning method**.

There are two popular methods to obtain the embedding for a sequence of tokens: 
* Mean pooling of the all hidden vectors of the last layer, which is commonly used in bidirectional embedding models.
* Last <EOS> token embedding, which is more popular for decoder-only LLM based embedding models. 
 
However, both methods have certain limitations. Mean pooling simply takes the average of token embeddings and may dilute the important information from key phrases, meanwhile the semantics of the last <EOS> token embedding may be dominated by last few tokens.

The latent attention layer aims to improve the **mean pooling method**. Denote the last layer hidden from decoder as the query $Q \in \mathbb{R}^{l \times d}$, where $l$ is the length of sequence, and $d$ is the hidden dimension. They are sent to attend the latent array $K=V \in \mathbb{R}^{r \times d}$, which are **trainable matrices**, used to obtain better representation, where $r$ is the number of latents in the dictionary. The output of this cross-attention is denoted by $O \in \mathbb{R}^{l \times d}$,

$$
O=\operatorname{softmax}\left(Q K^T\right) V.
$$

Intuitively, each token's represention in $O$ (which is a $d$ vector) is a linear combination of the $r$ row vectors in $V$(or $K$). 

This has the spirit of **sparse dictionary learning**{cite:p}`mairal2009online`, which aims to learn a **sparse set of atom vectors**, such that each representation can be transformed to a linear combination of atom vectors.

An additional 2-layer MLP was added to further transfrom the $O$ vectors.  Finally, a mean pooling after MLP layers to obtain the embedding of whole sequences. 

In the paper, authors used latent attention layer with $r$ of 512 and the number of heads as 8 for multi-head attention.

```{figure} ../img/chapter_application_IR/LLM_for_IR/Embedding/NV_embed/latent_attention_layer.png
---
scale: 80%
name: chapter_application_IR_LLM_fig_embedding_NV_embedding_latent_attention_layer
---
The illustration of proposed architecture design comprising of decoder-only LLM followed
by latent attention layer. Latent attention layer functions as a form of cross-attention where the decoder-only LLM output serves as queries (Q) and trainable latent array passes through the keyvalue inputs, followed by MLP. Image from {cite:p}`lee2024nv`.
```


The two-stage instruction tuning method include
* First-stage contrastive training with instructions on a variety of retrieval datasets, utilizing in-batch negatives and curated hard-negative examples. 
* Second stage contrastive instruction-tuning on a combination of retrieval and non-retrieval datasets (e.g., classification ) without applying the trick of in-batch negatives. 

The design rationale behind the two-stage finetunings are: 
* It is found that retrieval task presents greater difficulty compared to the non-retrieval tasks there is one stage training fully dedicated to the retrieval task.
* In second stage, as the retrieval and non-retrieval tasks are blended, it is necessary to remove in-batch negatives trick. Since the negative may come from the the class and are not true negatives. 







## Software

[haystack](https://github.com/deepset-ai/haystack)





## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```