(chapter_text_embedding_sec_text_embedding_LLM)=
# LLM Text Embedding

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



## LLM Embedding Distillation

{cite:p}`lee2024gecko`







## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```