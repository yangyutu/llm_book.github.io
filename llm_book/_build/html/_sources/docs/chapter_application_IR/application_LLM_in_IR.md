# Application of LLM in IR (WIP)






## Query Understanding & Optimization

[A Survey of Query Optimization in Large Language Models](https://arxiv.org/pdf/2412.17558)

[Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2305.14283)

[BlendFilter: Advancing Retrieval-Augmented Large Language Models via Query Generation Blending and Knowledge Filtering](https://arxiv.org/pdf/2402.11129)
### Query Categorization



#### Natural Language Queries

| Question Type | Description |
| :--- | :--- |
| Fact-based | Questions that seek specific facts or information. |
| Multi-hop | Questions that require reasoning over multiple pieces of information or relationships. |
| Numerical | Questions that involve numerical values or calculations. |
| Tabular | Questions that relate to data presented in tables. |
| Temporal | Questions that involve time-related information or events. |
| Multi-constraint | Questions that involve multiple constraints or conditions. |

Examples

| Question Type | Query Examples |
| :---: | :--- |
| Fact-based | What is the capital of France? <br> Who invented the telephone? <br> What is the population of Tokyo? |
| Multi-hop | Who is the CEO of the company that manufactures the iPhone? <br> What is the name of the river that flows through London? <br> What is the highest mountain in the country where the Taj Mahal is located? |
| Numerical | What is the average temperature in July in London? <br> What is the distance between New York and Los Angeles? <br> What is the GDP of Japan? |
| Tabular | Which country has the highest population density? <br> What is the average number of mobile phones sold by all the showrooms in the year 2007 ? <br> How many restaurants are in the American cuisine type? |
| Temporal | When did World War II end? <br> Who was the president of the United States in 1963? <br> What was the price of gold in 2008? |
| Multi-constraint | Find a hotel in Paris that has a swimming pool and is within walking distance of the Eiffel Tower. <br> What are some laptops that have a 15 -inch screen, 16 GB of RAM, and cost less than $\$ 1,000$ ? <br> Is it possible to constrain the flat face of a say 6 countersunk screws to a single face? |

### Query Optimization Overview

**Query Expansion** - aims to capture a wider range of relevant information and potentially uncover connections that may not have been apparent in the query.This process involves analyzing the initial query, identifying key concepts, and incorporating related terms, synonyms, or associated ideas to form a new query for creating a more comprehensive search.

**Query Decomposition** - aims to effectively break down complex, multihop queries into simpler, more manageable subqueries or tasks. This approach involves dissecting a query that requires facts from multiple sources or steps into smaller, more direct queries that can be answered individually.

**Query Disambiguation** - aims to identify and eliminate ambiguity in complex queries, ensuring they are unequivocal. This involves pinpointing elements of the query that could be interpreted in multiple ways and refining the query to ensure a single, precise interpretation.

**Query Abstraction** - aims to provide a broader perspective on the fact need, potentially leading to more diverse and comprehensive results. This involves identifying and distilling the fundamental intent and core conceptual elements of the query, then creating a higher-level representation that captures the essential meaning while removing specific details.

### GEFEED (Retrieval Feedback)

When using LLM to optimize query and retrievel processes, there has been efforts on using LLM to generate relevent context for the query {cite:p}`yu2022generate,wang2023query2doc` based on the **internal knowledge of LLM**. These relevant context can be used to further refine the query or the retrievel process. However, there are several fundamental drawback in this approach when it comes to knowledge intensive tasks. 
* LLM has a tendency to hallucinate content, generating information not grounded
by world knowledge, leading to untrustworthy outputs and a diminished capacity to provide accurate information.
* The quality and scope of the internal knowledge of LLM may be incomplete or out-of-date due to the reliability of the sources in the pre-training corpus. Moreover, due to model capacity limitation, **LLMs cannot memorize all world information**, particularly the long tail of knowledge from their training corpus {cite:p}`kandpal2023large`.

The key steps in GEFEED [{numref}`chapter_application_IR_LLM_fig_QueryUnderstandingOptimization_GEFEED_demo`] are:
* Given a query, the language model generates initial outputs (more than one)[{numref}`chapter_application_IR_LLM_fig_QueryUnderstandingOptimization_GEFEED_workflow`].
* A retrieval module retrieve expanded relevant information using the original query and generated outputs as a new query. 
* The language model reader produce the final output based on the expanded retrieved information.

The potential benefits from GEFEED are:
* By directly generating the expected answer, rather than performing query paraphrasing, the lack of lexical or semantic overlap with the question and the document can be reduced.
* As more relevant documents are retrieved from the corpus using expected answers, the recall and the precision of the retrieved documents can be both improved. 


```{figure} ../img/chapter_application_IR/LLM_for_IR/QueryUnderstandingOptimization/GEFEED_demo.png
---
scale: 60%
name: chapter_application_IR_LLM_fig_QueryUnderstandingOptimization_GEFEED_demo
---
REFEED operates by initially prompting a large language model to generate an answer in response to a given query, followed by the retrieval of documents from extensive document collections. Subsequently, the pipeline refines the initial answer by incorporating the information gleaned from the retrieved documents. Image from {cite:p}`yu2023improving`.
```

```{figure} ../img/chapter_application_IR/LLM_for_IR/QueryUnderstandingOptimization/GEFEED_workflow.png
---
scale: 60%
name: chapter_application_IR_LLM_fig_QueryUnderstandingOptimization_GEFEED_workflow
---
The language model is prompted to sample multiple answers, allowing for a more comprehensive retrieval feedback based on different answers. Image from {cite:p}`yu2023improving`.
```


## Query-Doc Ranking

[Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents](https://arxiv.org/pdf/2304.09542v2)

[A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models](https://arxiv.org/abs/2310.09497)

[Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://aclanthology.org/2024.findings-naacl.97/)

[Zero-Shot Listwise Document Reranking with a Large Language Model](https://arxiv.org/abs/2305.02156)

### Rank List Generation

```{figure} ../img/chapter_application_IR/LLM_for_IR/QueryDocRanking/RankList_permutation_generation/slidingwindow_ranklist_generation.png
---
scale: 60%
name: chapter_application_IR_LLM_fig_QueryDocRanking_slidingwindow_ranklist_generation
---
llustration of re-ranking 8 passages using sliding windows with a window size of 4 and a step size of 2. The blue color represents the first two windows, while the yellow color represents the last window. The sliding windows are applied in back-to-first order,. Image from {cite:p}`sun2023chatgpt`.
```


### Ranker Distillation

**Pairwise distillation**: Suppose we have a query $q$ and $D$ candidate documents  $\left(d_1, \ldots, d_M\right)$ for ranking. Let the LLM-based ranking results of the $D$ documents be $R=\left(r_1, \ldots, r_M\right)$ (e.g, $r_i=3$ means $d_i$ ranks third among the candidates). 

Let $s_i=f_\theta\left(q, d_i\right)$ be the student model's relevance prediction score between $\left(q, d_i\right)$. 

We can use pairwise Ranking loss to optimize the student model, which is given by

$$
\mathcal{L}_{\text {RankNet }}=\sum_{i=1}^M \sum_{j=1}^M \mathbb{1}_{r_i<r_j} \log \left(1+\exp \left(s_i-s_j\right)\right)
$$



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


## Application in RAG

[Small Models, Big Insights: Leveraging Slim Proxy Models to Decide When
and What to Retrieve for LLMs](https://aclanthology.org/2024.acl-long.242.pdf)

## Generative SERP

[GenSERP: Large Language Models for Whole Page Presentation](https://arxiv.org/abs/2402.14301)
## Collections

[Awesome Information Retrieval in the Age of Large Language Model](https://github.com/IR-LLM/Awesome-Information-Retrieval-in-the-Age-of-Large-Language-Model)


## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```