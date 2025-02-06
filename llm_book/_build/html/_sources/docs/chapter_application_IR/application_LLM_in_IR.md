(chapter_application_IR_LLM)=
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

(chapter_application_IR_LLM_query_doc_ranking)=
## Query-Doc Ranking

[Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents](https://arxiv.org/pdf/2304.09542v2)

[A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models](https://arxiv.org/abs/2310.09497)

[Zero-Shot Listwise Document Reranking with a Large Language Model](https://arxiv.org/abs/2305.02156)

(chapter_application_IR_LLM_query_doc_ranking_via_query_likelihood)=
### Ranking via Query Likelihood

```{figure} ../img/chapter_application_IR/LLM_for_IR/QueryDocRanking/query_likelihood/ranking_by_query_likelihood.png
---
scale: 70%
name: chapter_application_IR_LLM_fig_ranking_by_query_likelihood
---
Illustration of using query likelihood to approximate the query doc relevance. Image from {cite:p}`sachan2022improving`.
```

Given a passage $d_i$ and a query $q$, the relevance score is approximated by the likelihood of the $q$ conditioned on $d_i$ plus an **instruction prompt** $\mathbb{P}$. 

Specifically, we can estimate $\log p\left(q \mid {d}_i, \mathbb{P}\right)$ using a LLM to compute the average conditional loglikelihood of the query tokens:

$$
\log p\left(q \mid d_i ; \mathbb{P}\right)=\frac{1}{|q|} \sum_t \log p\left(q_t \mid q_{<t}, d_i ; \mathbb{P}\right)
$$

where $|q| = t$ denotes the number of query tokens. The instruction prompt is given by *Please write a question based on this passage* to the passage tokens as shown in {numref}`chapter_application_IR_LLM_fig_ranking_by_query_likelihood`.

{cite:p}`drozdov2023parade` further extends the above query-likelihood approach by including demonstrations. Specificially, let $z_1,...,z_k$ be positive query-document pair as demonstrations, then the modified query likelihood becomes

$$
\log p\left(q \mid d_i ; z_1,...,z_k ;\mathbb{P}\right)=\frac{1}{|q|} \sum_t \log p\left(q_t \mid q_{<t}, d_i ; z_1,...,z_k ;\mathbb{P}\right).
$$

The authors found that 
* Selecting demonstrations based on semantic similarity is not necessarily providing the best value;
* Instead, one can use difficulty-based selection to find challenging demonstrations to include in the prompt. We estimate difficulty using demonstration query likelihood (DQL):

$$
\operatorname{DQL}(z) \propto \frac{1}{\left|q^{(z)}\right|} \log P\left(q^{(z)} \mid d^{(z)}\right)
$$

then select the demonstrations with the lowest DQL. Intuitively, this should find hard samples that potentially correspond to large gradients had we directly trained the model instead of prompting.

### Ranking via Relevance Label Likelihood

```{figure} ../img/chapter_application_IR/LLM_for_IR/QueryDocRanking/relevance_label_likelihood/relevance_label_likelihood_demo.png
---
scale: 70%
name: chapter_application_IR_LLM_fig_ranking_by_relevance_label_likelihood
---
Illustration of using relevance likelihood to approximate the query doc relevance. One can use different rating class and scale to improve the results. Image from {cite:p}`zhuang2023beyond`.
```

Authors in {cite:p}`zhuang2023beyond` proposed that one can use LLM as zero-shot text ranker by prompting LLM to provide rating for a given query and doc pair. Example rating scale or class include
* Binary class: {Yes, No}
* Fine-grained relevance label: {Highly Relevant, Somewhat Relevant, Highly Relevant, Perfectly Relevant}
* Rating scale: {0, 1, 2, 3, 4}

We can obtain the log-likelihood of the LLM generating each relevance label:

$$
s_{i, k}=\operatorname{LLM}\left(l_k \mid q, d_i\right)
$$

where $l_k$ is the relevance class label.

Once we obtain the log-likelihood of each relevance label, we can derive the ranking scores.

**Expected relevance values**: First, we need to assign a series of relevance values $\left[y_0, ..., y_k\right]$ to all the relevance labels $\left[l_0, ..., l_k\right]$, where $y_k \in \mathbb{R}$. Then we can calculate the expected relevance value by:

$$
\begin{aligned}
f\left(q, d_i\right) & =\sum p_{i, k} \cdot y_k \\
\text { where } p_{i, k} & =\frac{\exp \left(s_{i, k}\right)}{\sum_{k^{\prime}} \exp \left(s_{i, k^{\prime}}\right)}
\end{aligned}
$$

**Peak relevance likelihood**: We can further simplify ranking score by only using the loglikelihood of **the peak relevance label** (e.g., *Perfectly Relevant* in this example). More formally, let $l_{k^*}$ denote the relevance label with the highest relevance. We can simply rank the documents by:

$$
f\left(q, d_i\right)=s_{i, k^*}.
$$

The key findings are:
* By using more fine grained relevance label will generally improve the zero-shot ranking performance. 
* It is hypothesized that the inclusion of fine-grained relevance labels in the prompt may guide LLMs to better differentiate documents, especially those ranked at
the top.


### Pairwise and Groupwise Text Ranking

```{figure} ../img/chapter_application_IR/LLM_for_IR/QueryDocRanking/pairwise_ranking/pairwise_ranking_demo.png
---
scale: 50%
name: chapter_application_IR_LLM_fig_ranking_by_pairwise_ranking
---
An illustration of pairwise ranking prompting. The scores in scoring mode represent the log-likelihood of the model generating the target text given the prompt. Image from {cite:p}`qin2023large`.
```

Using LLM for **pointwise ranking** via query or relevance class likelihood has shown good performance, but the success is mainly limited to large-scale models. The hypothesis is that pointwise ranking requires LLM to output calibrated predictions, which can be challenging for less-competent small-scale models. Authors from {cite:p}`qin2023large` proposed that one can leverage pairwise ranking to reduce the difficulity of the task.

As shown in {numref}`chapter_application_IR_LLM_fig_ranking_by_pairwise_ranking`, the pairwise ranking paradigm takes one query and a pair of documents as the input and output the comparison result. This potentially resolves the calibration issue.


One can further consider **group-wise ranking** by prompting LLM to order the relevance of $k$ candidate passages, as shown in {numref}`chapter_application_IR_LLM_fig_QueryDocRanking_groupwise_ranking`. 

```{figure} ../img/chapter_application_IR/LLM_for_IR/QueryDocRanking/groupwise_ranking/groupwise_ranking_demo.png
---
scale: 40%
name: chapter_application_IR_LLM_fig_QueryDocRanking_groupwise_ranking
---
Illustration of groupwise_ranking for $k$ passages. Image from {cite:p}`sun2023chatgpt`.
```

With the local order established by either pairwise ranking or groupwise ranking, one can achieve global ordered rank list using a sliding window approach [{numref}`chapter_application_IR_LLM_fig_QueryDocRanking_slidingwindow_ranklist_generation`]. 


```{figure} ../img/chapter_application_IR/LLM_for_IR/QueryDocRanking/RankList_permutation_generation/slidingwindow_ranklist_generation.png
---
scale: 40%
name: chapter_application_IR_LLM_fig_QueryDocRanking_slidingwindow_ranklist_generation
---
Illustration of re-ranking 8 passages using sliding windows with a window size of 4 and a step size of 2. The blue color represents the first two windows, while the yellow color represents the last window. The sliding windows are applied in back-to-first order. Image from {cite:p}`sun2023chatgpt`.
```

### Ranker Distillation

**Pairwise distillation**: Suppose we have a query $q$ and $D$ candidate documents  $\left(d_1, \ldots, d_M\right)$ for ranking. Let the LLM-based ranking results of the $D$ documents be $R=\left(r_1, \ldots, r_M\right)$ (e.g, $r_i=3$ means $d_i$ ranks third among the candidates). 

Let $s_i=f_\theta\left(q, d_i\right)$ be the student model's relevance prediction score between $\left(q, d_i\right)$. 

We can use pairwise Ranking loss to optimize the student model, which is given by

$$
\mathcal{L}_{\text {RankNet }}=\sum_{i=1}^M \sum_{j=1}^M \mathbb{1}_{r_i<r_j} \log \left(1+\exp \left(s_i-s_j\right)\right)
$$



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