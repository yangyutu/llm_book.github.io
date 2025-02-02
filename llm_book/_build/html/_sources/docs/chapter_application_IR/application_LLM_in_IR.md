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