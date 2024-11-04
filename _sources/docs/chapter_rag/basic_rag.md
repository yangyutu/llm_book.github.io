# RAG

## Motivation
LLMs have revolutionized natural language processing, but they still face several significant challenges, particularly in **knowledge intensive tasks**:

* **Hallucination**: LLMs can generate plausible-sounding but factually incorrect information when they are prompted with rare or ambiguous queries, e.g., what is kuula? (a fishing god).
* **Outdated knowledge**: The knowledge of LLMs is limited to their pre-training data, which can quickly become obsolete.
* **Untraceable reasoning**: The decision-making process of LLMs is often unclear, making it difficult to verify or understand their outputs.
* **Expensive cost to inject knowledge**: Although one can inject domain knowledge or updated knowledge via continuous pretraining or finetining, the cost of data collections and training is very high.

To address these challenges, researchers and developers have been exploring promising solutions. One such solution is to integrate of LLMs' inherent knowledge with external knowledge bases during model generation process. This approach is known as **Retrieval-Augmented Generation (RAG)** [{numref}`chapter_rag_fig_rag_demo`].

```{figure} ../img/chapter_rag/RAG_demo.png
---
scale: 60%
name: chapter_rag_fig_rag_demo
---
Illustration of RAG process applied to question answering. It mainly consists of basic steps. 1) Offline Indexing. Documents are split into chunks,
encoded into vectors, and stored in a vector database. 2) Retrieval. Retrieve the Top $k$ relevant chunks as context or knowledge supplement. 3)
Generation. The original question and the retrieved context are fed into LLM to generate the final answer. Image from {cite:p}`gao2023retrieval`.
```

Compared to LLM's responses that are relied on its own internal knowledge, RAG exhibits the following advantages:

**Improved Accuracy and Reliability**: By supplementing the LLM's knowledge with current, factual information from external sources, RAG can significantly reduce hallucinations and increase the accuracy of generated content.

**Superior Performance on Knowledge-Intensive Tasks**: RAG excels in tasks that require specific, detailed information, such as question-answering, fact-checking, and research assistance.

**Continuous Knowledge Updates**: Unlike traditional LLMs, which require retraining to incorporate new information, RAG systems can be updated by simply modifying the external knowledge base. This allows for more frequent and efficient knowledge updates.

**Domain-Specific Information Integration**: RAG enables the integration of specialized knowledge from particular fields or industries, making it possible to create more focused and accurate outputs for specific domains.


## RAG Frameworks

### Basic RAG 

RAG is a technique that combines the powerful language understanding generation capabilities of LLMs with the information retrieval ability of a retrieval system/search engine. This hybrid approach aims to leverage the strengths of both systems to produce more accurate, up-to-date, and verifiable outputs.

The **RAG** framework is built upon three fundamental components [{numref}`chapter_rag_fig_rag_framework_demo`]:
* **Indexing building**, which transforms raw documents into a format that enables efficient retrieval and knowledge integration. For example, one can split a document into multiple chunks, and encode each of them into dense embedding vectors (for dense retrieval) or inverted index (for sparse retrieval). Indexing building is usually done offline.
* **Retrieval**, which is the process of extracting relevant paragraph/chunks from the index in response to a query. This step involves online query understanding and processing - transform the query into embedding vector (for dense retrieval) or terms (for sparse retrieval) and retrieving documents using query vectors or terms. 
* **Generation**, which involves using the retrieved information along with the language model's inherent knowledge to produce a response. This step leverages the power of large language models to understand context, integrate the retrieved information, and generate coherent and relevant text. 


```{figure} ../img/chapter_rag/Basic_RAG_framework.png
---
scale: 35%
name: chapter_rag_fig_rag_framework_demo
---
Illustration of a basic RAG framework. 
```

````{prf:example} A minimal RAG example

The Vanilla RAG (Retrieval-Augmented Generation) operates in a simplified manner as follows:

- The text is divided into chunks.
- These chunks are then encoded into vectors using a Transformer encoder model, and all these vectors are stored in a vector database.
- Finally, a Language Model (LLM) prompt is created, and the model answers user queries based on the context retrieved from the top-k most relevant results found through vector indexing in the vector database.

During interaction, the same encoder model is used to vectorize user queries. Vector indexing is then performed to identify the top-k most relevant results from the vector database. These indexed text chunks are retrieved and provided as context to the LLM prompt for generating responses to user queries.

*Example LLM prompt*:

Give the answer to the user query delimited by triple backticks ```{query}``` using the information given in context delimited by triple backticks ```{context}```. If there is no relevant information in the provided context, try to answer yourself, but tell user that you did not have any relevant context to base your answer on. Be concise and output the answer of size less than 80 tokens.
````

### RAG Optimizations




Various optimizations can be applied to the basic RAG framework [{numref}`chapter_rag_fig_rag_framework_demo`] to improve the quality and reliability of the system's outputs. These optimizations address common challenges in RAG systems, including query-document mismatch, retrieval accuracy, and output reliability.

As shown in {numref}`chapter_rag_fig_rag_framework_optimization_demo`, we can divide optimizations into the following categories:

**Document Understanding & Augmentation**:Instead of performing mechanism chunking to split the document, we can apply language understanding models to split documents into semnatically coherent units. Besides, we can enriches documents with additional context, metadata (e.g., stamptime), and alternative representations (queries related to document) before they enter the indexing phase. This enrichment makes documents more discoverable and helps maintain their semantic context even when split into chunks for processing.

**Query Understanding and Rewriting**: This enhancement addresses one of the fundamental challenges in information retrieval: the vocabulary mismatch between query language and document language. The system can employ LLM to analyze and reformulate user queries, making them more effective for retrieval. It can help complex, multi-concept queries by decomposing the original query into multiple manageable subqueries. 

**Hybrid retrieval**: Instead of using only sparse retrieval or dense retrieval, one can combine them together to form a hybrid retrieval system. For example, encoder models used in dense retriever can produce additional features for rankers in the sparse retriever side. 

**Re-ranking**: After the initial retrieval, the Re-ranking step can significantly improves the quality and precision of retrieved content before it reaches the LLM. Documents are re-ranked using a much powerful model based on their contextual relevance to the user query, not just the vector semantic similarity. In addition, the re-ranking process often penalize similiar documents to promote a diverse set of relevant documents are sent to LLM.

**Output verification**: After the LLM's output, we can add additional step to validates the LLM's output against the retrieved and re-ranked sources. This final check ensures accuracy and consistency.

```{figure} ../img/chapter_rag/Basic_RAG_framework_optimization.png
---
scale: 35%
name: chapter_rag_fig_rag_framework_optimization_demo
---
Optimization of the basic RAG framework in different components. 
```


### RAG Challenges in Practice

The following key factors are essential to a successful application of RAG in real-world. These four keys are sequentially dependent and any issues on one-of-them will cause response of poor quality.

```{figure} ../img/chapter_rag/RAG_key_success_factors.png
---
scale: 35%
name: chapter_rag_fig_rag_framework_key_success_factors
---
Key factors underlying a successful RAG product. 
```




Following table summarize practical challenges and possible causes when applying RAG into actual product. 

| Issue Type | Document Understanding | Query Understanding & Ranking Service | LLM |
|------------|-------------------------|-----------------------------------|-----|
| Hallucination | Chunking, truncation, text extraction errors | Search results irrelevant | Model generated hallucination |
| Refusal to Answer | | Search results irrelevant & incomplete | Model not understanding content |
| Incomplete Response | Incomplete chunking | Search results irrelevant & incomplete | Model summary incomplete |
| Slow Response Time | | Search too slow  | Large model parameters |


<!-- ## RAG paradigam overview


% https://arxiv.org/pdf/2312.10997
 -->




## RAG Optimization: Documents
### Indexing Data Sources

In the indexing stage, there are different data sources, and each has its benefits and challenges.

```{table} Retrieval Data Sources
| Data Type | Examples | Benefits | Challenges|
| :--- | :--- | :--- | :--- | 
| Unstructured Data | Text, Web pages, Wikipedia, domain specific corpus | Large availability | Need quality control (e.g., remove bias, noisy content) during indexing time; Difficult to parse |
| Semi-structured Data | PDFs, structured markdowns, or Data that contains a combination of text, table, image information. | Cleaner content than unstructured text and web pages | Challenges are chunking while preserving table completeness. Converting table to text requires additional tools.| 
| Structured Data | Knowledge base, knowledge graph. | Organized, clean information | High cost to maintain up-to-date information; Need tools to generate KG search queries; requires additional effort to build, validate, and maintain structured databases. |
```

Performing search in a structured knowledge base usually involving additional preprocessing steps than performing search in unstructred/semi-structred data source 
{cite:p}`wang2023knowledgpt`. For example, to search entity in a knowledge base, we need to first extract entities from the query [{numref}`chapter_rag_fig_rag_knowledge_base_demo`]. 

```{figure} ../img/chapter_rag/data_source/RAG_knowledge_base.png
---
scale: 60%
name: chapter_rag_fig_rag_knowledge_base_demo
---
Comparison between retrieval results from document corpus and knowledge bases. Retrieval from knowledge
base (based on key phrases extracted by LLM) could avoid concept/entity missing issue. Image from {cite:p}`wang2023knowledgpt`.
```

### Data Source Augmentation 

For unstructured data sources, it can improve document understanding and feature derivation by augmenting the data source with additional information. 

For example, chunks can be enriched with metadata information such as page number, file name, author,category timestamp. Timestamp can be used to improve time-aware retrieval model, ensuring the fresh knowledge are ranked higher and avoiding outdated information.

Augmented data can also be artificially constructed. For example, adding summaries of paragraph, as well as introducing queries can be answered by the paragraphs (known as doc2query {cite:p}`nogueira2019document`).

```{figure} ../img/chapter_rag/data_source/doc_2_query.png
---
scale: 60%
name: chapter_rag_fig_rag_knowledge_base_demo
---
Use doc2query to enhance retrieval performance. Image from {cite:p}`nogueira2019document`.
```

### Document Splitting and Granularity

During document indexing stage, we need to split documents into different chunks. Such retrieval granularity ranges from fine to coarse, including Phrase, Sentence, Paragrpahs. 

Coarse-grained retrieval units fundamentally improve the **recall at the cost of precision**; that is, it can provide more relevant information for the problem, but they may also contain redundant content, which could distract the retriever and language models in downstream tasks. Intutively, encoding a large chunk of text into a single vector will have information loss.

On the other hand, fine-grained retrieval unit granularity increases the burden of retrieval and does not guarantee content completeness and semantic integrity (i.e., not enough context).

From a high level, an ideal splitting should consider the following factors:
* **Semantic coherence**: Chunks should maintain semantic coherence - split boundaries should respect natural semantic units and closely related information should stay in the same chunk.
* **Size consistency**: Chunks should be sized appropriately for the embedding model.
* **Information density**: Each chunk should contain sufficient information to be independently meaningful. 

There are different approaches to splitting:
* **Mechanical splitting** based on a fixed window size. This has the lowest processing cost, but there is no guaratee on maintaining semantic completeness for each chunk - it can create arbitrary breakpoints in the middle of sentences. One mitigation is to use overlapping sliding windows during splitting.
* **Structure-aware splitting** by leveraging document structures (headers, sections). This also has low processing cost and it respect the hierachical organization of the docuemnt. However, the chunk size can vary a lot as different documents can organize differently. Also, this method can only apply to relatively formal text data with such structural annotations.
* **Semantic-based splitting.** This method invovles using language understanding model to predict the semantic relationship between sentences and paragraphs. For example, we can use BERT to predict if two sentences are sementically close via the next sentence prediction task.  Sentences and paragraphs that are closely related to each other will be grouped into the same chunk. This method is much costly compared to previously two approaches, but it preserves topic coherence within the chunk.


## RAG Optimization: Query Understanding and Rewriting

### Motivation

Query understanding and rewriting is a crucial component in optimizing RAG  systems. The effectiveness of RAG heavily depends on the quality of the retrieval step, which in turn relies on how well the system understands and processes user queries. Raw user queries that ** don't directly match the way information is indexed offline** can lead to poor retrieval results.

Specifically, Key challenges that necessitate query rewriting include:
- **Vocabulary mismatch** between user queries and stored documents, particularly for **domain-specific queries** containing terminology and jargon
- **Implicit context** that needs to be made explicit
- **Complex queries** that combine multiple concepts
- **Contextual understanding** from multi-turn conversations

```{table} Retrieval Data Sources
| Query Type | Example | Rewrite Angle/Result | Explanation|
| :--- | :--- | :--- | :--- | 
| Vocabulary mismatch | Why is my car not starting | Car ignition failure diagnosis | Document side is likely to contain words like ignition. |
| Vocabulary mismatch | Can my boss fire me for being sick | Employment termination regulations regarding medical leave | Document side is likely to contain termination, medical leave.
| Implicit context | Current state tax rates | State tax rate for CA residents in 2024 | Incoporate time and location information into the query to improve precision. | 
| Complex queries | Can I take ibuprofen with my blood pressure medication while pregnant?| Decompose to three different subqueries on ibuprofen, blood pressure medication, and pregancy.
| Multi-turn conversational query | Turn 1: "Tell me about Tesla Model 3"; Turn 2: "What about its safety features?" | Safety features of Tesla Model 3  | Incoporate previous context into the rewritten query |
```

<!-- 

Motivation, why we need query understanding and rewrite


Understand what scenario we need to rewrite
* Highly specialized domains like law, medical
* Complex, multi-concept queries: convert to multiple subquestions, convert to multiple subqueries
* Multi-turn conversions: rewrite query based on historical conversation and context

How to rewrite?
* Use classical model to perform synoym expansion, acroynyn expansion, etc. 
* Use generative model to perform query write
* Use RAG + LLM to perform query understanding and rewrite
 -->



<!-- 

Highly Specialized Domains

In domains like law and medicine, effective query rewriting is essential because:
- Technical terminology may have multiple variations or synonyms
- Professional jargon needs to be mapped to standardized terms
- Concepts may be expressed differently in formal vs. informal language
- Domain-specific acronyms need expansion

````{prf:example}
- Original query: "What's the standard treatment for BP?"
- Rewritten: "What is the standard treatment for high blood pressure OR hypertension"
```` -->

### Approach to Vocalbulary Mismatch 

Vocalbulary mismatch often occur in querstion-answering in high-specialized domains, like law, science, engineering, etc. 
The reason is that concepts are often expressed differently in document language (more formal) vs. query language (less formal). Query understanding and rewriting is essential to bridge such gap by rewriting user's query concept to professional jargon. 

Traditional techniques focus on lexical and syntactic rule-based transformations, including
- Spelling correction and normalization
- Stop word removal and stemming
- Query expansion using WordNet or domain-specific dictionary or co-occurrence statistics
- Acronym restoring using predefined mappings

Tradictional approaches are usually computationally efficient, and it is predictable and interpretable. However, it often requires significant manual effort to craft rules and update dictionary. 

With the advancement of LLM, LLM can be used for more sophisticated query rewriting, as shown in the following example.

````{prf:example} LLM prompt to query rewrite
**Prompt:**
Given a query *What's the state tax rate?* from a user located at *San Jose, US* on *June, 2024*.
Rewrite this query to help retrieve comprehensive results from search engine like Google.
````
This approach leverages the excellent language understanding and generation of LLM, and it can capture implicit context and semantic variations without explicit rules. However, it also has the following drawbacks.
* High computational cost. LLM inference is much costly than traditional technique. One can reduce the cost by using effectively distilled SLM and by using the triggering model to decide when to invoke LLM.
* Lack of knowledge for queries involving rare entity names. This is an inherent drawback of LLM, which can be addressed by using a query-rewrite oriented RAG system.


### Approach to Complex Multi-Concept Queries

Many user queries combine multiple concepts or requirements that need to be decomposed for effective retrieval. The query decomposition involves the following steps:
- Breaking down complex queries into simpler, atomic questions
- Creating multiple focused, logically-related subqueries

````{prf:example}
The original query *Can I take ibuprofen with my blood pressure medication while pregnant?*" can be decomposed into:
  1. Is ibuprofen safe during pregnancy?
  2. What are the interactions between ibuprofen and blood pressure medication" 

````

LLM is best probably the best tool to handle complex queries. To guide LLM to produce useful and desired outcome, we can apply various prompting technique, like few shot prompting and CoT [{ref}`chapter_prompt_sec_CoT_prompting`]. For queries involving complex reaonsing, we can use {ref}`chapter_prompt_sec_step_back_prompting`.

### Approach to Multi-Turn Conversations

In conversational QA scenario (e.g., a chatbot), understanding user's query requires the understanding of previous conversational turns. In complex scenarioes, it needs the properly handling of 
* Resolving pronouns and references (user might use this, that, he, she to refer to entities in previous turns) 
* Potential conflicting aspect (user might need agree and then disagree as the conversation evolves)
* Topic switches (user might switch to another topic during one conversation session)

Given the complexity of multi-turn conversations, using LLM can offer a clean solution rather than using multiple specialized NLP modules/models.


## RAG Optimization: Retriever and Reader

### Retrieval Model Enhancement

In the basic RAG system, only a dense retrieval model based on vector embedding is used. In general, dense embeddings has good performance on recall as fundamentally it is an inexact, semantically based approach. On the other hand, sparse retrieval (e.g., inverted index plus BM25), which relies on exact tem matching, has good performance on precision, particularly for queries with low level details (e.g., specific number, year, name) and rare entities. 

Combining both sparse and dense approaches can capture different relevance features and can benefit from each other by leveraging complementary relevance information. For instance, both retrieval approaches can be used to generate initial recalled results and send to the next stage for re-ranking. 

There are other ways deep model can be used to enhance sparse model. For example, 
* **Term weight prediction**: Deep model can be used to predict contextualized term weights in a query, which can help sparse retrieval model to pay more attention to important terms give the query as a context. (See {ref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch_contextualized_term_importance`)
* **Enhance sparse retrieval ranker**: Sparse retrieval in practice often use much more sophositcated rankers than BM25. These ranker can benefit from query-document semantic similarity feature besides query document exact matching features. 

We are usually face the training data scarcity issue when we adapt a generic retrieval models to a highly specialized domain (e.g., medical, law, scientific). {cite:p}`dai2022promptagator` introduces a LLM-based approach, PROMPTAGATOR,  to enhance task-specific retrievers.

As shown in the {numref}`chapter_rag_fig_promptagator_demo`, PROMPTAGATOR consists of three components: 
* Prompt-based query generation, a task-specific prompt will be combined with a large language model to produce queries for all documents.
* Consistency filtering, which cleans the generated data based on round-trip consistency - query should be answered by the passage from which the query was generated. 
* Retriever training, in which a retriever will be trained using the filtered synthetic data.

```{figure} ../img/chapter_rag/retrieval_model/promptagator_training.png
---
scale: 70%
name: chapter_rag_fig_promptagator_demo
---
Illustration of PROMPTAGATOR, which generates synthetic data using LLM. Synthetic data, after consistency filtering, is used to train a retriever in labeled data scarcity domain. Image from {cite:p}`chapter_rag_fig_promptagator_demo`.
```



### Retrieval Result Quality Control

For RAG, maintaining high-quality retrieved data is crucial for generating accurate and coherent responses. Low quality content like redundancy and irrelvant will mislead LLM. In the following, we discuss several different approaches for quality control.

**Reranking**: Reranking is a commonly used, and effective quality control measure to improve the precision of retrieved results. Reranking can employ both rule-based methods (utilizing metrics like Diversity, Relevance, and MRR) and model-based approaches (e.g., BERT cross-encoder). The outcome of reranking is to select the most relevant and useful paragraphs for the LLM to consume.

**Context Compression**: A common misconception in the RAG process is the belief that retrieving as many relevant documents as possible and concatenating them to form a lengthy retrieval prompt is beneficial. However, excessive context can introduce more noise, diminishing the LLM's perception of key information .

**LLM-based quality evaluator**: We can also prompt the LLM to evaluate the retrieved content before generating the final answer. This allows the LLM to filter out documents with poor relevance.

<!-- ### Self-Aware LLM
However, we find that the retrieved knowledge does not always help and even has a
negative impact on original responses occasionally.

To better make use of both internal knowledge and external world knowledge,
we investigate eliciting the model’s ability to
recognize what they know and do not know
(which is also called “self-knowledge”) and
propose Self-Knowledge guided Retrieval augmentation (SKR), a simple yet effective method
which can let LLMs refer to the questions
they have previously encountered and adaptively call for external resources when dealing with new questions.

investigate eliciting the selfknowledge of LLMs and propose a simple yet effective Self-Knowledge guided Retrieval augmentation (SKR) method to flexibly call the retriever
for making better use of both internal and external
knowledge


The proposed direct prompting and in-context
learning methods can elicit self-knowledge of
LLMs to some extent. However, they have several limitations. First, both methods require designing prompts and calling the LLMs for each new
question, which makes it impractical. Second, incontext learning could also be unstable due to contextual bias and sensitivity

| Template |
| :--- |
| Do you need additional information to answer this question? |
| Would you like any extra prompts to help you? |
| Would you like any additional clues? |
| Can you answer this question based on what you  know? |
| Can you solve this question now? |


```{figure} ../img/chapter_rag/data_source/self_knowledge_LLM_demo.png
---
scale: 60%
name: chapter_rag_fig_rag_demo
---
omparison between two responses given by InstructGPT. The retrieved passages are relevant but not particularly helpful for solving the question, which influences the model’s judgment and leads to incorrect answers. Image from {cite:p}`wang2023self`.
``` -->

## Further RAG Discussion

### RAG vs Prompting and Fine Tuning

When adapt an generalist LLM to different usage scenarios, there are different approaches, including direct prompting, fine-tuning, and RAG.

Each method has distinct characteristics as illustrated in {numref}`chapter_rag_fig_rag_vs_prompt_FT`. We used a quadrant chart to illustrate the differences among three methods in two dimensions: external knowledge requirements and model adaption requirements. Prompt engineering leverages a model's inherent capabilities with minimum necessity for external knowledge and model adaption. RAG can leverage external knowledge via information retrieval, making it excellet for knowledge intentsive tasks. In contrast, FT is suitable for customizing models to specific structures, styles, or formats.

Prompt Engineering requires low modifications to the model. It is suitable for relative simple tasks without intensive external knowledge. RAG is particularly suitable dealing with dynamic knowledge tasks. For these tasks, indexing stage in RAG often has auto-refresh ability, enabling RAG to provide realtime knowledge updates and effective utilization of external knowledge sources with high interpretability. However, it comes with low latency scenarios, RAG has to spend extra time (~200-500ms) to perform retrieval. On the other hand, FT is more static, requiring retraining for knowledge updates and instruction following. It often demands significant computational resources for dataset preparation and training.

```{figure} ../img/chapter_rag/RAG_vs_FT_vs_prompt.png
---
scale: 80%
name: chapter_rag_fig_rag_vs_prompt_FT
---
RAG compared with other model optimization methods in the aspects of **External Knowledge Required** and **Model Adaption Required**.  Image from {cite:p}`gao2023retrieval`.
```


In the following, we also summarize different factors to consider when choosing among prompting, fine-tuning, and RAG.

```{table} Different factors to consider when choosing among prompting, fine-tuning, and RAG.
| Factors | Prompting | Fine-tuning | RAG|
| :--- | :--- | :--- | :--- | 
| Dynamic/up-to-date knowledge| ❌ | ✅ | ✅ |
| Reduce hullucilation | ❌ | ❌ | ✅ |
| Model behavior customizatio | ❌|✅|❌| 
| Training cost| ✅ |❌| ✅|
| Explanability| ❌ | ❌ | ✅ |
| General ability| ✅ |❌| ✅|
| Low latency | ✅ |❌| ✅|
```


### RAG vs Long Context LLM

Long context LLM is one active research area, as enabling long context can open opportunties to handle more complex tasks. 
For example, Gemini 1.5 (https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/) has a context window of 1 million tokens.  This capability makes it possible for long-document question answering, in which user can now incorporate the entire document directly into the prompt. 

**Do we still need information retrieval component when it comes to long context LLM?** Can we just put a large document as knowledge base in the prompt?
In fact, information retrieval plays an irreplaceable role. Providing LLMs with a large amount of context has the following drawbacks:
* Inference speed is signficicantly reduced due to long sequences. In comparison, 
* The long context usually contain many irrelevant and noisy parts, which will impact the model performance. 
* The process is not tracable or explanable. 

## Bibliography

Important resources:
- [Retrieval-based Language Models and Applications ACL2023 Tutorial](https://acl2023-retrieval-lm.github.io/)


```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```