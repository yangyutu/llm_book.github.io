# RAG

## Introduction

### Motivation
Large Language Models (LLMs) have revolutionized natural language processing, but they still face several significant challenges:

* **Hallucination**: LLMs can generate plausible-sounding but factually incorrect information.
* **Outdated Knowledge**: The knowledge of LLMs is limited to their training data, which can quickly become obsolete.
* **Untraceable Reasoning**: The decision-making process of LLMs is often unclear, making it difficult to verify or understand their outputs.

To address these challenges, researchers and developers have been exploring promising solutions. One such solution is the integration of LLMs' inherent knowledge with external knowledge bases. This approach is known as Retrieval-Augmented Generation (RAG).

### What is RAG

RAG is a technique that combines the powerful language generation capabilities of LLMs with the ability to retrieve relevant information from external sources. This hybrid approach aims to leverage the strengths of both systems to produce more accurate, up-to-date, and verifiable outputs.

The Retrieval-Augmented Generation (RAG) framework is built upon three fundamental components:
* **Retrieval**, which is the process of extracting relevant paragraph from external knowledge bases in response to a query. This step can be viewed as a search engine process where top relevant text blocks are returned given search key words.  
* **Generation**, which involves using the retrieved information along with the language model's inherent knowledge to produce a response. This step leverages the power of large language models to understand context, integrate the retrieved information, and generate coherent and relevant text. 
* **Augmentation**, which a set of methods employed to enhance the quality, reliability, and effectiveness of the generated output. These techniques can include fact-checking against trusted sources, combining information from multiple retrieved documents, assessing the model's confidence in different parts of the response, and providing explanations or citations to support the generated content. Augmentation is key to addressing challenges such as hallucination and improving the overall trustworthiness of the system.

### RAG Advantages


Compared to LLM's responses that are relied on its own internal knowledge, RAG exhibits the following advantages:

**Improved Accuracy and Reliability**: By supplementing the LLM's knowledge with current, factual information from external sources, RAG can significantly reduce hallucinations and increase the accuracy of generated content.

**Superior Performance on Knowledge-Intensive Tasks**: RAG excels in tasks that require specific, detailed information, such as question-answering, fact-checking, and research assistance.
Continuous Knowledge Updates: Unlike traditional LLMs, which require retraining to incorporate new information, RAG systems can be updated by simply modifying the external knowledge base. This allows for more frequent and efficient knowledge updates.

**Domain-Specific Information Integration**: RAG enables the integration of specialized knowledge from particular fields or industries, making it possible to create more focused and accurate outputs for specific domains.

By addressing the key limitations of traditional LLMs, RAG represents a significant step forward in natural language processing and AI-assisted information retrieval and generation. As this technology continues to evolve, it promises to enhance the capabilities and reliability of AI systems across a wide range of applications.

### A minimal RAG example


The Vanilla RAG (Retrieval-Augmented Generation) operates in a simplified manner as follows:

- The text is divided into chunks.
- These chunks are then encoded into vectors using a Transformer encoder model, and all these vectors are stored in a vector database.
- Finally, a Language Model (LLM) prompt is created, and the model answers user queries based on the context retrieved from the top-k most relevant results found through vector indexing in the vector database.

During interaction, the same encoder model is used to vectorize user queries. Vector indexing is then performed to identify the top-k most relevant results from the vector database. These indexed text chunks are retrieved and provided as context to the LLM prompt for generating responses to user queries.


Example prompt:
```
Give the answer to the user query delimited by triple backticks ```{query}``` using the information given in context delimited by triple backticks ```{context}```. If there is no relevant information in the provided context, try to answer yourself, but tell user that you did not have any relevant context to base your answer on. Be concise and output the answer of size less than 80 tokens.
```


### RAG paradigam overview


https://arxiv.org/pdf/2312.10997