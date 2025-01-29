# Conversational IR (WIP)

## Introduction

Search engines like Google and Bing have become integral tools to meet users' information needs. Recently, rapid advancements in NLP and LLM technologies have led to more intelligent and interactive user experiences in search engines. A notable advancement in this domain is **conversational search**, an emerging paradigm that provides natural language interactions for complex and accurate information access. Commercial conversational AI search engines, such as Perplexity and ChatGPT, have quickly attracted a large user base.

Compared to traditional search engines that rely on keywords or short phrases, conversational search leverages natural language dialogue for interactions, greatly enhancing the efficiency of information exchange and optimizing the user experience. 
This interactive approach have following advatnages:
* It allows user to issue more complex queries, ask explanation and follow-ups for in-depth reesarch questions.
* The systems better answer user queries by can proactively ask clarifying questions, make recommendations, and guide users in better expressing their needs through dialogue.


A typical conversational search system comprises several essential modules, each involving a variety of advanced technologies. As this field continues to evolve, it presents both significant opportunities and challenges. In this chapter, we will explore and analyze the most critical components that constitute a conversational search system, following the entire information flow. Our focus will include 
* Query understanding, including query reformulation, and search clarification
* Retrieval and ranking,
* Response generation. 

These components are integrated to work collaboratively, enabling more natural and intuitive interactions between users and the system, ultimately enhancing the overall search experience.


## Query Understanding and Optimization
### Contextual Understanding and Query Rewrite

For conversational search, accurately interpreting the user's current information needs is critical. Unlike traditional search engines that rely on isolated keyword-based queries, conversational search systems must consider the evolving context of an ongoing dialogue. As conversations progress, the context becomes increasingly complex and lengthy, posing significant challenges for traditional search engines, which often struggle with handling such multi-turn interactions.

**Contextual understanding** involves the ability of the conversational search system to grasp the nuances and shifts in the user's intent throughout the conversation. This requires advanced natural language processing (NLP) techniques to track and maintain the context across multiple exchanges, ensuring that the system can respond appropriately to follow-up questions and clarifications. Without a robust contextual understanding, the search engine may fail to recognize the continuity of the user's queries, leading to irrelevant or incomplete results.

**Query rewrite** plays a vital role in addressing these challenges. It involves transforming the entire conversational context, along with the current query, into a concise yet comprehensive representation of the user's present information needs. By distilling the context and refining the query, the system can generate more accurate and relevant search results. This process ensures that subsequent components, such as retrieval and response generation, can effectively process and respond to the user's query.

For instance, consider a user who starts with the query, "What is information retrieval?" and follows up with, "Tell me some of its famous scholars." A traditional search engine might fail to maintain the context, providing irrelevant results for the second query. In contrast, a conversational search system with effective query rewrite capabilities will recognize that the user is still referring to "information retrieval scholars" and provide accurate information accordingly.

In summary, contextual understanding and query rewrite are essential components of conversational search systems. They enable the system to manage complex, multi-turn interactions and deliver precise, contextually relevant results, thereby enhancing the overall user experience.

```{figure} ../img/chapter_application_IR/conversational_IR/query_rewrite/anaphora_ellipsis.png
---
scale: 60%
name: chapter_application_IR_conversational_IR_query_rewrite_anaphora_ellipsis
---
An example illustrating the semantic phenomena of **Anaphora** and **Ellipsis** in conversational search. Image from {cite:p}`mo2024survey`
```
### Query Clarification

In conversational search systems, query clarification is a pivotal mechanism that enhances user interaction by addressing ambiguities and refining search intents through dynamic dialogue. Users often express their search intents ambiguously, making it challenging for the system to immediately provide accurate results.

In traditional search engines, there is no intuitive and natural way to ask for clarifications. Users are typically presented with a list of results based on their initial query, which may not fully capture their intent. This often leads to a trial-and-error process where users refine their queries manually.

In contrast, conversational search systems can pose clarifying questions, such as “Do you mean [specific term]?” or “Can you provide more details about [topic]?”, rather than directly answering the query. When the system detects that clarification is needed, it proactively asks these questions to better understand the user’s intent. This approach ensures that the system provides more accurate and relevant search results, enhancing the overall user experience by making the search process more personalized and effective.

Implementing effective query clarification involves two primary challenges:
* **Ambiguity Detection**: Accurately identifying when a user's query is unclear or lacks sufficient detail.
* **Clarifying Question Generation**: Formulating appropriate and contextually relevant questions that guide the user toward specifying their intent.

Addressing these challenges is crucial for delivering a personalized and efficient search experience, as it ensures that the system comprehends the user's needs before retrieving information.

By incorporating query clarification, conversational search systems can handle complex and ambiguous user queries more effectively, leading to a more satisfying and efficient search experience.

