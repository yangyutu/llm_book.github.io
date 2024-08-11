# Basic RAG

The Vanilla RAG (Retrieval-Augmented Generation) operates in a simplified manner as follows:

- The text is divided into chunks.
- These chunks are then encoded into vectors using a Transformer encoder model, and all these vectors are stored in a vector database.
- Finally, a Language Model (LLM) prompt is created, and the model answers user queries based on the context retrieved from the top-k most relevant results found through vector indexing in the vector database.

During interaction, the same encoder model is used to vectorize user queries. Vector indexing is then performed to identify the top-k most relevant results from the vector database. These indexed text chunks are retrieved and provided as context to the LLM prompt for generating responses to user queries.


Example prompt:
```
Give the answer to the user query delimited by triple backticks ```{query}``` using the information given in context delimited by triple backticks ```{context}```. If there is no relevant information in the provided context, try to answer yourself, but tell user that you did not have any relevant context to base your answer on. Be concise and output the answer of size less than 80 tokens.
```