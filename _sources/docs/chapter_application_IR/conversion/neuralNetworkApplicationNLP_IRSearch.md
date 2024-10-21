
(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch)=
# Neural text ranking and information retrieval

```{index} IR, Neural IR, Information retrieval
```



## Overview of information retrieval

### Introduction 

#### Ad-hoc retrieval

Ad-hoc search and retrieval is a classic information retrieval (IR) task consisting of two steps: first, the user specifies his or her information need through a query; second, the information retrieval system fetches documents from a large corpus that are likely to be relevant to the query. Key elements in an ad-hoc retrieval system include
- **Query**, the textual description of information need.
- **Corpus**, a large collection of textual documents to be retrieved.
- **Relevance** is about whether a retrieved document can meet the user's information need.

There has been a long research and product development history on ad-hoc retrieval. Successful products in ad-hoc retrieval include Google search engine [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:adhocretrievaldemogoogle}] and Microsoft Bing [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:adhocretrievaldemobing}].  
One core component within Ad-hoc retrieval is text ranking. The returned documents from a retrieval system or a search engine are typically in the form of an ordered list of texts. These texts (web pages, academic papers, news, tweets, etc.) are ordered with respect to the relevance to the user's query, or the user's information need.

A major characteristic of ad-hoc retrieval is the heterogeneity of the query and the documents [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:querylengthDocLengthmsmarco}]. A user's query often comes with potentially unclear intent and is usually very short, ranging from a few words to a few sentences. On the other hand, documents are typically from a different set of authors with varying writing styles and have longer text length, ranging from multiple sentences to many paragraphs. Such heterogeneity poses significant challenges for vocabulary match and semantic match<sup>[^1]</sup> for ad-hoc retrieval tasks. 

\begin{figure}[H]
\begin{subfigure}[t]{0.45\textwidth}
	\centering

```{figure} images/../deepLearning/ApplicationIRSearch/DataExploration/query_length_MS_MARCO
```
\end{subfigure}\quad
\begin{subfigure}[t]{0.45\textwidth}
	\centering

```{figure} images/../deepLearning/ApplicationIRSearch/DataExploration/query_length_MS_MARCO
```
\end{subfigure}
\caption{Query length and document length distribution in Ad-hoc retrieval example using MS MARCO dataset.}\label{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:querylengthDocLengthmsmarco}
\end{figure}

There have been decades' research and engineering efforts on developing ad-hoc retrieval models and system. Traditional IR systems primarily rely on techniques to identify exact term matches between a query and a document and compute final relevance score between various weighting schemes. Such exact matching approach has achieved tremendous success due to scalability and computational efficiency - fetching a handful of relevant document from billions of candidate documents. Unfortunately, exact match often suffers from vocabulary mismatch problem where sentences with similar meaning but in different terms are considered not matched. Recent development of deep neural network approach {cite}`huang2013learning`, particularly Transformer based pre-trained large language models, has made great progress in semantic matching, or inexact match, by incorporating recent success in natural language understanding and generation. Recently, combining exact matching with semantic matching is empowering many IR and search products. 

\begin{figure}[H]
\begin{subfigure}[t]{1\textwidth}
	\centering

```{figure} images/../deepLearning/ApplicationIRSearch/ad_hoc_retrieval_demo_google
```
	\caption{Google search engine.}
	\label{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:adhocretrievaldemogoogle}
\end{subfigure}
\begin{subfigure}[t]{1\textwidth}
	\centering

```{figure} images/../deepLearning/ApplicationIRSearch/ad_hoc_retrieval_demo_bing
```
	\caption{Microsoft Bing search engine.}
	\label{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:adhocretrievaldemobing}
\end{subfigure}
\caption{Ad-hoc retrieval example using Google search and Microsoft Bing.}
\end{figure}

#### Open-domain question answering

Another application closely related IR is open-domain question answering (OpenQA), which has found a widespread adoption in products like search engine, intelligent assistant, and automatic customer service. OpenQA is a task to answer factoid questions that humans might ask, using a large collection of documents (e.g., Wikipedia, Web page, or collected document) as the information source. An  OpenQA example is like

**Q:** *What is the capital of China?*

\indent **A:** *Beijing*.

Contrast to Ad-hoc retrieval, instead of simply returning a
list of relevant documents, the goal of OpenQA is to identify (or extract) a span of text that directly answers the user’s question. Specifically, for *factoid* question answering, the OpenQA system primarily focuses on questions that can be answered with short phrases or named entities such as dates, locations, organizations, etc.

A typical modern OpenQA system adopts a two-stage pipeline {cite}`chen2017reading` [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:open-domainqa}]: (1) A document retriever selects a small set of relevant passages that probably contain the answer from a large-scale collection; (2) A document reader extracts the answer from relevant documents returned by the document retriever. Similar to ad-hoc search, relevant documents are required to be not only topically related to but also correctly address the question, which requires more semantics understanding beyond exact term matching features. One widely adopted strategy to improve OpenQA system with large corpus is to use an efficient document (or paragraph) retrieval technique to obtain a few relevant documents, and then use an accurate (yet expensive) reader model to read the retrieved documents and find the answer.

Nowadays many web search engines like Google and Bing have been evolving towards higher intelligence by incorporating OpenQA techniques into their search functionalities.

```{figure} images/../deepLearning/ApplicationIRSearch/open-domain_QA
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:open-domainqa
A typical open-domain architecture where a retriever retrieves passages from information source relevant to the question.
```

Compared with ad-hoc retrieval, OpenQA shows reduced heterogeneity between
the question and the answer passage/sentence yet add challenges of precisely understanding the question and locating passages that might contain answers. On one hand, the question is usually in natural language, which is longer than keyword queries and less ambiguous in intent. On the other hand, the answer passages/sentences are usually much shorter text spans than documents, leading to more concentrated topics/semantics. Retrieval and reader models need to capture the patterns expected in the answer passage/sentence based on the intent of the question, such as the matching of the context words, the existence of the expected answer type, and so on. 


### Modern IR Systems

A traditional IR system, or concretely a search engine, operates through several key steps. 

The first step is **crawling**.  A web search engines discover and collect web pages by crawling from site to site; Another vertical search engines such as e-commerce search
engines collect their product information from product description and other product meta data. The second step is **indexing**, which creates an inverted index that maps key words to document ids. The last step is **searching**. Searching is a process that accepts a text query as input, looks up relevant documents from the inverted index, ranks documents, and returns
a list of results, ranked by their relevance to the query. 

```{figure} images/../deepLearning/ApplicationIRSearch/traditional_IR_engine
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:traditionalirengine
Key steps in a traditional IR system.
```

The rapid progress of deep neural network learning {cite}`goodfellow2016deep` and their profound impact on natural language processing has also reshaped IR systems and brought IR into a deep learning age. Deep neural networks (e.g., Transformers {cite}`devlin2018bert`) have proved their unparalleled capability in semantic understanding over traditional IR margin yet they suffer from high computational cost and latency. This motivates the development of multi-stage retrieval and ranking IR system [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:retrieverankingarch}] in order to better balance trade-offs between effectiveness (i.e., quality and accuracy of final results) and efficiency (i.e., computational cost and latency). 

In this multi-stage pipeline, early stage models consists of simpler but more efficient models to reduce the candidate documents from billions to thousands; later stage models consists of complex models to perform accurate ranking for a handful of documents coming from early stages.

```{figure} images/../deepLearning/ApplicationIRSearch/retrieve_ranking_arch
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:retrieverankingarch
The multi-stage architecture of modern information retrieval systems.
```

In modern search engine, traditional IR models, which is based on term matching, serve as good candidates for early stage model due to their efficiency. The core idea of the traditional approach is to count repetitions of query terms in the document. Large counts indicates higher relevance. Different transformation and weighting schemes for those counts lead to a variety of possible TF-IDF ranking features. 

Later stage models are primarily deep learning model. Deep learning models in IR not only provide powerful representations of textual data that capture word and document semantics, allowing a machine to better under queries and documents, but also open doors to multi-modal (e.g., image, video) and multilingual search, ultimately paving the way for developing intelligent search engines that deliver rich contents to users. 

\begin{remark}[Why we need a semantic understanding model ]
For web-scale search engines like Google or Bing, typically a very small set of popular pages that can answer a good proportion of queries.\cite{mitra2016dual} The vast majority of queries contain common terms. It is possible to use term matching between key words in URL or title and query terms for text ranking; It is also possible to simply memorize the user clicks between common queries between their ideal URLs. For example, a query *CNN* is always matched to the CNN landing page. These simple methods clearly do not require a semantic understanding on the query and the document content. 

However, for new or tail queries as well as new and tail document, a semantic understanding on queries and documents is crucial. For these cases, there is a lack of click evidence between the queries and the documents, and therefore a model that capture the semantic-level relationship between the query and the document content is necessary for text ranking.
\end{remark}

### Challenges and opportunities in IR systems

#### Query understanding and rewriting

A user's query does not always have crystal clear description on the information need of the user. Rather, it often comes with potentially misspellings and unclear intent,  and is usually very short, ranging from a few words to a few sentences [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:tab:example_queries_MSMARCO}]. There are several challenges to understand the user's query.

Users might use vastly different query representations even though they have the same search intent. For example, suppose users like to get information about *distance between Sun and Earth*. Common used queries could be
- *how far earth sun*
- *distance from sun to earth*
- *distance of earth from sun*
- *how far earth is from the sun*

We can see that some of them are just key words rather than a full sentence and some of them might not have the completely correct grammar. 

There are also more challenging scenarios where queries are often poorly worded and far from describing the searcher’s actual information needs. Typically, we employ a query rewriting component to expand the search and increase recall,
i.e., to retrieve a larger set of results, with the hope that relevant results will not be missed. Such query rewriting component has multiple sub-components which are summarized below. 

**Spell checker**
Spell checking queries is an important and necessary feature of modern search. Spell checking enhance user experience by fixing basic spelling mistakes like *itlian restaurat* to *italian restaurant*.

**Query expansion **
Query expansion improves search result retrieval by adding or substituting terms to the user’s query. Essentially, these additional terms aim to minimize the mismatch between the searcher’s query and available documents. For example, the query *italian restaurant*, we can expand *restaurant* to *food* or *cuisine* to search all potential candidates.

**Query relaxation**
The reverse of query expansion is query relaxation, which expand the search scope when the user's query is too restrictive. For example, a search for *good Italian restaurant* can be relaxed to *italian restaurant*.

**Query intent understanding**
This subcomponent aims to figure out the main intent behind the query, e.g., the query *coffee shop* most likely has a local intent (an interest in nearby places) and the query *earthquake* may have a news intent. Later on, this intent will help in selecting and ranking the best documents for the query. 

Given a rewritten query, It is also important to correctly weigh specific terms in a query such that we can narrow down the search scope. Consider the query *NBA news*, a relevant document is expected to be about *NBA* and **news** but have more focus on *NBA*. There are traditional rule-based approach to determine the term importance as well as recent data-driven approach that determines the term importance based on sophisticated natural language and context understanding.

To improve relevance ranking, it is often necessary to incorporate additional context information (e.g., time, location, etc.) into the user's query. For example, when a user types in a query *coffee shop*, retrieve coffee shops by ascending distance to the user's location can generally improve relevance ranking. Still, there are challenges on deciding for which type of query we need to incorporate the context information.

```{figure} images/../deepLearning/ApplicationIRSearch/DataExploration/query_word_cloud
:name: fig:querywordcloud
Word cloud visualization for common query words using MS MARCO data. 
```

#### Exact match and semantic match

Traditional IR systems retrieve documents mainly by matching keywords in documents with those in search queries. While in many cases exact term match naturally ensure semantic match, there are cases, exact term matching can be insufficient. 

The first reason is due to the polysemy of words. That is, a word can mean different things depending on context. The meaning of *book* is different in *text book* and *book a hotel room*. Short queries particularly suffer from Polysemy because they are often devoid of context. 

The second reason is due to the fact that a concept is often expressed using different vocabularies and language styles in documents and queries. As a result, such a model would have difficulty in retrieving documents that have none of the query terms but turn out to be relevant.

Modern neural-based IR model enable semantic retrieval by learning latent representations of text from data and enable document retrieval based on semantic similarity. 

\begin{table}
	\footnotesize
	\centering
	\begin{tabular}{p{0.2\textwidth}p{0.75\textwidth}}
	\hline **Query** & "Weather Related Fatalities" \\
	\hline
	**Information Need** & A relevant document will report a type of weather event which has directly caused at least one fatality in some location. \\
	\hline **Lexical Document** & ".. Oklahoma and South Carolina each recorded three fatalities. There were two each in Arizona, Kentucky, Missouri, Utah and Virginia. Recording a single lightning death for the year 
	were Washington, D.C.; Kansas, Montana, North Dakota, .." \\
	\hline Semantic Document & .. Closed roads and icy highways took their toll as at least one motorist was killed in a 17 -vehicle pileup in Idaho, a 	tour bus crashed on an icy stretch of Sierra Nevada interstate and 100 -car string of accidents occurred near Seattle ... \\
	\hline
\end{tabular}
\caption{Retrieval results based on exact matching methods and semantic matching methods.}
\end{table}

An IR system solely rely on semantic retrieval is vulnerable to queries that have rare words. This is because rare words are infrequent or even never appear in the training data and learned representation associated with rare words query might be poor due to the nature of data-driven learning. On the other hand, exact matching approach are robust to rare words and can precisely retrieve documents containing rare terms.

Another drawback of semantic retrieval is high false positives: retrieving documents that are only loosely related to the query.

Nowadays, much efforts have been directed to achieve a strong and intelligent modern IR system by combining exact match and semantic match approaches in different ways. Examples include joint optimization of hybrid exact match and semantic match systems, enhancing exact match via semantic based query and document expansion, etc. 

#### Robustness to document variations

In response to users' queries and questions, IR systems needs to search a large collection of text documents, typically at the billion-level scale, to retrieve relevant ones. These documents are comprised of mostly unstructured natural language text, as compared to structured data like tables or forms. 

Documents can vary in length, ranging from sentences (e.g., searching for similar questions in a community question answering application like Quora) to lengthy web pages [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:tab:example_documents_MSMARCO}]. A long document might like a short document, covering a similar scope but with more words, which is also known as the **Verbosity hypothesis**. On the other hand, a long document might consist of a number of unrelated short documents concatenated together, which is known as **Scope hypothesis**. The wide variation of document forms lead to the IR different strategies. For example, following the Verbosity hypothesis a long document is represented by a single feature vector. Following the Scope hypothesis, one can break a long document into several semantically distinctive parts and represent each of them as separate feature vectors. We can consider each part as the unit of retrieval or rank the long document by aggregating evidence across its constituent parts.   
For full-text scientific articles, we might choose to only consider article titles and abstracts, and ignoring most of the numerical results and analysis. 

There are challenges on breaking a long document into semantically distinctive parts and encode each part into meaningful representation. Recent neural network methods extract semantic parts by clustering tokens in the hidden space and represent documents by multi-vector representations\cite{humeau2019poly, tang2021improving, luan2021sparse}.  

#### Computational efficiency

IR product such as search engines often serve a huge pool of user and need to handle tremendous volume of search requests during peak time (e.g., when there is breaking news events). To provide the best user experience, computational efficiency of IR models often directly affect user perceived latency. A long standing challenge is to achieve high accuracy/relevance in fetched documents yet to maintain a low latency. While traditional IR methods based on exact term match has excellent computational efficiency and scalability, it suffers from low accuracy due to the vocabulary and semantic mismatch problems. Recent progress in deep learning and natural language process are highlighted by complex transformer-based model {cite}`devlin2018bert` that achieved accuracy gain over traditional IR by a large margin yet experienced high latency issues. There are numerous ongoing studies (e.g., {cite}`mitra2017learning, mitra2019updated, gao2021coil`) aiming to bring the benefits from the two sides via hybrid modeling methodology.  

To alleviate the computational bottleneck from deep learning based dense retrieval, state-of-the-art search engines also adopts a multi-stage retrieval pipeline system: an efficient first-stage retriever uses a query to fetch a set of documents from the entire document collection, and subsequently one or more more powerful retriever to refine the results. 

\iffalse
#### Pre-trained language models

\begin{remark}[TODO]
External knowledge: in recent NLP research, it has been shown that the pretrained language models, especially the contextual embeddings [64], greatly boost the effectiveness of the down-stream NLP task. When pretrained embeddings or external knowledge are employed into the information retrieval area, care should he taken to make sure the representations are suitable for transferring into this domain [91].
\end{remark}

\cite{guu2020retrieval}

\cite{chang2020pre}

\cite{gao2021your}

\cite{gao2021unsupervised}

\cite{liu2021pre}

\cite{ma2021pre}
\fi
#### Context-dependent and personalized search

Personalization features have been the key drive factor for the huge success of recommender systems and computational advertising industry, where users are recommended items (e.g., goods, movies, musics, ads, etc) that they would be interested in. In a similar vein, incorporating personalization features into an IR system can benefit user experiences. Take search engine as an example, we can leverage user’s data, including location, search history, web browsing history, and even social network data, to help identify relevant contents for a specific user or cohort. For example, browsing history can offer insight like user’ topical interest and we can rank documents with similar topic higher. 

## Text ranking evaluation metrics

Consider a large corpus containing $N$ documents $D=\{d_1,...,d_N\}$. Given a query $q$, suppose the retriever and its subsequent re-ranker (if there is) ultimately produce an ordered list of $k$ relevant document $R_q = \{d_{i_1},...,d_{i_k}\}$, where documents $d_{i_1},...,d_{i_k} $ are ranked by some relevance measure with respect to the query. 

In the following, we discuss several commonly used metrics to evaluate an IR system. 

### Precision and recall

**Precision** and **recall** are metrics concerns the fraction of relevant documents retrieved for a query $q$, but they are not concerned with the ranking order. 
Specifically, precision computes the fraction of relevant documents with respect to  the total number of documents in the retrieved set $R_{q}$, where $R_{q}$ have $K$ documents; Recall computes the fraction of relevant documents with respect to the total number of relevant documents in the corpus $D$. More formally, we have
```{math}
\begin{align*}
\operatorname{Precision}_{q}@K &=\frac{\sum_{d \in R_{q}} \operatorname{rel}_{q}(d)}{\left|R_{q}\right|} = \frac{1}{K} \sum_{d \in R_{q}} \operatorname{rel}_{q}(d)\\
	\operatorname{Recall}_{q}@K &=\frac{\sum_{d \in R_{q}} \operatorname{rel}_{q}(d)}{\sum_{d \in D} \operatorname{rel}_{q}(d)}
\end{align*}
```
where $\operatorname{rel}_{q}(d)$ is a binary indicator function indicating if document $d$ is relevant to $q$. Note that the denominator $\sum_{d \in D} \operatorname{rel}_{q}(d)$ is the total number of relevant documents in $D$, which is a constant. 

Typically, we only retrieve a fixed number $K$ of documents, i.e., $|R_q| = K$, where $K$ typically takes 100 to 1000. The precision and recall at a given $K$ can be computed over the total number of queries $Q$, that is
```{math}
\begin{align*}
\operatorname{Precision}@{K} &=\frac{1}{|Q|}\sum_{q\in Q} \operatorname{Precision}_{q}@K \\
	\operatorname{Recall}@{K} &=\frac{1}{|Q|}\sum_{q\in Q} \operatorname{Recall}_{q}@K
\end{align*}
```

Because recall has a fixed denominator, recall will increase as $K$ increases.

### Mean reciprocal rank (MRR)

Mean reciprocal rank (MRR), introduced in {cite}`craswell2009mean` is another commonly used metric to measure relevance ranking quality. Different from precision and recall, MRR measures not only the fraction of relevant document, but also how the relevant documents are ranked in the returned list. 
First we define the **reciprocal rank** of the retrieved document associated with a query $q$.
$$
RR_{q}=\max _{d_i \in R_{q}} \frac{\operatorname{rel}_{q}(d)}{i}
$$
where $i$ is the rank of document $d$ in the retrieved document list. Clearly, only **one** relevant document $d$ can contribute to the value of $RR_q$; further, the contribution is diminishing if the document is ranked at the bottom.

With RR defined, MRR is simply the reciprocal rank averaged over all queries $Q$:
$$MRR = \frac{1}{|Q|}\sum_{q\in Q} RR_{q}.$$

\iffalse
### Mean average precision

Mean average precision (MAP) The average precision (Zhu, 2004) for a ranked list of documents $R$ is given by,
$$
A v e P_{q}=\frac{\sum_{\langle i, d\rangle \in R_{q}} \text { Precision }_{q, i} \times \operatorname{rel}_{q}(d)}{\sum_{d \in D} \operatorname{rel}_{q}(d)}
$$
where, Precision $_{q, i}$ is the precision computed at rank $i$ for the query $q$. The average precision metric is generally used when relevance judgments are binary, although variants using graded judgments have also been proposed (Robertson et al., 2010). The mean of the average precision over all queries gives the MAP score for the whole set.
\fi

### Mean Average Precision

The metric precision depends on the choice of $k$. Mean Average Precision (MAP) is an aggregate precision measure based on binary label of relevancy and is independent of the choice of $k$. There are several steps to get to MAP.
- First we need to compute precision at $k$ given a query $P(q) @ k$ as:
	$$
	P(q) @ k \equiv \frac{\sum_{i=1}^{k} r_{i}}{k}
	$$
	where $r_i$ is the binary relevance label for each doc in the retrieved list.
-  Then we define the average precision given a query $AP(q)$ at $k$ items as:
	$$
	AP(q) @ k \equiv \frac{1}{\sum_{i=1}^{k} r_{i}} \sum_{i=1}^{k} P(q) @ i \times r_{i}.
	$$
- Finally, MAP is just the mean of $AP(q)$ for all queries:
	$$
	MAP = \frac{\sum_{q\in Q} AP(q)}{|Q|}
	$$

Note that MAP is order-sensitive due to the introduction of the term $r_{i}$ in the calculation of AP. Intuitively, it is doing the average of precision at each ranking position, but penalizing the precision at positions with irrelevant item by strictly setting them to zeroes. MAP typically does not work for fine-grained relevance label other than the binary relevance level. 

\begin{example} Consider two queries and their retrieved document lists:
	$$
	\begin{aligned}
		&q_{1} \rightarrow d_{1}, d_{2} \\
		&q_{2} \rightarrow d_{3}, d_{4}, d_{5}
	\end{aligned}
	$$
	Assuming only $d_{2}, d_{3}, d_{5}$ are relevant document given their corresponding query. We have
	- $\mathrm{AP}$ of query $1: \frac{1}{1} \times\left(\frac{0}{1} \times 0+\frac{1}{2} \times 1\right)=\frac{1}{2}$
- 

	$\mathrm{AP}$ of query $2: \frac{1}{2} \times\left(\frac{1}{1} \times 1+\frac{1}{2} \times 0+\frac{2}{3} \times 1\right)=\frac{5}{6}$
	MAP: $\frac{1}{2} \times\left(\frac{1}{2}+\frac{5}{6}\right) \approx 67 \%$

\end{example}

### Normalized discounted cumulative gain (NDCG)

When we are evaluating retrieval and ranking results based on their relevance to the query, we normally evaluate the ranking result in the following way
- The ranking result is good if documents with high relevance appear in the top several positions in search engine result list.
- We expect documents with different degree of relevance should contribute to the final ranking in proportion to their relevance.

**Cumulative Gain (CG)** is the sum of the graded relevance scores of all documents in the search result list. CG only considers the relevance of the documents in the search result list, and does not consider the position of these documents in the result list. Given the ranking position of a result list, CG can be defined as:

$$CG@k = \sum_{i=1}^k s_i$$
where $s_i$ is the relevance score, or custom defined gain, of document $i$ in the result list. The relevance score of a document is typically provided by human annotators.

**Discounted cumulative gain (DCG)** is the discounted version of CG. The gain is accumulated from the top of the result list to the bottom, with the gain of each result discounted at lower ranks.

The traditional formula of DCG accumulated at a particular rank position $k$ is defined as
$$
DCG@k=\sum_{i=1}^{k} \frac{s_{i}}{\log_{2}(i+1)}=s_{1}+\sum_{i=2}^{k} \frac{s_{i}}{\log _{2}(i+1)}.
$$

An alternative formulation of $DCG$ places stronger emphasis on more relevant documents:
$$
{DCG}@k=\sum_{i=1}^{p} \frac{2^{rel_q(d_i)}-1}{\log _{2}(i+1)}.
$$

The **ideal DCG**, **IDCG**, is computed the same way but by sorting all the candidate  documents in the corpus by their relative relevance so that it produces the max possible DCG@k. The **normalized DCG**, **NDCG**, is then given by,
$$
NDCG@k=\frac{DCG@k}{IDCG@k}.
$$

\begin{example}

Consider 5 candidate documents with respect to a query. Let their ground truth relevance scores be $$s_1=10, s_2=0,s_3=0,s_4=1,s_5=5,$$
which corresponds to a perfect rank of $$s_1, s_5, s_4, s_2, s_3.$$
Let the predicted scores be $$y_1=0.05, y_2=1.1, y_3=1, y_4=0.5, y_5=0.0,$$
which corresponds to rank $$s_2, s_3, s_4, s_1, s_5.$$

For $k=1,2$, we have
$$DCG@k = 0, NDCG@k=0.$$

For $k=3$, we have
$$DCG@k = \frac{s_4}{\log_2 (3+1)} = 0.5, IDCG@k = 10.0 + \frac{5.0}{\log_2 3} + \frac{1.0}{\log_2 4} = 13.65, NDCG@k=0.0366.$$

For $k=4$, we have
$$DCG@k = \frac{s_4}{\log_2 4} + \frac{s_1}{\log_2 5} = 4.807, IDCG@k = IDCG@3 + 0.0 = 13.65, NDCG@k=0.352.$$ 

\end{example}

### Online metrics

When a text ranking model is deployed to serve user's request, we can also measure the model performance by tracking several online metrics.

**Click-through rate** When a user types a query and starts a search session, we can measure the success of a search session on user's reactions. On a per-query level, we can define success via click-through rate.
The click-through rate (CTR) measures the ratio of clicks to impressions.
$$\operatorname{CTR} = \frac{\text{Number of clicks}}{\text{Number of impressions}}$$
where an impression means a page displayed on the search result page a search engine result page and a click means that the user clicks the page.

One problem with the click-through rate is we cannot simply treat a click as the success of document retrieval and ranking. For example, a click might be immediately followed by a click back as the user quickly realizes the clicked doc is not what he is looking for. We can alleviate this issue by removing clicks that have a short dwell time.

**Time to success**: Click-through rate only considers the search session of a single query. In real application case, a user's search experience might span multiple query sessions until he finds what he needs. For example, the users
initially search *action movies* and they do not find that the ideal results and refine the initial query to a more specific one: *action movies by Jackie Chan*. Ideally, we can measure the time spent by the user in identifying the page he wants as a metrics.

## Traditional sparse IR fundamentals

### The exact match framework

Most traditional approaches to ad-hoc retrieval simply count repetitions of the query terms in the document text and assign proper weights to matched terms to calculate a final matching score. This framework, also known as exact term matching, despite its simplicity, serves as a foundation for many IR systems. A variety of traditional IR methods fall into this framework and they mostly differ in different weighting (e.g., tf-idf) and term normalization (e.g., dogs to dog) schemes.

In the exact term matching, we represent a query and a document by a set of their constituent terms, that is, $q = \{t^{q}_1,...,t^q_M\}$ and $d = \{t^{d}_1,...,t^d_M\}$. The matching score between $q$ and $d$ with respect to a vocabulary $\cV$ is given by:
$$
S(q, d)= \sum_{t\in \cV} f(t)\cdot\bm{1}(t\in q\cap d) = \sum_{t \in q \cap d} f(t)
$$
where $f$ is some function of a term and its associated statistics, the three most important of which are 
- Term frequency (how many times a term occurs in a document);
- Document frequency (the number of documents that contain at least once instance of the term);
- Document length (the length of the document that the term occurs in).

Exact term match framework estimates document relevance based on the count of only the query terms in the document. The position of these occurrences and relationship with other terms in the document are ignored.

BM25 are based on exact matching of query and document words, which
limits the in- formation available to the ranking model and may lead to problems such
vocabulary mismatch

### Vector space model

In the vector space model, we represent each query or document by a vector in a high dimensional space. The vector representation has the dimensionality equal to the vocabulary size, and in which each vector component corresponds to a term in the vocabulary of the collection. This query vector representation stands in contrast to the term vector representation of the previous section, which included only the terms appearing in the query. Given a query vector and a set of document vectors, one for each document in the collection, we rank the documents by computing a similarity measure between the query vector and each document

The most commonly used similarity scoring function for a document vector $\vec{d}$ and a query vector $\vec{q}$ is the cosine similarity $\operatorname{Sim}(\vec{d}, \vec{q})$ is computed as
$$
\operatorname{Sim}(\vec{d}, \vec{q})=\frac{\vec{d}}{|\vec{d}|} \cdot \frac{\vec{q}}{|\vec{q}|}.
$$
The component value associated with term $t$ is typically the product of term frequency $tf(t)$ and inverse document frequency $idf(t)$. In addition, cosine similarity has a length normalization component that implicitly handles issues related to document length.

Over the years there have been a number of popular variants for both the TF and the IDF functions have been proposed and evaluated. A basic version of $tf(t)$ is given by
$$
tf(t,d)= \begin{cases}\log \left(f_{t, d}\right)+1 & \text { if } f_{t, d}>0 \\ 0 & \text { otherwise. }\end{cases}
$$
where  $f_{t, d}$ is the actual term frequency count of $t$ in document $d$
Here the basic intuition is that a term appearing many times in a document should be assigned a higher weight for that document, and the its value should not necessarily increase linearly with the actual term frequency $f_{t, d}$, hence the logarithm. Although two occurrences of a term should be given more weight than one occurrence, they shouldn't necessarily be given twice the weight.

A common $idf(t)$ functions is given by
$$
idf(t)=\log \left(N / N_{t}\right)
$$
where $N_t$ is the number of documents in the corpus that contain the term $t$ and $N$ is the total number of documents. Here the basic intuition behind the $idf$ functions is that a term appearing in many documents should be assigned a lower weight than a term appearing in few documents. 

### BM25

One of the most widely adopted exact matching method is called BM25 (short for Okapi BM25)\cite{yang2018anserini, robertson2009probabilistic, croft2010search}. BM25 combines overlapping terms, term-frequency (TF), inverse document frequency (IDF), and document length into following formula
$$
BM25(q, d)=\sum_{t_{q} \in q\cap d} i d f\left(t_{q}\right) \cdot \frac{t f\left(t_{q}, d\right) \cdot\left(k_{1}+1\right)}{t f\left(t_{q}, d\right)+k_{1} \cdot\left(1-b+b \cdot \frac{|d|}{a v g d l}\right)}
$$
where $tf(t_q, d)$ is the query's term frequency in the document $d$, $|d|$ is the length (in terms of words) of document $d$, $avgdl$ is the average length of documents in the collection $D$, and $k_{1}$ and $b$ are parameters that are usually tuned on a validation dataset. In practice, $k_{1}$ is sometimes set to some default value in the range $[1.2,2.0]$ and $b$ as $0.75$. The $i d f(t)$ is computed as,
$$
idf(t)=\log \frac{N-N_t+0.5}{N_t+0.5}.
$$

At first sight, BM25 looks quite like a traditional $tf\times idf$ weight - a product of two components, one based on $tf$ and one on $idf$. However, there is one significant difference. The $tf$ component in the BM25 uses some saturation mechanism to discount the impact of frequent terms in a document. 
Intuitively, a document $d$ has a higher BM25 score if 
- Many query terms also frequently occur in the document; 
- These frequent co-occurring terms have larger idf values (i.e., they are not common terms). 

BM25 does not concerns with word semantics, that is whether the word is
a noun or a verb, or the meaning of each word. It is only sensitive to which are common words and which are rare words, and the document length. If one query contains both common words and rare words, this method puts more weight on the rare words and returns documents with more rare words in the query. Besides, a term saturation mechanism is applied to decrease the matching signal when a matched word appears too frequently in the document. A document-length normalization mechanism is used to discount term weight when a document is longer than average documents in the collection. 

Two parameters in BM25, $k_1$ and $b$, are designed to perform **term frequency saturation**
and **document-length normalization**,respectively. 
- The constant $k_{1}$ determines how the $tf$ component of the term weight changes as the frequency increases. If $k_{1}=0$, the term frequency component would be ignored and only term presence or absence would matter. If $k_{1}$ is large, the term weight component would increase nearly linearly with the frequency. 
- The constant $b$ regulates the impact of the length normalization, where $b=0$ corresponds to no length normalization, and $b=1$ is full normalization. 

If the query is long, then we might also use similar weighting for query terms. This is appropriate if the queries are paragraph-long information needs, but unnecessary for short queries.
$$
BM25(q, d)=\sum_{t_q \in q\cap d} idf(t_q) \cdot \frac{\left(k_{1}+1\right) tf(t_q,d) }{k_{1}\left((1-b)+b \times  |d|/avgdl\right)+tf(t_q,d)} \cdot \frac{\left(k_{3}+1\right) tf(t_q,q)}{k_{2}+tf(t_q,q)}
$$
with $tf(t_q, q)$ being the frequency of term $t$ in the query $q$, and $k_{2}$ being another positive tuning parameter that this time calibrates term frequency scaling of the query. 

\begin{example}
Consider the following example from {cite}`croft2010search`. Consider a query with two terms, *president* and *lincoln*, each of which occurs only once in the query $(tf(t,q)=1)$. 

Let's assume that the collection we are searching has $N=500,000$ documents, and that in this collection, *president* occurs in 40,000 documents $\left(N_{1}=40,000\right)$ and *lincoln* occurs in 300 documents $\left(N_{2}=300\right)$. 

Consider a document in which *president* occurring 15 times $\left(f_{1}=15\right)$ and *lincoln* occurring 25 times $\left(f_{2}=25\right)$. The document length is $90 \%$ of the average length $(dl /avdl=0.9)$. The parameter values we use are $k_{1}=1.2, b=0.75$, and $k_{2}=100$. With these values, 
$$K=k_{1}\left((1-b)+b \times  |d|/avgdl\right)=1.2 \cdot(0.25+0.75 \cdot 0.9)=1.11,$$ and the document BM25 score is:
```{math}
\begin{align*}
BM25(Q, D)&=\\
	&\log \frac{(0+0.5) /(0-0+0.5)}{(40000-0+0.5) /(500000-40000-0+0+0.5)}\times \frac{(1.2+1) 15}{1.11+15}\\
	&+\log \frac{(0+0.5) /(0-0+0.5)}{(300-0+0.5) /(500000-300-0+0+0.5)}\times \frac{(1.2+1) 25}{1.11+25} \\
	&=\log 460000.5 / 40000.5 \cdot 33 / 16.11 +\log 499700.5 / 300.5 \cdot 55 / 26.11 \cdot 101 / 101\\
	&=2.44 \cdot 2.05 +7.42 \cdot 2.11 \\
	&=5.00+15.66=20.66
\end{align*}
```
\end{example}

\iffalse

### Multiple streams and BM25F

In the classical BM25 model, we simply treat each document as a single body of text without differentiating internal structures such as titles, abstracts and main body in the document. In the Web context, with extensive hyperlinks, it is usual to enhance the original texts with the anchor text of incoming links.

It can be beneficial to take into account the structure of documents into account. a query match on the title might be expected to provide stronger evidence of possible relevance than an equivalent match on the body text. It is now well known in the Web context that matching on anchor text is a very strong signal.

More formally, we assume all documents which are structured into a set of $S$ streams. The assumption is that there is a single flat stream structure, common to all documents. 

We have a set of $S$ streams, and we wish to assign relative weights $v_{s}$ to them. For a given document, each stream has its associated length (the total length of the document would normally be the sum of the stream lengths). Each term in the document may occur in any of the streams, with any frequency; the total across streams of these term stream frequencies would be the usual term-document frequency. The entire document becomes a vector of vectors.

- streams $\quad \bar{s}=1, \ldots, S$
- stream lengths $l_{s}$
- stream weights $v_{s}$
- document $\left(\mathbf{t f}_{1}, \ldots, \mathbf{t f}_{|\mathbf{V}|}\right)$ vector of vectors
- $\mathbf{t f}_{i}$ vector $\left(t f_{1 i}, \ldots, t f_{S i}\right)$

where $t f_{s i}$ is the frequency of term $i$ in stream $s$.

The simplest extension of BM25 to weighted streams is to calculate a weighted variant of the total term frequency. This also implies having a similarly weighted variant of the total document length.
```{math}
\begin{align*}
\widetilde{t f}_{i} &=\sum_{s=1}^{S} v_{s} t f_{s i} \\
	\widetilde{d l} &=\sum_{s=1}^{S} v_{s} s l_{s} \\
	\widetilde{a v d l} &=\text { average of } \tilde{d} l \text { across documents } \\
	w_{i}^{\text {simpleBM25F }} &=\frac{\widetilde{t f}_{i}}{k_{1}\left((1-b)+b \frac{\widetilde{d} l}{a v d l}\right)+\widetilde{t f_{i}}} w_{i}^{\mathrm{RSJ}}
\end{align*}
```

\begin{remark}
It is worth mentioning a very transparent interpretation of the simple version - although it does not apply directly to the version with variable $b$, it may give some insight. If the stream weights $v_{s}$ are integers, then we can see the simple BM25F formula as an ordinary BM25 function applied to a document in which some of the streams have been replicated. For example, if the streams and weights are $\left\{v_{\text {title }}=5\right.$, $v_{\text {abstract }}=2$, $\left.v_{\text {body }}=1\right\}$, then formula $3.18$ is equivalent to $3.15$ applied to a document in which the title has been replicated five times and the abstract twice.
\end{remark}

\fi

\iffalse
## Sparse IR serving
(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:traditional_inverted_indexing)=
### Inverted indexing

```{index} inverted index
```

Exact match models not only bring huge success to traditional IR systems but also remain indispensable in modern IR system using neural match and ranking models. One key advantage of exact match lies in efficiency. The extreme fast lookup of query terms in a billion-scale document pool can be efficiently implemented using **inverted indexing**. \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:traditional_invertedlists} shows one simple implementation based inverted lists. The inverted list maps back from a term to a list of documents where the term occurs, together with additional information associated with the term in the document (e.g., term frequency, or tf). With summation over exact matches, scoring of each query term only goes to documents that contain matching terms. There are advanced data structures like trees to support efficient traversal and Boolean operations. 

```{figure} images/../deepLearning/ApplicationIRSearch/TraditionalIR/inverted_lists
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:traditional_invertedlists
An illustration of implementing inverted indexing using lists.
```

An incoming query will be decomposed into different terms and retrieve the list of documents appending each matched term. These documents are then union together and sent to subsequent stages for further ranking.

### Document term generation

\begin{remark}[normalization]
Need to normalise words in the indexed text as well as query terms to the same form

Example: We want to match U.S.A. to USA

We most commonly implicitly define equivalence classes of terms.

Alternatively, we could do asymmetric expansion:
window $\rightarrow$ window, windows
windows $\rightarrow$ Windows,
windows, window
Windows $\rightarrow$ Windows

Either at query time, or at index time

More powerful, but less efficient

\end{remark}

\begin{remark}[multilingual complications]
A single index usually contains terms of several languages.
Documents or their components can contain multiple
languages/format, for instance a French email with a Spanish pdf attachment
What is the document unit for indexing?
- a file?
- an email?
- an email with 5 attachments?
- an email thread?
Answering the question "What is a document?" is not trivial.
Smaller units raise precision, drop recall
Also might have to deal with XML/hierarchies of HTML documents etc.
\end{remark}

\begin{remark}[index numbers]
Date 20/3/91, Item name B-52, IP address 100.2.86.144, phone number 800-234-2333 	

Older IR systems may not index numbers
\end{remark}

\begin{remark}[word segmentation]
For language like Chinese, need to perform word segmentation.

For language like Dutch, German, Swedish, there are compounding words without whitespace.

For example, for German compounding word
*Lebensversicherungsgesellschaftsangestellter*, it should be segmented into 
leben, s, versicherung, s, gesellschaft, s, angestellter
(in English: *Life insurance company employee*.)

\end{remark}

#### Document offline ranking

The original design of inverted indexing is that each term will have a list that contains all the documents having this term. This design does not scale well when the collection of documents are large and constantly increasing. 

#### Online query processing

\fi

\iffalse
### RM3

\cite{lavrenko2017relevance}

RM3 [20] is a popular query expansion technique. It adds related terms to the query to compensate for the vocabulary gap between queries and documents. BM25+RM3 has been proven to be strong {cite}`lin2019neural`.

### Probabilistic Naive Bayes retrieval

Let the binary vector of document $d$ represented by vector $\bm{x}$, $\bm{x} = (x_1, ..., x_M) \in \{0,1\}^{M}$, where $x_t = 1$ if term $t$ is present in document $d$ and $x_t = 0$ if $t$ is not present in $d$, and $M$ is the vocabulary size.

Similarly, we can represent a query by $\bm{q} = (q_1,...,q_M)$. The odds that a pair of $(\bm{x}, \bm{q})$ is relevant is expressed as

$$O(R|\bm{x}, \bm{q}) = \frac{P(R=1|\bm{x}, \bm{q})}{P(R=0|\bm{x}, \bm{q})},$$
where $R$ is a binary label: 0 is irrelevant and 1 is relevant.

Using Bayes theorem, we have
```{math}
\begin{align*}
&P(R=1 \mid \bm{x}, \bm{q})=\frac{P(\bm{x} \mid R=1, \bm{q}) P(R=1 \mid \bm{q})}{P(\bm{x} \mid \bm{q})} \\
	&P(R=0 \mid \bm{x}, \bm{q})=\frac{P(\bm{x} \mid R=0, \bm{q}) P(R=0 \mid \bm{q})}{P(\bm{x} \mid \bm{q})}
\end{align*}
```

Now the relevance odds can be expressed as

$$O(R|\bm{x}, \bm{q}) = \frac{P(R=1 \mid \bm{q})}{P(R=0 \mid \bm{q})} \cdot \frac{P(R=1|\bm{q})}{P(R=0|\bm{q})} \cdot \frac{P(\bm{x} \mid R=1, \bm{q})}{P(\bm{x} \mid R=0, \bm{q})} = O(R|\bm{q}) \cdot\frac{P(\bm{x} \mid R=1, \bm{q})}{P(\bm{x} \mid R=0, \bm{q})}.$$
where $O(R|\bm{q})$ is the marginalized odds: given the query $\bm{q}$, the odd that it is relevant to a document.

To compute $P(\bm{x} \mid R=0, \bm{q})$ and $P(\bm{x} \mid R=1, \bm{q})$, we can further make the **Naive Bayes conditional independence** assumption that the presence or absence of a word in a document is independent of the presence or absence of any other word (given the query):
$$
\frac{P(\bm{x} \mid R=1, \bm{q})}{P(\bm{x} \mid R=0, \bm{q})}=\prod_{t=1}^{M} \frac{P\left(x_{t} \mid R=1, \bm{q}\right)}{P\left(x_{t} \mid R=0, \bm{q}\right)}
$$
So:
$$
O(R \mid \bm{x}, {q})=O(R \mid \bm{q}) \cdot \prod_{t=1}^{M} \frac{P\left(x_{t} \mid R=1, \bm{q}\right)}{P\left(x_{t} \mid R=0, \bm{q}\right)}
$$
Because each $x_{t}$ is either 0 or 1 , we can separate the terms to give:
$$
O(R \mid {x}, \bm{q})=O(R \mid \bm{q}) \cdot \prod_{t: x_{t}=1} \frac{P\left(x_{t}=1 \mid R=1, \bm{q}\right)}{P\left(x_{t}=1 \mid R=0, \bm{q}\right)} \cdot \prod_{t: x_{t}=0} \frac{P\left(x_{t}=0 \mid R=1, \bm{q}\right)}{P\left(x_{t}=0 \mid R=0, \bm{q}\right)} .
$$
Henceforth, let $p_{t}=P\left(x_{t}=1 \mid R=1, \bm{q}\right)$ be the probability of a term appearing in a document relevant to the query, and $u_t = P\left(x_{t}=1 \mid R=0, \bm{q}\right)$ be the probability of a term appearing in a irrelevant document. 

Then we have
$$O(R \mid \bm{q}, {x})=O(R \mid \bm{q}) \cdot \prod_{t: x_{t}=1} \frac{p_{t}}{u_{t}} \cdot \prod_{t: x_{t}=0} \frac{1-p_{t}}{1-u_{t}}$$

An additional simplification can be achieved by assuming that terms not occuring in the query are equally likely to occur in relevant and irrelevant documents; that is, if $q_t$ = 0 then $p_t$ = $u_t$.

Then we arrive at 
$$O(R \mid \bm{q}, {x})=O(R \mid \bm{q}) \cdot \prod_{t: x_{t}=1, q_t=1} \frac{p_{t}}{u_{t}} \cdot \prod_{t: x_{t}=0, q_t=1} \frac{1-p_{t}}{1-u_{t}}$$

The interpretation of the relevance odds is:
- A pair of query and document are more likely to be relevant if terms occurring in the query also frequently occurring in the document (e.g., large $p_t$).
- A pair of query and document are more likely to be irrelevant if terms occurring in the query rarely occur in the document (e.g., large $u_t$).

 ### Language-model based approach

 #### Framework

Language modeling is a quite general formal approach to IR, with many variant realizations. The original and basic method for using language models in IR is the query likelihood model. In it, we construct from each document $d$ in the collection a language model $M_{d}$. Our goal is to rank documents by $P(d \mid q)$, where the probability of a document is interpreted as the likelihood that it is relevant to the query. Using Bayes rule (as introduced in probirsec), we have:
$$
P(d \mid q)=P(q \mid d) P(d) / P(q)
$$
$P(q)$ is the same for all documents, and so can be ignored. The prior probability of a document $P(d)$ is often treated as uniform across all $d$ and so it can also be ignored, but we could implement a genuine prior which could include criteria like authority, length, genre, newness, and number of previous people who have read the document. But, given these simplifications, we return results ranked by simply $P(q \mid d)$, the probability of the query $q$ under the language model derived from $d$. The Language Modeling approach thus attempts to model the query generation process:
Documents are ranked by the probability that a query would be observed as a random sample from the respective document model.

#### Unigram language model

The most common way to do this is using the multinomial unigram language model, which is equivalent to a multinomial Naive Bayes model (page $*13.3*$ ), where the documents are the classes, each treated in the estimation as a separate "language". Under this model, we have that:
$$
\mathrm{P}\left(\mathrm{q} \mid \mathrm{M}_{d}\right)=K_{q} \prod_{t \in V} P\left(t \mid M_{d}\right)^{\mathrm{tf}_{t, d}}
$$
where, again $K_{q}=L_{d} ! /\left(\mathrm{tf}_{t_{1}, d} ! \mathrm{tf}_{t_{2}, d} ! \cdots \mathrm{tf}_{t_{M}, d} !\right)$ is the multinomial coefficient for the query $q$, which we will henceforth ignore, since it is a constant for a particular query.
For retrieval based on a language model (henceforth $L M$ ), we treat the generation of queries as a random process. The approach is to
1. Infer a LM for each document.
2. Estimate $P\left(q \mid M_{d_{i}}\right)$, the probability of generating the query according to each of these document models.
3. Rank the documents according to these probabilities.
The intuition of the basic model is that the user has a prototype document in mind, and generates a query based on words that appear in this document. Often, users have a reasonable idea of terms that are likely to occur in documents of interest and they will choose query terms that distinguish these documents from others in the collection. Collection statistics are an integral part of the language model, rather than being used heuristically as in many other approaches.

### Challenges and approaches

These are just two examples of the "vocabulary mismatch problem" [Furnas et al., 1987], which represents a fundamental challenge in information retrieval. There are three general approaches to tackling this challenge: enrich query representations to better match document representations, enrich document representations to better match query representations, and attempts to go beyond exact term matching:

#### Enriching query representations

One obvious approach to bridge the gap between query and document terms is to enrich query representations with query expansion techniques [Carpineto and Romano, 2012]. In relevance feedback, the representation of the user's query is augmented with terms derived from documents that are known to be relevant (for example, documents that have been presented to the user and that the user has indicated is relevant): two popular formulations are based on the vector space model [Rocchio, 1971] and the probabilistic retrieval framework [Robertson and Spark Jones, 1976]. 

In pseudo-relevance feedback [Croft and Harper, 1979], also called "blind" relevance feedback, top-ranking documents are simply assumed to be relevant, thus providing a source for additional query terms. 

Query expansion techniques, however, do not need to involve relevance feedback: examples include Xu and Croft [2000], who introduced global techniques that identify word relations from the entire collection as possible expansion terms (this occurs in a corpus preprocessing step, independent of any queries), and Voorhees [1994], who experimented with query expansion using lexical-semantic relations from WordNet [Miller, 1995]. A useful distinction when discussing query expansion techniques is the dichotomy between pre-retrieval techniques, where expansion terms can be computed without examining any documents from the collection, and post-retrieval techniques, which are based on analyses of documents from an initial retrieval. 

#### Enriching document representations

 Another obvious approach to bridge the gap between query and document terms is to enrich document representations. This strategy works well for noisy transcriptions of speech [Singhal and Pereira, 1999] and short texts such as tweets [Efron et al., 2012]. Although not as popular as query expansion techniques, researchers nevertheless explored this approach throughout the $1980 \mathrm{&nbsp;s}$ and $1990 \mathrm{&nbsp;s}$ [Salton and Buckley, 1988b, Voorhees and Hou, 1993]. The origins of document expansion trace even earlier to Kwok [1975], who took advantage of bibliographic metadata for expansion, and finally, Brauen et al. [1968], who used previously issued user queries to modify the vector representation of a relevant document. 

#### Semantic match
Other examples of attempts to go beyond exact match include techniques that attempt to perform matching in some semantic space induced from data, for example, based on latent semantic analysis [Deerwester et al., 1990] or latent Dirichlet allocation [Wei and Croft, 2006].

At a high level, retrieval models up until this time contrast with “soft” or semantic matching enabled by continuous representations in neural networks, where query terms do not have to match document terms exactly in order to contribute to relevance. Semantic matching refers to techniques and attempts to address a variety of linguistic phenomena, including **synonymy, paraphrase, term variation, and different expressions of similar intents**, specifically in the context of information access [Li and Xu, 2014]. Following this usage, “relevance matching” is often used to describe the correspondences between queries and texts that account for a text being relevant to a query (see Section 2.2). Thus, relevance matching is generally understood to comprise both exact match and semantic match components. However, there is another major phase in the development of ranking techniques before we get to semantic matching and how neural networks accomplish it.

\fi

## Classic semantic dense retrieval models
### Overview

#### Motivation

For ad-hoc search, traditional exact-term matching models (e.g., BM25) are playing critical roles in both traditional IR systems [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:traditionalirengine}] and modern multi-stage pipelines [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:retrieverankingarch}]. Unfortunately, exact-term matching inherently suffers from the vocabulary mismatch problem due to the fact that a concept is often expressed using different vocabularies and language styles in documents and queries.

Early latent semantic models such as latent semantic analysis (LSA) illustrated the idea of identifying semantically relevant documents for a query when lexical matching is insufficient. However, their effectiveness in addressing the language discrepancy between documents and search queries are limited by their weak modeling capacity (i.e., simple, linear models). Also, these model parameters are typically learned via the unsupervised learning, i.e., by grouping different terms that occur in a similar context into the same semantic cluster. 

The introduction of deep neural networks for semantic modeling and retrieval was pioneered in {cite}`huang2013learning`. Recent deep learning model utilize the neural networks with large learning capacity and user-interaction data for supervised learning, which has led to significance performance gain over LSA.  Similarly in the field of OpenQA {cite}`karpukhin2020dense`, whose first stage is to retrieve relevant passages that might contain the answer, semantic-based retrieval has also demonstrated performance gains over traditional retrieval methods. 

\iffalse
\begin{remark}
	Most NLP tasks concern semantic matching, i.e., identifying the semantic meaning and infer- ring the semantic relations between two pieces of text, while the ad-hoc retrieval task is mainly about relevance match- ing, i.e., identifying whether a document is relevant to a given query.

	In this section, we discuss the differences between text
	matching in ad-hoc retrieval and other NLP tasks. The matching in many NLP tasks, such as paraphrase identi- fication, question answering and automatic conversation, is mainly concerned with semantic matching, i.e., identifying the semantic meaning and inferring the semantic relations between two pieces of text. In these semantic matching tasks, the two texts are usually homogeneous and consist of a few natural language sentences, such as questions/answer sentences, or dialogs. To infer the semantic relations between natural language sentences, semantic matching emphasizes the following three factors:

	**Similarity matching signals**: It is important, or critical to capture the semantic similarity/relatedness between words, phrases and sentences, as compared with exact matching signals. For example, in paraphrase identification, one needs to identify whether two sentences convey the same meaning with different expressions. In automatic conversation, one aims to find a proper response semantically related to the previous dialog, which may not share any common words or phrases between them.

	**Compositional meanings**: Since texts in semantic matching usually consist of natural language sentences with grammatical structures, it is more beneficial to use the compositional meaning of the sentences based on such grammatical structures rather than treating them as a set/sequence of words [25]. For example, in question answering, most questions have clear grammatical structures which can help identify the compositional meaning that reflects what the question is about.

	**Global matching requirement**: Semantic matching usually treats the two pieces of text as a whole to infer the semantic relations between them, leading to a global matching requirement. This is partially related to the fact that most texts in semantic matching have limited lengths and thus the topic scope is concentrated. For example, two sentences are considered as paraphrases if the whole meaning is the same, and a good answer fully answers the question.
\end{remark}
\fi

#### Two architecture paradigms

The current neural  architecture paradigms for IR can be categorized into two classes: **representation-based** and **interaction-based** [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:twoparadigms}]. 

In the representation-based architecture, a query and a document are encoded independently into two embedding vectors, then their relevance is estimated based on a single similarity score between the two embedding vectors. 

Here we would like to make a critical distinction on symmetric vs. asymmetric encoding:
- For **symmetric encoding**, the query and the entries in the corpus are typically of the similar length and have the same amount of content and they are encoded using the same network. Symmetric encoding is used for symmetric semantic search. An example would be searching for similar questions. For instance, the query could be *How to learn Python online?* and the entry that satisfies the search is like *How to learn Python on the web?*. 
- For **asymmetric encoding**, we usually have a short query (like a question or some keywords) and we would like to find a longer paragraph answering the query; they are encoded using two different networks. An example would be information retrieval. The entry is typically a paragraph or a web-page.

In the interaction-based architecture, instead of directly encoding $q$ and $d$ into individual embeddings, term-level interaction features across the query and the document are first constructed. Then a deep neural network is used to extract high-level matching features from the interactions and produce a final relevance score.

```{figure} images/../deepLearning/ApplicationIRSearch/DeepRetrievalModels/Two_paradigms
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:twoparadigms
Two common architectural paradigms in semantic retrieval learning: representation-based learning (left) and interaction-based learning (right).
```

These two architectures have different strengths in modeling relevance and final model serving. For example, a representation-based model architecture makes it possible to pre-compute and cache document representations offline, greatly reducing the online computational load per query. However, the pre-computation of query-independent document representations often miss term-level matching features that are critical to construct high-quality retrieval results. On the other hand, interaction-based architectures are often good at capturing the fine-grained matching feature between the query and the document. 

Since interaction-based models can model interactions between word pairs in queries and document, they are effective for re-ranking, but are cost-prohibitive for first-stage retrieval as the expensive document-query interactions must be computed online for all ranked documents.

Representation-based models enable low-latency, full-collection retrieval with a dense index. By representing queries and documents with dense vectors, retrieval is reduced to nearest neighbor search, or a maximum inner product search (MIPS) {cite}`shrivastava2014asymmetric` problem if similarity is represented by an inner product.

In recent years, there has been increasing effort on accelerating maximum inner product and nearest neighbor search, which led to high-quality implementations of libraries for nearest neighbor search such as hnsw {cite}`malkov2018efficient`, FAISS {cite}`johnson2019billion`, and SCaNN {cite}`guo2020accelerating`. 

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:classicRepresentationLearning)=
### Classic representation-based learning

#### DSSM

Deep structured semantic model (DSSM) {cite}`huang2013learning` improves the previous latent semantic models in two aspects: 1) DSSM is supervised learning based on labeled data, while latent semantic models are unsupervised learning; 2) DSSM utilize deep neural networks to capture more semantic meanings. 

The high-level architecture of DSSM is illustrated in \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:dssm}. First, we represent a query and a document (only its title) by a sparse vector, respectively. Second, we apply a non-linear projection to map the query and the document sparse vectors to two low-dimensional embedding vectors in a common semantic space. Finally, the relevance of each document given the query is calculated as the cosine similarity between their embedding vectors in that semantic space. 

```{figure} images/../deepLearning/ApplicationIRSearch/DeepRetrievalModels/DSSM/dssm
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:dssm
The architecture of DSSM. Two MLP encoders with shared parameters are used to encode a query and a document into dense vectors. Query and document are both represented by term vectors. The final relevance score is computed via dot product between the query vector and the document vector.
```

To represent word features in the query and the documents, DSSM adopt a word level sparse term vector representation with letter 3-gram vocabulary, whose size is approximately $30k \approx 30^3$. Here 30 is the approximate number of alphabet letters. This is also known as a letter trigram word hashing technique. In other words, both query and the documents will be represented by sparse vectors with dimensionality of $30k$.

The usage of letter 3-gram vocabulary has multiple benefits compared to the full vocabulary:
- Avoid OOV problem with finite-size vocabulary or term vector dimensionality.
- The use of letter n-gram can capture morphological meanings of words.

One problem of this method is collision, i.e., two different
words could have the same letter n-gram vector representation because this is a bag-of-words representation that does not take into account orders. But the collision probability is rather low.

\begin{table}[H]
	\scriptsize
	\centering
	\begin{tabular}{l|c|c|c|c}
		\hline & \multicolumn{2}{c}{ Letter-Bigram } & \multicolumn{2}{c}{ Letter-Trigram } \\
		\hline Word Size & Token Size & Collision & Token Size & Collision \\
		\hline $40k$ & 1107 & 18 & 10306 & 2 \\
		\hline $500k$ & 1607 & 1192 & 30621 & 22 \\
		\hline
	\end{tabular}
	\caption{Word hashing token size and collision numbers as a function of the vocabulary size and the type of letter ngrams.}
\end{table}

**Training**. The neural network model is trained on the clickthrough data to map a query and its relevant document to vectors that are similar to each other and vice versa. The click-through logs consist of a list of queries and their clicked documents. It is assumed that a query is relevant, at least partially, to the documents that are clicked on for that query. 

The semantic relevance score between a query $q$ and a document $d$ is given by:
$$
R(q, d)=\operatorname{Cosine}\left(y_{Q}, y_{D}\right)
$$
where $E_{q}$ and $E_{q}$ are the embedding vectors of the query and the document, respectively. The conditional probability of a document being relevant to a given query is now defined through a Softmax function
$$
P(d \mid q)=\frac{\exp (\gamma R(q, d))}{\sum_{d^{\prime} \in D} \exp \left(\gamma R\left(q, d^{\prime}\right)\right)}
$$
where $\gamma$ is a smoothing factor as a hyperparameter. $D$ denotes the set of candidate documents to be ranked. While $D$ should ideally contain all possible documents in the corpus, in practice, for each query $q$, $D$ is approximated by including the clicked document $d^{+}$ and four randomly selected un-clicked documents.

In training, the model parameters are estimated to maximize the likelihood of the clicked documents given the queries across the training set. Equivalently, we need to minimize the following loss function
$$
L=-\sum_{(q, d^+)}\log P\left(d^{+} \mid q\right).
$$

**Evaluation** DSSM is compared with baselines of traditional IR models like TF-IDF, BM25, and LSA. Specifically, the best performing DNN-based semantic model, L-WH DNN, uses three hidden layers, including the layer with the Letter-trigram-based Word Hashing, and an output layer, and is discriminatively trained on query-title pairs.

{
\begin{tabular}{c|c|c|c}
	\hline Models & NDCG@1 & NDCG@3 & NDCG@10 \\
	\hline \hline TF-IDF & $0.319$ & $0.382$ & $0.462$ \\
	\hline BM25 & $0.308$ & $0.373$ & $0.455$ \\
	\hline LSA & $0.298$ & $0.372$ & $0.455$ \\
	\hline L-WH DNN & $\mathbf{0.362}$ & $\mathbf{0.425}$ & $\mathbf{0.498}$ \\
	\hline
\end{tabular}	
}

#### CNN-DSSM

DSSM treats a query or a document as a bag of words, the fine-grained contextual structures embedding in the word order are lost. The DSSM-CNN\cite{shen2014latent} [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:cnndssm}] directly represents local contextual features at the word n-gram level; i.e., it projects each raw word n-gram to a low dimensional feature vector where semantically similar word $\mathrm{n}$ grams are projected to vectors that are close to each other in this feature space. 

Moreover, instead of simply summing all local word-n-gram features evenly, the DSSM-CNN performs a max pooling operation to select the highest neuron activation value across all word n-gram features at each dimension. This amounts to extract the sentence-level salient semantic concepts. 

Meanwhile, for any sequence of words, this operation forms a fixed-length sentence level feature vector, with the same dimensionality as that of the local word n-gram features.

Given the letter-trigram based word representation, we represent a word-n-gram by concatenating the letter-trigram vectors of each word, e.g., for the $t$-th word-n-gram at the word-ngram layer, we have:
$$
l_{t}=\left[f_{t-d}^{T}, \ldots, f_{t}^{T}, \ldots, f_{t+d}^{T}\right]^{T}, \quad t=1, \ldots, T
$$
where $f_{t}$ is the letter-trigram representation of the $t$-th word, and $n=2 d+1$ is the size of the contextual window. In our experiment, there are about $30 \mathrm{&nbsp;K}$ unique letter-trigrams observed in the training set after the data are lower-cased and punctuation removed. Therefore, the letter-trigram layer has a dimensionality of $n \times 30 K$.

```{figure} images/../deepLearning/ApplicationIRSearch/DeepRetrievalModels/CNN_DSSM/CNN_DSSM
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:cnndssm
The architecture of CNN-DSSM. Each term together with its left and right contextual words are encoded together into term vectors. 
```

### Classic interaction-based matching

\iffalse
#### Motivations

Representation-based learning does a job in semantic matching, i.e., 
identifying the semantic meaning and infer- ring the semantic relations between two pieces of text. While some NLP tasks (e.g., sentiment analysis, sentence similarity task) concern semantic matching, the ad-hoc retrieval task sometimes not just require semantic matching but also exact term matching, particularly when the query contain specific, rare words (e.g., a city's name). 

Representation based matching defers the interaction between two sentences to until their individual representation matures (at the output of the encoders), therefore runs at the risk of losing details (e.g., a city name) important for the matching task in representing the sentences.

Successful relevance matching requires proper handling of the exact matching signals, query term importance, and di- verse matching requirements. 

The matching in ad-hoc retrieval, on the contrary, is mainly about relevance matching, i.e., identifying whether a document is relevant to a given query. In this task, the query is typically short and keyword based, while the document can vary considerably in length, from tens of words to thousands or even tens of thousands of words. To estimate the relevance between a query and a document, relevance matching is focused on the following three factors:

Exact matching signals: Although term mismatch is a critical problem in ad-hoc retrieval and has been tackled using different semantic similarity signals, the exact matching of terms in documents with those in queries is still the most important signal in ad-hoc retrieval due to the indexing and search paradigm in modern search engines. For example, Fang and Zhai [7] proposed the semantic term matching constraint which states that matching an original query term exactly should always contribute no less to the relevance score than matching a semantically related term multiple times. than matching a semantically related term multiple times. This also explains why some traditional retrieval models, e.g., BM25, can work reasonably well purely based on exact matching signals.

Query term importance: Since queries are mainly short and keyword based without complex grammatical structures in ad-hoc retrieval, it is important to take into account term importance, while the compositional relation among the query terms is usually the simple "and" relation in operational search. For example, given the query "bitcoin news", a relevant document is expected to be about "bitcoin" and "news", where the term "bitcoin" is more important than "news" in the sense that a document describing other aspects of "bitcoin" would be more relevant than a document describing "news" of other things. In the literature, there have been many formal studies on retrieval models showing the importance of term discrimination $[\overline{[}, \mathbb{6}]$.

Diverse matching requirement: In ad-hoc retrieval, a relevant document can be very long and there have been different hypotheses concerning document length [22] in the literature, leading to a diverse matching requirement. Specifically, the Verbosity Hypothesis assumes that a long document is like a short document, covering a similar scope but with more words. In this case, the relevance matching might be global if we assume short documents have a concentrated topic. On the contrary, the Scope Hypothesis assumes a long document consists of a number of unrelated short documents concatenated together. In this way, the relevance matching could happen in any part of a relevant document, and we do not require the document as a whole to be relevant to a query.

\fi

#### DRMM

```{figure} images/../deepLearning/ApplicationIRSearch/DeepRetrievalModels/DRMM/DRMM
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:drmm
The architecture of DRMM.
```

The deep relevance matching model (DRMM)\cite{guo2016deep} is an interaction-focused model which employs a deep neural network architecture \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:drmm} to extract high-level matching features from term level interaction features.  

Specifically, the model first constructs local interactions between each pair of terms from a query and a document based on term embeddings (e.g., word2vec). The interaction score can be as simple as the cosine similarity score. For each query term, we transform its interaction scores with terms in the document into a fixed-length matching histogram on range of similarity score (e.g., the range will be [-1, 1] for cosine similarity score). Based on this fixed-length matching histogram, we then employ a feed forward neural network to extract high-level matching patterns and finally to produce a matching score. 

Note that there is one matching score associated with each term in the query and the overall matching score is generated by aggregating all the matching scores using a term gating mechanism.

A term Gating Network is used to explicitly incorporate query term importance. The term gating network produces an aggregation weight for each query term controlling how much the relevance score on that query term contributes to the final relevance score. Specifically, the Softmax function is employed as the gating function.
$$
g_{i}=\frac{\exp \left(\boldsymbol{w}_{g} \boldsymbol{x}_{i}^{(q)}\right)}{\sum_{j=1}^{M} \exp \left(\boldsymbol{w}_{g} \boldsymbol{x}_{j}^{(q)}\right)}, \quad i=1, \ldots, M
$$
where ${w}_{g}$ is network weight vector and ${x}_{i}^{(q)}, i=1, \ldots, M$ denotes the input feature of the $i$-th query term. The input feature can be either the word embedding or the idf (inverse document frequency) of the term. 

**Training** The DRMM can trained using hinge loss on relevance data collected from search logs. Given a triple $\left(q, d^{+}, d^{-}\right)$, where document $d^{+}$is ranked higher than document $d^{-}$with respect to query $q$, the loss function is defined as:
$$
\mathcal{L}\left(q, d^{+}, d^{-}\right)=\max \left(0,1-s\left(q, d^{+}\right)+s\left(q, d^{-}\right)\right)
$$
where $s(q, d)$ denotes the predicted matching score for $(q, d)$.

#### KNRM

The Deep Relevance Matching Model (DRMM) in the previous section uses histograms and deep neural network as feature extractors to summarize the word-level similarities into ranking signals. The word level similarities are calculated from pre-trained word2vec embeddings, and the histogram counts the number of word pairs at different similarity levels. However, because the histogram counts are not differentiable, word2vec embeddings cannot be adjusted during the training. 

Kernel-based neural ranking model (KNRM)\cite{xiong2017end} construct similar interaction matrix as DRMM, but employs learnable Gaussian kernels to create differentiable *soft histograms*. Using learnable Gaussian kernels allows the embeddings to be learned during training.

The key component in the KNRM architecture [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:knrm}] is the kernel pooling, which uses kernels to convert term-level interaction matrix $M$ to query-document ranking features $\phi(M)$. Specifically,
```{math}
\begin{align*}
\phi(M) &=\sum_{i=1}^{n} \log \vec{K}\left(M_{i}\right) \\
	\vec{K}\left(M_{i}\right) &=\left\{K_{1}\left(M_{i}\right), \ldots, K_{K}\left(M_{i}\right)\right\}
\end{align*}
```
$\vec{K}\left(M_{i}\right)$ applies $K$ kernels to the $i$-th query word's row of the interaction matrix, summarizing (pooling) it into a $K$-dimensional feature vector. The log-sum of each query word's feature vector forms the query-document ranking feature vector $\phi$. The kernel can simply be an RBF kernel:
$$
K_{k}\left(M_{i}\right)=\sum_{j} \exp \left(-\frac{\left(M_{i j}-\mu_{k}\right)^{2}}{2 \sigma_{k}^{2}}\right) .
$$

By employing kernels of different parameters, the kernel pooling can implement feature extractions ranging from exact matches (using $\mu_{0}=1.0$ and $\sigma=10^{-3} \cdot \mu$) to other soft matches (using the other 10 kernels taking values like $\mu_{1}=0.9, \mu_{2}=0.7, \ldots, \mu_{10}=-0.9$). 

The ranking features $\phi(M)$ are combined by a ranking layer to produce the final ranking score:
$$
f(q, d)=\tanh \left(w^{T} \phi(M)+b\right) .
$$
$w$ and $b$ are the layer parameters to learn. The training of K-NRM uses the pairwise learning to rank loss:
$$
L(w, b, \mathcal{V})=\sum_{q} \sum_{(d^{+}, d^{-}) \in D_{q}^{+,-}} \max \left(0,1-f\left(q, d^{+}\right)+f\left(q, d^{-}\right)\right)
$$
$D_{q}^{+,-}$ are the pairwise preferences from the ground truth: $d^{+}$ ranks higher than $d^{-}$. The parameters to learn include the ranking parameters $w, b$, and the word embeddings $V$.

```{figure} images/../deepLearning/ApplicationIRSearch/DeepRetrievalModels/KNRM/KNRM
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:knrm
The architecture of KNRM. Given input query words and document words, the embedding layer maps them into distributed representations, the translation layer calculates the word-word similarities and forms the translation matrix, the kernel pooling layer generate soft-TF counts as ranking features, and the learning to rank layer combines the soft-TF to the final ranking score.
```

\begin{remark}[Modeling n-gram interaction features]
KNRM uses kernels to extract interaction features at the unigram level. Conv-KNRM {cite}`dai2018convolutional` extends KNRM by first using convolutional layers to construct n-gram embeddings and produce interaction matrix consisting of n-grams (uni-gram, bi-gram, etc.)
\end{remark}

## Ranking model training

### Overview

\cite{ai2019learning}

\begin{remark}
	Unlike in classification or regression, the main goal of a ranking 	problem is not to assign a label or a value to individual items, but, 	given a list of items, to produce an ordering of the items in that list in such a way that the utility of the entire list is maximized. In other
	words, in ranking we are more concerned with the relative ordering 	of the relevance of items—for some notion of relevance—than their absolute magnitudes.

	Modeling relativity in ranking has been extensively studied in the past, especially in the context of learning-to-rank. Learning to-rank aims to learn a scoring function that maps feature vectors to real-valued scores in a supervised setting. Scores computed by such a function induce an ordering of items in the list. The majority of existing learning-to-rank algorithms learn a parameterized function by optimizing a loss that acts on pairs of items (pairwise) or a list of items (listwise) [5, 7, 8, 37]. The idea is that such loss functions guide the learning algorithm to optimize preferences between pairs of items or to maximize a ranking metric such as NDCG [6, 20, 32], thereby indirectly modeling relative relevance.
\end{remark}

\begin{remark}[why list-wise ranking]
Though effective, most existing learning-to-rank frameworks
are restricted to the paradigm of univariate scoring functions: the
relevance of an item is computed independently of other items in
the list. This setting could prove sub-optimal for ranking problems
for two main reasons. First, univariate functions have limited power
to model cross-item comparison. Consider an ad hoc document retrieval scenario where a user is searching for the name of an artist. If
all the results returned by the query (e.g., “calvin harris”) are recent,
the user may be interested in the latest news or tour information. If,
on the other hand, most of the query results are older (e.g., “frank
sinatra”), it is more likely that the user seeks information on artist
discography or biography. Thus, the relevance of each document
depends on the distribution of the whole list. Second, user interaction with search results shows a strong tendency to compare items.	
users compare a document with its surrounding documents prior to a click, and that a ranking model that uses the direct comparison mechanism can be more effective, as it mimics user behavior more
closely.

\end{remark}

\begin{remark}[pointwise method]
Pointwise transforms the problem into a multi-classification or regression problem. If it comes down to a multi-classification problem, for a certain Query, label the degree of relevance between the document and this Query, and the labels are divided into limited categories, so that the problem is turned into a multi-classification problem; if it comes down to a regression problem, for a certain Query , then the relevance Score is calculated for the relevance of the document to this Query, so that the problem can be attributed to a regression problem. Application Pointwise models include Subset Ranking, OC SVM, McRank, Prank, etc.	

The Pointwise algorithm is simple to implement and easy to understand, but it only models the relevance of a single document for a given Query, and only considers the absolute relevance of a single document. Pointwise only learns the global relevance of the document and the Query, and sorts the order. have a certain impact. In some scenarios, the impact of the first few documents on the sorting results is very important. For example, the content of the first page of the search engine is very important, but Pointwise does not consider this impact, and does not consider the order of sorting. do punishment.
\end{remark}

\begin{remark}[training loss function and score function]\hfill
- RankNet, LambdaRank are methods with pairwise loss function and univariate score function
- Biencoders with N-pair loss function are methods with list-wise loss function and univariate score function
- Cross-encoder with Binary cross entropy loss are methods with point-wise loss and univariate score function

\end{remark}

### Training data

In a typical model learning setting, we construct training data from user search log, which contains queries issued by users and the documents they clicked after issuing the query. The basic assumption is that a query and a document are relevant if the user clicked the document. 

Model learning in information retrieval typically falls into the category of contrastive learning. The query and the clicked document form a positive example; the query and irrelevant documents form negative examples. For retrieval problems, it is often the case that positive examples are available explicitly, while negative examples are unknown and need to be selected from an extremely large pool. The strategy of selecting negative examples plays an important role in determining quality of the encoders. In the most simple case, we randomly select unclicked documents as irrelevant document, or negative example. We defer the discussion of advanced negative example selecting strategy to \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:negativeSamplingStrategies}.

When there is a shortage of annotation data or click behavior data, we can also leverage weakly supervised data for training {cite}`dehghani2017neural,ram2021learning,haddad2019learning,nie2018multi`. In the weakly supervised data, labels or signals are obtained from an unsupervised ranking model, such as BM25. For example, given a query, relevance scores for all documents can be computed efficiently using BM25. Documents with highest scores can be used as positive examples and documents with lower scores can be used as negatives or hard negatives. 

### Feature engineering

### Model training objective functions

#### Pointwise regression objective

The idea of pointwise regression objective is to model the numerical relevance score for a given query-document. During inference time, the relevance scores between a set of candidates and a given query can be predicted and ranked.  

During training, given a set of query-document pairs $\left(q_{i}, d_{i, j}\right)$ and their corresponding relevance score $y_{i, j} \in [0, 1]$ and their prediction $f(q_i,d_{i,j})$. A pointwise regression objective tries to optimize a model to predict the relevance score via minimization
$$L = -\sum_{i} \sum_{j} (f(q_i,d_{i,j})) - y_{i,j})^2.
$$

Using a regression objective offer flexible for the user to model different levels of relevance between queries and documents. However, such flexibility also comes with a requirement that the target relevance score should be accurate in absolute scale.
While human annotated data might provide absolute relevance score, human annotation data is expensive and small scale. On the other hand, absolute relevance scores that are approximated by click data can be noisy and less optimal for regression objective. To make training robust to label noises, one can consider pairwise ranking objectives.     

\begin{remark}
	This particularly important in weak supervision, as the scores are imperfect values—using the ranking objective alleviates this issue by forcing the
	model to learn a preference function rather than reproduce absolute
	scores. In
\end{remark}

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:pointwise_ranking_loss)=
#### Pointwise ranking objective

The idea of pointwise ranking objective is to simplify a ranking problem to a binary classification problem. Specifically, given a set of query-document pairs $\left(q_{i}, d_{i, j}\right)$ and their corresponding relevance label $y_{i, j} \in \{0, 1\}$, where 0 denotes irrelevant and 1 denotes relevant. A pointwise learning objective tries to optimize a model to predict the relevance label. 

A commonly used pointwise loss functions is the binary Cross Entropy loss:
$$
L=-\sum_{i} \sum_{j} y_{i, j} \log \left(p\left(q_{i}, d_{i, j}\right)\right)+\left(1-y_{i, j}\right) \log \left(1-p\left(q_{i}, d_{i, j}\right)\right)
$$
where $p\left(q_{i}, d{i, j}\right)$ is the predicted probability of document $d_{i,j}$ being relevant to query $q_i$.

The advantages of pointwise ranking objectives are two-fold. First, pointwise ranking objectives are computed based on each query-document pair $\left(q_{i}, d_{i, j}\right)$ separately, which makes it simple and easy to scale. Second, the outputs of neural models learned with pointwise loss functions often have real meanings and value in practice. For instance, in sponsored search, the predicted the relevance probability can be used in ad bidding, which is more important than creating a good result list in some application scenarios.

In general, however, pointwise ranking objectives are considered to be less effective in ranking tasks. Because pointwise loss functions consider no document preference or order information, they do not guarantee to produce the best ranking list when the model loss reaches the global minimum. Therefore, better ranking paradigms that directly optimize document ranking based on pairwise loss functions and even listwise loss functions.

\iffalse
#### Pairwise ranking objective

Pairwise ranking objectives focus on optimizing the relative preferences between documents rather than their labels. In contrast to pointwise methods where the final ranking loss is the sum of loss on each document, pairwise loss functions are computed based on the permutations of all possible document pairs [96]. It usually can be formalized as
$$
L(f ; \mathcal{S}, \mathcal{T}, \mathcal{Y})=\sum_{i} \sum_{(j, k), y_{i, j} \succ y_{i, k}} L\left(f\left(s_{i}, t_{i, j}\right)-f\left(s_{i}, t_{i, k}\right)\right)
$$
where $t_{i, j}$ and $t_{i, k}$ are two documents for query $s_{i}$ and $t_{i, j}$ is preferable comparing to $t_{i, k}$ (i.e., $y_{i, j} \succ y_{i, k}$ ). For instance, a well-known pairwise loss function is Hinge loss:
$$
L(f ; \mathcal{S}, \mathcal{T}, \mathcal{Y})=\sum_{i} \sum_{(j, k), y_{i, j} \succ y_{i, k}} \max \left(0,1-f\left(s_{i}, t_{i, j}\right)+f\left(s_{i}, t_{i, k}\right)\right)
$$
Hinge loss has been widely used in the training of neural ranking models such as DRMM [21] and K-NRM [ [85]. Another popular pairwise loss function is the pairwise cross entropy defined as
$$
L(f ; \mathcal{S}, \mathcal{T}, \mathcal{Y})=-\sum_{i} \sum_{(j, k), y_{i, j} \succ y_{i, k}} \log \sigma\left(f\left(s_{i}, t_{i, j}\right)-f\left(s_{i}, t_{i, k}\right)\right)
$$
where $\sigma(x)=\frac{1}{1+\exp (-x)}$. Pairwise cross entropy is first proposed in RankNet by Burges et al. [97], which is considered to be one of the initial studies on applying neural network techniques to ranking problems.

Ideally, when pairwise ranking loss is minimized, all preference relationships between documents should be satisfied and the model will produce the optimal result list for each query. This makes pairwise ranking objectives effective in many tasks where performance is evaluated based on the ranking of relevant documents. In practice, however, optimizing document preferences in pairwise methods does not always lead to the improvement of final ranking metrics due to two reasons: (1) it is impossible to develop a ranking model that can correctly predict document preferences in all cases; and (2) in the computation of most existing ranking metrics, not all document pairs are equally important. This means that the performance of pairwise preference prediction is not equal to the performance of the final retrieval results as a list. Given this problem, previous studies $[98,99,100,101]$ further proposed listwise ranking objectives for learning to rank.
\fi

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:triplet_loss)=
#### Pairwise ranking via triplet loss

Pointwise ranking loss aims to optimize the model to directly predict relevance between query and documents on absolute score. From embedding optimization perspective, it train the neural query/document encoders to produce similar embedding vectors for a query and its relevant document and dissimilar embedding vectors for a query and its irrelevant documents. 

On the other hand, pairwise ranking objectives focus on optimizing the relative preferences between documents rather than their labels. In contrast to pointwise methods where the final ranking loss is the sum of loss on each document, pairwise loss functions are computed based on the different combination of document pairs.

One of the most common pairwise ranking loss function is the **triplet loss**. Let $\mathcal{D}=\left\{\left\langle q_{i}, d_{i}^{+}, d_{i}^{-}\right\rangle\right\}_{i=1}^{m}$ be the training data organized into $m$ triplets. Each triplet contains one query $q_{i}$ and one relevant document $d_{i}^{+}$, along with one irrelevant (negative) documents $d_{i}^{-}$. Negative documents are typically randomly sampled from a large corpus or are strategically constructed [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:negativeSamplingStrategies}]. 
Visualization of the learning process in the embedding space is shown in \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:triplet}. Triplet loss helps guide the encoder networks to pull relevant query and document closer and push irrelevant query and document away.   

The loss function is given by
$$L =- \sum_{\left\langle q_{i}, d_{i}^{+}, d_{i}^{-}\right\rangle}\max(0, m - \operatorname{Sim}(q_i, d_i^+) + \operatorname{Sim}(q_i, d^-_i))$$
where $\operatorname{Sim}(q, d)$ is the similarity score produced by the network between the query and the document and $m$ is a hyper-parameter adjusting the margin. Clearly, only when $\operatorname{Sim}(q_i, d_i^+) - \operatorname{Sim}(q_i, d^-_i) > m$ there will be no loss incurred. Commonly used $\operatorname{Sim}$ functions include **dot product** or **Cosine similarity** (i.e., length-normalized dot product), which are related to distance calculation in the Euclidean space and hyperspherical surface. 

```{figure} images/../deepLearning/ApplicationIRSearch/TrainingLoss/triplet
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:triplet
The illustration of the learning process (in the embedding space) using triplet loss.
```

Triplet loss can also operating in the angular space
$$
\operatorname{sim}(q, d)=1-\arccos \left(\frac{\psi_{\beta}(q) \cdot \psi_{\alpha}(d)}{\left\|\psi_{\beta}(q)\right\|\left\|\psi_{\alpha}(d)\right\|}\right) / \pi
$$
As illustrated in Figure 1, the training objective is to score the positive example $d^{+}$by at least the margin $\mu$ higher than the negative one $d^{-}$. As part of our loss function, we use the triplet margin objective:
$$
l\left(q, d^{+}, d^{-}\right):=\max \left(0, \operatorname{sim}\left(q, d^{-}\right)-\operatorname{sim}\left(q, d^{+}\right)+\mu\right)
$$

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:N_pair_loss)=
#### N-pair loss

Triplet loss optimize the neural by encouraging positive pair $(q_i, d^+_i)$ to be more similar than its negative pair $(q_i, d^+_i)$. One improvement is to encourage $q_i$ to be more similar $d^+_i$ compared to $n$ negative examples $ d_{i, 1}^{-}, \cdots, d_{i, n}^{-}$, instead of just one negative example. This is known as N-pair loss {cite}`sohn2016improved`, and it is typically more robust than triplet loss.

Let $\mathcal{D}=\left\{\left\langle q_{i}, d_{i}^{+}, D_i^-\right\rangle\right\}_{i=1}^{m}$, where $D_i^- = \{d_{i, 1}^{-}, \cdots, d_{i, n}^{-}\}$ are a set of negative examples (i.e., irrelevant document) with respect to query $q_i$,  be the training data that consists of $m$ examples. Each example contains one query $q_{i}$ and one relevant document $d_{i}^{+}$, along with $n$ irrelevant (negative) documents $d_{i, j}^{-}$. The $n$ negative documents are typically randomly sampled from a large corpus or are strategically constructed [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:negativeSamplingStrategies}]. 

Visualization of the learning process in the embedding space is shown in \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:npairloss}. Like triplet loss, N-pair loss helps guide the encoder networks to pull relevant query and document closer and push irrelevant query and document away. Besides that, when there are are negatives are involved in the N-pair loss, their repelling to each other appears to help the learning of generating more uniform embeddings\cite{wang2020understanding}. 

The loss function is given by
$$L =-\sum_{\left\langle q_{i}, d_{i}^{+}, D_{i}^{-}\right\rangle}\log \frac{\exp(\operatorname{Sim}\left(e_{q_{i}}, e_{d_{i}^{+}}\right))}{\exp(\operatorname{Sim}\left(e_{q_{i}}, e_{d_{i}^{+}}\right))+\sum_{d^-_i\in D^-} \exp(\operatorname{Sim}\left(e_{q_{i}}, e_{d_{i}^{-}}\right))}$$
where $\operatorname{Sim}(e_q, e_d)$ is the similarity score function taking query embedding $e_q$ and document embedding $e_d$ as the input. 
```{figure} images/../deepLearning/ApplicationIRSearch/TrainingLoss/N_pair_loss
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:npairloss
The illustration of the learning process (in the embedding space) using N-pair loss.
```

#### N-pair dual loss

```{figure} images/../deepLearning/ApplicationIRSearch/TrainingLoss/N_pair_loss_dual
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:npairlossdual
The illustration of the learning process (in the embedding space) using N-pair dual loss.
```

The N-pair loss uses query as the anchor to adjust the distribution of document vectors in the embedding space. Authors in {cite}`li2021more` proposed that document can also be as the anchor to adjust the distribution of query vectors in the embedding space. This leads to loss functions consisting of two parts
```{math}
\begin{align*}
L &= L_{prime} + L_{dual} \\
	L_{prime} &=-\sum_{\left\langle q_{i}, d_{i}^{+}, D_{i}^{-}\right\rangle}\log \frac{\exp(\operatorname{Sim}\left(e_{q_{i}}, e_{d_{i}^{+}}\right))}{\exp(\operatorname{Sim}\left(e_{q_{i}}, e_{d_{i}^{+}}\right))+\sum_{d^-_i\in D^-} \exp(\operatorname{Sim}\left(e_{q_{i}}, e_{d_{i}^{-}}\right))} \\
	L_{dual} &=-\sum_{\left\langle d_{i}, q_{i}^{+}, Q_{i}^{-}\right\rangle}\log \frac{\exp(\operatorname{Sim}\left(e_{d_{i}}, e_{q_{i}^{+}}\right))}{\exp(\operatorname{Sim}\left(e_{d_{i}}, e_{q_{i}^{+}}\right))+\sum_{q^-_i\in Q^-} \exp(\operatorname{Sim}\left(e_{d_{i}}, e_{q_{i}^{-}}\right))}
\end{align*}
```
where $\operatorname{Sim}(e_q, e_d)$ is a **symmetric** similarity score function for the query and the document embedding vectors, $L_{prime}$ is the N-pair loss, and $L_{dual}$ is the N-pair dual loss. 

To compute dual loss, we need to prepare training data $\mathcal{D}_{dual}=\left\{\left\langle d_{i}, q_{i}^{+}, Q_i^-\right\rangle\right\}_{i=1}^{m}$, where $Q_i^- = \{q_{i, 1}^{-}, \cdots, q_{i, n}^{-}\}$ are a set of negative queries examples (i.e., irrelevant query) with respect to document $d_i$. Each example contains one document $d_{i}$ and one relevant query $d_{i}^{+}$, along with $n$ irrelevant (negative) queries $q_{i, j}^{-}$. 

\iffalse
### Learning-to-rank objective
#### RankNet

Consider a pair of candidate documents $(i, j)$ with respect to the query $q$. Assume we have a model $f(q, d)$ that produces relevance scores $s_i$ and $s_j$, respectively:
$$s_i = f(q, d_i), s_j = f(q, d_j).$$

We model the probability that $d_i$ is more relevant than $d_j$ with respect to $q$ by
$$P(i \succ j) = \frac{1}{1+\exp ^{-\left(s_{i}-s_{j}\right)}} = \sigma(s_i - s_j).$$

Now let $y_{i j} \in\{0,1\}$ be the actual label of the given document pair $(i, j)$, where $y_{ij} = 1$ denotes $d_i$ is more relevant and vice versa for $y_{ij} = 0$. 

Considering all the possible pairs, we can optimize the model via a loss function resembling cross entropy, which is given by 
$$
L=-\sum_{i \neq j} y_{i j} \log P_{i j}+\left(1-y_{i j}\right) \log \left(1-P_{i j}\right)
$$
where we use $P_{ij}$ to denote $P(i \succ j)$. This cross entropy loss formulation is originated in the **RankNet**\cite{burges2005learning}. 

By exploiting the symmetry in the gradient calculation with respect to $s_i$ and $s_j$, we can further simplify the training computation. 

The improvement is based on a factorization of the calculation of gradient of the cross entropy loss, under its pairwise update context.
Given the point cross entropy loss as $L$ :
```{math}
\begin{align*}
L(s_i, s_j)&=-y_{i j} \log P_{i j}-\left(1-y_{i j}\right) \log \left(1-P_{i j}\right) \\
&=-y_{i j} \log \sigma(s_i - s_j)-\left(1-y_{i j}\right) \log \left(1 - \sigma(s_i - s_j)\right)
\end{align*}
```

It can be shown that $\frac{\partial L}{\partial s_i} = -\frac{\partial L}{\partial s_j}$; Specifically,
$$\frac{\partial L}{\partial s_i} = \sigma(s_i - s_j) - y_{ij}, \frac{\partial L}{\partial s_j} = y_{ij} - \sigma(s_i - s_j).$$

The gradient with respect to a model parameter $w_{k}$ can be written as:
$$
\frac{\partial L}{\partial w_{k}} =\frac{\partial L}{\partial s_{i}} \frac{\partial s_{i}}{\partial w_{k}}+\frac{\partial L}{\partial s_{j}} \frac{\partial s_{j}}{\partial w_{k}} $$

It can be shown that 

Now rewrite the gradient in total losses for all training pairs $\{i, j\}$ that satisfied $i \succ j$ :
```{math}
\begin{align*}
\frac{\partial L_{T}}{\partial w_{k}} &=\sum_{\{i, j\}}\left[\frac{\partial L}{\partial s_{i}} \frac{\partial s_{i}}{\partial w_{k}}+\frac{\partial L}{\partial s_{j}} \frac{\partial s_{j}}{\partial w_{k}}\right] \\
	&=\sum_{i} \frac{\partial s_{i}}{\partial w_{k}}\left(\sum_{\forall j<i} \frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{i}}\right)+\sum_{j} \frac{\partial s_{j}}{\partial w_{k}}\left(\sum_{\forall i \succ j} \frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{j}}\right) \\
	&=\sum_{i} \frac{\partial s_{i}}{\partial w_{k}}\left[\sum_{\forall j<i} \frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{i}}+\sum_{\forall j<i} \frac{\partial L\left(s_{j}, s_{i}\right)}{\partial s_{i}}\right] \\
	&=\sum_{i} \frac{\partial s_{i}}{\partial w_{k}}\left[\sum_{\forall j<i} \frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{i}}-\sum_{\forall j \succ i} \frac{\partial L\left(s_{j}, s_{i}\right)}{\partial s_{j}}\right] \\
	&=\sum_{i} \frac{\partial s_{i}}{\partial w_{k}} \lambda_{i}
\end{align*}
```

The lambda of a given document is:
$$
\begin{aligned}
	\lambda_{i} &=\left[\sum_{\forall j<i} \frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{i}}-\sum_{\forall j \succ i} \frac{\partial L\left(s_{j}, s_{i}\right)}{\partial s_{j}}\right] \\
	&=\left[\sum_{\forall j<i} \lambda_{i j}-\sum_{\forall j \succ i} \lambda_{i j}\right] \\
	\lambda_{i j} \text { such that: } \\
	& \lambda_{i j} \equiv \frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{i}} \cdot\left|\Delta N D C G_{i j}\right|
\end{aligned}
$$
The proposed method is to adjust the pairwise lambda $\lambda_{i j}$ such that:
where $\triangle N D C G_{i j}$ is the change in NDCG when the position of $i$ and $j$ are swapped.
The researcher found that by such adjustment, without theoretical proof, the model is empirically optimizing NDCG, and hence yield better overall results.
#### LambdaNet

Two important enhancements have been achieved from RankNet to LambdaNet.
1. Training speed-up thanks to factorization of gradient calculation
2. Optimization towards a ranking metric
Gradient Factorization
For the first point, LambdaNet is a mathematically improved version of RankNet. The improvement is based on a factorization of the calculation of gradient of the cross entropy loss, under its pairwise update context.
Given the point cross entropy loss as $L$ :
$$
L=y_{i j} \log _{2} P_{i j}+\left(1-y_{i j}\right) \log _{2}\left(1-P_{i j}\right)
$$
The gradient (the 1 st-order derivative of the loss w.r.t. a model parameter $w_{k}$ ) can be written as:
$$
\frac{\partial L}{\partial w_{k}}=\frac{\partial L}{\partial s_{i}} \frac{\partial s_{i}}{\partial w_{k}}+\frac{\partial L}{\partial s_{j}} \frac{\partial s_{j}}{\partial w_{k}}
$$
In plain words, the impact of a change in model parameter $w_{k}$ will go through the resulting changes in the model scores and then the changes in loss. Now rewrite the gradient in total losses for all training pairs $\{i, j\}$ that satisfied $i \succ j$ :
$$
\begin{aligned}
	\frac{\partial L_{T}}{\partial w_{k}} &=\sum_{\{i, j\}}\left[\frac{\partial L}{\partial s_{i}} \frac{\partial s_{i}}{\partial w_{k}}+\frac{\partial L}{\partial s_{j}} \frac{\partial s_{j}}{\partial w_{k}}\right] \\
	&=\sum_{i} \frac{\partial s_{i}}{\partial w_{k}}\left(\sum_{\forall j<i} \frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{i}}\right)+\sum_{j} \frac{\partial s_{j}}{\partial w_{k}}\left(\sum_{\forall i \succ j} \frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{j}}\right)
\end{aligned}
$$
with the fact that:
$$
\frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{i}}=-\frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{j}}=\log _{2} e\left[\left(1-y_{i j}\right)-\frac{1}{1+e^{s_{1}-s_{j}}}\right]
$$
and a re-indexing of the second-term, we end up with:

Ranking Metric Optimization
Since we model the score difference of a pair of documents in a query as a probability measure, the model is optimizing the pairwise correctness of ranking, which may not be the ultimately desirable objective.
Remember that the ranking objective is indeed measured by (ideally) a position-sensitive graded measure such as NDCG. But in the above setup NDCG is not directly linked to the minimization of cross entropy. A straightforward and also simple solution is to use NDCG as an early stop criteria and determine by using a validation dataset.

LambdaRank proposes yet another solution. The researcher found that during the gradient update using the lambda notion, for each pair instead of calculating just the lambda, we can adjusted lambda by the change in NDCG for that pair provided that the position of the two item swaped with each other.
The lambda of a given document is:
$$
\begin{aligned}
	\lambda_{i} &=\left[\sum_{\forall j<i} \frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{i}}-\sum_{\forall j \succ i} \frac{\partial L\left(s_{j}, s_{i}\right)}{\partial s_{j}}\right] \\
	&=\left[\sum_{\forall j<i} \lambda_{i j}-\sum_{\forall j \succ i} \lambda_{i j}\right] \\
	\lambda_{i j} \text { such that: } \\
	& \lambda_{i j} \equiv \frac{\partial L\left(s_{i}, s_{j}\right)}{\partial s_{i}} \cdot\left|\Delta N D C G_{i j}\right|
\end{aligned}
$$
The proposed method is to adjust the pairwise lambda $\lambda_{i j}$ such that:
where $\triangle N D C G_{i j}$ is the change in NDCG when the position of $i$ and $j$ are swapped.
The researcher found that by such adjustment, without theoretical proof, the model is empirically optimizing NDCG, and hence yield better overall results.

\begin{remark}[LambdaMART]
	LambdaMART is simply a LambdaNet but replaces the underlying neural network model with gradient boosting regression trees (or more general, gradient boosting machines, GBM). GBM is proven to be very robust and performant in handling real world problem.
	The model wins several real-world large-scale LTR contests.
\end{remark}

\begin{remark}[LambdaLoss]
	In the original LambdaRank and LambdaMART framework, no theoretical work has been done to mathematically prove that ranking metric is being optimized after the adjustment of the lambda calculation. The finding is purely based on empirical works, i.e., by observing the results from varying dataset and simulation with experiments.

	Researchers from Google recently {cite}`wang2018Lambda` published a generalized framework called LambdaLoss, which serves as an extension of the original ranking model and comes with a thorough theoretical groundwork to justify that the model is indeed optimizing a ranking metric.
\end{remark}

\begin{remark}
	From {cite}`wang2018Lambda`

	A commonly used pairwise loss function is the logistic loss
	$$
	\begin{aligned}
		l(\mathbf{y}, \mathbf{s}) &=\sum_{i=1}^{n} \sum_{j=1}^{n} \mathbb{I}_{y_{i}>y_{j}} \log _{2}\left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right) \\
		&=\sum_{y_{i}>y_{j}} \log _{2}\left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right)
	\end{aligned}
	$$
	where $I$ is the indicator function and $\sigma$ is a hyper-parameter. This loss is based on cross entropy and used in the RankNet algorithms [35]. The intuition is to apply a penalty on the out-of-order pair $(i, j)$ that has $y_{i}>y_{j}$ but $s_{i}<s_{j}$. Note that we use $\log _{2}$ instead of $\log$ purposely to facilitate the discussion in Section $5 .$

	3.3 Ranking Metrics
	There are many existing ranking metrics such as NDCG and MAP used in IR problems. A common property of these metrics is that they are rank-dependent and place more emphasis on performance of the top ranked documents. For example, the commonly adopted NDCG metric for a single query over the document list ranked by decreasing scores $\boldsymbol{s}$ is defined as
	$$
	\mathrm{NDCG}=\frac{1}{\operatorname{maxDCG}} \sum_{i=1}^{n} \frac{2^{y_{i}}-1}{\log _{2}(1+i)}=\sum_{i=1}^{n} \frac{G_{i}}{D_{i}}
	$$
	where
	$$
	G_{i}=\frac{2^{y_{i}}-1}{\operatorname{maxDCG}}, D_{i}=\log _{2}(1+i)
	$$
	are gain and discount functions respectively and maxDCG is a normalization factor per query and computed as the DCG for the list ranked by decreasing relevance labels y of the query. Please note that maxDCG is a constant factor per query in NDCG.

	Ideally, learning-to-rank algorithms should use ranking metrics as learning objectives. However, it is easy to see that sorting by scores is needed to obtain ranks. This makes ranking metrics either flat or discontinuous everywhere, so they cannot be directly optimized efficiently.
	3.4 LambdaRank
	Bridging the gap between evaluation metrics and loss functions has been studied actively in the past [23]. Among them, LambdaRank or its tree-based variant LambdaMART has been one of the most effective algorithms to incorporate ranking metrics in the learning procedure. The basic idea is to dynamically adjust the loss during the training based on ranking metrics. Using NDCG as an example, $\triangle \mathrm{NDCG}$ is defined as the absolute difference between the NDCG values when two documents $i$ and $j$ are swapped
	$$
	\Delta \operatorname{NDCG}(i, j)=\left|G_{i}-G_{j}\right|\left|\frac{1}{D_{i}}-\frac{1}{D_{j}}\right|
	$$
	LambdaRank uses the logistic loss in Eq 3 and adapts it by reweighing each document pair by $\triangle \mathrm{NDCG}$ in each iteration
	$$
	l(\mathbf{y}, \mathrm{s})=\sum_{y_{i}>y_{j}} \Delta \mathrm{NDCG}(i, j) \log _{2}\left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right)
	$$

\end{remark}

\fi
(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:negativeSamplingStrategies)=
## Training data sampling strategies

### Principles

From the ranking perspective, both retrieval and re-ranking requires the generation of some order on the input samples. For example, given a query $q$ and a set of candidate documents $(d_1,...,d_N)$. We need the model to produce an order list $d_2 \succ d_3 ... \succ d_k$ according to their relevance to the query. 

To train a model to produce the expected results during inference, we should ensure the training data distribution to matched with the inference time data distribution. 
Particularly, the inference time the candidate document distribution and ranking granularity  differ vastly for retrieval tasks and re-ranking tasks [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:retrievalrerankingtask}]. Specifically, 
- For the retrieval task, we typically need to identify top $k (k=1000-10000)$ relevant documents from the entire document corpus. This is achieved by ranking all documents in the corpus with respect to the relevance of the query. 
- For the re-ranking task, we need to identify the top $k (k=10)$ most relevant documents from the relevant documents generated by the retrieval task.  

Clearly, features most useful in the retrieval task (i.e., distinguish relevant from irrelevant) are often not the same as the features most useful in re-ranking task (i.e., distinguish most relevant from less relevant). Therefore, the training samples for retrieval and re-ranking need to be constructed differently.

```{figure} images/../deepLearning/ApplicationIRSearch/TrainingDataSampling/retrieval_reranking_task
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:retrievalrerankingtask
Retrieval tasks and re-ranking tasks are faced with different the candidate document distribution and ranking granularity.
```

Constructing the proper training data distribution is more challenging to retrieval stage than the re-ranking stage. In re-ranking stage, data in the training and inference phases are both the documents from previous retrieval stages. In the retrieval stage, we need to construct training examples in a mini-batch fashion in a way that each batch approximates the distribution in the inference phase as close as possible. 

This section will mainly focus on constructing training examples for retrieval model training in an efficient and effective way. Since the number of negative examples (i.e., irrelevant documents) significantly outnumber the number of positive examples. Constructing training examples particularly boil down to constructing proper negative examples. 

### Negative sampling methods I: heuristic methods

#### Overview

The essence of the negative sampling algorithm is to set or adjust the sampling distribution during negative sampling based on certain methods. According to the way the negative sampling algorithm sets the sampling distribution, we can divide the current negative sampling algorithms into two categories: Heuristic Negative Sampling Algorithms and Model-based Negative Sampling Algorithms.

In {cite}`karpukhin2020dense`, there are three different types of negatives: (1) Random: any random passage from the corpus; (2) BM25: top passages returned by BM25 which don’t contain the answer but match most question tokens; (3) Gold: positive passages paired with other questions which appear in the training set.

One approach to improving the effectiveness of single-vector bi-encoders is hard negative mining, by training with carefully selected negative examples that emphasize discrimination between relevant and non-relevant texts.

both large in-batch negative sampling and asynchronous ANN index updates are computationally demanding.

Compared with the two heuristic algorithms mentioned above, the model-based negative sampling algorithm is easier to pick high-quality negative examples, and it is also the more cutting-edge sampling algorithm at present. Here are several model-based negative sampling algorithms:

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:in-batch-negatives)=
#### Random negatives and in-batch negatives

Random negative sampling is the most basic negative sampling algorithm. The algorithm uniformly sample documents from the corpus and treat it as a negative. Clearly, random negatives can generate negatives that are too easy for the model. For example, a negative document that is topically different from the query. These easy negatives lower the learning efficiency, that is, each batch produces limited information gain to update the model. Still, random negatives are widely used because of its simplicity.

In practice, random negatives are implemented as in-batch negatives.  In the contrastive learning framework with N-pair loss [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:N_pair_loss}], we construct a mini-batch of query-doc examples like $\{(q_1, d_1^+, d_{1,1}^-, d_{1,M}^-), ..., (q_N, d_N^+, d_{N,1}^-, d_{N,M}^M)\}$, Naively implementing N-pair loss would increase computational cost from constructing sufficient negative documents corresponding to each query. In-batch negatives\cite{karpukhin2020dense} is trick to reuse positive documents associated with other queries as extra negatives [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:inbatchnegatives}]. The critical assumption here is that queries in a mini-batch are vastly different semantically, and positive documents from other queries would be confidently used as negatives. The assumption is largely true since each mini-batch is randomly sampled from the set of all training queries, in-batch negative document are usually true negative although they might not be hard negatives.

```{figure} images/../deepLearning/ApplicationIRSearch/TrainingDataSampling/NegativeSampling/in_batch_negatives
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:inbatchnegatives
The illustration of using in-batch negatives in contrastive learning.
```

Specifically, assume that we have $N$ queries in a mini-batch and each one is associated with a relevant  positive document. By using positive document of other queries, each query will have an additional $N - 1$ negatives.  

Formally, we can define our batch-wise loss function as follows:
$$
\mathcal{L}:=\sum_{1 \leq i \leq N}\left(\sum_{1 \leq j \leq N} l\left(q_{i}, d_{i}^{+}, d_{j}^{-}\right)+\sum_{1 \leq k \leq N, k \neq i} l\left(q_{i}, d_{i}^{+}, d_{k}^{+}\right)\right)
$$
where $l\left(q_{i}, d_{i}^{+}, d_{j}^{-}\right)$ is the loss function for a triplet.

In-batch negative offers an efficient implementation for random negatives. Another way to mitigate the inefficient learning issue is simply use large batch size (>4,000) {cite}`qu2020rocketqa`. This can be implemented using distributed multi-GPU training.

#### Popularity-based negative sampling

Popularity-based negative sampling use document popularity as the sampling weight to sample negative documents. The popularity of a document can be defined as some combination of click, dwell time, quality, etc. Compared to random negative sampling, this algorithm replaces the uniform distribution with a popularity-based sampling distribution, which can be pre-computed offline. 

The major rationale of using popularity-based negative examples is to improve representation learning. Popular negative documents represent a harder negative compared to a unpopular negative since they tend to have to a higher chance of being more relevant; that is, lying closer to query in the embedding space. If the model is trained to distinguish these harder cases, the over learned representations will be likely improved. 

Popularity-based negative sampling is also used in word2vec training {cite}`mikolov2013distributed`. For example, the probability to sample a word $w_i$ is given by:
$$
P\left(w_i\right)=\frac{f\left(w_i\right)^{3 / 4}}{\sum_{j=0}^n\left(f\left(w_j\right)^{3 / 4}\right)},
$$
where $f(w)$ is the frequency of word $w$. This equation, compared to linear popularity, has the tendency to increase the probability for less frequent words and decrease the probability for more frequent words.

#### Topic-aware negative sampling

In-batch random negatives would often consist of  topically-different documents, leaving little information gain for the training. To improve the information gain from a single random batch, we can constrain the queries and their relevant document are drawn from a similar topic\cite{hofstatter2021efficiently}.

The procedures are
- Cluster queries using query embeddings produced by basic query encoder.
- Sample queries and their relevant documents from a randomly picked cluster. A relevant document of a query form the negative of the other query.

Since queries are topically similar, the formed in-batch negatives are harder examples than randomly formed in-batch negative, therefore delivering more information gain each batch.  

Note that here we group queries into clusters by their embedding similarity, which allows grouping queries without lexical overlap. We can also consider lexical similarity between queries as additional signals to predict query similarity. 

### Negative sampling methods II: model-based methods
#### Static hard negative examples

Deep model improves the encoded representation of queries and documents by contrastive learning, in which the model learns to distinguish positive examples and negative examples. A simple random sampling strategy tend to produce a large quantity of easy negative examples since easy negative examples make up the majority of negative examples. Here by easy negative examples, we mean a document that can be easily judged to be irrelevant to the query. For example, the document and the query are in completely different topics.   

The model learning from easy negative example can quickly plateau since easy negative examples produces vanishing gradients to update the model. An improvement strategy is to supply additional hard negatives with randomly sampled negatives. In the simplest case, hard negatives can be selected based on a traditional BM25model {cite}`karpukhin2020dense, nogueira2019passage` or other efficient dense retriever: hard negatives are those have a high relevant score to the query but they are not relevant.  

As illustrated in \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:impacthardnegativeonretrieval}, a model trained only with easy negatives can fail to distinguish fairly relevant documents from irrelevant examples; on the other hand, a model trained with some hard negatives can learn better representations:
- Positive document embeddings are more aligned {cite}`wang2020understanding`; that is, they are lying closer with respect to each other.
- Fairly relevant and irrelevant documents are more separated in the embedding space and thus a better decision boundary for relevant and irrelevant documents. 

```{figure} images/../deepLearning/ApplicationIRSearch/TrainingDataSampling/Impact_hard_negative_on_retrieval
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:impacthardnegativeonretrieval
Illustration of importance of negative hard examples, which helps learning better representations to distinguish irrelevant and fairly relevant documents.
```

In generating these negative examples, the negative-generation model (e.g., BM25) and the model under training are de-coupled; that is the negative-generation model is not updated during training and the hard examples are static. Despite this simplicity, static hard negative examples introduces two shortcomings:
- Distribution mismatch, the negatives generated by the static model might quickly become less hard since the target model is constantly evolving.
- The generated negatives can have a higher risk of being false negatives to the target model because negative-generation model and the target model are two different models.  

#### Dynamic hard negative examples

```{figure} images/../deepLearning/ApplicationIRSearch/TrainingDataSampling/NegativeSampling/ANCE_negative_sampling_demo
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:ancenegativesamplingdemo
Dynamic hard negative sampling from ANCE asynchronous training framework. Negatives are drawn from index produced using models at the previous checkpoint. Image from {cite}`xiong2020approximate`.
```

Dynamic hard negative mining is a scheme proposed in ANCE\cite{xiong2020approximate}. The core idea is to use the target model at previous checkpoint as the negative-generation model [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:ancenegativesamplingdemo}]. However, this negative mining approach is rather computationally demanding since corpus index need updates at every checkpoint. 

\begin{remark}[from ANCE]

	Based on our analysis, we propose Approximate nearest neighbor Negative Contrastive Estimation (ANCE), a new contrastive representation learning mechanism for dense retrieval. Instead of random or in-batch local negatives, ANCE constructs global negatives using the beingoptimized DR model to retrieve from the entire corpus. 

	This fundamentally aligns the distribution of negative samples in training and of irrelevant documents to separate in testing. From the variance reduction perspective, these ANCE negatives lift the upper bound of per instance gradient norm, reduce the variance of the
	stochastic gradient estimation, and lead to faster learning convergence.
\end{remark}

### Label denoising
#### False negatives

Hard negative examples produced from static or dynamic negative mining methods are effective to improve the encoder's performance. However, when selecting hard negatives with a less powerful model (e.g., BM25), we are also running the risk of introduce more false negatives (i.e., negative examples are actually positive) than a random sampling approach. Authors in {cite}`qu2020rocketqa` proposed to utilize a well-trained, complex  model (e.g., a cross-encoder) to determine if an initially retrieved hard-negative is a false negative. Such models are more powerful for capturing semantic similarity among query and documents. Although they are less ideal for deployment and inference purpose due to high computational cost, they are suitable for filtering. From the initially retrieved hard-negative documents, we can filter out documents that are actually relevant to the query. The resulting documents can be used as denoised hard negatives. 

#### False positives

Because of the noise in the labeling process (e.g., based on click data), it is also possible that a positive labeled document turns out to be irrelevant. To reduce false positive examples, one can develop more robust labeling process and merge labels from multiple sources of signals. 

## Transformer architectures for retrieval and ranking

### Transformers

#### Why Transformers?

BERT (Bidirectional Encoder Representations from Transformers) {cite}`devlin2018bert` and its transformer variants {cite}`lin2021survey` represent the state-of-the-art modeling strategies in a broad range of natural language processing tasks. The application of BERT in information retrieval and ranking was pioneered by {cite}`nogueira2019passage, nogueira2019multi`. The fundamental characteristics of BERT architecture is self-attention. By pretraining BERT on large scale text data, BERT encoder can produce contextualized embeddings can better capture semantics of different linguistic units. By adding additional prediction head to the BERT backbone, such BERT encoders can be fine-tuned to retrieval related tasks. In this section, we will go over the application of different BERT-based models in neural information retrieval and ranking tasks. 

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:monoBERT)=
#### Mono-BERT for point-wise ranking

```{figure} images/../deepLearning/ApplicationIRSearch/DeepRetrievalModels/Berts/BERT/mono_bert_arch
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:monobertarch
The architecture of Mono-BERT for document relevance ranking. The input is the concatenation of the query token sequence and the candidate document token sequence. Once the input sequence is passed through the model, we use the [CLS] embedding as input to a single layer neural network to obtain a posterior probability $p_{i}$ of the candidate $d_{i}$ being relevant to query $q$. 
```

The first application of BERT in document retrieval is using BERT as a cross encoder, where the query token sequence and the document token sequence are concatenated via [SEP] token and encoded together. This architecture [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:monobertarch}], called mono-BERT, was first proposed by {cite}`nogueira2019passage, nogueira2019multi`.

To meet the token sequence length constraint of a BERT encoder (e.g., 512), we might need to truncate the query (e.g, not greater than 64 tokens) and the candidate document token sequence such that the total concatenated token sequence have a maximum length of 512 tokens.

Once the input sequence is passed through the model, we use the [CLS] embedding as input to a single layer neural network to obtain a posterior probability $p_{i}$ of the candidate $d_{i}$ being relevant to query $q$. The posterior probability can be used to rank documents.

The training data can be represented by a collections of triplets $(q, J_P^q, J_N^q), q\in Q$, where $Q$ is the set of queries, $J_{P}^q$ is the set of indexes of the relevant candidates associated with query $q$ and $J_{N}^q$ is the set of indexes of the nonrelevant candidates.

The encoder can be fine-tuned using cross-entropy loss:
$$
L_{\text {mono-BERT}}=-\sum_{q\in Q}( \sum_{j \in J_{P}^q} \log \left(s_{j}\right)-\sum_{j \in J_{N}^q} \log \left(1-s_{j}\right) ).
$$

\iffalse
#### Mono T5

\cite{nogueira2020document}

\fi

#### Duo-BERT for pairwise ranking

```{figure} images/../deepLearning/ApplicationIRSearch/DeepRetrievalModels/Berts/BERT/duo_bert_arch
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:duobertarch
The duo-BERT architecture takes the concatenation of the query and two candidate documents as the input. Once the input sequence is passed through the model, we use the [CLS] embedding as input to a single layer neural network to obtain a posterior probability that the first document is more relevant than the second document. 
```

Mono-BERT can be characterized as a *pointwise* approach for ranking. Within the *framework of learning to rank*, {cite}`nogueira2019passage, nogueira2019multi` also proposed duo-BERT, which is  a *pairwise* ranking approach. In this pairwise approach, the duo-BERT ranker model estimates the probability $p_{i, j}$ of the candidate $d_{i}$ being more relevant than $d_{j}$ with respect to query $q$.

The duo-BERT architecture [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:duobertarch}] takes the concatenation of the query $q$, the candidate document $d_{i}$, and the candidate document $d_{j}$ as the input. We also need to truncate the query, candidates $d_{i}$ and $d_{j}$ to proper lengths (e.g., 62 , 223 , and 223 tokens, respectively), so the entire sequence will have at most 512 tokens. 

Once the input sequence is passed through the model, we use the [CLS] embedding as input to a single layer neural network to obtain a posterior probability $p_{i,j}$. This posterior probability can be used to rank documents $i$ and $j$ with respect to each other. If there are $k$ candidates for query $q$, there will be $k(k-1)$ passes to compute all the pairwise probabilities. 

The model can be fine-tune using with the following loss per query.
```{math}
\begin{align*}
L_{\text {duo }}=&-\sum_{i \in J_{P}, j \in J_{N}} \log \left(p_{i, j}\right) \\
	&-\sum_{i \in J_{N}, j \in J_{P}} \log \left(1-p_{i, j}\right)
\end{align*}
```

At inference time, the obtained $k(k -1)$ pairwise probabilities are used to produce the final document relevance ranking given the query. Authors in {cite}`nogueira2019multi` investigate five different aggregation methods (SUM, BINARY, MIN, MAX, and SAMPLE) to produce the final ranking score.
```{math}
\begin{align*}
\operatorname{SUM}:  s_{i} &=\sum_{j \in J_{i}} p_{i, j} \\
	\operatorname{BINARY}: s_{i} &=\sum_{j \in J_{i}} \bm{1}_{p_{i, j} > 0.5} \\
	\operatorname{MIN}: s_{i}  &=\min _{j \in J_{i}} p_{i, j} \\
	\operatorname{MAX}: s_{i} &=\max _{j \in J_{i}} p_{i, j} \\
	\operatorname{SAMPLE}: s_{i}&=\sum_{j \in J_{i}(m)} p_{i, j}
\end{align*}
```
where $J_i = \{1 <= j <= k, j\neq i\}$ and $J_i(m)$ is $m$ randomly sampled elements from $J_i$. 

The SUM method measures the pairwise agreement that candidate $d_{i}$ is more relevant than the rest of the candidates $\left\{d_{j}\right\}_{j \neq i^{*}}$. The BINARY method resembles majority vote. The Min (MAX) method measures the relevance of $d_{i}$ only against its strongest (weakest) competitor. The SAMPLE method aims to decrease the high inference costs of pairwise computations via sampling. Comparison studies using MS MARCO dataset suggest that SUM and BINARY give the best results.

#### Multistage retrieval and ranking pipeline

```{figure} images/../deepLearning/ApplicationIRSearch/DeepRetrievalModels/Berts/BERT/multistage_retrieval_ranking_bert
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:multistageretrievalrankingbert
Illustration of a three-stage retrieval-ranking architecture using BM25, monoBERT and duoBERT. Image from {cite}`nogueira2019multi`.
```

With BERT variants of different ranking capability, we can construct a multi-stage ranking architecture to select a handful of most relevant document from a large collection of candidate documents given a query. Consider a typical architecture comprising a number of stages from $H_0$ ot $H_N$. $H_0$ is a exact-term matching stage using from an inverted index. $H_0$ stage take billion-scale document as input and output thousands of candidates $R_0$. For stages from $H_1$ to $H_N$, each stage $H_{n}$ receives a ranked list $R_{n-1}$ of candidates from the previous stage and output candidate list $R_n$. Typically $|R_n| \ll |R_{n-1}|$ to enable efficient retrieval. 

An example three-stage retrieval-ranking system is shown in \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:multistageretrievalrankingbert}. In the first stage $H_{0}$, given a query $q$, the top candidate documents $R_{0}$ are retrieved using BM25. In the second stage $H_{1}$, monoBERT produces a relevance score $s_{i}$ for each pair of query $q$ and candidate $d_{i} \in R_{0}.$ The top candidates with respect to these relevance scores are passed to the last stage $H_{2}$, in which duoBERT computes a relevance score $p_{i, j}$ for each triple $\left(q, d_{i}, d_{j}\right)$. The final list of candidates $R_{2}$ is formed by re-ranking the candidates according to these scores .

**Evaluation.** Different multistage architecture configurations are evaluated using the MS MARCO dataset. We have following observations:
- Using a single stage of BM25 yields the worst performance.
- Adding an additional monoBERT significantly improve the performance over the single BM25 stage architecture.
- Adding the third component duoBERT only yields a diminishing gain.

Further, the author found that	employing the technique of Target Corpus Pre-training (TCP)\ gives additional performance gain. Specifically, the BERT backbone will undergo a two-phase pre-training. In the first phase, the model is pre-trained using the original setup, that is Wikipedia (2.5B words) and the Toronto Book corpus ( 0.8B words) for one million iterations. In the second phase, the model is further pre-trained on the MS MARCO corpus.

{\scriptsize
\begin{tabular}{l|ll}
	\hline Method & Dev & Eval \\
	\hline Anserini (BM25) & $18.7$ & $19.0$ \\
	+ monoBERT & $37.2$ & $36.5$ \\
	+ monoBERT + duoBERT $_{\text {MAX }}$ & $32.6$ & $-$ \\
	+ monoBERT + duoBERT $_{\text {MIN }}$ & $37.9$ & $-$ \\
	+ monoBERT + duoBERT $_{\text {SUM }}$ & $38.2$ & $37.0$ \\
	+ monoBERT + duoBERT $_{\text {BINARY }}$ & $38.3$ & $-$ \\
	+ monoBERT + duoBERT $_{\text {SUM }}+$ TCP & $39.0$ & $37.9$ \\
	\hline
\end{tabular}
}

### DC-BERT

```{figure} images/../deepLearning/ApplicationIRSearch/DeepRetrievalModels/Berts/DC_BERT/DC_bert
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:dcbert
The overall architecture of DC-BERT [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:dcbert}] consists of a dual-BERT component for decoupled encoding, a Transformer component for question-document interactions, and a classifier component for document relevance scoring.
```

One way to improve the computational efficiency is to employ dual BERT encoders for partial separate encoding and then employ an additional shallow module for cross encoding. One example is the architecture shown in \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:dcbert}, which is called DC-BERT and proposed in {cite}`nie2020dc`. The overall architecture of DC-BERT  consists of a dual-BERT component for decoupled encoding, a Transformer component for question-document interactions, and a binary classifier component for document relevance scoring.

The document encoder can be run offline to pre-encodes all documents and caches all term representations. During online inference, we only need to run the BERT query encodes online. Then the obtained contextual term representations are fed into high-layer Transformer interaction layer. 

**Dual-BERT component**. DC-BERT contains two pre-trained BERT models to independently encode the question and each retrieved document. During training, the parameters of both BERT models are fine-tuned to optimize the learning objective.

**Transformer component.** The dual-BERT components produce contextualized embeddings for both the query token sequence and the document token sequence. Then we add global position embeddings $\mathbf{E}_{P_{i}} \in \mathbb{R}^{d}$ and segment embedding again to re-encode the position information and segment information (i.e., query vs document). Both the global position and segment embeddings are initialized from pre-trained BERT, and will be fine-tuned. The number of Transformer layers $K$ is configurable to trade-off between the model capacity and efficiency. The Transformer layers are initialized by the last $K$ layers of pre-trained BERT, and are fine-tuned during the training.

**Classifier component.** The two CLS token output from the Transformer layers will be fed into a linear binary classifier to predict whether the retrieved document is relevant to the query. Following previous work (Das et al., 2019; Htut et al., 2018; Lin et al., 2018), we employ paragraph-level distant supervision to gather labels for training the classifier, where a paragraph that contains the exact ground truth answer span is labeled as a positive example. We parameterize the binary classifier as a MLP layer on top of the Transformer layers:
$$
p\left(q_{i}, d_{j}\right)=\sigma\left(\operatorname{Linear}\left(\left[o_{[C L S]} ; o_{[C L S]}^{\prime}\right]\right)\right)
$$
where $\left(q_{i}, d_{j}\right)$ is a pair of question and retrieved document, and $o_{[C L S]}$ and $o_{[C L S]}^{\prime}$ are the Transformer output encodings of the [CLS] token of the question and the document, respectively. The MLP parameters are updated by minimizing the cross-entropy loss.

DC-BERT uses one Transformer layer for question-document interactions. Quantized BERT is a 8bit-Integer model. DistilBERT is a compact BERT model with 2 Transformer layers.

We first compare the retriever speed. DC-BERT achieves over 10x speedup over the BERT-base retriever, which demonstrates the efficiency of our method. Quantized BERT has the same model architecture as BERT-base, leading to the minimal speedup. DistilBERT achieves about 6x speedup with only 2 Transformer layers, while BERT-base uses 12 Transformer layers.

With a 10x speedup, DC-BERT still achieves similar retrieval performance compared to BERT- base on both datasets. At the cost of little speedup, Quantized BERT also works well in ranking documents. DistilBERT performs significantly worse than BERT-base, which shows the limitation of the distilled BERT model. We
{\scriptsize
		\begin{tabular}{|c|cc|cc|}
		\hline
		& \multicolumn{2}{c|}{SQuAD}       & \multicolumn{2}{c|}{Natural Questions} \\ \hline
		Model          & \multicolumn{1}{c|}{PTB@10} & Speedup & \multicolumn{1}{c|}{P@10}    & Speedup    \\ \hline
		BERT-base      & \multicolumn{1}{c|}{71.5}   & 1.0x    & \multicolumn{1}{c|}{65.0}      & 1.0x       \\ \hline
		Quantized BERT & \multicolumn{1}{c|}{68.0}   & 1.1x    & \multicolumn{1}{c|}{64.3}      & 1.1x       \\ \hline
		DistilBERT     & \multicolumn{1}{c|}{56.4}   & 5.7x    & \multicolumn{1}{c|}{60.6}      & 5.7x       \\ \hline
		DC-BERT        & \multicolumn{1}{c|}{70.1}   & 10.3x   & \multicolumn{1}{c|}{63.5}      & 10.3x      \\ \hline
	\end{tabular}
}

To further investigate the impact of our model architecture design, we compare the performance of DC-BERT and its variants, including 1) DC-BERT-Linear, which uses linear layers instead of Transformers for interaction; and 2) DC-BERT-LSTM, which uses LSTM and bi- linear layers for interactions following previous work (Min et al., 2018). We report the results in Table 3. Due to the simplistic architecture of the interaction layers, DC-BERT-Linear achieves the best speedup but has significant performance drop, while DC-BERT-LSTM achieves slightly worse performance and speedup than DC-BERT.

{\scriptsize
\begin{tabular}{lcc}
	\hline Retriever Model & Retriever P@10 & Retriever Speedup \\
	\hline DC-BERT-Linear & $57.3$ & $43.6 \mathrm{x}$ \\
	DC-BERT-LSTM & $61.5$ & $8.2 \mathrm{x}$ \\
	DC-BERT & $63.5$ & $10.3 \mathrm{x}$ \\
	\hline
\end{tabular}	

}

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:colBERT)=
### ColBERT

#### Model architecture and training
```{figure} images/../deepLearning/ApplicationIRSearch/DeepRetrievalModels/Berts/Col_BERT/Col_bert
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:colbert
The architecture of ColBERT, which consists of an early separate encoding phase and a late interaction phase.
```

ColBERT {cite}`khattab2020colbert` is another example architecture that consists of an early separate encoding phase and a late interaction phase, as shown in \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:colbert}. ColBERT employs a single BERT model for both query and document encoders but distinguish input sequences that correspond to queries and documents by prepending a special token [Q] to queries and another token [D] to documents.

The query Encoder take query tokens as the input. Note that if a query is shorter than a pre-defined number $N_q$, it will be padded with BERT’s special [mask] tokens up to length $N_q$; otherwise, only the first $N_q$ tokens will be kept. It is found that the mask token padding serves as some sort of query augmentation and brings perform gain. In additional, a [Q] token is placed right after BERT’s sequence start token [CLS]. The query encoder then computes a contextualized representation for the query tokens.

The document encoder has a very similar architecture. A [D] token is placed right after BERT’s sequence start token [CLS]. Note that after passing through the encoder, embeddings correponding to punctuation symbols are filtered out. 

Given BERT's representation of each token, an additional linear layer with no activation is used to reduce the dimensionality reduction. The reduced dimensionality $m$ is set much smaller than BERT's fixed hidden dimension.

Finally, given $q= q_{1} \ldots q_{l}$ and $d=d_{1} \ldots d_{n}$, an additional CNN layer is used to allow each embedding vector to interact with its neighbor, yielding  the bags of embeddings $E_{q}$ and $E_{d}$ in the following manner.
```{math}
\begin{align*}
&E_{q}:=\operatorname{Normalize}\left(\operatorname{CNN}\left(\operatorname{BERT}\left([Q] q_{0} q_{1} \ldots q_{l} \# \# \ldots \#\right)\right)\right) \\
	&E_{d}:=\operatorname{Filter}\left(\operatorname{Normalize}\left(\operatorname{CNN}\left(\operatorname{BERT}\left([D] d_{0} d_{1} \ldots d_{n} \right)\right)\right)\right)
\end{align*}
```
Here \# refers to the [mask] tokens and $\operatorname{Normalize}$ denotes $L_2$ length normalization.

In the late interaction phase, every query embedding interacts with all document embeddings via a MaxSimilarity operator, which computes maximum similarity (e.g., cosine similarity), and the scalar outputs of these operators are summed across query terms.

Formally, the final similarity score between the $q$ and $d$ is given by
$$
S_{q, d} =\sum_{i \in I_q} \max _{j \in I_d} E_{q_{i}} \cdot E_{d_{j}}^{T},
$$
where $I_q = \{1,...,l\}$, $I_d = \{1, ..., n\}$ are the index sets for query token embeddings and document token embeddings, respectively. 
ColBERT is differentiable end-to-end and we can fine-tune the BERT encoders and train from scratch the additional parameters (i.e., the linear layer and the $[Q]$ and $[D]$ markers' embeddings). Notice that the final aggregation interaction mechanism has no trainable parameters. 

The retrieval performance of ColBERT is evaluated on MS MARCO dataset. Compared with traditional exact term matching retrieval, ColBERT has shortcomings in terms of latency but MRR is significantly better.

{
\scriptsize
\begin{tabular}{lcccc}
	\hline Method & MRR@10(Dev) & MRR@10 (Local Eval) & Latency (ms) & Recall@50 \\
	\hline BM25 (official) & $16.7$ & $-$ & $-$ & $-$ \\
	BM25 (Anserini) & $18.7$ & $19.5$ & 62 & $59.2$ \\
	doc2query & $21.5$ & $22.8$ & 85 & $64.4$ \\
	DeepCT & $24.3$ & $-$ & 62 (est.) & $69[2]$ \\
	docTTTTTquery & $27.7$ & $28.4$ & 87 & $75.6$ \\
	\hline ColBERT $_{\text {L2 }}$ (re-rank) & $34.8$ & $36.4$ & $-$ & $75.3$ \\
	ColBERTL2 (end-to-end) & $36.0$ & $36.7$ & 458 & $82.9$
\end{tabular}
}

Similarly, we can evaluate ColBERT's re-ranking performance against some strong baselines, such as BERT cross encoders {cite}`nogueira2019passage, nogueira2019multi`. ColBERT has demonstrated significant benefits in reducing latency with little cost of re-ranking performance. 

{
\scriptsize
\begin{tabular}{lccc}
	\hline Method & MRR@10 (Dev) & MRR@10 (Eval) & Re-ranking Latency (ms) \\
	\hline BM25 (official) & $16.7$ & $16.5$ & $-$ \\
	\hline KNRM & $19.8$ & $19.8$ & 3 \\
	Duet & $24.3$ & $24.5$ & 22 \\
	fastText+ConvKNRM & $29.0$ & $27.7$ & 28 \\
	BERT base & $34.7$ & $-$ & 10,700 \\
	BERT large & $36.5$ & $35.9$ & 32,900 \\
	\hline ColBERT (over BERT base ) & $34.9$ & $34.9$ & 61 \\
	\hline
\end{tabular}
}

\iffalse

#### Offline computation and indexing for re-ranking and retrieval 

\begin{remark}[TODO]

\end{remark}

By design, ColBERT isolates almost all of the computations between queries and documents, largely to enable pre-computing document representations offline. At a high level, our indexing procedure is straight-forward: we proceed over the documents in the collection in batches, running our document encoder $f_{D}$ on each batch and storing the output embeddings per document. Although indexing.

**Top $k$ re-ranking** Recall that ColBERT can be used for re-ranking the output of another retrieval model, typically a term-based model, or directly for end-to-end retrieval from a document collection. In this section, we discuss how we use ColBERT for ranking a small set of $k$ (e.g., $k=1000$ ) documents given a query q. Since $k$ is small, we rely on batch computations to exhaustively score each document. we reduce its matrix across document terms via a max-pool (i.e., representing an exhaustive implementation of our MaxSim computation) and reduce across query terms via a summation. Fi- nally, we sort the k documents by their total scores.

Relative to existing neural rankers (especially, but not exclu-
sively, BERT-based ones), this computation is very cheap that, in fact, its cost is dominated by the cost of gathering and transferring the pre-computed embeddings.

{
\begin{tabular}{ccc}
	Method & Re-ranking Latency (ms) & FLOPs/query \\
	\hline
	BERT base & 10,700 & $97 \mathrm{&nbsp;T}(13,900 \times)$ \\
	BERT large & 32,900 & $340 \mathrm{&nbsp;T}(48,600 \times)$ \\
	ColBERT & 61 & $7 \mathrm{&nbsp;B}(1 \times)$ \\
	\hline
\end{tabular}	
}

**Top $k$ retrieval** Subsequently, when serving queries, we use a two-stage procedure to retrieve the top- $k$ documents from the entire collection. Both stages rely on ColBERT's scoring: the first is an approximate stage aimed at filtering while the second is a refinement stage. For the first stage, we concurrently issue $N_{q}$ vector-similarity queries (corresponding to each of the embeddings in $E_{q}$ ) onto our faiss index. This retrieves the top- $k^{\prime}\left(\right.$ e.g., $\left.k^{\prime}=k / 2\right)$ matches for that vector over all document embeddings. We map each of those to its document of origin, producing $N_{q} \times k^{\prime}$ document IDs, only $K \leq N_{q} \times k^{\prime}$ of which are unique. These $K$ documents likely contain one or more embeddings that are highly similar to the query embeddings. For the second stage, we refine this set by exhaustively re-ranking only those $K$ documents in the above re-ranking.

{
			\begin{tabular}{cc}
			Method & Retrieval Latency (ms)  \\
			\hline
			BM25 (Anserini) & 62  \\
			ColBERT & 458  \\
			\hline
		\end{tabular}	

}

\fi

## Multi-vector representations

### Introduction

In classic representation-based learning for semantic retrieval [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:classicRepresentationLearning}], we use two encoders (i.e., bi-encoders) to separately encoder a query and a candidate document into two dense vectors in the embedding space, and then a score function, such as cosine similarity, to produce the final relevance score. In this paradigm, there is a single global, static representation for each query and each document. Specifically, the document's embedding remain the same regardless of the document length, the content structure of document (e.g., multiple topics) and the variation of queries that are relevant to the document. It is very common that a document with hundreds of tokens might contain several distinct subtopics, some important semantic information might be easily missed or biased by each other when compressing a document into a dense vector.  As such, this simple bi-encoder structure may cause serious information loss when used to encode documents. <sup>[^2]</sup>

On the other hand,  cross-encoders based on BERT variants [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:classicRepresentationLearning}] utilize multiple self-attention layers not only to extract contextualized features from queries and documents but also capture the interactions between them. Cross-encoders only produce intermediate representations that take a pair of query and document as the joint input. While BERT-based cross-encoders brought significant performance gain,  they are computationally prohibitive and impractical for online inference. 

In this section, we focus on different strategies {cite}`humeau2019poly, tang2021improving, luan2021sparse` to encode documents by multi-vector representations, which enriches the single vector representation produced by a bi-encoder. With additional computational overhead, these strategies can gain much improvement of the encoding quality while retaining the fast retrieval strengths of Bi-encoder.

### Token-level multi-vector representation

To enrich the representations of the documents produced by Bi-encoder, some researchers extend the original Bi-encoder by employing more delicate structures like later- interaction

ColBERT [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:colBERT}] can be viewed a token-level multi-vector representation encoder for both queries and documents. Token-level representations for documents can be pre-computed offline. During online inference, late interactions of the query's multi-vectors representation and the document's  multi-vectors representation are used to  improve the robustness of dense retrieval, as compared to inner products of single-vector representations. Specifically,

Formally, given $q= q_{1} \ldots q_{l}$ and $d=d_{1} \ldots d_{n}$ and their token level embeddings $\{E_{q_1},\ldots E_{q_l}\}$ and $\{E_{d_1},...,E_{d_n}\}$ and the final similarity score between the $q$ and $d$ is given by
$$
S_{q, d} =\sum_{i \in I_q} \max _{j \in I_d} E_{q_{i}} \cdot E_{d_{j}}^{T},
$$
where $I_q = \{1,...,l\}$, $I_d = \{1, ..., n\}$ are the index sets for query token embeddings and document token embeddings, respectively. 

While this method has shown signficant improvement over bi-encoder methods, it has a main disadvantage of high storage requirements. For example, ColBERT requires storing all the WordPiece token vectors of each text in the corpus. 

### Semantic clusters as pseudo query embeddings

The primary limitation of Bi-encoder is information loss when we condense the document into a query agnostic dense vector representation. Authors in {cite}`tang2021improving` proposed the idea of representing a document by its semantic salient fragments [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:pseudoqueryembeddings}]. These semantic fragments can be modeled by token embedding vector clusters in the embedding space. By performing clustering algorithms (e.g., k-means) on token embeddings, the generated centroids can be used as a document's multi-vector presentation. Another interpretation is that these centroids can be viewed as multiple potential queries corresponding to the input document; as such, we can call them *pseudo query embeddings*. 

```{figure} images/../deepLearning/ApplicationIRSearch/MultivectorRepresentation/pseudo_query_embedding/pseudo_query_embeddings
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:pseudoqueryembeddings
Deriving semantic clusters by clustering token-level embedding vectors. Semantic cluster centroids are used as multi-vector document representation, or as pseudo query embeddings. The final relevance score between a query and a document is computed using attententive pooling of centroid cluster and doc product. 
```

There are a couple of steps to calculate the final relevance score between a query and a document. First, we encode the query $q$ into a dense vector $e_q$ (via the CLS token embedding) and the document $d$ into multiple vectors via token level encoding and clustering $c_1,...,c_K$. Second, the query-conditional document representation $e_d$ is obtained by attending to the centroids using $e_q$ as the key. Finally, we can compute the similarity score via dot product between $e_q$ and $e_d$. 

In summary, we have
```{math}
\begin{align*}
a_{j}&=\operatorname{Softmax}\left(e_{q} \cdot c_{j}\right) \\
	e_{d}&=\sum_{j=1}^{k} a_{j} c_{j} \\
	y &=e_{q} \cdot e_{d}
\end{align*}
```

In practice, we can save the centroid embeddings in memory and retrieve them using the real queries. 

\iffalse

\begin{remark}[compare with Poly-Encoder]
	Comparing with existing work, we find that the Poly-Encoder(learnt-k) (Humeau et al., 2019) is equivalent to learning multiple fixed global pseudo query embeddings $\left\{c_{j}\right\}_{j=1}^{k}$ across all of the documents. That model treats the pseudo query embeddings as learnable parameters which are kept fixed during the inference. It uses the linear combinations of document token embeddings $\left\{d_{i}\right\}_{i=1}^{m}$ as the compressed document embeddings, taking similarity scores between $\left\{d_{i}\right\}_{i=1}^{m}$ and $\left\{c_{j}\right\}_{j=1}^{k}$ as the combination weights. Conversely, the PolyEncoder(first-k) (Humeau et al., 2019) and MEBERT(Luan et al., 2020) use the first $k$ document token embeddings as the pseudo query embeddings, i.e., $\left\{c_{j}\right\}_{j=1}^{k}=\left\{d_{i}\right\}_{i=1}^{k}$ and adopt the pseudo i.e., $\left\{c_{j}\right\}_{j=1}=\left\{d_{i}\right\}_{i=1}^{k}$ and adopt the pseudo query embeddings as compressed document embeddings. In contrast to Poly-Encoder(learnt-k), they rely on dynamic pseudo query embeddings. Experimental results on conversation datasets show PolyEncoder(first-k) is better than the former. However, only adopting the first- $k$ document embeddings seems to be a coarse strategy since a lot of information may exist in the latter part of the document. To this end, we present an approach which generates multiple adaptive semantic embeddings for each document by exploring all of the contents in the document.	

\end{remark}

### Polyencoder

another more sophisticated approach is to employ different encoders for queries and documents, where the document encoder abstracts the content into multiple embeddings—each embedding captures some aspects of the document, while the query encoder obtains a single embedding for each query

\cite{humeau2019poly}

### ME-BERT
\cite{luan2021sparse}

### Phrase representation
\cite{lee2020learning}
\cite{lee2021phrase}

\fi

## Enhancing sparse IR via dense methods

### Query and document expansion

#### Overview

Query expansion and document expansion techniques provide two potential solutions to the inherent vocabulary mismatch problem in traditional IR systems. The core idea is to add extra relevant terms to queries and documents respectively to aid relevance matching. 

Consider an ad-hoc search example of *automobile sales per year in the US*. 
- Document expansion can be implemented by appending *car* in documents that contains the term *automobile* but not *car*. Then an exact-match retriever can fetch documents containing either *car* and *automobile*.
- Query expansion can be accomplished by retrieving results from both the query *automobile sales per year in the US* and the query *car sales per year in the US*.

There are many different approaches to coming up suitable terms to expand queries and documents in order to make relevance matching easier. These approaches range from traditional rule-based methods such as synonym expansion to recent learning based approaches by mining user logs. For example, augmented terms for a document can come from queries that are relevant from user click-through logs. 
```{figure} images/../deepLearning/ApplicationIRSearch/QueryDocExpansion/queryExpansion/query_expansion_arch
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:queryexpansionarch
Query expansion module in an IR system.
```

Both query and document expansion can be fit into typical IR architectures through an de-coupled module. A query expansion module takes an input query and output a (richer) expanded query [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:queryexpansionarch}]. They are also known as query rewriters or expanders. The module might remove terms deemed unnecessary in the user’s query, for example stop words and add extra terms facilitate the engine to retrieve documents with a high recall. 

```{figure} images/../deepLearning/ApplicationIRSearch/QueryDocExpansion/documentExpansion/doc_expansion_arch
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:docexpansionarch
Document expansion module in an IR system.
```

Similarly, document expansion naturally fits into the retrieval and ranking pipeline [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:docexpansionarch}]. The index will be simply built upon the expanded corpus to provide a richer set of candidate documents for retrieval. The extra computation for document expansion can be all carried out offline. Therefore, it presents the same level of effectiveness like query expansion but at lower latency costs (for example, using less computationally intensive rerankers). 

Query expansion and document expansion have different pros and cons. Main advantages of query expansions include
- Compare to document expansion, query expansion techniques can be quickly implemented and experimented without modifying the entire indexing. On the other hand, experimenting document expansion techniques can be costly and time-consuming, since the entire indexing is affected. 
- Query expansion techniques are generally more flexible. For example, it is easy to switch on or off different features at query time (for example, selectively apply expansion only to certain intents or certain query types). 
	The flexibility of query expansion also allow us insert an expansion module in different stages of the retrieval-ranking pipeline. 

On the other hand, one unique advantage for document expansion is that: documents are typically much longer than queries, and thus offer more context for a model to choose appropriate expansion terms. Neural based natural language generation models, like Transformers {cite}`vaswani2017attention`, can benefit from richer contexts and generate cohesive natural language terms to expand original documents. 

#### Document expansion via query prediction

Authors in {cite}`nogueira2019document` proposed DocT5Query, a document expansion strategy based on a seq-to-seq natural language generation model to enrich each document. 

For each document, the transformer generation model predicts a set of queries that are likely to be relevant to the document. Given a dataset of (query, relevant document) pairs, we use a transformer model is trained to takes the document as input and then to produce the target query<sup>[^3]</sup>.

```{figure} images/../deepLearning/ApplicationIRSearch/QueryDocExpansion/documentExpansion/Doc2query_arch
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:doc2queryarch
Given a document, our Doc2query model predicts a query, which is appended to the document. Expansion is applied to all documents in the corpus, which are then indexed and searched as before. Image from {cite}`nogueira2019document`.
```

Once the model is trained, we can use the model to predict top-$k$ queries using beam search and append them to each document in the corpus. 

Examples of query predictions on MS MARCO compared to real user queries.
- Input Document:  *July is the hottest month in Washington DC with an average temperature of 80F and the coldest is January at 38F with the most daily sunshine hours at 9 in July. The wettest month is May with an average of $100 \mathrm{&nbsp;mm}$ of rain.*
- Predicted Query:  *weather in washington dc*
- Target Query:  *what is the temperature in washington*

Another example:
- Input Document: *sex chromosome - (genetics) a chromosome that determines the sex of an individual; mammals normally have two sex chromosomes chromosome - a threadlike strand of DNA in the cell nucleus that carries the genes in a linear order; humans have 22 chromosome pairs plus two sex chromosomes.*
- Predicted Query: *what is the relationship between genes and chromosomes*
- Target Query: *which chromosome controls sex characteristics *

This document expansion technique has demonstrated its effectiveness on the MS MARCO dataset when it is combined with BM25. The query prediction improves the performance from two aspects:
- Predicted queries tend to copy some words from the input document (e.g., Washington DC, chromosome), which is sort of equivalent to performing term re-weighting (i.e., increasing the importance of key terms).
- Predicted queries might also contain words not present in the input document (e.g., weather), which can be characterized as expansion by synonyms and other related terms. 

A widely used relevance feedback algorithm was developed by Rocchio {cite}`rocchio1971relevance` for vector space models. 
 Let $\left\{d_{1}, d_{2}, \cdots, d_{k}\right\}$ be the vectors top $k$ documents retrieved by SNRM in response to the query vector $q$. The updated query vector is computed as:
$$
\vec{q}^{*}=\vec{q}+\alpha \frac{1}{k} \sum_{i=1}^{k} \vec{d}_{i}
$$
where $\alpha$ controls the weight of the feedback vector. In practice we only keep the top $t$ (e.g., $t=10 -20$) terms with the highest values in the updated query vector $\vec{q}^{*}$.

A continued study in this line shows that replacing the transformer with more powerful seq-to-seq transformer model T5 {cite}`raffel2019exploring`  can bring further performance gain.

### Pseudo relevance feedback

#### Basics

Pseudo relevance feedback (PRF) is another commonly used technique to boost the performance of traditional IR models and to reduce the effect of query-document vocabulary mismatches and improve the estimate the term weights. The interest of using PRF has been recently expanded into the neural IR models {cite}`li2018nprf`.

```{figure} images/../deepLearning/ApplicationIRSearch/PseudoRelevanceFeedback/PRF_arch
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:prfarch
A typical architecture for pseudo relevance feedback implementation.
```

\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:prfarch} shows a typical architecture for pseudo relevance feedback implementation. There are two rounds of retrieval. In the initial retrieval, the retriever fetches a batch of relevant documents based on the original query. We can use the top-$k$ documents to expand and refine the original query. In the second round retrieval, the retriever fetches relevant documents as the final result based on the expanded query. Intuitively, the first-round retrieved documents help identify terms not present in the original query that are discriminative of relevant texts. After the expansion, the expanded query effective mitigate the vocabulary gap between original query and the corpus. 

#### Neural pseudo relevance feedback (NPRF)

**Overview**
Given a query q, NPRF estimates the relevance of a target document $d$ relative to $q$ using following key procedures []
	1. Create initial retrieval result. Given a document corpus $\cD$, a simple ranking method (e.g., BM25) $\operatorname{rel}_{q}(q, d)$ is applied to each $d\in \cD$ to obtain the top-$m$ documents, denoted as $D_{q}$ for $q$.
	1. Compute document-document relevance. We extract the relevance between each $d_{q}\in D_{q}$ and the target $q$, using a neural ranking method $\operatorname{rel}_{d}(d_{q}, d)$.
	1. Compute final relevance.  The relevance scores $\operatorname{rel}_{d}(d_{q}, d)$ from previous step weighted by $rel_{q}(q, d_{q})$ to arrive at $\operatorname{rel}'_{d}\left(d_{q}, d\right)$. The weighting which serves as an estimator for the confidence of the contribution of $d_{q}$ relative to $q$. Finally, we relevance between $q$ and $d$ is given by the aggregation of these adjusted relevance scores, $$\operatorname{rel}_{D}(q, D_{q}, d) = \sum_{d_q\in D_q} \operatorname{rel}'_{d}(d_q, d).$$

```{figure} images/../deepLearning/ApplicationIRSearch/PseudoRelevanceFeedback/NPRF_arch
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:prfarch
Overall architecture and procedure for neural pseudo relevance feedback (NPRF) implementation.
```

**Model architecture**

The NPRF framework begins with an initial ranking for the input query $q$ determined by $\operatorname{rel}_{q}(., .)$, which forms $D_{q}$, the set of the top- $m$ documents $D_{q}$. The ultimate query-document relevance score $\operatorname{rel}_{D}\left(q, D_{q}, d\right)$ is computed as follows.

Extracting document interactions. Given the target document $d$ and each feedback document $d_{q} \in D_{q}, r e l_{d}(., .)$ is used to evaluate the relevance between $d$ and $d_{q}$, resulting in $m$ real-valued relevance scores, where each score corresponds to the estimated relevance of $d$ according to one feedback document $d_{q}$.

As mentioned, two NIRMs are separately used to compute rel $_{d}\left(d_{q}, d\right)$ in our experiments. Both models take as input the cosine similarities between each pair of terms in $d_{q}$ and $d$, which are computed using pre-trained word embeddings as explained in Section 3.1. Given that both models consider only unigram matches and do not consider term dependencies, we first summarize $d_{q}$ by retaining only the top- $k$ terms according to their $t f$-idf scores, which speeds up training by reducing the document size and removing noisy terms. In our pilot experiments, the use of top- $k$ tf-idf document summarization did not influence performance. For different $d_{q} \in D_{q}$, the same model is used as $\operatorname{rel}_{d}(., .)$ for different pairs of $\left(d_{q}, d\right)$ by sharing model weights.

Combining document interactions. When determining the relevance of a target document $d$, there exist two sources of relevance signals to consider: the target document's relevance relative to the feedback documents $D_{q}$ and its relevance relative to the query $q$ itself. In this step, we combine rel ${ }_{d}\left(d_{q}, d\right)$ for each $d_{q} \in D_{q}$ into an overall feedback document relevance score

$\operatorname{rel}_{D}\left(q, D_{q}, d\right)$. When combining the relevance scores, the agreement between $q$ and each $d_{q}$ is also important, since $d_{q}$ may differ from $q$ in terms of information needs. The relevance of $d_{q}$ from the initial ranking rel ${ }_{q}\left(q, d_{q}\right)$ is employed to quantify this agreement and weight each $\operatorname{rel}_{d}\left(d_{q}, d\right)$ accordingly.

When computing such agreements, it is necessary to remove the influence of the absolute ranges of the scores from the initial ranker. For example, ranking scores from a language model (Ponte and Croft, 1998) and from BM25 (Robertson et al., 1995) can differ substantially in their absolute ranges. To mitigate this, we use a smoothed $\min -\max$ normalization to rescale $\operatorname{rel}_{q}\left(q, d_{q}\right)$ into the range $[0.5,1]$. The min-max normalization is applied by considering $\min \left(\operatorname{rel}_{q}\left(q, d_{q}\right) \mid d_{q} \in\right.$ $\left.D_{q}\right)$ and $\max \left(r e l_{q}\left(q, d_{q}\right) \mid d_{q} \in D_{q}\right)$. Hereafter, $r_{q} l_{q}\left(q, d_{q}\right)$ is used to denote this relevance score after min-max normalization for brevity. The (normalized) relevance score is smoothed and then weighted by the relevance evaluation of $d_{q}$, producing a weighted document relevance score rel $_{d}{ }^{\prime}\left(d_{q}, d\right)$ for each $d_{q} \in D_{q}$ that reflects the relevance of $d_{q}$ relative to $q$. This computation is described in the following equation.
$$
\operatorname{rel}_{d}{ }^{\prime}\left(d_{q}, d\right)=\operatorname{rel}_{d}\left(d_{q}, d\right)\left(0.5+0.5 \times \operatorname{rel}_{q}\left(q, d_{q}\right)\right)
$$
As the last step, we propose two variants for combining the $r e l_{d}{ }^{\prime}\left(d_{q}, d\right)$ for different $d_{q}$ into a single score $r e l_{D}\left(q, D_{q}, d\right)$ : (i) performing a direct summation and (ii) using a feed forward network with a hyperbolic tangent ( $\tanh$ ) non-linear activation. Namely, the first variant simply sums up the scores, whereas the second takes the ranking positions of individual feedback documents into account.

**Optimization and Training**
Each training sample consists of a query $q$, a set of $m$ feedback documents $D_{q}$, a relevant target document $d^{+}$and a non-relevant target document $d^{-}$according to the ground truth. The Adam optimizer (Kingma and $\mathrm{Ba}, 2014$ ) is used with a learning rate $0.001$ and a batch size of 20 . Training normally converges within 30 epochs, with weights uniformly initialized. A hinge loss is employed for training as shown below.
$$
\operatorname{loss}\left(q, D_{q}, d^{+}, d^{-}\right)= \max \left(0,1-\operatorname{rel}\left(q, D_{q}, d^{+}\right)+\operatorname{rel}\left(q, D_{q}, d^{-}\right)\right)
$$

### Contextualized term importance

#### Context-aware term importance: Deep-CT

In ad-hoc search, queries are mainly short and keyword based without complex grammatical structures. To be able to fetch most relevant results, it is  important to take into account term importance. For example,  given the query *bitcoin news*, a relevant document is expected to be about *bitcoin* and *news*, where the term *bitcoin* is more important than *news* in the sense that a document describing other aspects of bitcoin would be more relevant than a document describing news of other things.

In the traditional IR framework, term importance is calculated using inverse document frequency. A term is less important if it is a common term appearing in a large number of documents. These frequency-based term weights have been a huge success in traditional IR systems due to its simplicity and scalability. The problematic aspect is that Tf-idf determines the term importance solely based on word counts rather than the semantics. High-frequency words  does not necessarily indicate their central role to the meaning of the text, especially for short texts where the word frequency distribution is quite flat. Considering the following two passages returned for the query stomach {cite}`dai2019context`, one is relevant and one is not:
- Relevant: In some cases, an upset stomach is the result of an allergic reaction to a certain type of food. It also may be caused by an irritation. Sometimes this happens from consuming too much alcohol or caffeine. Eating too many fatty foods or too much food in general may also cause an upset stomach.
- Less relevant: All parts ofthe body (muscles , brain, heart, and liver) need energy to work. This energy comes from the food we eat. Our bodies digest the food we eat by mixing it with fluids( acids and enzymes) in the stomach. When the stomach digests food, the carbohydrate (sugars and starches) in the food breaks down into another type of sugar, called glucose.

In both passages, the word *stomach* appear twice; but the second passage is actually off-topic. This example also suggests that the importance of a term depends on its context, which helps understand the role of the word playing in the text. 

Authors in {cite}`dai2019context` proposed DeepCT, which uses the contextual word representations from BERT to estimate term importance to improve the traditional IR approach. Specifically, given a word in a specific text, its contextualized word embedding (e.g., BERT) is used a feature vector that characterizes the word's syntactic and semantic role in the text. Then DeepCT estimates the word's importance score via a weighted summation:
$$
\hat{y}_{t, c}= {w} T_{t, c}+b
$$
where $T_{t, c} \in \R^D$ is token $t$ 's contextualized embedding in the text $c$; and, ${w}\in \R^D$ and $b$ are the weights and bias.

The model parameters of DeepCT are the weight and bias, and they can be estimated from a supervised learning task, per-token regression, given by 
$$
L_{MSE}=\sum_{c} \sum_{t}\left(y_{t, c}-\hat{y}_{t, c}\right)^{2}.
$$

The ground truth term weight $y_{t, c}$ for every word in either the query or the document are estimated in the following manner. 
- The importance of a term in a document $d$ is estimated by the occurrence of the term in relevant queries [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:deepcttermimportancedemo}]. More formally, it is given by
	$$
	QTR(t, d)=\frac{\left|Q_{d, t}\right|}{\left|Q_{d}\right|}
	$$
	$Q_{d}$ is the set of queries that are relevant to document $d$. $Q_{d, t}$ is the subset of $Q_{d}$ that contains term $t$. The intuition is that words that appear in relevant queries are more important than other words in the document.
- The importance of a term in a document $d$ is estimated by the occurrence of the term in relevant queries [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:deepcttermimportancedemo}]. More formally, it is given by
	$$
	TR(t, q)=\frac{\left|D_{q, t}\right|}{\left|D_{q}\right|}
	$$
	$D_{q}$ is the set of documents that are relevant to the query $q$. $D_{q, t}$ is the subset of relevant documents that contains term $t$. The intuition is that a query term is more important if it is mentioned by more relevant documents.

```{figure} images/../deepLearning/ApplicationIRSearch/termImportance/deepCT/deepCT_term_importance_demo
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:deepcttermimportancedemo
Illustration of calculating context-aware term importance for a query (left) and a document (right). Term importance in a query is estimated from relevant documents of the query; term importance in a document is estimated from relevant queries of the document.
```

\iffalse

#### Learnable context-aware term importance: Deep-Impact

DeepCT has an interesting “quirk”: in truth, it only learns the term frequency (tf) component of term weights, but still relies on the remaining parts of the BM25 scoring function via the gen- eration of pseudo-documents. This approach also has a weakness: it only assigns weights to terms that are already present in the document, which limits retrieval to exact match. This is an impor- tant limitation that is addressed by the use of dense representations, which are capable of capturing semantic matches.

Deep CT learning independent term-level scores without taking into account the term co-occurrences in the document.

DeepCT (Dai and Callan, 2019), which uses a transformer to learn term weights based on a re- gression model, with the supervision signal coming from the MS MARCO passage ranking test collection

DeepImpact brought together two key ideas: the use of document expansion to iden- tify dimensions in the sparse vector that should have non-zero weights and a term weighting model based on a pairwise loss between relevant and non- relevant texts with respect to a query. Expansion

DeepImpact aim at learning the final term impact jointly across all query terms occurring in a passage. 
DeepImpact learns richer interaction patterns among the impacts, when compared to training each im- pact in isolation.

To address vocabulary mismatch, DeepImpact leverages DocT5Query to enrich every document with new terms likely to occur in queries for which the document is relevant. 

**Network architecture**  The overall architecture of the Deeplmpact neural network is depicted in Figure 1. Deeplmpact feeds a contextual LM encoder the original document terms (in white) and the injected expansion terms (in gray), separating both by a [SEP] separator token to distinguish both contexts. The LM encoder produces an embedding for each input term. The first occurrence of each unique term is provided as input to the impact score encoder, which is a two-layer MLP with ReLU activations. This produces a single-value **positive score** for each unique term in the document, representing its impact. Given a query $q$, we model the score of document $d$ as simply the sum of impacts for the intersection of terms in $q$ and $d$. That is,
$$s(q, d) = \sum_{t\in q\cap d} \operatorname{ScoreEncoder}(h_t).$$

For each triple, two scores for the corresponding two documents are computed. The model is optimized via pairwise Softmax cross- entropy loss over the computed scores of the documents. 

```{figure} images/../deepLearning/ApplicationIRSearch/termImportance/deepImpact/deepImpact_term_importance_demo
:name: fig:deepimpacttermimportancedemo
DeepImpact architecture.
```

**Quantization and Query Processing.** In our approach we predict real-valued document-term scores, also called impact scores, that we store in the inverted index. Since storing a floating point value per posting would blow up the space requirements of the inverted index, we decided to store impacts in a quantized form. The quantized impact scores belong to the range of $\left[1,2^{b}-1\right]$, where $b$ is the number of bits used to store each value. We experimented with b = 8 using linear quantization, and did not notice any loss in precision w.r.t. the original scores. Since

### Contextualized sparse representation

#### Motivation

An important point to make here is that neural networks, particularly transformers, have not made sparse representations obsolete. Both dense and sparse learned representations clearly exploit transformers-the trick is that the latter class of techniques then "projects" the learned knowledge back into the sparse vocabulary space. This allows us to reuse decades of innovation in inverted indexes (e.g., integer coding techniques to compress inverted lists) and efficient query evaluation algorithms (e.g., smart skipping to reduce query latency): for example, the Lucene index used in our uniCOIL experiments is only $1.3 \mathrm{&nbsp;GB}$, compared to $\sim 40$ GB for COIL-tok, 26 GB for TCTColBERTv2, and 154 GB for ColBERT. We note, however, that with dense retrieval techniques, fixedwidth vectors can be approximated with binary hash codes, yielding far more compact representations with sacrificing much effectiveness (Yamada et al., 2021). Once again, no clear winner emerges at present.

The complete design space of modern information retrieval techniques requires proper accounting of the tradeoffs between output quality (effectiveness), time (query latency), and space (index size). Here, we have only focused on the first aspect. Learned representations for information retrieval are clearly the future, but the advantages and disadvantages of dense vs. sparse approaches along these dimensions are not yet fully understood. It'll be exciting to see what comes next!

#### COIL-token and uni-COIL

The recently proposed COIL architecture (Gao et al., 2021a) presents an interesting case for this conceptual framework. Where does it belong? The authors themselves describe COIL as "a new exact lexical match retrieval architecture armed with deep LM representations". COIL produces representations for each document token that are then directly stored in the inverted index, where the term frequency usually goes in an inverted list. Although COIL is perhaps best described as the intellectual descendant of ColBERT (Khattab and Zaharia, 2020), another way to think about it within our conceptual framework is that instead of assignmodel assigns each term a vector "weight". Query evaluation in COIL involves accumulating inner products instead of scalar weights.

```{figure} images/../deepLearning/ApplicationIRSearch/ContextualizedSparseEmbedding/COIL_tok_embedding
:name: fig:coiltokembedding

```

In another interesting extension, if we reduce the token dimension of COIL to one, the model degenerates into producing scalar weights, which then becomes directly comparable to DeepCT, row (2a) and the "no-expansion" variant of DeepImpact, row (2c). These comparisons isolate the effects of different term weighting models. We dub this variant of COIL "uniCOIL", on top of which we can also add doc2query-T5, which produces a fair comparison to DeepImpact, row ( $2 \mathrm{&nbsp;d})$. The original formulation of COIL, even with a token dimension of one, is not directly amenable to retrieval using inverted indexes because weights can be negative. To address this issue, we added a ReLU operation on the output term weights of the base COIL model to force the model to generate non-negative weights.

### Discussion

\begin{tabular}{lllr}
	\hline \multicolumn{2}{l}{ Sparse Representations } & & MRR@10 \\
	\hline & Term Weighting & Expansion & \\
	\hline (1a) & BM25 & None & $0.184$ \\
	(1b) & BM25 & doc2query-T5 & $0.277$ \\
	\hline (2a) & DeepCT & None & $0.243$ \\
	(2b) & DeepCT & doc2query-T5 & $?$ \\
	(2c) & DeepImpact & None & $?$ \\
	(2d) & DeepImpact & doc2query-T5 & $0.326$ \\
	(2e) & COIL-tok $(d=32)$ & None & $0.341$ \\
	(2f) & COIL-tok $(d=32)$ & doc2query-T5 & $0.361$ \\
	(2g) & uniCOIL & None & $0.315$ \\
	$(2 \mathrm{&nbsp;h})$ & uniCOIL & doc2query-T5 & $0.352$ \\
	\hline
\end{tabular}

\fi

## Knowledge distillation

### Introduction

Knowledge distillation aims to transfer knowledge from a well-trained, high-performing yet cumbersome teacher model to a lightweight student model with significant performance loss. Knowledge distillation has been a widely adopted method to achieve efficient neural network architecture, thus reducing overall inference costs, including memory requirements as well as inference latency. Typically, the teacher model can be an ensemble of separately trained models or a single very large model trained with a very strong regularizer such as dropout. The student model uses the distilled knowledge from the teacher network as additional learning cues. The resulting student model is computationally inexpensive and has accuracy better than directly training it from scratch.

As such, tor retrieval and ranking systems, knowledge distillation is a desirable approach to develop efficient models to meet the high requirement on both accuracy and latency. 

For example, one can distill knowledge from a more powerful cross-encoder (e.g., BERT cross-encoder in \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:monoBERT}) to a computational efficient bi-encoders. Empirically, this two-step procedure might be more effective than directly training a bi-encoder from scratch.

In this section, we first review the principle of knowledge distillation. Then we go over a couple examples to demonstrate the application of knowledge distillation in developing retrieval and ranking models.

\iffalse

Another alternative approach is to distill a more complex model (e.g., term-level representation learning method or interaction-focused model) to a document-level representation learning architecture. For example, Lin et al. [132] distilled the knowledge from ColBERT’s expressive MaxSim operator for computing relevance scores into a simple dot product, thus enabling a single- step ANN search. Their key insight is that during distillation, tight coupling between the teacher model and the student model enables more flexible distillation strategies and yields better learned representations. The approach improves query latency and greatly reduces the onerous storage requirement of ColBERT, while only making modest sacrifices in terms of effectiveness. Tahami et al. [196] utilized knowledge distillation to compress the complex BERT cross-encoder network as a teacher model into the student BERT bi-encoder model. This increases the prediction quality of BERT-based bi-encoders without affecting its inference speed. They evaluated the approach on three domain-popular datasets, and results show that the proposed method achieves statistically significant gains.

\fi
### Knowledge distillation training framework

In the classic knowledge distillation framework {cite}`hinton2015distilling, tang2019distilling`[\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:teacherstudentdistillationscheme}], the fundamental principle is that the teacher model produces soft label $q$ for each input feature $x$. Soft label $q$ can be viewed as a softened probability vector distributed over class labels of interest. 
Soft targets contain valuable information on the rich similarity structure over the data. Use MNIST classification as an example, a reasonable soft target will tell that 2 looks more like 3 than 9. These soft targets can be viewed as a strategy to mitigate the over-confidence issue and reduce gradient variance when we train neural networks using one-hot hard labels. Similar mechanism is leveraged in smooth label to improves model generalization. 

Allows the smaller Student model to be trained on much smaller data than the original cumbersome model and with a much higher learning rate

Specifically, the logits $z$ from the techer model are outputted to generate soft labels via
$$
q_{i}^T=\frac{\exp \left(z_{i} / T\right)}{\sum_{j} \exp \left(z_{j} / T\right)},
$$
where $T$ is the temperature parameter controlling softness of the probability vector, and the sum is over the entire label space. When $T=1$, it is equivalent to standard Softmax function. As $T$ grows, $q$ become softer and approaches uniform distribution $T=\infty$. On the other hand, as $T\to 0$, the $q$ approaches a one-hot hard label. 

```{figure} images/../deepLearning/ModelCompression/KnowledgeDistillation/teacher_student_distillation_scheme
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:teacherstudentdistillationscheme
The classic teacher-student knowledge distillation framework.
```

The loss function for the student network training is a weighted sum of the hard label based cross entropy loss and soft label based KL divergence. The rationale of KL (Kullback-Leibler) divergence is to use the softened probability vector from the teacher model to guide the learning of the student network. Minimizing the KL divergence constrains the student model's probabilistic outputs to match soft targets of the teacher model.

The loss function is formally given by
$$L =(1-\alpha) L_{C E}\left(p, y\right)+\alpha T^{2} {L}_{K L}\left(p^T, q^T\right)$$
where $L_{CE}$ is the regular cross entropy loss between predicted probability vector $p$ and the one-hot label vector
$${L}_{CE}\left(p, y\right) =-\sum_{j}{y}_{j} \log {p}_{j};$$
$L_{KL}$ is the KL divergence loss between the softened predictions at temperature $T$ from the student and the teacher networks, respectively:
$${L}_{KL}\left({p}^T, q^{T}\right) = -\sum_{j} {q}_{j}^{T} \log \frac{p_{j}^T}{{q}_{j}^{T}}.$$
Note that the same high temperature is used to produce distributions from the student model. 

Note that $L_{KL}(p^T, q^T) = L_{CE}(p^T, q^T) + H(q^T, q^T)$, with $H(q^T, q^T)$ being the entropy of probability vector $q^T$ and remaining as a constant during the training. As a result, we also often reduce the total loss to
$$L =(1-\alpha) L_{C E}\left(p, y\right)+\alpha T^{2} {L}_{CE}\left(p^T, q^T\right).$$

Finally, the multiplier $T^2$ is used to re-scale the gradient of KL loss and $\alpha$ is a scalar controlling the weight contribution to each loss. 

Besides using softened probability vector and KL divergence loss to guide the student learning process, we can also use MSE loss between the logits from the teacher and the student networks. Specifically, 
$$L_{MSE} = \norm{z^{(T)} - z^{(S)}}^2$$
where $z^{(T)}$ and $z^{(S)}$ are logits from the teacher and the student network, respectively. 

\begin{remark}[connections between MSE loss and KL loss]
	In {cite}`hinton2015distilling`, given a single sample input feature $x$, the gradient of ${L}_{KL}$ with respect to $z_{k}^{(S)}$ is as follows<sup>[^4]</sup>:
	$$
	\frac{\partial {L}_{KL}}{\partial {z}_{k}^{s}}=T\left(p_{k}^{T}-{q}_{k}^{T}\right).
	$$
	When $T$ goes to $\infty$, using the approximation $\exp \left({z}_{k}/ T\right) \approx 1+{z}_{k} / T$, the gradient is simplified to:
	$$
	\frac{\partial {L}_{KL}}{\partial {z}_{k}^{(S)}} \approx T\left(\frac{1+z_{k}^{(S)} / T}{K+\sum_{j} {z}_{j}^{(S)} / T}-\frac{1+{z}_{k}^{(T)} / T}{K+\sum_{j} {z}_{j}^{(T)} / T}\right)
	$$
	where $K$ is the number of classes.

	Here, by assuming the zero-mean teacher and student logit, i.e., $\sum_{j} {z}_{j}^{(T)}=0$ and $\sum_{j} {z}_{j}^{(S)}=0$, and hence $\frac{\partial {L}_{K L}}{\partial {z}_{k}^{(S)}} \approx \frac{1}{K}\left({z}_{k}^{(S)}-{z}_{k}^{(T)}\right)$. This indicates that minimizing ${L}_{KL}$ is equivalent to minimizing the mean squared error ${L}_{MSE}$, under a sufficiently large temperature $T$ and the zero-mean logit assumption for both the teacher and the student.
\end{remark}

### Example distillation strategies

#### Single cross-encoder teacher distillation

```{figure} images/../deepLearning/ApplicationIRSearch/KnowledgeDistllation/cross_encoder_distillation
:name: fig:crossencoderdistillation

```

#### Single bi-encoder teacher distillation

Authors in {cite}`vakili2020distilling, lu2020twinbert` pioneered the strategy of distilling powerful BERT cross-encoder into BERT bi-encoder to retain the benefits of the two model architectures: the accuracy of cross-encoder and the efficiency of bi-encoder.  

Knowledge distillation follows the classic soft label framework. Bi-encoder student model training can use pointwise ranking loss, which is equivalent to binary relevance classification problem given a query and a candidate document. More formally, given training examples $(q_i, d_i)$ and their labels $y_i\in \{0, 1\}$. The BERT cross-encoder as teacher model to produce soft targets for irrelevance label and relevance label.

Although cross-encoder teacher can offer accurate soft labels, it cannot directly extend to the **in-batch negatives** technique and **N-pair loss** [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:N_pair_loss}] when training the student model. The reason is that query and document embedding cannot be computed separately from a cross-encoder. Implementing in-batch negatives using cross-encoder requires exhaustive computation on all combinations between a query and possible documents, which amount to $|B|^2$ ($|B|$ is the batch size) query-document pairs.

Authors in {cite}`lin2021batch` proposed to leverage bi-encoder variant such as Col-BERT [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:colBERT}] can also be leveraged as a teacher model, which has the advantage that it is more feasible to perform exhaustive comparisons between queries and passages since they are passed through the encoder independently [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch::fig:inbatchdistillation}]. 

```{figure} images/../deepLearning/ApplicationIRSearch/KnowledgeDistllation/in_batch_distillation
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch::fig:inbatchdistillation
Compared to cross-encoder teacher, bi-encoder teacher computes query and document embeddings independents, which enables the application of the in-batch negative trick. Image from {cite}`lin2021batch`.
```

#### Ensemble teacher distillation

As we have seen in previous sections, large Transformer based models such as BERT cross-encoders or bi-encoders are popular choices of teacher models when we perform knowledge distillation. These fine-tuned BERT models often show high performance variances across different runs. From ensemble learning perspective, using an ensemble of models as a teacher model could potentially not only achieves better distillation results, but also reduces the performance variances. 

The critical challenge arising from distilling an ensemble teacher model vs a single teacher model is how to reconcile soft target labels generated by different teacher models. 

Authors in {cite}`zhuang2021ensemble` propose following method to fuse scores and labels. Formally, consider query $q_{i}$, its $j$-th candidate document $d_{i j}$, and $K$ teacher models. Let the predicted ranking score by the $k$-th teacher be represented as $\hat{s}_{i j}^{(k)}$. 

The simplest aggregated teacher label is to directly use the mean score, namely
$$
s_{i j}=\frac{1}{K}\hat{s}_{i j}^{(k)}.
$$

The simple average scheme would work poorly when teacher models can have outputs with very different scales. A more robust way to fuse scores is to reciprocal rank, given by
$$
s_{i j}^{(k)}=\frac{1}{K} \sum_{k=1}^{K}\frac{1}{C+\hat{r}_{i j}^{(k)}}
$$
where $\hat{r}_{i j}^{(k)}$ is the predicted rank of the $j$-th candidate text by the $k$-th teacher, and $C$ is the constant as model hyperparameters.

With the fused score, softened probability vector can be obtained by taking Softmax with temperature as the scaling factor.

\iffalse

\cite{hofstatter2020improving}
#### Multi-teacher distillation

\cite{choi2021improving}

```{figure} images/../deepLearning/ApplicationIRSearch/KnowledgeDistllation/multiteacher_distillation
:name: fig:multiteacherdistillation
: Architecture of TRMD-ColBERT. (left) Fine-tuned monoBERT is used as a cross-encoder teacher. monoBERT is used to distill the knowledge from its CLS representation to the student. (middle) TRMD-ColBERT estimates the score by combining
		scores from two rankers, each of which uses the same representations as those used by each teacher ranker. TRMD-ColBERT
		is trained by distilling the knowledge from the representation of each teacher to the student. (right) Fined-tuned ColBERT is
		used as a bi-encoder teacher. ColBERT is used to distill knowledge from its representations to the student.
```

**Architecture**

As shown in Figure 1, TRMD-ColBERT combines two rankers on top of the BERT encoder such that it can learn from two different teachers. With this structural modification, the bi-encoder student (ColBERT) can directly integrate the knowledge of both teachers (monoBERT and ColBERT). Each ranker in the resulting TRMDColBERT uses the same BERT representation as what the ranker of its respective teacher uses.

While ColBERT has two CLS representations (for query and document), a ranker in monoBERT uses only one CLS representation that reflects query-document interaction. Hence, adding monoBERT ranker to ColBERT is not straightforward. To address this issue, we combine two CLS representation vectors of ColBERT into one vector. Following TwinBERT [10], we apply residual function [3] to combine the two vectors, where the formal definition of the residual function is as follows.

Here, $x$ is the max of the query's and document's CLS vectors and $F$ is a linear mapping function from $x$ to the residual with parameters is a linear mapping function from $x$ to the residual with parameters the resulting CLS representation to estimate the score $S_{\text {monoBERT }}$. TRMD-ColBERT also utilizes a ColBERT ranker for relevance estimation, producing $S_{\text {ColBERT. }}$ A ranker corresponding to ColBERT uses the same method to calculate the relevance score as in the original ColBERT teacher. The total score of TRMD-ColBERT, $S_{\text {total }}$, is obtained by adding the two scores from the two rankers.
TRMD-Twizmere colberal TRMD-TwinBERT can be constructed in the same manner as in constructing TRMD-ColBERT. One notable point is that TwinBERT ranker can use either CLS representation or query/document representation. In our experiment, we chose TwinBERT using query/document representation to make TwinBERT ranker use a different representation from monoBERT's choice of representation.

**Learning**

Knowledge distillation [4] has been used to train a small but highperforming model by forcing a small student model to learn from a large teacher model or from multiple teachers [2, 14]. TRMD trains a student model by distilling knowledge from two teachers with different BERT encoder types, a cross-encoder and a bi-encoder. As shown in Figure 1, monoBERT and ColBERT become the teachers, a single bi-encoder ColBERT becomes the student model, and each teacher distills the knowledge contained in its BERT output representation to the student's representation. Before applying knowledge distillation, we independently fine-tune the pre-trained teacher models such that their individual performance is optimized.
During the training of the student model with knowledge distillation, we introduce three loss terms. First, the hard prediction loss $\mathcal{L}_{\text {hard } \text { pred }}$ is defined as the hinge loss between the student's prediction score and the true relevance score of the document, and it is shown in Equation (3). Second, CLS loss $\mathcal{L}_{C L S}$ encourages the student model to learn the CLS representation from the monoBERT teacher using MSE as in Equation (4). Third, representation loss $\mathcal{L}_{R E P}$ forces the student's representations to resemble the representations of ColBERT using MSE between the two representations as in Equation (5).
$$
\begin{aligned}
	\mathcal{L}_{\text {hard }-\text { pred }} &=\text { Hingeloss }\left(\operatorname{softmax}\left(z^{S}\right)\right) \\
	\mathcal{L}_{C L S} &=M S E\left(C L S^{T}, C L S^{S}\right) \\
	\mathcal{L}_{R E P} &=M S E\left(R E P^{T}, R E P^{S}\right)
\end{aligned}
$$
In Equations (3), (4), and (5), $z$ is a vector whose elements are the predicted scores of a positive document and a negative document, CLS stands for a CLS representation vector of BERT, REP is the whole representation generated by BERT including CLS, query and document representation vectors, and $(\cdot)^{T}$ and $(\cdot)^{S}$ indicate the teacher model and the student model, respectively. We define the total loss for the distillation process as the sum of these three losses as in Equation (6).
$$
\mathcal{L}_{\text {total }}=\mathcal{L}_{\text {hard-pred }}+\mathcal{L}_{C L S}+\mathcal{L}_{R E P}
$$

\fi

\iffalse

## Neural sparse representation learning model

### Motivation

### SNRM

\cite{zamani2018neural}

**Motivation**

Neural ranking models learn dense representations causing essentially every query term to match every document term, making it highly inefficient or intractable to rank the whole collection. The reliance on a first stage ranker creates a dual problem: First, the interaction and combination effects are not well understood. Second, the first stage ranker serves as a “gate-keeper” or filter, effectively blocking the potential of neural models to uncover new relevant documents.

Our approach addresses this head-on: by enforcing and rewarding
sparsity in the representation learning, we create a latent represen- tation that aims to capture meaningful semantic relations while still parsimoniously matching documents. This

**Architecture**

Our neural framework consists of three major components: the query representation $\phi_{Q}$, the document representation $\phi_{D}$, and the matching function $\psi$. The retrieval score for each query document pair $(q, d)$ is then computed as follows:
$$
\text { retrieval } \operatorname{score}(q, d)=\psi\left(\phi_{Q}(q), \phi_{D}(d)\right)
$$
where $\phi_{D}$ is a query independent component, and thus can be computed offline. In contrast to previous neural ranking models that learn low-dimensional dense representations, $\phi_{D}$ should output a high-dimensional sparse vector. This will allow us to construct an inverted index based on the learned representations. Further, $\phi_{Q}$ and $\phi_{D}$ in SNRM share parameters between these components. 

Based on this argument, we need a representation learning model in which the representation sparsity is a function of the input length. We propose an architecture based on ngram representation learning. The intuition behind our model is that we first learn a sparse representation for each continuous $n$ words in the given document or query. The learned sparse representations are then aggregated via average pooling. In fact, our document representation (and similarly our query representation) is obtained as follows:
$$
\phi_{D}(d)=\frac{1}{|d|-n+1} \sum_{i=1}^{|d|-n+1} \phi_{\text {ngram }}\left(w_{i}, w_{i+1}, \cdots, w_{i+n-1}\right)
$$
where $w_{1}, w_{2}, \cdots, w_{|d|}$ denote the terms appearing in document $d$ with the same order. $\phi_{\text {ngram }}$ learns a sparse representation for the given ngram. The query representation is also computed using the same function. This approach provides two important advantages:
- The number of representations averaged in Equation (2) has a direct relationship with the input document/query length. Therefore, the level of the sparsity in the final learned representations depends on the length of the input text. This results in more sparse representations for queries in comparison to documents, which is desired.
- Using the sliding window for encoding the input words as a set of ngrams helps to capture the local interaction among terms, hence the model considers term dependencies. Term dependencies have been widely known to be useful for improving the retrieval performance [31].

We model our ngram representation function $\phi_{\text {ngram }}$ by a fullyconnected feed-forward network that reads the a sliding window over the sequence of their embeddings and en- codes their ngrams.

These fully-connected layers have an hourglass structure that forces the information of the input data to be compressed and passed through a small number of units in the middle layers that are meant to learn the low dimensional manifold of the data. Then the number of hidden units increases in upper layers to give us a high-dimensional output, e.g., 20,000 . Note that in terms of the structure and the dimension of layers, the model looks like an autoencoder. However, unlike autoencoders, we have no reconstruction loss. 

```{figure} images/../deepLearning/ApplicationIRSearch/SparseRepresentationLearning/SNRM/sparse_encoder_arch
:name: fig:sparseencoderarch
Learning a latent sparse representation for a document. 
```

**Training**

To train the SNRM framework we have two objectives: the retrieval objective with the goal of improving retrieval accuracy and the sparsity objective with the goal of increasing sparseness in the learned query and document representations.

Let $T=\left\{\left(q_{1}, d_{11}, d_{12}, y_{1}\right), \cdots,\left(q_{N}, d_{N 1}, d_{N 2}, y_{N}\right)\right\}$ denote a set of $N$ training instances; each containing a query string $q_{i}$, two document candidates $d_{i 1}$ and $d_{i 2}$, and a label $y_{i} \in\{-1,1\}$ indicating which document is more relevant to the query. In the following, we explain how we optimize our model to achieve both of the mentioned goals.
Retrieval Objective We train our model using a pairwise setting as depicted in Figure 2 a. We employ hinge loss (max-margin loss function) that has been widely used in the learning to rank literature for pairwise models [25]. Hinge loss is a linear loss function that penalizes examples violating a margin constraint. The hinge loss for the $i^{\text {th }}$ training instance is defined as follows:
$$
\mathcal{L}=\max \left\{0, \epsilon-y_{i}\left[\psi\left(\phi_{Q}\left(q_{i}\right), \phi_{D}\left(d_{i 1}\right)\right)-\psi\left(\phi_{Q}\left(q_{i}\right), \phi_{D}\left(d_{i 2}\right)\right)\right]\right\}
$$
where $\epsilon$ is a hyper-parameter determining the margin of hinge loss.

Sparsity Objective In addition to improving the retrieval accuracy, our model aims at maximizing the sparsity ratio, which is defined as follows:
$$
\text { sparsity ratio }(\vec{v})=\frac{\text { total number of zero elements in } \vec{v}}{|\vec{v}|}
$$

that approximates the true minimum for $L_{0}$ is NP-hard [36]. Therefore, a tractable surrogate loss function should be considered. An alternative would be minimizing $L_{1}$ norm (i.e., $\left.L_{1}(\vec{v})=\sum_{i=1}^{|\vec{v}|}\left|\vec{v}_{i}\right|\right)$. 

The final loss function for the $i^{\text {th }}$ training instance is defined as follows:
$$
\mathcal{L}\left(q_{i}, d_{i 1}, d_{i 2}, y_{i}\right)+\lambda L_{1}\left(\phi_{Q}\left(q_{i}\right)\left\|\phi_{D}\left(d_{i 1}\right)\right\| \phi_{D}\left(d_{i 2}\right)\right)
$$
where $\|$ means concatenation. The hyper-parameter $\lambda$ controls the sparsity of the learned representations.

```{figure} images/../deepLearning/ApplicationIRSearch/SparseRepresentationLearning/SNRM/sparse_encoder_training
:name: fig:sparseencodertraining

```

\begin{remark}
Although it is clear that we can minimize $L_{1}$ as a term in our loss function, it is not immediately obvious how minimizing $L_{1}$ would lead to sparsity in the query and document representations. Employing the $L_{1}$ norm to promote sparsity has a long history, dating back at least to 1930 s for the Fourier transform extrapolation from partial observations [11]. $L_{1}$ has been also employed in the information theory literature for recovering band-limited signals [19]. Later on, sparsity minimization has received significant attention as a method for hyperparameter optimization in regression, known as the Lasso [45].
The theoretical justification for the fact that $L_{1}$ minimization would lead to sparsity in our model relies on two points: (1) the choice of rectified linear unit (ReLU) as the activation function forces the non-positive elements to be zero $(\operatorname{ReLU}(x)=\max \{0, x\})$, and (2) the gradient of $L_{1}(\vec{v})$ for an element of $\vec{v}$ is constant and thus independent of its value. Therefore, the gradient optimization approach used in the backpropagation algorithm [42] reduces the elements of the query and document representation vectors independent of their values. This moves small values toward zero and thus the desired sparsity is obtained.
\end{remark}

### SparTerm

\cite{bai2020sparterm}

**Motivation**
SRNM {cite}`zamani2018neural` learns latent sparse representations for the query and document based on dense neural models, in which the “latent” token plays the role of the traditional term during inverted indexing. One challenge about SNRM is that it loses the interpretability of the original terms, which is critical to industrial systems

In this paper, we propose a novel framework SparTerm to learn Term-based Sparse representations directly in the full vocabulary space. Equipped with the pre-trained language model, the proposed SparTerm learns a function to map the frequency-based BoW representation to a sparse term importance distribution in the whole vocabulary, which offers the flexibility to involve both term-weighting and expansion in the same framework. 

**Architecture**

More specifically, SparTerm comprises an importance predictor and a gating controller. The importance predictor maps the raw input text to a dense importance distribution in the vocabulary space, which is different from traditional term weighting methods that only consider literal terms of the input text. To ensure the sparsity and flexibility of the final representation, the gating controller is introduced to generate a binary and sparse gating signal across the dimension of vocabulary size, indicating which tokens should be activated. These two modules cooperatively yield a term-based sparse representation based on the semantic relationship of the input text with each term in the vocabulary.

```{figure} images/../deepLearning/ApplicationIRSearch/SparseRepresentationLearning/SparTerm/sparTerm_arch
:name: fig:spartermarch
Model Architecture of SparTerm. Our overall architecture contains an importance predictor and a gating controller. The importance predictor generates a dense importance distribution with the dimension of vocabulary size, while the gating controller outputs a sparse and binary gating vector to control term activation for the final representation. These two modules cooperatively ensure the sparsity and flexibility of the final representation.
```

3.2 The Importance Predictor
Given the input passage $p$, the importance predictor outputs semantic importance of all the terms in the vocabulary, which unify term weighting and expansion into the framework. As shown in Figure 2(b), prior to importance prediction, BERT-based encoder is employed to help get the deep contextualized embedding $h_{i}$ for each term $w_{i}$ in the passage $p$. Each $h_{i}$ models the surrounding context from a certain position $i$, thus providing a different view of which terms are semantically related to the topic of the current passage. With a token-wise importance predictor, we obtain a dense importance distribution $I_{i}$ of dimension $v$ for each $h_{i}$ :
$$
I_{i}=\text { Transform }\left(h_{i}\right) E^{\mathrm{T}}+b
$$
where Transform denotes a linear transformation with GELU activation and layer normalization, $E$ is the shared word embedding matrix and $b$ the bias term. Note that the token-wise importance prediction module is similar to the masked language prediction layer in BERT, thus we can initialize this part of parameters directly from pre-trained BERT. The final passage-wise importance distribution can be fetched simply by the summation of all token-wise importance distributions:
$$
I=\sum_{i=0}^{L} \operatorname{Relu}\left(I_{i}\right)
$$
where $L$ is the sequence length of passage $p$ and Relu activation function is leveraged to ensure the nonnegativity of importance logits.
3.3 The Gating Controller
The gating controller generates a binary gating signal of which terms to activate to represent the passage. First, the terms appearing in the original passage, which we referred to as literal terms, should be activated by the controller by default. Apart from the literal terms, some other terms related to the passage topic are also expected to be activated to tackle the "lexical mismatch" problem of BOW representation. Accordingly, we propose two kinds of gating controller: literal-only gating and expansion-enhanced gating, which can be applied in scenarios with different requirements for lexical matching.

Literal-only Gating. If simply setting $\mathcal{G}(p)=B O W(p)$, where $B o W(p)$ denotes the binary BoW vector for passage $p$, we get the literal-only gating controller. In this setting, only those terms existing in the original passage are considered activated for the passage representation. Without expansion for non-literal terms, the sparse representation learning is reduced to a pure term re-weighting scheme. Nevertheless, in the experiment part, we empirically show that this gating controller can achieve competitive retrieval performance by learning importance for literal terms.

Exapnsion-enhanced Gating. The expansion-enhanced gating controller activates terms that can hopefully bridge the "lexical mismatch" gap. Similar to the importance prediction process formulated by Equation (2) and Equation (3), we obtain a passage-wise dense term gating distribution $G$ of dimension $v$ with independent network parameters, as shown in Figure 2(c). Note that although the gating distribution $G$ and the importance distribution $I$ share the same dimension $v$, they are different in logit scales and mathematical implications. I represents the semantic importance of each term in vocabulary, while $G$ quantifies the probability of each term to participate in the final sparse representation. To ensure the sparsity of $p^{\prime}$, we apply a binarizer to $G$ :
$$
G^{\prime}=\operatorname{Binarizer}(G)
$$
where the Binarizer denotes a binary activation function which outputs only 0 or 1 . The gating vector for expansion terms $G_{e}$ is obtained by:
$$
G_{e}=G^{\prime} \odot(\neg B o W(p))
$$
where the bitwise negation vector $\neg B o W(p)$ is applied to ensure orthogonal to the literal-only gating. Simply adding the expansion gating and the literal-only gating, we get the final expansion-enhanced gating vector $G_{l e}$ :
$$
G_{l e}=G_{e}+B o W(p)
$$
Involving both literal and expansion terms, the final sparse representation can be a "free" distribution in the vocabulary space. Note that in the framework of SparTerm, expanded terms are not directly appended to the original passage, but are used to control the gating signal of whether allowing a term participating the final representation. This ensures the input text to the BERT encoder is always the natural language of the original passage.

\begin{tabular}{|l|l|}
	\hline Term expansion & Description and examples \\
	\hline Passage2query & Expand words that tend to appear in corresponding queries, i.e. "how far", "what causes". \\
	\hline Synonym & Expand synonym for original core words, i.e. "cartoon"->"animation". \\
	\hline Co-occurred words & Expand frequently co-occurred words for original core words, i.e. "earthquakes"->"ruins". \\
	\hline Summarization words & Expand summarization words that tend to appear in passage summarization or taggings. \\
	\hline
\end{tabular}

**Training**
In this section, we introduce the training strategy of the importance predictor and expansion-enhanced gating controller.

Training the importance predictor. The importance predictor is trained end-to-end by optimizing the ranking objective. Let $R=\left\{\left(q_{1}, p_{1,+}, p_{1,-}\right), \ldots,\left(q_{N}, p_{N,+}, p_{N,-}\right)\right\}$ denote a set of $N$ training instances; each containing a query $q_{i}$, a posotive candidate passage $p_{i,+}$ and a negative one $p_{i,-}$, indicating that $p_{i,+}$ is more relevant to the query than $p_{i,-}$. The loss function is the negative log likelihood of the positive passage:
$$
L_{r a n k}\left(q_{i}, p_{i,+}, p_{i,-}\right)=-\log \frac{e^{\operatorname{sim}\left(q_{i}^{\prime}, p_{i,+}^{\prime}\right)}}{e^{\operatorname{sim}\left(q_{i}^{\prime}, p_{i,+}^{\prime}\right)}+e^{\operatorname{sim}\left(q_{i}^{\prime}, p_{i,-}^{\prime}\right)}}
$$
where $q_{i}^{\prime}, p_{i,+}^{\prime}, p_{i,-}^{\prime}$ is the sparse representation of $q_{i}, p_{i,+}, p_{i,-}$ obtained by Equation (1), sim denotes any similarity measurement such as dot-product. Different with the training objective of DeepCT ??, we don't directly fit the statistical term importance distribution, but view the importance as intermediate variables that can be learned by distant supervisory signal for passage ranking. Endto-end learning can involve every terms in the optimization process, which can yield smoother importance distribution, but also of enough distinguishability.

Training the exapnsion-enhanced gating controller. We summarize four types of term expansion in Table 1, all of which can be optimized in our SparTerm framework. Intuitively, the pretrained BERT already has the ability of expanding synonym words and co-occured words by the Masked Language Model(MLM) pretraining task. Therefore, in this paper, we focus on expanding passage2query-alike and summarization terms. Given a passagequery/summary parallel corpus $\mathrm{C}$, where $p$ is a passage, $t$ the corresponding target text, and $T$ of dimension $v$ is the binary bag-ofwords vector of $t$. We use the binary cross-entropy loss to maximize probability values of all the terms in vocabulary:
$$
L_{\text {exp }}=-\lambda_{1} \sum_{j \in\left\{m \mid T_{m}=0\right\}} \log \left(1-G_{j}\right)-\lambda_{2} \sum_{k \in\left\{m \mid T_{m}=1\right\}} \log G_{k}
$$
where $G$ is the dense gating probability distribution for $p, \lambda_{1}$ and $\lambda_{2}$ two tunable hyper-parameters. $\lambda_{1}$ is the loss weight for terms expected not to be expanded, while $\lambda_{2}$ is for terms that appear in the target text. In the experiment, we set $\lambda_{2}$ much larger than $\lambda_{1}$ to encourage more terms to be expanded.

End-to-end joint training. Intuitively, the supervisory ranking signal can also be leveraged to guide the training of the gating controller, thus we can train the importance predictor and gating controller jointly:
$$
L=L_{\text {rank }}+L_{\exp }
$$

### Ultra-high dimensional BERT representation (UHD-BERT)

\cite{jang2021ultra}

```{figure} images/../deepLearning/ApplicationIRSearch/HybridRetrieval/UHD/UHD_arch
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:uhdarch
Overall architecture with the training scheme for building UHD sparse representations.
```

```{figure} images/../deepLearning/ApplicationIRSearch/HybridRetrieval/UHD/UHD_encoder_arch
:name: fig:uhdencoderarch
The Encoder structure.
```

**Overall architecture** \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:uhdarch} depicts the overall architecture of our Ultra-High Dimensional (UHD) model. UHD falls into the bi-encoder architecture paradigm, which comprises a query encoder, a document encoder, and a scoring function. While the query and document encoders are run separately, the weights are shared for the query and document encoders. After the final query and document representations are formed with sparsification and bucketization to be explained below, they are matched with dot product as the scoring function.

The encoder is composed of three modules: 1) the BERT module to convert text into dense token embeddings, 2) the Winner-Take-All (WTA) module that sparsifies the BERT module's outputs, and 3) the Max-pool module that performs non-interfering aggregation of sparse token embeddings. We define the final output from the WTA module as a bucket (Figure 2), implying that the final representation to be used for matching contains multiple buckets. The BERT and WTA modules are trained jointly with an objective to maximize the similarity between a query bucket and individual relevant document buckets.

**Encoder architecture**

Let $q=\left\{q_{1}, q_{2}, \ldots, q_{|q|}\right\}$ be a query and $d=$ $\left\{d_{1}, d_{2}, \ldots, d_{|d|}\right\}$ be a document. We obtain contextualized dense representations from BERT for the tokens in $q$ and $d$. Aside from the bucketed representations to be explained in Section 3.5, we here assume a common option of using only the last layer for a query or document representation.

An embedding for $q$ or $d$ is obtained as follows:
$$
\begin{aligned}
	&E_{q}=\operatorname{BERT}(q) \in \mathbb{R}^{|q| \times h} \\
	&E_{d}=\operatorname{BERT}(d) \in \mathbb{R}^{|d| \times h}
\end{aligned}
$$
where $h$ is the hidden size of BERT. Queries and documents are encoded separately with the same encoder.

To sparsify each dense token embedding, we adopt a WTA layer (Makhzani and Frey, 2015), which is an $n$-dimensional linear layer in which only top- $k$ activations $(k \ll n$ ) are preserved while the others are set to zero. Let $t_{i}$ be the $i^{\text {th }}$ token of a query or a document and $E_{t_{i}}$ be its dense embedding. Then a high-dimensional sparse representation $S_{t_{i}}$ is built as follows:
$$
\begin{aligned}
	z_{t_{i}} &=E_{t_{i}} \cdot W+b, W \in \mathbb{R}^{h \times n}, b \in \mathbb{R}^{n} \\
	S_{t_{i}}[\text { dim }] &= \begin{cases}z_{t_{i}}[\text { dim }], & \text { if } z_{t_{i}}[\operatorname{dim}] \in \text { top- } k\left(z_{t_{i}}\right) \\
		0, & \text { otherwise }\end{cases}
\end{aligned}
$$
where $\operatorname{dim} \in[1, n]$ is dimension of $z_{t_{i}}$ and $S_{t_{i}}$.
In order to drive the learning flow to winning signals, the WTA module considers only the gradients flowing back through the winning dimensions. In addition, we impose weight sparsity constraint proposed in Ahmad and Scheinkman (2019), which is like applying dropout (Srivastava et al., 2014) individually to output layer's nodes. The benefits of adopting WTA are: 1) We can control sparsity of a resulting embedding precisely and conveniently by adjusting $k$, an ability considered important for generating sparse representations reliably, and 2) $k$ can be modified at inference time so that the output's sparsity can be altered for an application's need without re-training the model. 

Sparse token embeddings generated by the encoder are aggregated with token-wise max-pool followed by L2-normalization to produce a single sparse embedding $B_{t}$, a bucket:
$$
B_{t}=\max -\operatorname{pool}\left(S_{t_{1}}, S_{t_{2}}, \ldots, S_{t_{|t|}}\right) \in \mathbb{R}^{n}
$$

**Representations with Multiple Buckets**

We use multiple buckets to encode information with different levels of abstraction coming from different layers of BERT, as shown in (Jo and Myaeng, 2020). We expect UHD representations extracted from multiple layers contain richer information than from a single layer (e.g., the last layer often used for a downstream task).

We first extract $V$ representations corresponding to the number of the BERT layers for all tokens $t$.
$$
E_{t}^{j}=B E R T^{j}(t) \in \mathbb{R}^{|t| \times h}
$$
where $j$ is a BERT layer $\in[1, V]$. WTA layers are then independently applied to all BERT layers as in Section $3.3$ to obtain $V$ buckets.
$$
B_{t}^{j}=W T A^{j}\left(E_{t}^{j}\right) \in \mathbb{R}^{n}
$$
After applying the bucketing mechanism, we obtain $B_{q}^{j}$ and $B_{d}^{j}$ for $q$ and $d$ so that a relevance score for the query and document is computed by a bucket-wise dot product as follows.
$$
\operatorname{Rel}(q, d)=\sum_{j} B_{q}^{j} \cdot B_{d}^{j}
$$

**Training** The entire model is trained to make a query similar to the relevant documents and dissimilar to the irrelevant ones. Given a query $q$, a set of relevant (positive) documents $D^{p}$, and a set of irrelevant (negative) documents $D^{n}$, we calculate the loss:
$$
\mathcal{L}=\sum_{\substack{d^{p} \in D^{p} \\ d^{n} \in D^{n}}} \max \left(0,1-\operatorname{Rel}\left(q, d^{p}\right)+\operatorname{Rel}\left(q, d^{n}\right)\right)
$$
Given a query, we regard the positives of other queries as in-batch negatives. 

**Binarization for efficient retrieval**
Our model allows for exploiting an inverted index. We regard each dimension in our bucketed sparse representations, which is indexable, as a signal or a latent term. For instance, if $n$ (dimensionality) is 81,920 , a document is represented with a combination of a few latent terms out of 81,920 . Only the non-zero dimensions of a document enter the inverted index with their weights. The level of efficiency in symbolic IR can be achieved since only a small fraction of the dimension in our UHD representation contains a non-zero value. Even higher efficiency can be gained by using the binarized output for indexing and ranking. For binarization, we convert non-zero values to 1 , leaving others as 0 .
We first encode all documents in a collection using the trained encoder to construct an inverted index. Each bucket conceptually has its own independent inverted index, resulting in $|B|$ (e.g. the number of BERT layers) inverted indices. 

**Evaluation results**

MS MARCO Retrieval results
{

			\begin{tabular}{ccccc}
			\hline Model & MRR@10 & R50 & R200 & R1000 \\
			\hline BM25 & $18.7$ & $59.2$ & $73.8$ & $85.7$ \\
			Doc2query & $21.5$ & $64.4$ & $77.9$ & $89.1$ \\
			DeepCT & $24.3$ & 69 & 82 & 91 \\
			DocTTTTTquery & $27.7$ & $75.6$ & $86.9$ & $94.7$ \\
			SparTerm & $27.94$ & $72.48$ & $84.05$ & $92.45$ \\
			UHD-BERT & $30.04$ & $77.77$ & $88.81$ & $96.01$ \\
			\hline
		\end{tabular}

}

### Inverted indexing construction

component. We look at each index of the learned representation as a "latent term". In other words, let $M$ denote the dimensionality of document representation, e.g., 20,000. Thus, we assume that there exist $M$ latent terms. Therefore, if the $i^{\text {th }}$ element of $\phi_{D}(d)$ is nonzero, then the document $d$ would be added to the inverted index for the latent term $i$. The value of this element is the weight of the latent term $i$ in the learned high-dimensional latent vector space.

## Hybrid retrieval models

### Motivation

\begin{remark}[TODO]
	Our phrase representation combines both dense and sparse vectors. Dense vectors are effective for encoding local syntactic and semantic cues lever- aging recent advances in contextualized text encoding, while sparse vectors are superior at encoding precise lexical information.
\end{remark}

\begin{remark}[TODO]
	Typical first-stage retrievers adopt a bag-of-words retrieval model that computes the relevance score based on heuristics defined over the exact word overlap between queries and documents. Models such as BM25 [32] remained state-ofthe-art for decades and are still widely used today. Though successful, lexical retrieval struggles when matching goes beyond surface forms and fails when query and document mention the same concept using different words (vocabulary mismatch ), or share only high-level similarities in topics or language styles.
	An alternative approach for first-stage retrieval is a neural-based, dense embedding retrieval: query words are mapped into a single vector query representation to search against document vectors. Such methods learn an inner product space where retrieval can be done efficiently leveraging recent advances in maximum inner product search (MIPS) [34,15,12]. Instead of heuristics, embedding retrieval learns an encoder to understand and encode queries and documents, and the encoded vectors can softly match beyond text surface form. However, single vector representations have limited capacity [I], and are unable to produce granular token-level matching signals that are critical to accurate retrieval [1],33].	

\end{remark}

\begin{remark}[TODO dense embedding problems/methods]
Despite the impressive results, the inherent properties of dense representations - low dimensional and dense - can pose a severe efficiency challenge for first-stage or full ranking. Since each dimension in a dense embedding is overloaded and entangled (i.e., polysemous) with the limited number of dimensions available, it is susceptible to false matches with large index sizes (Reimers and Gurevych, 2020). Also, all the dimensions must participate in representing words, queries, and documents regardless of the amount of information content as well as in matching (Zamani et al., 2018), which is inefficient. As a result, it is meaningless to build an inverted index for the dimensions, without which it is difficult to build an efficient and effective first-stage or full ranker.

Other drawbacks of the dense embedding approaches to IR include: 1) The retrieval results are hardly interpretable like other neural approaches, making it difficult to improve the design through failure analyses or implement conditional/selective processing (Belinkov and Glass, 2019); and 2) It is not straightforward to adopt the well studied termbased symbolic IR techniques, such as pseudorelevance feedback, for further improvements.

\end{remark}

\begin{remark}[TODO advantages of sparse models]

\end{remark}

### Simple score interpolation

#### Word-Embedding based model

\cite{mitra2016dual}

With this motivation, we define a simple yet, as we will demonstrate, effective ranking function we call the Dual Embedding Space Model:
$$
\operatorname{DESM}(Q, D)=\frac{1}{|Q|} \sum_{q_{i} \in Q} \frac{\mathbf{q}_{i}^{T} \overline{\mathbf{D}}}{\left\|\mathbf{q}_{i}\right\|\|\overline{\mathbf{D}}\|}
$$
where
$$
\overline{\mathbf{D}}=\frac{1}{|D|} \sum_{\mathbf{d}_{j} \in D} \frac{\mathbf{d}_{j}}{\left\|\mathbf{d}_{j}\right\|}
$$
Here $\overline{\mathbf{D}}$ is the centroid of all the normalized vectors for the words in the document serving as a single embedding for the whole document. In this formulation of the DESM, the document embeddings can be pre-computed, and at the time of ranking, we only need to sum the score contributions across the query terms. 

$M M(Q, D)=\alpha D \operatorname{ESM}(Q, D)+(1-\alpha) B M 25(Q, D)$ $\alpha \in \mathbb{R}, 0 \leq \alpha \leq 1$
To choose the appropriate value for $\alpha$, we perform a parameter sweep between zero and one at intervals of $0.01$ on the implicit feedback based training set described in Section 3.1.

#### BERT bi-encoder and BM25

\cite{wang2021bert}
The general BERT re-ranker allows both possibilities by defining the score $s(p)$ of a passage $p$ as the interpolation between the two scores:
$$
s(p)=\alpha \hat{s}_{B M 25}(p)+(1-\alpha) s_{B E R T}(p)
$$
where $\hat{s}_{B M 25}(p)$ is the normalised BM25 score for passage $p$ (see equation 16 by Lin et al. [14]), $s_{B E R T}(p)$ is the BERT score for $p$, and the hyperparameter $\alpha$ controls the relative importance of BM25 and BERT scores.

In this paper we provide a thorough empirical investigation of the importance of interpolating BERT and BM25 scores for those BERT-based DRs that do not consider BM25 scores during training and inference. We show that, unlike for the BERT re-ranker, this interpolation provides significant gains in effectiveness compared to using the BM25 or the DRs scores alone. We also show that DRs, when not interpolated with $\mathrm{BM} 25$, perform poorly with respect to deep evaluation measures, an aspect ignored in previous works, that have instead only focused on shallow measures (early rank positions). Finally, we provide evidence of why interpolation is required: our analysis in fact shows that BERT-based DRs are better than bag-of-words models (BM25) at modelling and detecting strong relevance signals. This is sufficient in providing good gains over BM25 for shallow measures. However, they fail to model the more complex, weaker relevance signals, for which instead BM25 does a good job: this results in DRs being outperformed by BM25 for deep measures. The interpolation of both methods is able to make up for each other's weaknesses.

\begin{remark}[not necessary for re-ranking]
It is not necessary to interpolate BERT re-ranker and bag-of-words scores to generate the final ranking. In fact, the BERT re-ranker scores alone can be used by the re-ranker: the BERT re-ranker score appears to already capture the relevance signal provided by BM25.

\end{remark}

### Duet

```{figure} images/../deepLearning/ApplicationIRSearch/HybridRetrieval/Duet/duet
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:duet
The duet architecture composed of the local model (left) and the distributed model (right). The local sub-network takes an interaction matrix of query and document terms as input, whereas the distributed sub-network learns embeddings of the query and document text before matching.
```

Authors in {cite}`mitra2017learning, mitra2019updated` proposed an early hybrid architecture called Duet, which combines signals from exact-match features as well as semantic-based matching features for retrieval and ranking. \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:duet} provides a detailed schematic view of the duet architecture. The distributed model is a representation-based bi-encoder to encode query and documents into features and compute a semantic matching score; the local model is an interaction-based model that operates over an term-level interaction matrix  and compute an exact matching score. The final score under the duet setup is the sum of the two scores, that is,
$$
s(q, d)=s_{local}(q, d)+s_{distributed}(q, d).
$$

**local model** The local model estimates document relevance based on patterns of exact matches of query terms in the document. Each term is represented by its one-hot encoding in a fixed vocabulary size $V$.  and the interaction matrix is a binary $N\times M$ matrix, where $N$ and $M$ are the length of the query and the document, respectively. Treating the interaction matrix as a 2D matrix, we can further use convolutional layers to extract high-level features and an MLP to produce an exact match score with $c$ filters, a kernel size of $n_{\mathrm{d}} \times 1$, and a stride of 1. 

**Distributed model** The distributed model aims to learn dense lower-dimensional vector representations of the query and the document text, and then computes the similarity between them in the learned embedding space using an MLP. Similar to {cite}`huang2013learning`, words in the query and the the document are represented by frequency-count vector with respect to a character 3-gram vocabulary. MLP and convolutional layers are used to extract features and compute scores from this frequency vectors. 

**Training** Using final similarity score, the hybrid model can be trained from end-to-end using N-pair loss function [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:N_pair_loss}].

### COIL

#### The model

Duet model architecture is an early attempt to bring benefits of exact matching and inexact matching into an end-to-end model. Yet its interaction matrix feature and token level features are primarily based on one-hot encoding or count-based encoding. Along the line of hybrid retrieval, authors in {cite}`gao2021coil` proposed COIL (Contextualized Inverted List) approach that combine exact match and semantic match in the BERT-based bi-encoder architecture, which draws upon significant process in contextualized embedding representations for words and sentences. The proposed architecture [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:coilembedding}] resembles ColBERT [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:colBERT}], where token-level embeddings of co-occurring terms are fed into max similarity aggregator for late interactions. Unlike ColBERT, where all query terms and all document terms are involved in the max similarity computation, here only exact matched terms from query and documents are used to compute max similarity. The semantic match is then achieved via cosine similarity of CLS special token embedding.  

**Contextualized exact match.** We encode tokens from a query $q = (q_1,...,q_N)$ and tokens from a document $d = (d_1,...,d_M)$ with a BERT-based pre-trained language model (LM) with
```{math}
\begin{align*}
{v}_{i}^{q} &={W}_{tok} \operatorname{LM}(q, q_i)+{b}_{tok} \\
	{v}_{j}^{d} &={W}_{tok} \operatorname{LM}(d, d_j)+{b}_{tok}
\end{align*}
```
where ${W}_{tok}^{k \times d_m}$ is a matrix that project the embeddings with $d_m$ dimension from the LM to into a vector space of lower dimension $k$. 

The contextualized exact lexical match scoring function between query document is then defined based on vector similarities between exact matched query document token pairs:
$$
s_{tok}(q, d)=\sum_{q_{i} \in q \cap d} \max _{d_{j}=q_{i}}\left({v}_{i}^{q \top} {v}_{j}^{d}\right)
$$
Note that, in the essence of exact match, the summation only goes through only co-occurring terms, $q_{i} \in q \cap d$. For each query token $q_{i}$, we finds all tokens $d_{j}$ in the document, computes their similarity with $q_{i}$ using the contextualized token vectors. The maximum similarities are motivated to capture the most salient signal. 

**Contextualized semantic match.** Like classic lexical systems, $s_{tok}$ defined does not consider mismatch terms and will introduce the vocabulary mismatch issue. This can remedied by leveraging the CLS token representation for semantic matching since the CLS token embeddings aggregate sentence level semantic information. Using a similar dimensionality reduction scheme, we have
```{math}
\begin{align*}
v_{cls}^{q} &={W}_{cls} \mathrm{LM}(q, q_{CLS})+{b}_{cls} \\
	v_{cls}^{d} &={W}_{cls} \mathrm{LM}(d, d_{CLS})+{b}_{cls}.
	s_{cls}(q, d) = v_{cls}^{q} \cdot v_{c l s}^{d}
\end{align*}
```

The similarity between $v_{cls}^{q}$ and $v_{cls}^{d}$ provides a high level semantic matching and mitigates the issue of vocabulary mismatch. The full form of COIL is:
$$
s_{full}(q, d)=s_{tok}(q, d) + s_{cls}(q, d).
$$

**Training.** The model can be trained from end-to-end using N-pair loss [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:N_pair_loss}]. Specifically, let $\mathcal{D}=\left\{\left\langle q_{i}, d_{i}^{+}, D_i^-\right\rangle\right\}_{i=1}^{m}$, where $D_i^- = \{d_{i, 1}^{-}, \cdots, d_{i, n}^{-}\}$ are a set of negative examples (i.e., irrelevant document) with respect to query $q_i$,  be the training data that consists of $m$ examples. Each example contains one query $q_{i}$ and one relevant document $d_{i}^{+}$, along with $n$ irrelevant (negative) documents $d_{i, j}^{-}$. 

The loss function per example is given by
$$L =-\log \frac{\exp(s_{full}\left(q, d^{+}\right))}{\exp(s_{full}\left(q, d^{+}\right))+\sum_{d^-\in D^-} \exp(s_{full}\left(q, d^{-}\right))}$$
where $\operatorname{Sim}(e_q, e_d)$ is the similarity score function taking query embedding $e_q$ and document embedding $e_d$ as the input. 

```{figure} images/../deepLearning/ApplicationIRSearch/HybridRetrieval/COIL/COIL_embedding
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:coilembedding
The architecture of COIL to compute exact match score, inexact match score, and the aggregated final score. Only exact matched terms (shown in solid red and blue lines) from query and documents are used to compute max similarity.
```

#### Indexing and serving

```{figure} images/../deepLearning/ApplicationIRSearch/HybridRetrieval/COIL/coil_indexing_method
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:coilindexingmethod
Indexing and serving using COIL model. CLS token embedding for each document is pre-computed and cached. For inverted indexing, one inverted list for each token is created. The inverted list contains the document id in where the token is present and and the token's contextualized embeddings. Image from {cite}`gao2021coil`.
```

Inverted indexing is designed to achieve efficient online inference for traditional IR systems [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:traditional_invertedlists}]. Inverted indexing can be adapted to serve COIL model as well, as shown in \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:coilindexingmethod}. After training, COIL model can pre-compute offline the document representations on the token level (i.e., token embeddings) and the document level (i.e., CLS special token embedding) and builds up a search index. Documents in the collection are encoded offline into token and CLS vectors. Formally, for a unique token $t$ in the vocabulary $V$, we collect its contextualized vectors from all of its mentions from documents in collection $\cD$, building token $t$ 's contextualized inverted list:
$$
I^{t}=\left\{\boldsymbol{v}_{j}^{d} \mid d_{j}=t, d \in \cD\right\}
$$
where $v_{j}^{d}$ is the BERT-based token encoding. We also collect and build the index for the CLS token embeddings of all documents, given by $I^{c l s}=\left\{v_{c l s}^{d} \mid d \in \cD\right\}$.

### CoRT

```{figure} images/../deepLearning/ApplicationIRSearch/HybridRetrieval/CoRT/CoRT_arch
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:cortarch
Bi-encoder architecture for CoRT.
```

The model architecture of CoRT, illustrated in Figure 1, follows the idea of a Siamese Neural Network (Bromley et al., 1993). Passages and queries are encoded using an identical model with shared weights except for one detail: The passage encoder $\psi_{\alpha}$ and the query encoder $\psi_{\beta}$ use different segment embeddings (Devlin et al., 2019). CoRT computes relevance scores as angular similarity between query and passage representations while training a pairwise ranking objective.

**Training** Training CoRT corresponds to updating the parameters of the encoder $\psi$ towards representations that reflect relevance between queries and passages through vector similarity. Each training sample is a triple comprising a query $q$, a positive passage $d^{+}$ and a negative passage $d^{-}$. While positive passages are taken from relevance assessments, negative passages are sampled from term-based rankings (i.e. BM25) to support the **complementary property of CoRT.** The relevance score for a query-passage pair $(q, d)$ is calculated using the angular cosine similarity function ${ }^{2}$.

**Indexing and retrieval** we only index one representation per passage rather than one per token. Finally, we combine the resulting ranking of CoRT with the respective BM25 ranking by interleaving the positions beginning with CoRT to create a single merged ranking of equal length. During this process, each passage that was already added by the other ranking is omitted. For example, merging two ranking lists beginning with $[a, b, c, d, \ldots]$ and $[e, c, f, a, \ldots]$ would result in $[a, e, b, c, \not, f, d, \not d, \ldots]$. The interleaving procedure stops as soon as the desired ranking size has been reached. The result is a compound ranking of CoRT and BM25, which we denote with $\mathrm{CoRT}_{\mathrm{BM} 25}$.

### CLEAR
\cite{gao2021complement}

CLEAR (Complementary Retrieval Model), a retrieval model that seeks to complement classical lexical exact-match models such as BM25 with semantic matching signals from a neural embedding matching model.

clear explicitly trains the neural embedding to encode language struc- tures and semantics that lexical retrieval fails to capture with a novel residual-based embedding learning method.

Importantly, unlike existing tech- niques that train embeddings directly for ranking independently [40,4], clear explicitly trains the embedding retrieval model with a residual method: the em- bedding model is trained to build upon the lexical model’s exact matching signals and to fix the mistakes made by the lexical model by supplementing semantic level information, effectively learning semantic matching not captured by the lexical model, which we term the un-captured residual.

In early works, embedding models are all trained independently from the lexical models and rely on simple post-training fusion to form a hybrid score. To the best of our knowl- edge, ours is the first work that investigates jointly training latent embeddings and lexical retrieval for first-stage ad hoc retrieval.

clear consists of a lexical retrieval model and an embedding retrieval model. Between these two models, one’s weakness is the other’s strength: lexical retrieval performs exact token matching but cannot handle vocabulary mismatch; mean- while, the embedding retrieval supports semantic matching but loses granular (lexical level) information. To ensure that the two types of models work together and fix each other’s weakness, we propose a residual-based learning framework that teaches the neural embeddings to be complementary to the lexical retrieval.

We propose a novel residual-based learning framework to ensure that the lexical retrieval model and the embedding retrieval model work well together. While BM25 has just two trainable parameters, the embedding model has more flexibility. To make the best use of the embedding model, we must avoid the embedding model “relearning” signals already captured by the lexical model. Instead, we focus its capacity on semantic level matching missing in the lexical model.

In general, the neural embedding model training uses hinge loss [36] defined over a triplet: a query $q$, a relevant document $d^{+}$, and an irrelevant document $d^{-}$serving as a negative example:
$$
\mathcal{L}=\left[m-s_{\mathrm{emb}}\left(q, d^{+}\right)+s_{\mathrm{emb}}\left(q, d^{-}\right)\right]_{+}
$$
where $[x]_{+}=\max \{0, x\}$, and $\mathrm{m}$ is a static loss margin. In order to train embeddings that complement lexical retrieval, we propose two techniques: sampling negative examples $d^{-}$from lexical retrieval errors, and replacing static margin $m$ with a variable margin that conditions on the lexical retrieval's residuals.

Error-based Negative Sampling We sample negative examples ( $d^{-}$in Eq. 5 ) from those documents mistakenly retrieved by lexical retrieval. Given a positive query-document pair, we uniformly sample irrelevant examples from the top $N$ documents returned by lexical retrieval with probability $p$. With such negative samples, the embedding model learns to differentiate relevant documents from confusing ones that are lexically similar to the query but semantically irrelevant.
Residual-based Margin Intuitively, different query-document pairs require different levels of extra semantic information for matching on top of exact matching signals. Only when lexical matching fails will the semantic matching signal be necessary. Our negative sampling strategy does not tell the neural model the degree of error made by the lexical retrieval that it needs to fix. To address this challenge, we propose a new residual margin. In particular, in the hinge loss, the conventional static constant margin $m$ is replaced by a linear residual margin function $m_{r}$, defined over $\mathrm{s}_{\operatorname{lex}}\left(q, d^{+}\right)$and $\mathrm{s}_{\text {lex }}\left(q, d^{-}\right)$, the lexical retrieval scores:
$$
m_{r}\left(\mathrm{&nbsp;s}_{\text {lex }}\left(q, d^{+}\right), \mathrm{s}_{\text {lex }}\left(q, d^{-}\right)\right)=\xi-\lambda_{\text {train }}\left(\mathrm{s}_{\text {lex }}\left(q, d^{+}\right)-\mathrm{s}_{\text {lex }}\left(q, d^{-}\right)\right)
$$
where $\xi$ is a constant non-negative bias term. The difference $\mathrm{s}_{\text {lex }}\left(q, d^{+}\right)-\mathrm{s}_{\text {lex }}\left(q, d^{-}\right)$ corresponds to a residual of the lexical retrieval. We use a scaling factor $\lambda_{\text {train }}$ to adjust the contribution of residual. Consequently, the full loss becomes a function of both lexical and embedding scores computed on the triplet,
$$
\mathcal{L}=\left[m_{r}\left(\mathrm{&nbsp;s}_{\mathrm{lex}}\left(q, d^{+}\right), \mathrm{s}_{\mathrm{lex}}\left(q, d^{-}\right)\right)-s_{\mathrm{emb}}\left(q, d^{+}\right)+s_{\mathrm{emb}}\left(q, d^{-}\right)\right]_{+}
$$
For pairs where the lexical retrieval model already gives an effective document ranking, the residual margin $m_{r}(\mathrm{Eq} .6)$ becomes small or even becomes negative. In such situations, the neural embedding model makes little gradient update, and it does not need to, as the lexical retrieval model already produces satisfying results. On the other hand, if there is a vocabulary mismatch or topic difference, the lexical model may fail, causing the residual margin to be high and thereby driving the embedding model to accommodate in gradient update. Through the course of training, the neural model learns to encode the semantic patterns that are not captured by text surface forms. When training finishes, the two models will work together, as CLEAR.

**Retrieval** CLEAR retrieves from the lexical and embedding index respectively, taking the union of the resulting candidates, and sorts using a final retrieval score: a weighted average of lexical matching and neural embedding scores:
$$
s_{\text {CLEAR }}(q, d)=\lambda_{\text {test }} s_{\text {lex }}(q, d)+s_{\mathrm{emb}}(q, d)
$$
We give CLEAR the flexibility to take different $\lambda_{\text {train }}$ and $\lambda_{\text {test }}$ values. Though both are used for interpolating scores from different retrieval models, they have different interpretations. Training $\lambda_{train}$ serves as a global control over the resid- ual based margin. On the other hand, testing $\lambda_{test}$ controls the contribution from the two retrieval components. clear achieves low retrieval latency by having each of the two retrieval
models adopt optimized search algorithms and data structures. For the lexi- cal retrieval model, clear index the entire collection with a typical inverted index. For the embedding retrieval model, clear pre-computes all document embeddings and indexes them with fast MIPS indexes such as FAISS [15] or SCANN [12], which can scale to millions of candidates with millisecond latency. As a result, clear can serve as a first-stage, full-collection retriever.

### Score fusion

\cite{lin2021batch}

As shown in Luan et al. (2020); Gao et al. (2020), a single dense embedding cannot sufficiently represent passages, especially when the passages are long, and they further demonstrate that sparse retrieval can complement dense retrieval by a linear combination of their scores. However, it is not practical to compute scores over all query and passage pairs, especially when the corpus is large. Thus, we propose an alternative approximation, which is easy sparse and dense retrieval using Anserini (Yang et al., 2018 ${ }^{4}$ and Faiss (Johnson et al., 2017), ${ }^{5}$ respectively.

For each query $\mathbf{q}$, we use sparse and dense representations to retrieve top 1000 passages, $\mathcal{D}_{s p}$ and $\mathcal{D}_{d s}$, with their relevance scores, $\phi_{s p}\left(\mathbf{q}, \mathbf{d} \in \mathcal{D}_{s p}\right)$ and $\phi_{d s}\left(\mathbf{q}, \mathbf{d} \in \mathcal{D}_{d s}\right)$, respectively. Then, we compute the scores for each retrieved passages, $\mathbf{d} \in \mathcal{D}_{s p} \cup \mathcal{D}_{d s}$, as follows:
$\phi(\mathbf{q}, \mathbf{d})= \begin{cases}\alpha \cdot \phi_{s p}(\mathbf{q}, \mathbf{d})+\min _{\mathbf{d} \in \mathcal{D}_{d s}} \phi_{d s}(\mathbf{q}, \mathbf{d}), & \text { if } \mathbf{d} \notin D_{d s} \\ \alpha \cdot \min _{\mathbf{d} \in \mathcal{D}_{s p}} \phi_{s p}(\mathbf{q}, \mathbf{d})+\phi_{d s}(\mathbf{q}, \mathbf{d}), & \text { if } \mathbf{d} \notin D_{s p} \\ \alpha \cdot \phi_{s p}(\mathbf{q}, \mathbf{d})+\phi_{d s}(\mathbf{q}, \mathbf{d}), & \text { otherwise. }\end{cases}$
Eq. (7) is an approximation of linear combination of sparse and dense relevant scores. For approximation, if $\mathrm{d} \notin \mathcal{D}_{s p}\left(\right.$ or $\left.\mathcal{D}_{d s}\right)$, we directly use the minimum score of $\phi_{s p}\left(\mathbf{q}, \mathbf{d} \in \mathcal{D}_{s p}\right)$, or $\phi_{d s}(\mathbf{q}, \mathbf{d} \in$ $\mathcal{D}_{d s}$ ) as a substitute.

\fi

\iffalse
## Query suggestion

### Query similarity from Query-URL bipartite graph method

Although the click information on SERP URLs are often noisy, aggregating clicks from a large number of users tends to reflect the relevance between queries and URLs. Such rich query-URL relevance information can be used for generating high quality query suggestions. As an example, the co-occurrence based method may fail to generate suggestion for a tail (typo) query "faecboek." If we can leverage the top clicked URLs on the SERP of the query, it is likely to generate relevant suggestions. In practice, such approach can help address the issues on tail queries that lack enough co-occurrence information.

Typically, query-URL bipartite graph-based methods use clicks from queries on URLs as signals. They usually work as follows. First, a probabilistic matrix is constructed using click counts. Next, a starting node (i.e., a test query) is chosen. Third, a random walk (RW) is performed on the graph using the probabilistic matrix. Forth, final suggestion is generated using RW results.

## Practical searching and ranking

### Problem statement

\begin{remark}[classification of search engine]
	A search engine could be a **general search engine** like Google or Bing or a **specialized search engine** like Amazon products search.

	Nowadays, search engine are typically personalized. That is, the searcher is a logged-in user or not. And the user is logged in and you have access to their profile as well as their historical search data.

\end{remark}

Build a generic search engine that returns relevant results for queries like "Richard Nixon", "Programming languages" etc.
This will require you to build a machine learning system that provides the most relevant results for a search query by ranking them in order of relevance. Therefore, you will be working on the search ranking problem

### Metrics

### Architecture

#### Overview

```{figure} images/../deepLearning/ApplicationRecommenderSys/practical_search_engine/search_engin_fundamental_components
:name: fig:searchenginfundamentalcomponents

```

```{figure} images/../deepLearning/ApplicationRecommenderSys/search/search_engine_arch
:name: fig:searchenginearch

```

Web search engines get their information by crawling from site to site, while some e-commerce search engines collect their information according to their own providers without crawling.

The indexing process is to store and organize their collected information on their servers, as well as prepare some positive and negative signals for the following search process. 

Searching is a process that accepts a text query as input and returns
a list of results, ranked by their relevance to the query. The search process can be further divided into three steps, which include query understanding, retrieval, and ranking.

#### Query rewriting

Queries are often poorly worded and far from describing the searcher’s
actual information needs. Hence, we use query rewriting to increase recall,
i.e., to retrieve a larger set of relevant results. Query rewriting has multiple
components which are mentioned below. 

**Spell checker**

Spell checking queries is an integral part of the search experience and is
assumed to be a necessary feature of modern search. Spell checking allows
you to fix basic spelling mistakes like “itlian restaurat” to “italian
restaurant”.

**Query expansion **

Query expansion improves search result retrieval by adding terms to the
user’s query. Essentially, these additional terms minimize the mismatch
between the searcher’s query and available documents.
Hence, after correcting the spelling mistakes, we would want to expand
terms, e.g., for the query “italian restaurant”, we should expand “restaurant”
to food or recipe to look at all potential candidates (i.e., web pages) for this
query

**Query relaxation**

The reverse of query expansion, i.e., query relaxation, serves the same purpose. For example, a search for *good Italian restaurant* can be relaxed to *italian restaurant*.

#### Query understanding

This stage includes figuring out the main intent behind the query, e.g., the query “gas stations” most likely has a local intent (an interest in nearby places) and the query “earthquake” may have a newsy intent. Later on, this
intent will help in selecting and ranking the best documents for the query. 

#### Document selection

The web has billions of documents. Therefore, our first step in document
selection is to find a fairly large set of documents that seems relevant to the
searcher’s query. For example, some common queries like “sports”, can
match millions of web pages. Document selection’s role will be to reduce this
set from those millions of documents to a smaller subset of the most relevant
documents.
Document selection is more focused on recall. It uses a simpler technique to
sift through billions of documents on the web and retrieve documents that
have the potential of being relevant

#### Ranker
The ranker will actively utilize machine learning to find the best order of
documents (this is also called learning to rank).
If the number of documents from the document selection stage is
significantly large (more than 10k) and the amount of incoming traffic is also
huge (more than 10k QPS or queries per second), you would want to have
multiple stages of ranking with varying degrees of complexity and model
sizes for the ML models. Multiple stages in ranking can allow you to only
utilize complex models at the very last stage where ranking order is most
important. This keeps computation cost in check for a large scale search
system.

#### Blender
Blender gives relevant results from various search verticals, like, images,
videos, news, local results, and blog posts.
For the “italian restaurant” search, we may get a blend of websites, local
results, and images. The fundamental motivation behind blending is to
satisfy the searcher and to engage them by making the results more
relevant.
Another important aspect to consider is the diversity of results, e.g., you
might not want to show all results from the same source(website).
The blender finally outputs a search engine result page (SERP) in response to
the searcher’s query

### Query understanding and rewriting

Query understanding is a fundamental part of search engine. It is responsible to **precisely infer the intent of the query formulated by search user, to correct spelling errors in the query, to reformulate the query to capture its intent more accurately, and to guide search user in the formulation of query with precise intent.** Query understanding methods generally take place before the search engine retrieves and
ranks search results. If we can understand the information needs of search queries in the best way, we can better serve users. 

#### Spelling correction

### Document selection

#### Document selection overview
From the one-hundred billion documents on the internet, we want to
retrieve the top one-hundred thousand that are relevant to the searcher’s
query by using information retrieval techniques.
The searcher’s query does not match with only a single document. 

#### Inverted index

Inverted index: an index data structure that stores a mapping from **content, such as words or numbers**, to its location in a set of documents.

```{figure} images/../deepLearning/ApplicationRecommenderSys/practical_search_engine/inverted_index
:name: fig:invertedindex
Inverted index
```

\begin{example}
	For instance, the searcher’s query is *Italian restaurants*. The query expansion component tells us to look for *Italian food*.

	The document selection criteria would then be as follows

	match the term Italian **and** match the term restaurant **or** food.

	We will go into the index and retrieve all the documents based on the above selection criteria.
\end{example}

#### Relevance scoring scheme

The
selection criteria derived from the query may match a lot of documents with
a different degree of relevance.

Let’s see how the relevance score is calculated. One basic scoring scheme is
to utilize a simple weighted linear combination of the factors involved. The
weight of each factor depends on its importance in determining the
relevance score. Some of these factors are:
- Terms match
- Document popularity (popular documents will have a higher score). The intuition is that a popular document historically has a high engagement rate with an average user. 
- Query intent match (i.e., local intent, news intent, etc.)
- Personalization match. (It scores how well a document meets the searcher’s individual requirements based on a lot of aspects. For instance, the searcher’s age, gender, interests,and location.)

The diagram below shows how the linear scorer will assign a relevance
score to a document.

```{figure} images/../deepLearning/ApplicationRecommenderSys/practical_search_engine/document_selection_relevant_score_scheme
:name: fig:documentselectionrelevantscorescheme

```

The weight of each factor in determining the score is selectedmanually, through the intuition, in the above scorer. Machine learningcan also be used to decide these weights.

### Ranking features

#### Overview

```{figure} images/../deepLearning/ApplicationRecommenderSys/practical_search_engine/search_engine_flow
:name: fig:searchengineflow
Search engine flow
```

The ranker will actively utilize machine learning to find the best order of documents (this is also called learning to rank).If the number of documents from the document selection stage is significantly large (more than 10k) and the amount of incoming traffic is also huge (more than 10k QPS or queries per second), you would want to have multiple stages of ranking with varying degrees of complexity and model sizes for the ML models. 

Multiple stages in ranking can allow you to only utilize complex models at the very last stage where ranking order is most important. This keeps computation cost in check for a large scale search system. For example, one configuration can be that your document selection returns 100k documents, and you pass them through two stages of ranking. In stage one, you can use fast (nanoseconds) linear ML models to rank them. In stage two, you can utilize computationally expensive models (like deep learning models) to find the most optimized order of top 500 documents given by stage one.

The four such actors for search are:
- Searcher
- Query
- Document
- Context

The context for a search query is browser history. However, it is a lot more than just search history. It can also include the searcher’s age, gender, location, and previous queries and the time of day.

\paragraph{Searcher-specific features}

Assuming that the searcher is logged in, you can tailor the results according
to their **age, gender and interests** by using this information as features for
your model.

\paragraph{Query specific features}

**Query historical engagement** \\
For relatively popular queries, historical engagement can be very important.
You can use query’s prior engagement as a feature. For example, let’s say the
searcher queries “earthquake”. We know from **historical data that this query results in engagement with “news component”, i.e. most people who
	searched “earthquake”, were looking for news regarding a recent
	earthquake.** Therefore, you should consider this factor while ranking the
documents for the query.

\begin{remark}
	Here has the idea of collaborative filtering. We use the behavior of a large of group of user to help us find the most relevant query result.
	The feature can be represented as a sparse categorical feature.
\end{remark}

**Query intent** \\
The “query intent” feature enables the model to identify the kind of
information a searcher is looking for when they type their query. The model
uses this feature to assign a higher rank to the documents that match the
query’s intent. For instance, if a person queries “pizza places”, the intent
here is local. Therefore, the model will give high rank to the pizza places that
are located near the searcher.

A few examples of query intent are news, local, commerce.** We can get query intent from the query understanding component.**

\begin{remark}
	Note that the query intent understanding component can also use collaborative filtering idea. 

	For example, for a query like pizza places. If most of the searchers click the document corresponding to their nearby places. We can use a model to learn that pizza places are likely a local intent.

\end{remark}

#### Document specific features

**Page rank**\\

The rank of a document can serve as a feature. To estimate the relevance of the document under consideration, we can look at the number and quality of the documents that link to it. **Page rank** feature encodes the importance the page based on the connections.

\begin{remark}
	$$
	P R\left(p_{i}\right)=\frac{1-d}{N}+d \sum_{p_{j} \in M\left(p_{i}\right)} \frac{P R\left(p_{j}\right)}{L\left(p_{j}\right)}
	$$
	where $p_{1}, p_{2}, \ldots, p_{N}$ are the pages under consideration, $M\left(p_{i}\right)$ is the set of pages that link to $p_{i}, L\left(p_{j}\right)$ is the number of outbound links on page $p_{j}$, and $N$ is the total number of pages.  This residual probability, $d$, is usually set to 0.85, estimated from the frequency that an average surfer uses his or her browser's bookmark feature. 
\end{remark}

**Document engagement radius**

The document engagement radius can be another important feature. A document on a coffee shop in Seattle would be more relevant to people living within a ten-mile radius of the shop. However, a document on the Eiffel Tower might interest people all around the world. Hence, in case our query has a local intent, we will choose the document with the local scope of appeal rather than that with a global scope of appeal.

#### Context specific features

**Time of search**

When a searcher has queried for restaurants. In this case, a contextual feature can be the time of the day. This will allow the model to display or rank restaurants that are open at that hour.
When a searcher has searched for goods and it is near holiday season. This feature can also help the model the rank shopping relevant webs higher.

**Recent events**

The searcher may appreciate any recent events related to the query. Forexample, upon querying “Vancouver”, the results include recent top stories about Vancouver.

#### Searcher-document features

**Searcher-document Distance**

For queries related to locations, stores, restaurants, we can use distance between the searcher and the matching locations as a feature to measure the relevance of the documents. Consider the case where a person has searched for restaurants in their vicinity. Documents regarding nearby restaurants will be selected forranking. The ranking model can then rank the documents based on thedistance between the coordinates of the searcher and the restaurants in the document.

**Personal Historical engagement**

Another interesting feature could be the searcher’s historical engagement with the result type of the document. For instance, if a person has engaged with video documents more in the past, it indicates that video documents are generally more relevant for that person. Historical engagement with a particular website or document can also be an important signal as the user might be trying to “re-find” the document.

\begin{remark}
	We have discussed previously the Query historical engagement feature, which is a feature based on generic user behavior. Here personal historical engagement is a user-specific feature.
\end{remark}

### Query-document feature

**Text match**

One feature can be the text match. Text match can not only be in the title of the document, but it can also be in the metadata or content of a document.Look at the following example. It contains a text match between the query and document title and another between the query and document content.

## Information retrieval and neural matching

#### Comparison with recommender system

Recommender are generally working passively, and there is generally no query entered by the user. The user's interest is implicitly expressed by his past behavior, some attributes of the user (such as occupation, age, gender, etc.), and the current context. The recommended entity can be e-commerce goods, movies, tags, friends, etc. The two main entities in the recommendation: user and item, they are different types of things, unlike the query and doc in the search, which are generally text.

```{figure} images/../deepLearning/ApplicationRecommenderSys/search/recommendation_overview
:name: fig:recommendationoverview

```

```{figure} images/../deepLearning/ApplicationRecommenderSys/search/search_vs_recommendation
:name: fig:searchvsrecommendation

```

#### A unified perspective from matching

Most of the systems used in machine learning, such as search, recommendation, and advertising, can actually be divided into two stages: recall and sorting. Recall is a process of pulling candidate sets, which is often a matching problem, and many matching features will It is an important basis for the sorting stage. Furthermore, search, recommendation, and advertisement are actually a matching problem:

Search: query vs doc match \\

Advertising: query vs ad matching or user vs ad matching \\

Recommendation: user vs item matching

\fi

## Approximate nearest neighbor search

### Overview

Applying dense retrieval in the first-stage of the ad-hoc retrieval system involves performing nearest neighbor search among web-scale documents in the high-dimensional embedding space. Exact nearest neighbor search is inherently expensive due to the curse of dimensionality and the large number of documents. Consider a $D$-dimensional Euclidean space $\mathbb{R}^{D}$, the problem is to find the nearest element $\mathrm{NN}(x)$, in a finite set $\mathcal{Y} \subset \mathbb{R}^{D}$ of $n$ vectors, minimizing the distance to the query vector $x \in \mathbb{R}^{D}$ is given by:
$$
\mathrm{NN}(x)=\arg \min _{y \in \mathcal{Y}} d(x, y).
$$
A brute force exhaustive distance calculation has the complexity of $\mathcal{O}(n D)$. Several multi-dimensional indexing methods, such as the popular KD-tree {cite}`friedman1977algorithm` or other branch and bound techniques, have been proposed to reduce the search time. However, nowadays the dominating approaches are approximate nearest neighbor search via vector quantization, which is the primary focus of this section. 

### Vector quantization

#### Approximate representation and storage
Quantization is a technique widely used to reduce the cardinality of high dimensional representation space, in particular when the input data is real-valued. 

Formally, a **quantizer** is a function $q$ mapping a multi-dimensional vector $x \in \mathbb{R}^{D}$ to a pre-defined centroid $q(x) = c_i$, where $c_i \in \cC = \{c_1,...,c_{k}\}$. 
The values $c_i \in \R^D$ are called **centroids**, and the set $\cC$ is the *codebook* of size $k$.

The set $\mathcal{V}_{i}$ of vectors mapped to a given index $i$ is referred to as a (Voronoi) cell, and defined as
$$
\mathcal{V}_{i} \triangleq\left\{x \in \mathbb{R}^{D}: q(x)=c_{i}\right\}.
$$
The $k$ cells of a quantizer form a partition of $\mathbb{R}^{D}$. By definition, all the vectors lying in the same cell $\mathcal{V}_{i}$ are reconstructed by the same centroid $c_{i}$. That is, they are all **approximated** by centroid $c_i$. 

How do we measure the approximation quality of such representation? The quality of a quantizer is usually measured by the mean squared error between the input vector $x$ and its reproduction value $q(x):$
$$
\operatorname{MSE}(q)=\mathbb{E}_{X}\left[d(q(x), x)^{2}\right]=\int p(x) d(q(x), x)^{2} d x
$$
where $d(x, y)=\|x-y\|$ is the Euclidean distance between $x$ and $y$, and where $p(x)$ is the probability distribution function corresponding the randomly sampled $x$. Approximate calculation of the integral can be achieved by Monte-Carlo sampling.

In order for the quantizer to be optimal, it has to satisfy two properties known as the Lloyd optimality conditions. First, a vector $x$ must be quantized to its nearest codebook centroid, in terms of the Euclidean distance:
$$
q(x)=\arg \min _{\mathrm{s}_{i} \in \mathcal{C}} d\left(x, c_{1}\right)
$$
As a result, the cells are delimited by hyperplanes. The second Lloyd condition is that the reconstruction value must be the expectation of the vectors lying in the Voronoi cell:
$$
c_{i}=\mathbb{E}_{X}[x \mid i]=\int_{V_{i}} p(x) x d x .
$$
The Lloyd quantizer, which corresponds to the **$k$-means clustering algorithm**, finds a near-optimal codebook by iteratively assigning the vectors of a training set to centroids and re-estimating these centroids from the assigned vectors. 
```{figure} images/../deepLearning/ApplicationIRSearch/ApproximateNearestNeighbor/codebook_construction
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:codebookconstruction
Codecook construction can be achieved using k-means algorithm to compute $K$ centroids from database vectors.
```

The **storage** for $N$ vectors now reduce to storage of their index values plus the centroids in the codebook. Each index value requires $\log_{2} K$ bits. On the other hand, storing the original vectors typically take more than $\log_2(k)$ bits.

Two important benefits to compressing the dataset are that (1) memory access times are generally the limiting factor on processing speed, and (2) sheer memory capacity can be a problem for big datasets.

In \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:codebookmemorysavingdemo}, we illustrate the storage saving by representing a $D$ dimensional vector by a codebook of 256 centroids. We only need 8-bits ($2^8 = 256$) to store a centroid id. Each vector is now replace by a 8-bit integers.

```{figure} images/../deepLearning/ApplicationIRSearch/ApproximateNearestNeighbor/codebook_memory_saving_demo
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:codebookmemorysavingdemo
Illustration of memory saving benefits of vector quantization. A $D$-dimensional float vector is stored as its nearest centroid integer id, which only occupies $\log_2 K$ bit.  
```

#### Approximating distances using quantized codes

Given the representation choices for the query vector $x$ and the database vector $y$, we define two options in approximating the distance $d(x, y)$ [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:distance_compute}]. 

**Symmetric distance computation (SDC)**: both the vectors $x$ and $y$ are represented by their respective centroids $q(x)$ and $q(y)$. The distance $d(x, y)$ is approximated by the distance $\hat{d}(x, y) \triangleq d(q(x), q(y))$.

**Asymmetric distance computation (ADC)**: the database vector $y$ is represented by $q(y)$, but the query $x$ is not encoded. The distance $d(x, y)$ is approximated by the distance $\tilde{d}(x, y) \triangleq d(x, q(y))$

```{figure} images/../deepLearning/ApplicationIRSearch/ApproximateNearestNeighbor/distance_compute
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:distance_compute
Illustration of the symmetric (left) and asymmetric distance (right) computation. The distance $d(x, y)$ is estimated with either the distance $d(q(x), q(y))$ (left) or the distance $d(x, q(y))$ (right). 
```

Suppose now we have a query vector $x$ and we want to find its nearest neighbors among all the $y$ in the database $\cY$.

There are benefits in performing symmetric distance computation. To perform symmetric distance computation, we can pre-compute a $K\times K$ table to cache the Euclidean distance between all centroids. After computing the encoding $q(x)$, we can get $d(q(x), q(y))$ by table lookup.
On the other hand, to perform asymmetric distance computation between $x$ and all $y\in \cY$, we can directly calculate the Euclidean distance between the query vector $x$ and centroid $q(y)$ in the codebook. 

### Product quantization

#### From vector quantization to product quantization
Let us consider a quantizer producing 64 bits codes, i.e., which can contain $k=2^{64} \approx 1.8\times 10^{19} $ centroids. It is prohibitive run the k-means algorithm and practically impossible to store the $D \times k$ floating point values representing the $k$ centroids.

Product quantization serves as an efficient solution to address these computation and memory consumption issues in vector quantization. The key idea of product quantization is **grouping and splitting**[ \autoref{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:codebookmemorysavingdemoproductquantization}]. The input vector $x$ is split into $m$ distinct subvectors $u_{j}, 1 \leq$ $j \leq m$ of dimension $D^{*}=D / m$, where $D$ is a multiple of $m$. The subvectors are then quantized separately using $m$ distinct quantizers. A given vector $x$ is therefore quantized as follows:
$$\underbrace{x_{1}, \ldots, x_{D^{*}}}_{u_{1}(x)}, \ldots, \underbrace{x_{D-D^{*}+1}, \ldots, x_{D}}_{u_{m}(x)} \rightarrow q_{1}\left(u_{1}(x)\right), \ldots, q_{m}\left(u_{m}(x)\right),$$
where $q_{j}$ is a quantizer used to quantize the $j^{\text {th }}$ subvector using the codebook $\mathcal{C}_{j} = \{c_{j,1},...,c_{j,k^*}\}$. Here we assume that
all subquantizers have the same finite number $k^*$ centroids.

```{figure} images/../deepLearning/ApplicationIRSearch/ApproximateNearestNeighbor/codebook_memory_saving_demo_product_quantization
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:codebookmemorysavingdemoproductquantization
Illustration of memory saving benefits of vector product quantization. A $D$-dimensional float vector is first split into $m$ subvectors, and each subvector is stored as its nearest centroid integer id, which only occupies $\log_2 K^*$ bit. 
```

In this case, a reproduction value of the product quantizer is identified by an element of the product index set $\mathcal{I}=$ $\mathcal{I}_{1} \times \ldots \times \mathcal{I}_{m}$. The codebook is therefore defined as the Cartesian product
$$
\mathcal{C}=\mathcal{C}_{1} \times \ldots \times \mathcal{C}_{m}
$$
and a centroid of this set is the concatenation of centroids of the $m$ subquantizers. In that case, the total number of centroids is given by
$$
k=\left(k^{*}\right)^{m}
$$
Note that in the extremal case where $m=D$, the components of a vector $x$ are all quantized separately. Then the product quantizer turns out to be a scalar quantizer, where the quantization function associated with each component may be different.

The strength of a product quantizer is to produce a large set of centroids from several small groups of centroids; each group is associated with its own subquantizer. 

\iffalse
Storing the codebook $\mathcal{C}$ explicitly is not efficient. Instead, we store the $m \times k^{*}$ centroids of all the subquantizers, i.e., $m D^{*} k^{*}=k^{*} D$ floating points values. Quantizing an element requires $k^{*} D$ floating point operations. Table I summarizes the resource requirements associated with k-means, HKM and product $k$-means. The product quantizer is clearly the the only one that can be indexed in memory for large values of $k$.

{\begin{tabular}{lcc} 
	& memory usage & assignment complexity \\
	\hline k-means & $k D$ & $k D$ \\
	product k-means & $m k^{*} D^{*}=k^{1 / m} D$ & $m k^{*} D^{*}=k^{1 / m} D$ \\
	\hline
\end{tabular}
}

In order to provide good quantization properties when choosing a constant value of $k^{*}$, each subvector should have, on average, a comparable energy. One way to ensure this property is to multiply the vector by a random orthogonal matrix prior to quantization. However, for most vector types this is not required and not recommended, as consecutive components are often correlated by construction and are better quantized together with the same subquantizer. As the subspaces are orthogonal, the squared distortion associated with the product quantizer is
$$
\operatorname{MSE}(q)=\sum_{j} \operatorname{MSE}\left(q_{j}\right)
$$
where $\operatorname{MSE}\left(q_{j}\right)$ is the distortion associated with quantizer $q_{j}$. Figure 1 shows the MSE as a function of the code length for different $\left(m, K^{*}\right)$ tuples, where the code length is $l=m \log _{2} K^{*}$, if $K^{*}$ is a power of two. The curves are obtained for a set of 128-dimensional SIFT descriptors, see section $\mathrm{V}$ for details. One can observe that for a fixed number of bits, it is better to use a small number of subquantizers with many centroids than having many subquantizers with few bits. At the extreme when $m=1$, the product quantizer becomes a regular k-means codebook.

High values of $K^{*}$ increase the computational cost of the quantizer, as shown by Table I. They also increase the memory usage of storing the centroids $\left(K^{*} \times D\right.$ floating point values), which further reduces the efficiency if the centroid look-up table does no longer fit in cache memory. In the case where $m=1$, we can not afford using more than 16 bits to keep this cost tractable. Using $K*=256, m=8$ is often a reasonable choice.

\fi

#### Approximating distances using quantized codes

Like vector quantization, we also have different options in approximating distance calculation between query vector $x$ and database vector $y$.

**Symmetric distance computation (SDC)**: 
$$
\hat{d}(x, y)=d(q(x), q(y))=\sqrt{\sum_{j} d\left(q_{j}(x), q_{j}(y)\right)^{2}}
$$
where the distance $d\left(c_{j, i}, c_{j, i^{\prime}}\right)^{2}$ is read from a look-up table associated with the $j^{\text {th }}$ subquantizer. Each look-up table contains all the squared distances between pairs of centroids $\left(i, i^{\prime}\right)$ of the subquantizer, or $\left(k^{*}\right)^{2}$ squared distances $^{1}$.

**Asymmetric distance computation (ADC)**: 
$$
\tilde{d}(x, y)=d(x, q(y))=\sqrt{\sum_{j} d\left(u_{j}(x), q_{j}\left(u_{j}(y)\right)\right)^{2}}
$$
where the squared distances $d\left(u_{j}(x), c_{j, i}\right)^{2}: j=$ $1 \ldots m, i=1 \ldots k^{*}$, are computed prior to the search.
For nearest neighbors search, we do not compute the square roots in practice: the square root function is monotonically increasing and the squared distances produces the same vector ranking.

### Approximate non-exhaustive nearest neighbor search

#### Hierarchical quantization and inverted file indexing

While performing approximate nearest neighbor search with vector quantization or product quantization can already achieve computation acceleration and storage saving in the distance calculation, the search is still exhaustive in which we are computing distance between the query vector and all the vectors in the database. Exhaustive search is not scalable to database containing billions of vectors and scenarios having high query through-puts.   

To avoid exhaustive search we can design a hierarchical quantization strategy to reduce the number of candidates that we will run distance calculation and use product quantization to speed up the distance calculation.

The candidate reduction is achieved via a technique called **inverted file index** (IVF). IVF applies vector quantization via  k-means clustering to produce a large number (e.g., 100) of dataset partitions. At the query time, we identify the a number (e.g., 10) of partitions that are the nearest to the query and only compare the query vector to database vector in the these partitions. The residual distance between each vector and its associated partition centroid is approximated by a residual product quantization. 

In sparse retrieval, inverted indexing refers to the mapping from a term to a list of documents in the database that contain the term. It resemble the index in the back of a textbook, which maps words or concepts to page numbers

In the context of vector quantization for efficient dense retrieval, we use k-means clustering to partition all vectors in the dataset.  
For each partition, an inverted file list refers to the document vectors and its corresponding document ids belonging to this partition.

Given a query, once we determine which partition the query belongs to (using a quantizer), we reduce the search space to the documents in the same partition.  

So far, there are two layers of quantization, which are realized through a **coarse quantizer** and a **residual  product quantizer**. More formally, we denote the centroid $q_{\mathrm{c}}(y)$ associated with a vector $y$. Then the product quantizer $q_{\mathrm{p}}$ is used to encode the residual vector
$$
r(y)=y-q_{\mathrm{c}}(y)
$$
corresponding to the offset in the Voronoi cell. The energy of the residual vector is small compared to that of the vector itself. The vector is approximated by
$$
\tilde{y} = q_{\mathrm{c}}(y)+q_{\mathrm{p}}\left(y-q_{\mathrm{c}}(y)\right).
$$
Commonly we represent $y$ by the tuple $\left(q_{\mathrm{c}}(y), q_{\mathrm{p}}(r(y))\right)$. Like binary representation of a number, the coarse quantizer part represents the most significant bits, while the product quantizer part represents the least significant bits.

\begin{remark}[shared product quantizer for residuals]
The product quantizer can be learned on a set of residual vectors. Ideally, we can learn a product quantizer for each partition since the residual vectors likely to be dependent on the coarse quantizer. One can further reduce memory cost significantly by using the same product quantizer across all coarse quantizers, although this probably gives inferior results
\end{remark}

\iffalse
#### Inverted file indexing and non exhaustive search

**Indexing structure**
We use the coarse quantizer to implement an inverted file structure as an array of lists $\mathcal{L}_{1} \ldots \mathcal{L}_{k^{\prime}}$. If $\mathcal{Y}$ is the vector dataset to index, the list $\mathcal{L}_{i}$ associated with the centroid $c_{i}$ of $q_{\mathrm{c}}$ stores the set $\left\{y \in \mathcal{Y}: q_{\mathrm{c}}(y)=c_{i}\right\}$.
In inverted list $\mathcal{L}_{i}$, an entry corresponding to $y$ contains a vector identifier and the encoded residual $q_{\mathrm{p}}(r(y)):$

{\begin{tabular}{lc}
	\hline field & length (bits) \\
	\hline identifier & $8-32$ \\
	code for residual & $m\left\lceil\log _{2} k^{*}\right\rceil$ \\
	\hline
\end{tabular}
}
The identifier field is the overhead due to the inverted file structure. Depending on the nature of the vectors to be stored, the identifier is not necessarily unique. For instance, to describe images by local descriptors, image identifiers can replace vector identifiers, i.e., all vectors of the same image have the same identifier. Therefore, a 20-bit field is sufficient to identify an image from a dataset of one million. This memory cost can be reduced further using index compression [27], [28], which may reduce the average cost of storing the identifier to about 8 bits, depending on parameters ${ }^{2}$. Note that some geometrical information can also be inserted in this entry, as proposed in [20] and [27].

**Search algorithm** The inverted file is the key to the non-exhaustive version of our method. When searching the nearest neighbors of a vector $x$, the inverted file provides a subset of $\mathcal{Y}$ for which distances are estimated: only the inverted list $\mathcal{L}_{i}$ corresponding to $q_{\mathrm{c}}(x)$ is scanned.

However, $x$ and its nearest neighbor are often not quantized to the same centroid, but to nearby ones. To address this problem, we use the multiple assignment strategy of [29]. The query $x$ is assigned to $w$ indexes instead of only one, which correspond to the $w$ nearest neighbors of $x$ in the codebook of $q_{\mathrm{c}}$. All the corresponding inverted lists are scanned. Multiple assignment is not applied to database vectors, as this would increase the memory usage.

Figure 5 gives an overview of how a database is indexed and searched.

\begin{method}Indexing a vector $y$ proceeds as follows:
	1. quantize $y$ to $q_{\mathrm{c}}(y)$
	1. compute the residual $r(y)=y-q_{\mathcal{c}}(y)$
	1. quantize $r(y)$ to $q_{\mathrm{p}}(r(y))$, which, for the product quantizer, amounts to assigning $u_{j}(y)$ to $q_{j}\left(u_{j}(y)\right)$, for $j=1 \ldots m$.
	1. add a new entry to the inverted list corresponding to $q_{\mathrm{c}}(y)$. It contains the vector (or image) identifier and the binary code (the product quantizer's indexes).

\end{method}

\begin{method}Searching the nearest neighbor(s) of a query $x$ consists of
	1. 1) quantize $x$ to its $w$ nearest neighbors in the codebook $q_{c}$;

	For the sake of presentation, in the two next steps we simply denote by $r(x)$ the residuals associated with these $w$ assignments. The two steps are applied to all $w$ assignments.
	1. compute the squared distance $d\left(u_{j}(r(x)), c_{j, i}\right)^{2}$ for each subquantizer $j$ and each of its centroids $c_{j, i}$
	1. compute the squared distance between $r(x)$ and all the indexed vectors of the inverted list. Using the subvector-to-centroid distances computed in the previous step, this consists in summing up $m$ looked-up values.
	1. select the $K$ nearest neighbors of $x$ based on the estimated distances. This is implemented efficiently by maintaining a Maxheap structure of fixed capacity, that stores the $K$ smallest values seen so far. After each distance calculation, the point identifier is added to the structure only if its distance is below the largest distance in the Maxheap.

\end{method}

\begin{remark}[computational cost]
Only Step 3 depends on the database size. Compared with $\mathrm{ADC}$, the additional step of quantizing $x$ to $q_{\mathrm{c}}(x)$ consists in computing $k^{\prime}$ distances between $D$ dimensional vectors. Assuming that the inverted lists are balanced, about $n \times w / k^{\prime}$ entries have to be parsed. Therefore, the search is significantly faster than ADC, as shown in the next section.
\end{remark}

```{figure} images/../deepLearning/ApplicationIRSearch/ApproximateNearestNeighbor/IVFADC_arch
:name: fig:ivfadcarch
Overview of the inverted file with asymmetric distance
		computation (IVFADC) indexing system. Top: insertion of a vector during indexing stage.
		Bottom: search $k$ nearest neighbor for a query vector $x$.

```

\fi

\iffalse

Traditional databases are made up of structured tables containing symbolic information. For example, an image collection would be represented as a table with one row per indexed photo. Each row contains information such as an image identifier and descriptive text. Rows can be linked to entries from other tables as well, such as an image with people in it being linked to a table of names.

AI tools, like text embedding (word2vec) or convolutional neural net (CNN) descriptors trained with deep learning, generate high-dimensional vectors. These representations are much more powerful and flexible than a fixed symbolic representation, as we’ll explain in this post. Yet traditional databases that can be queried with SQL are not adapted to these new representations. First, the huge inflow of new multimedia items creates billions of vectors. Second, and more importantly, finding similar entries means finding similar high-dimensional vectors, which is inefficient if not impossible with standard query languages.

```{figure} images/../deepLearning/ApplicationRecommenderSys/similaritySearch/vector_similarity_search
:name: fig:vectorsimilaritysearch

```

\fi

## Query understanding

### Query preprocessing

Character Filtering:
- Unicode Normalization
- Removing Accents
- Ignoring Capitalization

Tokenization:
- White space characters
- Punctuation
- Hyphens and Apostrophes
- In chinese/japanese, it’s necessary to employ a word segmentation algorithm specific to that language.

## Query Rewriting

### Introduction

Query rewriting (QRW), which targets to alter a given query to alternative queries that can improve relevance performance by reducing the mismatches, is a critical task in modern search engines.

### Traditional query rewriting

### Embedding-based approach

### Sequence-to-sequence learning

\cite{he2016learning}
In this paper, we propose a learning to rewrite framework that consists of a candidate generating phase and a candidate ranking phase. The candidate generating phase provides us the flexibility to reuse most of existing query rewriters; while the candidate ranking phase allows us to explicitly optimize search relevance. Experimental results on a commercial search engine demonstrate the effectiveness of the proposed framework. Further experiments are conducted to understand the important components of the proposed framework.

#### Candidate generation

 We propose to use the Sequence-to-Sequence LSTM model [21] to build a new candidate generator. In model training, we treat the original query as input sequence, and use its rewrite queries as target output sequences. In prediction, the most probable output sequences are obtained by a beam-search method elaborated at the end of this section and are used as the query rewrite candidates.

In a sequence to sequence LSTM, we want to estimate the conditional probability $p\left(y_1, \cdots, y_I \mid x_1, \cdots, x_J\right)$ where $x_1, \cdots, x_J$ is an input sequence and $y_1, \cdots, y_I$ is its corresponding output sequence whose length $I$ may differ from J. The LSTM computes this conditional probability by first obtaining the fixed dimensional representation $v$ of the input sequence $x_1, \cdots, x_J$ given by the last hidden state of the LSTM, and then computing the probability of $y_1, \cdots, y_I$ with a standard LSTM-LM formulation whose initial hidden state is set as the representation $v$ of $x_1, \cdots, x_J$ :
$$
p\left(y_1, \cdots, y_I \mid x_1, \cdots, x_J\right)=\prod_{i=1}^I p\left(y_i \mid v, y_1, \cdots, y_{i-1}\right),
$$
where $p\left(y_i \mid v, y_1, \cdots, y_{i-1}\right)$ is represented with a softmax over all the words in the vocabulary. Note that we require that each query ends with a special end-of-query symbol " $<$ EOQ $>$ ", which enables the model to define a distribution over sequences of all possible lengths. The overall scheme is outlined in figure 2, where the shown LSTM computes the representation of the terms in the query $q t_1, q t_2, \cdots, q t_m$, $\angle \mathrm{EOQ}>$ and then uses this representation to compute the probability of $r t_1, r t_2, \cdots, r t_n,<\mathrm{EOQ}>$.

```{figure} images/../deepLearning/ApplicationIRSearch/QueryRewriting/learning_to_rewrite/seq_2_seq_rewrite_arch
:name: fig:seq2seqrewritearch
Scheme of Sequence to Sequence LSTM for Generating Query Rewrite.
```

\begin{remark}[training]
We learn a large deep LSTM on large-scale query-rewrite query pairs. More details about how to prepare these pairs will be discussed in the experimental section. We trained it by maximizing the log probability of a correct rewrite query $r=r t_1, r t_2, \cdots, r t_n,<E O Q>$ given the query $q=$ $q t_1, q t_2, \cdots, q t_m,<E O Q>$, thus the training objective is
$$
\frac{1}{|D|} \sum(q, r) \in D \log p(r \mid q),
$$
where $D$ is the training data set and $p(r \mid q)$ is calculated according to Eq. (2). Once training is complete, we feed original queries to the model and produce rewrite candidates by finding the most likely rewrites according to the LSTM - We search for the most likely query rewrites using a simple leftto-right beam search decoder instead of an exact decoder.
\end{remark}

#### Learning to rank candidates

\begin{remark}[generating training data label]
In this work, we specifically focus on boosting the relevance performance via query rewriting, thus the learning target should indicate the quality of the rewrite candidates from the perspective of search relevance. Intuitively a better rewrite candidate could attract more clicks to its documents. In other words, the number of clicks on the returned document from a rewrite candidate could be a good indicator about its quality in terms of relevance.  

For each query and query rewrite pair $(q, r)$, we estimate the click numbers if we alter $q$ to $r$ from the query-document click graph as illustrated in Figure 3 :
- We use the query $r$ to retrieve top $k$ documents from the search engine as $\mathcal{U}=\left\{u_1, u_2, \cdots, u_k\right\}$ where $u_i$ indicates the ranked $i$-th document in the returned list. For example, we retrieve $k=4$ documents for the rewrite candidate $r_1$.
- The click number of the document $u_i$ in $\mathcal{U}$ with the information intent of $q c_{r, u_i}$ is estimated as $n_{q, u_i}$ if there are clicks between query $q$ and document $u_i$ in the query-document graph (e.g., $c_{r_1, d_2}=n_{q, d_2}$ in Figure 3) and 0 (e.g., $c_{r_2, d_2}=0$ in Figure 3 ) otherwise. The rationale of using $\left(q, u_i\right)$ instead of $\left(r, u_i\right)$ is that $r$ could drift the intent of $q$ and we want to maintain the original intent. For example, if $r$ changes the intent of $q, u_i$ could be completely irrelevant to $q$ and it is reasonable to estimate $c_{r, u_i}=0$. Let $\mathcal{C}_r=\left\{c_{r, u_1}, c_{r, u_2}, \ldots, c_{r, u_k}\right\}$ be the estimated click numbers for its top $k$ retrieved documents.

With the estimated click numbers $\mathcal{C}_r$ and document positions as shown in Figure 3, we can generate the learning target. Next we illustrate how to generate the point-wise learning target $y_{q, r}$ from $\mathcal{C}_r$. Our basic idea is to aggregate the estimated click numbers in $\mathcal{C}_r$ to a unified target $y_{q, r}$. In this paper, we investigate the following aggregating strategies:
- Clicknum: Intuitively the total click numbers can indicate the quality of a rewrite candidate in terms of relevance. Therefore the Clicknum strategy is to sum click numbers in $\mathcal{C}_r$ as $y_{q, r}$ :
$$
y_{q, r}:=\sum_{i=1}^k c_{r, u_i}
$$
- Discounted clicknum: In addition to click numbers, the positions of documents in the ranking list are also important. Ideally we should rank documents with a large number of clicks higher; hence we need to penalize these documents with a large number of clicks but

lower positions. With these intuitions, the discounted clicknum strategy defines $y_{q, r}$ from $\mathcal{C}_r$ as:
$$
y_{q, r}:=\sum_{i=1}^k \frac{c_{r, u_i}}{i+1},
$$
where the contribution of $c_{r, u_i}$ in $y_{q, r}$ is penalized by its position in the ranking list.
- Discounted log clicknum: Click numbers in $\mathcal{C}_T$ could vary dramatically. Some have an extremely large number of clicks; while others have a few. In this scenario, the documents with large numbers of clicks could dominate the learning target. Therefore the discounted log clicknum strategy applies a $\log$ function to click numbers as:
$$
y_{q, r}:=\sum_{i=1}^k \frac{\log _2\left(c_{r, u_i}\right)}{i+1} .
$$
- Logdiscounted log clicknum: The Discounted clicknum strategy could over-penalize the click numbers. Similar to Discounted log clicknum, Logdiscounted log clicknum also applies a log function to positions as:
$$
y_{q, r}:=\sum_{i=1}^k \frac{\log _2\left(c_{r, u_i}\right)}{\log (i+1)} .
$$
\end{remark}

```{figure} images/../deepLearning/ApplicationIRSearch/QueryRewriting/learning_to_rewrite/ranking_label_generation
:name: fig:rankinglabelgeneration
Generating click numbers Cr from the query-document click graph
```

\begin{remark}[ranking features]
For each pair of query and rewrite candidate $(q, r)$, we can build three groups of feature functions - 
- query features: features are extracted from only the original query $q$;
- rewrite features: features are extracted from only the rewrite candidate $r$
- pair features: features are extracted from both $q$ and $r$

Before introducing the details about these features, we first introduce notations we use in their definitions. Let $f_q$ and $f_r$ be the query frequencies obtained from the search log. Let $\mathcal{W}=\left\{W_1, W_2, \ldots, W_N\right\}$ be the dictionary with $N$ words. We use $\mathbf{q}=\left\{w_{q, 1}, w_q, 2, \ldots, w_{q, N}\right\}$ to indicate the vector representation of $q$ where $w_{q, i}$ is the frequency of $W_i$ in $q$. Similarly we represent $r$ as $\mathbf{r}=$ $\left\{w_{r, 1}, w_{r, 2}, \ldots, w_{r, N}\right\}$. We further assume that $\mathcal{U}_q$ and $\mathcal{U}_r$ are the sets of URLs connecting to $q$ and $r$ in the querydocument graph, respectively. The definitions and descriptions of these features are summarized in Table 1.	
\end{remark}

\begin{table}[H]
\footnotesize
\centering
\begin{tabular}{p{0.1\textwidth}p{0.85\textwidth}}
	\hline Feature Group & Feature \\
	\hline \multirow{5}{*}{ Query Features } & $h_1-$ Number of words in $q$ as $\sum_{i=1}^N w_{q, i}$ \\
	& $h_2-$ Number of stop words in $q: S_q$ \\
	& $h_3-$ Language model score of the query $q: L_q$ \\
	& $h_4-$ The query frequencies of the query $q: f_q$ \\
	& $h_5-$ The average length of words in $q: A_q$ \\
	\hline \multirow{5}{*}{ Rewrite Features }& $h_6-$ Number of words in $r$ as $\sum_{i=1}^N w_{r, i}$ \\
	& $h_7-$ Number of stop words in $r: S_r$ \\
	& $h_8-$ Language model score of the query rewrite $r: L_r$ \\
	& $h_9-$ The query frequencies of the query rewrite $r: f_r$ \\
    & $h_{10}-$ The average length of words in $r: A_r$ \\
	\hline \multirow{8}{*}{ Pair Features }& $h_{11}-$ Jaccard similarity of URLs as $\frac{\left|\mathcal{U}_q \cap \mathcal{U}_r\right|}{\left|\mathcal{U}_q \cup \mathcal{U}_r\right|}$ \\
& $h_{12}-$ Difference between the frequencies of the original query $q$ and the rewrite candidate $q: f_q-f_r$ \\
& $h_{13}-$ Word-level cosine similarity between $q$ and $r: \frac{\sum_{i=1}^N w_{q, i} w_{r, i}}{\sqrt{\sum_{i=1}^N w_{q, i}^2} \sqrt{\sum_{i=1}^N w_{r, i}^2}}$ \\
& $h_{14}-$ Difference between the number of words between $q$ and $r: \sum_{i=1}^N w_{q, i}-\sum_{i=1}^N w_{r, i}$ \\
& $h_{15}-$ Number of common words in $q$ and $r$ \\
& $h_{16}-$ Difference of language model scores between $q$ and $r: L_q-L_r$ \\
& $h_{17}-$ Difference between the number of stop words between $q$ and $r: S_q-S_r$ \\
& $h_{18}-$ Difference between the average length of words in $q$ and $r: A_q-A_r$ \\
\hline
\end{tabular}
\end{table}

## Spelling correction

### Introduction
Spelling correction is a must-have for any modern search engine. Estimates of the fraction of misspelled search queries vary, but a variety of studies place it between 10\% and 15\%. For today’s searchers, a search engine without robust spelling correction simply doesn’t work.

That doesn’t mean, however, that you should build a spelling correction system from scratch. Off-the-shelf spelling correction systems, such as Aspell<sup>[^5]</sup> or Hunspell<sup>[^6]</sup>, are highly customizable and should suit most people’s needs.

### Architecture

#### Overall architecture
Offline, before any queries take place:
- Indexing Tokens. Building the index used at query time for candidate generation.
- Building a language model. Computing the model to estimate the a priori probability of an intended query.
- Building an error model. Computing the model to estimate the probability of a particular misspelling, given an intended query.

At query time:
- Candidate generation. Identifying the spelling correction candidates for the query.
- Scoring. Computing the score or probability for each candidate.
- Presenting suggestions. Determining whether and how to present a spelling correction.

#### Indexing tokens

Indexing for spelling correction is a bit different than for document retrieval. The fundamental data structure for document retrieval is an inverted index (also known as a posting list) that maps tokens to documents. In contrast, indexing for spelling correction typically maps substrings of tokens (character n-grams) to tokens.
Most misspelled tokens differ from the intended tokens by at most a few characters (i.e, a small edit distance). An index for approximate string matching enables discovery of these near-misses through retrieval of strings - in our case, tokens - by their substrings. We generate this index by identifying the unique tokens in the corpus, iterating through them, and inserting the appropriate substring-to-string mappings into a substring index, such as an $\mathrm{n}$-gram index or a suffix tree.
Although this description assumes a batch, offline indexing process, it is straightforward to update the index incrementally as we insert, remove, or update documents.

This index grows with the size of the corpus vocabulary, so it can become quite large. Moreover, as vocabulary grows, the index becomes more dense: in particular, short substrings are mapped to large numbers of tokens. Hence, a spelling correction system must make tradeoffs among storage, efficiency, and quality in indexing and candidate generation.
It is also possible to index tokens based on how they sound, such as canonicalizing them into Metaphone codes. This approach is most useful for words with unintuitive spellings, such as proper names or words adopted from other languages.

### Building models

The goal of spelling correction is to find the correction, out of all possible candidate corrections (including the original query), that has the highest probability of being correct. In order to do that, we need two models: a language model that tells us a priori probability of a query, and an error model that tells us the probability of a query string given the intended query. With these two models and Bayes’ theorem, we can score candidate spelling correction candidates and rank them based on their probability.

#### Building a language model

The language model estimates the probability of an intended query — that is, the probability, given no further information, that a searcher would set out to type a particular query.

Let’s consider the simplifying assumption that all of our queries are single tokens. In that case, we can normalize the historical frequencies of unique queries to establish a probability distribution. Since we have to allow for the possibility of seeing a query for the first time, we need smoothing (e.g, Good-Turing) to avoid assigning it a probability of zero. We can also combine token frequency in the query logs with token frequency in the corpus to determine our prior probabilities, but it’s important not to let the corpus frequencies drown out the query logs that demonstrate actual searcher intent.

This approach breaks down when we try to model the probability of multiple-token queries. We run into two problems. The first is scale: the number of token sequences for which we need to compute and store probabilities grows exponentially in the query length. The second is sparsity: as queries get longer, we are less able to accurately estimate their low probabilities from historical data. Eventually we find that most long token sequences have never been seen before.

The solution to both problems is to rely on the frequencies of n-gram frequencies for small values of n (e.g. unigrams, bigrams, and trigrams). For larger values of n, we can use backoff or interpolation. For more details, read this book chapter on n-grams by Daniel Jurafsky and James Martin.

#### Building an Error Model

The error model estimates the probability of a particular misspelling, given an intended query. You may be wondering why we’re computing the probability in this direction, when our ultimate goal is the reverse — namely, to score candidate queries given a possible misspelling. But as we’ll see, Bayes’ theorem allows us to combine the language model and the error model to achieve this goal.

Spelling mistakes represent a set of one or more errors. These are the most common types of spelling errors, also called edits:
- Insertion. Adding an extra letter, e.g., truely. An important special case is a repeated letter, e.g., pavillion.
- Deletion. Missing a letter, e.g., chauffer. An important special case is missing a repeated letter, e.g., begining.
- Substitution. Substituting one letter for another, e.g., appearence. The most common substitutions are incorrect vowels.
- Transposition. Swapping consecutive letters, e.g. acheive.

We generally model spelling mistakes using a noisy channel model that estimates the probability of a sequence of errors, given a particular query. We can tune such a model heuristically, or we can train a machine-learned model from a collection of example spelling mistakes.

Note that the error model depends on the characteristics of searchers, and particularly on how they access the search engine. In particular, people make different (and more frequent) mistakes on smaller mobile keyboards than on larger laptop keyboards.

### Candidate generation

Candidate generation is analogous to document retrieval: it’s how we obtain a tractable set of spelling correction candidates that we then score and rank.

Just as document retrieval leverages the inverted index, candidate generation leverages the spelling correction index to obtain a set of candidates that is hopefully includes the best correction, if there is one.

To get an idea of how candidate generation works, consider how you might retrieval all tokens within an edit distance of 1 of the misspelled query retreival. A token within edit distance must end with etreival, or start with r and end with treival, or start with re and end with reival, or start with ret and end with eival, etc. Using a substring index, you could retrieve all of these candidates with an OR of ANDs.

That’s not the most efficient or general algorithm for candidate generation, but hopefully it gives you an idea of how such algorithms work.

The cost of retrieval depends on the aggressivity of correction, which is roughly the maximum edit distance — that is, the number of edits that can be compounded in a single mistake. The set of candidates grows exponentially with edit distance. The probability of a candidate decreases with its edit distance, so the cost of more aggressive candidate generation yields diminishing returns.
### Scoring

Hopefully the right spelling correction is among the generated candidates. But how do we pick the best candidate? And how do we determine whether the query was misspelled in the first place?
For each candidate, we have its prior probability from the language model. We can use the error model to compute the probability of the query, given the candidate. So now we apply Bayes' theorem to obtain the conditional probability of the candidate, given the query:

Prob (candidate $\mid$ query) $\propto \operatorname{Prob}$ (query $\mid$ candidate) * Prob (candidate)
We use the proportionality symbol $(\propto)$ here instead of equality because we've left out the denominator, which is the a priori probability of the query. This denominator doesn't change the relative scores of candidates.

In general, we go with the top-scoring candidate, which may be the query itself. We use the probability to establish our confidence in the candidate.

### Tutorial

How to write a Spelling corrector
http://norvig.com/spell-correct.html

## Speller

### Introduction
Spelling mistakes constitute a large share of errors in search engine queries (e.g., Nordlie 1999 [1], Spink et al. 2001 [2], Wang et al. 2003 [3]) and written text in general (Flor and Futagi, 2012 [4] ). Taking Bing for example, $10 \% \sim 15 \%$ query traffic contain typos. The capability of automatically and accurately correcting user input errors greatly enhances user experience. Without a speller, Bing users may see up to $15 \%$ searches return irrelevant results. Meanwhile, spellers are a ubiquitous and fundamental component for various Microsoft products, not only in Bing search engine, but also in productivity and collaboration tools (M365) and Microsoft Cognitive Service in Azure Al.

Different applications may need different spellers. For example, the Bing speller handles search queries that are often short and keyword-based. It needs frequent update to adapt to fast Web/index change and emerging query trends by recognizing new entities, events, etc. On the other hand, office spellers fix typos in formal documents with long sentences in natural languages (also called document proofing). The typo patterns are very different from those in Web search queries, and they do not change as fast. Therefore, it is not necessary to frequently update office spellers.

#### Speller error type

There is a long and rich literature of spelling correction. We can categorize previous works from various dimensions.

First, based on whether the correction target appears in a pre-defined lexicon, spelling correction research is broadly grouped into two categories: **correcting Non-word Errors (NWE)** and **Real Word Errors (RWE)**. 
- In NWE task, any word not found in a pre-defined vocabulary (e.g., a lexicon or a confusion set) is considered as a misspelling word.
- The RWE task is also referred to Context Sensitive Spelling Correction (CSSC). If the misspelling word is a valid word in the vocabulary, it is a real word error, which means the error is context sensitive. For example, both "peace" and "piece" are valid words in dictionary, but in the context "a\_ of cake", "peace" is the misspelled token. 

 A straightforward approach to NWE spelling correction is applying a similarity function to rank a list of lexical words which are like the misspelling word. Therefore, how to define the similarity function becomes the key to this approach. Various methods, including edit distance, phonetic similarity, as well as keyboard layout proximity, have been proposed. 

 (Choudhury et al. 2007) [9] builds a weighted network of words called SpellNet and finds that the probability of RWE in a language is equal to the average weighted degree of SpellNet. (Church et al. 2007) [10] compress a trigram language model for spelling task. (Yanen et al. 2011) [11] designs a unified HMM model and utilize bigram language model in the state transition.

Second, typing error can also be categorized as typographic error and cognitive errors (Kukich 1992) [12]. Typographic errors are the result of mistyped keys and can be described in terms of keyboard key proximity. Cognitive errors on the other hand, are caused by a misunderstanding of the correct spelling of a word. They include phonetic errors, in which similar sounding letter sequences are substituted for the correct sequence; and homonym errors, in which a word is substituted for another word with the same pronunciation but a different meaning.

### Language models for speller

Language Models play a critical role in spellers by predicting the likelihood of validity for a given piece of text. The Bing Speller is equipped with two types of LMs: N-Gram LM and deep LM (e.g., BERT pre-trained LMs). An N-gram LM is built directly from the Web corpus, such as the title and body of Web documents. It is good at capturing entities on the Web. Complementary to an N-gram LM, a deep LM is powerful at handling semantic queries. A speller implements LM using N-grams at different length scales: smaller scale (i.e., shorter) LM is applied in candidate generation (e.g., the char-level LM and word-level LM used in SMT are often short N-grams), while larger scale (i.e., longer) LM is used in ranking phase.

#### Char-level N grams
Ideally, we would like a vocabulary to cover all possible valid words (associated with their popularity). However, for a search engine like Bing, it may not be realistic to build a comprehensive vocabulary since the Web is massive and dynamic. It is a huge engineering challenge to build and maintain such a vocabulary. An alternative solution is to use a language model (LM) to tell, in terms of probability, if a word is valid. Char-level N-grams is one classic implementation of such a language model. To some degree, a vocabulary is a special case of an LM, which returns only true or false ( 0 or 1 probability). A general LM represents new words using probabilities, which is smoother than a crisp vocabulary. Moreover, LM has better generalization capability on unseen words, which is preferred by search engines. Char-level Ngrams are used to build Char-level N-gram Language Models (details in Section 3.2.1), which are further used in SMT candidate generation (details in Section 7).

#### Word level N grams

While Char-level N-grams are used to build an LM to tell the validity of words, Word-level N-grams are used to build an LM to address RWE (real word error) in queries. A Word-level LM checks the correctness of a word in the context of the whole query; it can be considered as the extension of a Char-level LM. Word-level N-grams can be generated from various sources, including the user queries, and the title, body, anchor and URL of Web documents. They are commonly used in Viterbi decoder (Section 6), SMT query decoder (Section 7.3) and N-gram LM features at speller ranking stage (Section 9.2).

## Query rewriting

### Overview

 Query rewriting automatically transforms search queries in order to better represent the searcher’s intent. Query rewriting strategies generally serve two purposes: increasing recall and increasing precision.

Query rewriting is a powerful tool for taking what we understand about a query and using it to increase both recall and precision. Many of the problems that search engines attempt to solve through ranking can and should be addressed through query rewriting.

This post provides an overview of query rewriting. We’ll dive into the details of specific techniques in future posts.

#### Techniques to increasing Recall

A key motivation for query rewriting search queries is to increase recall — that is, to retrieve a larger set of relevant results. In extreme cases, increasing recall is the difference between returning some (hopefully relevant) results and returning no results.

The two main query rewriting strategies to increase recall are query expansion and query relaxation.

**Query Expansion**

Query expansion broadens the query by adding additional tokens or phrases. These additional tokens may be related to the original query terms as synonyms or abbreviations (we’ll discuss how to obtain these in future posts); or they may be obtained using the stemming and spelling correction methods we covered in previous posts.

If the original query is an AND of tokens, query expansion replaces it with an AND of ORs. For example, if the query ip lawyer originally retrieves documents containing ip AND lawyer, an expanded query using synonyms would retrieve documents containing (ip OR “intellectual property”) AND (lawyer OR attorney).

Although query expansion is mostly valuable for increasing recall, it can also increase precision. Matches using expanded tokens may be more relevant than matches restricted to the original query tokens. In addition, the presence of expansion terms serves as a relevance signal to improve ranking.

**Query Relaxation**

Query relaxation feels like the opposite of query expansion: instead of adding tokens to the query, it removes them. Specifically, query relaxation increases recall by removing — or optionalizing — tokens that may not be necessary to ensure relevance. For example, a search for cute fluffy kittens might return results that only match fluffy kittens.

A query relaxation strategy can be naïve, e.g, retrieve documents that match all but one of the query tokens. But a naive strategy risks optionalizing a token that is critical to the query’s meaning, e.g., replacing cute fluffy kittens with cute fluffy. More sophisticated query relaxation strategies using query parsing or analysis to identify the main concept in a query and then optionalize words that serve as modifiers.

Both query expansion and query relaxation aim to increase recall without sacrificing too much precision. They are most useful for queries that return no results, since there we have the least to lose and the most to gain. In general, we should be increasingly conservative about query expansion — and especially about query relaxation — as the result set for the original query grows.

#### Techniques to increasing Precision

Query rewriting can also be used to increase precision — that is, to reduce the number of irrelevant results. While increasing recall is most valuable for avoiding small or empty result sets, increasing precision is most valuable for queries that would otherwise return large, heterogeneous result sets.

**Query Segmentation**

Sometimes multiple tokens represent a single semantic unit, e.g., dress shirt in the query white dress shirt. Treating this segment as a quoted phrase, i.e., rewriting the query as white “dress shirt” can significantly improve precision, avoiding matches for white shirt dresses.

Query segmentation is related to tokenization: we can think of these segments as larger tokens. But we generally think of tokenization at the character level and query segmentation at the token level. We will discuss query segmentation algorithms in a future post.

**Query Scoping**

Documents often have structure. Articles have titles and authors; products have categories and brands; etc. Query rewriting can improve precision by scoping, or restricting, how different parts of the query match different parts of documents.

Query scoping often relies on query segmentation. We determine an entity type for each query segment, and then restrict matches based on an association between entity types and document fields.

Query rewriting can also perform scoping at the query level, e.g., restricting the entire set of results to a single category. This kind of scoping is typically framed as a classification problem.

### Query expansion
A key application of query rewriting is increasing recall — that is, matching a larger set of relevant results. In extreme cases, increasing recall means the difference between returning some results and returning no results. More typically, query expansion casts a wider net for results that are relevant but don’t match the query terms exactly.

Overview

Query expansion broadens the query by introducing additional tokens or phrases. The search engine automatically rewrites the query to include them. For example, the query vp marketing becomes (vp OR “vice president”) marketing.

The main challenge for query expansion is obtaining those additional tokens and phrases. We also need to integrate query expansion into scoring and address the interface considerations that arise from query expansion.

Sources for Query Expansion Terms

We’ve covered spelling correction and stemming in previous posts, so we won’t revisit them here. In any case, they aren’t really query expansion. Spelling correction generally replaces the query rather than expanding it. Stemming is usually implemented by replacing tokens in queries and documents with their canonical forms, although it can also be implemented using query expansion.

Query expansion terms are typically abbreviations or synonyms.

Abbreviations

Abbreviations represent exactly the same meaning as the words they abbreviate, e.g., inc means incorporated. Our challenge is recognizing abbreviations in queries and documents.

Using a Dictionary

The simplest approach is to use a dictionary of abbreviations. There are many commercial and noncommercial dictionaries available, some intended for general-purpose language and others for specialized domains. We can also create our own. Dictionaries work well for abbreviations that are unambiguous, e.g., CEO meaning chief executive officer. We simply match strings, possibly in combination with stemming or lemmatization.

But abbreviations are often ambiguous. Does st mean street or saint? Without knowing the query or document context, we can’t be sure. Hence our dictionary has to be conservative to minimize the risk of matching abbreviations to the wrong words. We face a harsh trade-off between precision and recall.

Using Machine Learning

A more sophisticated approach is to model abbreviation recognition as a supervised machine learning problem. Instead of simply recognizing abbreviations as strings, we train a model using examples of abbreviations in context (e.g., the sequence of surrounding words), and we represent that context as components in a feature vector. This approach works better for identifying abbreviations in documents than in queries, since the former supply richer context.

How do we collect these examples? We could use an entirely manual process, annotating documents to find abbreviations and extracting them along with their contexts. But such an approach would be expensive and tedious.

A more practical alternative is to automatically identify potential abbreviations using patterns. For example, it’s common to introduce an abbreviation by parenthesizing it, e.g., gross domestic product (GDP). We can detect this and other patterns automatically. Pattern matching won’t catch all abbreviations, and it will also encounter false positives. But it’s certainly more scalable than a manual process.

Another way to identify abbreviations is to use unsupervised machine learning. We look for pairs of word or phrases that exhibit both surface similarity (e.g., matching first letters or one word being a prefix of the other) and semantic similarity. A popular tool for the latter is Word2vec: it maps tokens and phrases to a vector space such that the cosine of the angle between vectors reflects the semantic similarity inferred from the corpus. As with a supervised approach, this approach will both miss some abbreviations and introduce false positives.

Synonyms

Most of the techniques we’ve discussed for abbreviations apply to synonyms in general. We can identify synonyms using dictionaries, supervised learning, or unsupervised learning. As with abbreviations, we have to deal with the inherent ambiguity of language.

An important difference from abbreviations is that, since we can no longer rely on the surface similarity of abbreviations, we depend entirely on inferring semantic similarity. That makes the problem significantly harder, particularly for unsupervised approaches. We’re likely to encounter false positives from antonyms and other related words that aren’t synonyms.

Also, unlike abbreviations, synonyms rarely match the original term exactly. They may be more specific (e.g., computer -> laptop), more general (ipad -> tablet), or similar but not quite identical (e.g, web -> internet).

Hence, we not only need to discover and disambiguate synonyms; we also need to establish a similarity threshold. If we’re using Word2vec, we can require minimum cosine similarity.

Also, if we know the semantic relationship between the synonym and the original term, we can take it into account. For example, we can favor a synonym that is more specific than the original term, as opposed to one that is more general.

Scoring Results

Query expansion uses query rewriting to increase the number of search results. How do we design the scoring function to rank results that match because of query expansion?

The simplest approach is to treat matches from query expansion just like matches to the original query. This approach works for abbreviation matches that completely preserve the meaning of the original query terms — assuming that we don’t make any mistakes because of ambiguity. Synonym matches, however, may introduce subtle changes in meaning.

A more sophisticated approach is to apply a discount to matches from query expansion. This discount may be a constant, or it can reflect the expected change in meaning (e.g., a function of the cosine similarity). This approach, while heuristic, integrates well with hand-tuned scoring functions, such as those used in many Lucene-based search engines.

The best — or at least most principled — approach is to integrate query expansion into a machine learned ranking model using features (in the machine learning sense) that indicate whether a document matched the original query terms or terms introduced through query expansion. These features should also indicate whether the expansion was through an abbreviation or a synonym, the similarity of the synonym, etc.

Integrating query expansion into a machine-learned ranking model is a bit tricky. We can’t take full advantage of pre-existing training data from a system that hasn’t performed query expansion. Instead, we start with a heuristic model to collect training data (e.g., one of the previously discussed approaches) and then use that data to learn weights for query expansion features.

### Query relaxation

In the previous post, we discussed query expansion as a way to increase recall. In this post we’ll discuss the other major technique for increasing recall: query relaxation.

Query relaxation feels like the opposite of query expansion. Instead of adding tokens to the query, we remove them. Ignoring tokens makes the query less restrictive and thus increases recall. An effective query relaxation strategy removes only tokens that aren’t necessary to communicate the searcher’s intent.

Let’s consider four approaches to query relaxation, in increasing order of complexity: stop words, specificity, syntactic analysis, and semantic analysis.

Stop Words

The simplest form of query relaxation is ignoring stop words. Stop words are words like the and of: they generally don’t contribute meaning to the query; hence, removing them preserves the intent while increasing recall.

But sometimes stop words matter. There’s a difference between king hill and king of the hill. And there’s even a post-punk band named The The. These edge cases notwithstanding, stop words are usually safe to ignore.

Most open-source and commercial search engines come with a list of default stop words and offer the option of ignoring them during query processing (e.g. Lucene’s StopFilter). In addition, Google has published lists of stop words in 29 languages.

Specificity

Query tokens vary in their specificity. For example, in the search query black hdmi cable, the token hdmi is more specific than cable, which is in turn more specific than black. Specificity generally indicates how essential each query token is to communicating the searcher’s intent. Using specificity, we can determine that it’s more more reasonable to relax the query black hdmi cable to hdmi cable than to black cable.

Inverse Document Frequency

We can measure token specificity using inverse document frequency (idf). The inverse (idf is actually the logarithm of the inverse) means that rare tokens — that is, tokens that occur in fewer documents — have a higher idf than those that occur more frequently. Using idf to measure token specificity is a generalization of stop words, since stop words are very common words and thus have low idf.

Information retrieval has used idf for decades, ever since Karen Spärck Jones’s seminal 1972 paper on “A statistical interpretation of term specificity and its application in retrieval”. It’s often combined with term frequency (tf) to obtain tf-idf, a function that assigns weights to query tokens for scoring document relevance. For example, Lucene implements a TFIDFSimilarity class for scoring.

But be careful about edge cases. Unique tokens, such as proper names or misspelled words, have very high idf but don’t necessarily represent a corresponding share of query intent. Tokens that aren’t in the corpus have undefined idf — though that can be fixed with smoothing, e.g., adding 1 before taking the logarithm. Nonetheless, idf is a useful signal of token specificity.

Lexical Databases

A completely different approach to measuring specificity is to use a lexical database (also called a “knowledge graph”) like WordNet that arranges concepts into semantic hierarchies. This approach is useful for comparing tokens with a hierarchical relationship, e.g., dog is more specific than animal. It’s less useful for tokens without a hierarchical relationship, e.g., black and hdmi.

A lexical database also enables a more nuanced form of query relaxation. Instead of ignoring a token, we can replace it with a more general term, also known as a hypernym. We can also use a lexical database for query expansion.

Syntactic Analysis

Another approach to query relaxation to use a query’s syntactic structure to determine which tokens are optional.

A large fraction of search queries are noun phrases. A noun phrase serves the place of a noun — that is, it represents a thing or set of things. A noun phrase can be a solitary noun, e.g., cat, or it can a complex phrase like the best cat in the whole wide world.

We can analyze search queries using a part-of-speech tagger, such as NLTK, which in turn allows us to parse the overall syntactic structure of the query. If the query is a noun phrase, parsing allows us to identify its head noun, as well as any adjectives and phrases modifying it.

A reasonable query relaxation strategy preserves the head noun and removes one or more of its modifiers. For example, the most important word in the best cat in the whole wide world is the head noun, cat.

But this strategy can break down. For example, if the query is free shipping, the adjective free is at least as important to the meaning as the head noun shipping. Syntax does not always dictate semantics. Still, emphasizing the head noun and the modifiers closest to it usually works in practice.

Semantic Analysis

The most sophisticated approach to query relaxation goes beyond token frequency and syntax and considers semantics — that is, what the tokens mean, particularly in relation to one another.

For example, we can relax the query polo shirt to polo, since shirt is implied. In contrast, relaxing dress shirt to dress completely changes the query’s meaning. Syntax isn’t helpful: in both cases, we’re replacing a noun phrase with a token that isn’t even the head noun. And shirt is no more specific than polo or dress. So we need to understand the semantics to relax this query successfully.

We can use the Word2vec model to embed words and phrases into a vector space that captures their semantics. This embedding allows us to recognize how much the query tokens overlap with one another in meaning, which in turn helps us estimate the consequence of ignoring a token. Word2vec also allows us to compute the similarity between a token and a phrase containing that token. If they’re similar, then it’s probably safe to relax the query by replacing the phrase with the token.

Relax but be Careful

Like query expansion, query relaxation aims to increase recall while minimizing the loss of precision. If we already have a reasonable quantity and quality of results, then query relaxation probably isn’t worth the risk.

Query relaxation is most useful for queries that return few or no results, since those are the queries for which we have the least to lose and the most to gain. But remember that more results doesn’t necessarily mean better results. Use query relaxation, but use it with care.

### Query segmentation

The previous two posts focused on using query rewriting to increase recall. We can also use query rewriting to increase precision — that is, to reduce the number of irrelevant results. While increasing recall helps us avoid small or empty result sets, increasing precision helps us avoid large result sets that are full of noise.

In this post, we’ll discuss the first of two query rewriting strategies used to increase precision: query segmentation. We’ll first talk about how to perform query segmentation, and then about how to use query segmentation to increase precision through query rewriting.

Semantic Units

Query segmentation attempts to divide the search query into a sequence of semantic units, each of which consists of one or more tokens. For a single-token query like machine, there’s only one possible segmentation. Multiple-token queries, like machine learning framework, admit multiple possible segmentations, such as “machine learning” framework and machine “learning framework”. In theory, the number of possible segmentations for a query grows exponentially with the number of tokens; in practice, there are only a handful of plausible segmentations.

The goal of query segmentation is to identify the query’s semantic units. In our machine learning framework example it’s clear that the correct segmentation is “machine learning” framework — that is, the searcher is interested in frameworks for machine learning, rather than something to do with machines and learning frameworks. Identifying this segmentation is critical to understanding the query and thus returning precise matches.

So how do we automatically identify the correct segmentation?

Dictionary Approach

The simplest approach is to obtain a dictionary of phrases appropriate to the application domain, and then to automatically treat the phrases as segments when they occur in queries. For example, a search engine for computer science articles could use a dictionary of common terms from that domain. Hopefully such a dictionary would include the term machine learning.

A dictionary-based approach requires a mechanism to resolve overlapping phrases, e.g., to segment the query machine learning framework if the dictionary includes both machine learning and learning framework. To keep things simple, we can associate each phrase with a score — such as its observed frequency in a subject-specific corpus — and favor the phrase with the highest score when there’s overlap.

Such a strategy tends to work reasonably well, but it can’t take advantage of query context. We’ll return to this point in a moment.

Statistical Approach

If we can’t buy or borrow a dictionary, we can create one from a document corpus. We analyze the corpus to find collocations — that is, sequences of words that co-occur more often than would be expected by chance.

We can make this determination using statistical measures like mutual information, t-test scores, or log-likelihood. It’s also possible to obtain collocations using Word2vec.

Once we have a list of collocations and associated scores, we can create a dictionary by keeping the collocations with scores above a threshold. Choosing the threshold is a trade-off between precision and coverage. We can also review the list manually to remove false positives.

Supervised Machine Learning

A more sophisticated approach to query segmentation is to model it as a supervised machine learning problem.

Query segmentation is a binary classification problem at the token level. For each token, we need to decide whether it continues the current segment or begins a new one. Given a collection of correctly segmented queries, we can train a binary classifier to perform query segmentation.

As with all machine learning, our success depends on how we represent the examples as feature vectors. Features could include token frequencies, mutual information for bigrams, part-of-speech tags, etc. Bear in mind that some features are more expensive to compute than others. Slow computation may be an acceptable expense or inconvenience for training, but it can be a show-stopper if it noticeably slows down query processing.

A supervised machine learning approach is more robust than one based on a dictionary, since it can represent context in the feature vector. But it’s a more complex and expensive to implement. It’s probably better to start with a simple approach and only invest further if the results aren’t good enough.

Using Query Segmentation for Query Rewriting

Now that we’ve obtained a query segmentation, what do we do with it? We rewrite the query to improve precision!

A straightforward approach is to auto-phrase segments — that is, to treat them as if they were quoted phrases. Returning to our example, we’d rewrite machine learning framework as “machine learning” framework, filtering out results that contain the tokens machine and learning but not as a phrase. The approach certainly increases precision, but it can be so drastic as to lead to no results. For this reason, many search engines boost phrase matches but don’t require them.

A more nuanced approach is to couple auto-phrasing with query expansion approaches like stemming, lemmatization, and synonym expansion. Continuing with our example, machine learning framework could include results that match “machine learned” framework or “machine learning” infrastructure. This approach can achieve strong precision and recall.

Finally, if we’re using query relaxation, it’s important for relaxation to respect query segmentation by not breaking up segments.

### Query scoping

In the previous post, we discussed how query segmentation improves precision by grouping query words into semantic units. In this post, we’ll discuss query scoping, another query rewriting technique that improves precision by matching each query segment to the right attribute.

Leveraging Structure

Documents or products often have an explicit structure that mirrors how people search for them.

For example, if you search for black michael kors dress on a site that sells clothing, you intend black as a color, Michael Kors as a brand, and dress as a category.

Similarly, if you search for microsoft ceo on LinkedIn, you’re looking for Microsoft as an employer and CEO as a job title.

Query scoping rewrites queries to take advantage of this structure.

Query Tagging

Query scoping begins with query tagging. Query tagging maps each query segment to the intended corpus attribute. It’s a machine learning problem — specifically, a non-binary classification problem. In the information retrieval literature, query tagging is a special case of named-entity recognition (NER), which can be applied to arbitrary text documents.

In our clothing example, the attributes include category, brand, and color. LinkedIn’s attributes include person name, company name, and job title. In general, attributes are specific to the domain and corpus.

Ideally, document structure is explicitly represented in the search index. If not, we may be able to extract document structure through algorithms for document understanding, also known as information extraction.

Regardless of how we obtain document structure, we need to establish the corpus attributes in advance. Query tagging is a supervised machine learning problem, which means that we can’t map query segments to attributes we didn’t anticipate. At best, we can label those segments as “unknown” because the classifier isn’t able to pick an attribute for them.

Training Data

We generate training data for query tagging from a representative set of segmented, labeled queries. For example, our training data could include examples like (black, michael kors, dress) -> (Color, Brand, Category).

Obtaining a representative set of queries from search logs is straightforward for a search application that’s already in production. To train a query tagger for a new search application, we’ll need to be more creative, either synthesizing a set of queries or borrowing a search log from a similar application. It may be easier to launch a search application without tagging, and then train a tagger after we’ve collected a critical mass of search traffic.

Let’s assume that we already have a query segmentation algorithm, as discussed in the previous post, and that we’ve applied it to our set of queries. Query tagging assumes — and thus requires — query segmentation.

We then need to label the query segments. We can use human labelers, but this process is slow and expensive. Human labelers also need to be familiar with the domain in order to provide robust judgements.

A more efficient approach is to obtain labels from our search logs — specifically from clicks on search results. For each click, we look at the query-result pair and label each query segment with the attribute it matches in the results. For example, if we look at a click for the search black michael kors dress, we’ll probably find that black matches the color attribute, Michael Kors matches the brand attribute, and dress matches the category attribute.

This approach works when each segment matches precisely one attribute in the clicked result. If a segment matches multiple attributes, we can exclude the click from our training data, or we can randomly select from the attributes. If a segment doesn’t match an attribute, we label it as “unknown”.

Finally, we have to represent the training examples as feature vectors. We can create features from tokens, stemmed or lemmatized tokens, or we can look below the token levels at the characters. If we have additional context about the searcher, we can also incorporate it into the feature vector.

Building the Model

Once we have training data, we can use various approaches to build a machine-learned model. A common choice for query tagging is a conditional random field (CRF). The Stanford NLP package provides a CRF named-entity recognizer that is suitable for query tagging.

It may be tempting to use a cutting-edge approach like deep learning. But bear in mind that more data — and better data — usually beats better algorithms. Robust query tagging relies mostly on robust generation of training data.

From Tagging to Scoping

Having tagged each query segment, we rewrite the query to scope the query.

A straightforward approach is to only match each segment against its associated attribute, as if the searcher had done so explicitly using an advanced search interface. This approach tends to optimize for precision, but it can be a bit unforgiving when it comes to recall.

A problem with this literal approach is that some attributes are difficult to distinguish from one another. For example, it’s often difficult to determine whether a query segment in a LinkedIn search matches a skill or part of a job title (e.g., a Developer who knows Java vs. Java Developer). Rather than count our ability to make this distinction, we can ignore it, effectively combining skill and job title into a single attribute.

If we have a similarity matrix for our attributes, or we implement the query tagger as a multiclass classifier, then we can can use the query tagger to restrict matches to the best attributes for each segment. Determining the number of attributes to match is a precision / recall tradeoff.

Finally, since query scoping is an aggressive way to increase precision, we can combine it with techniques like query expansion and query relaxation to increase recall. By requiring query expansion and relaxation to respect query scoping, we can achieve a good balance of precision and recall.

Summary

Query scoping is a powerful technique to increase precision by leveraging the explicit structure of the corpus and the implicit structure of queries. It requires more work than most of the other techniques we’ve discussed, but it’s worth it: it’s one of the most effective ways to capture query intent.

### QW papers

Translating queries into snippets for improved query expansion

Statistical Machine Translation for Query Expansion in Answer Retrieval

Context- and Content-aware Embeddings for Query Rewriting in Sponsored Search

QUEEN: Neural Query Rewriting in E-commerce

Learning to rewrite queries

Query expansion using local and global document analysis

Few-Shot Generative Conversational Query Rewriting

Question Rewriting for Conversational Question Answering

## Benchmark datasets

### MS MARCO

[MS MARCO](https://microsoft.github.io/msmarco/Datasets) (Microsoft MAchine Reading Comprehension) {cite}`nguyen2016ms` is a large scale dataset widely used to train and evaluate models for the document retrieval and ranking tasks as well as tasks like key phrase extraction for question
answering. MS MARCO dataset is sampled from Bing search engine user logs, with Bing retrieved passages given queries and human annotated relevance labels. There are more than 530,000 questions in the "train" data partition, and the evaluation is usually performed on around 6,800 questions in the "dev" and "eval" data partition. The ground-truth labels for the "eval" partition are not published. The original data set contains more than $8.8$ million passages.

There are two tasks: Passage ranking and document ranking; and two subtasks in each case: full ranking and re-ranking.

Each task uses a large human-generated set of training labels. The two tasks have different sets of test queries. Both tasks use similar form of training data with usually one positive training document/passage per training query. In the case of passage ranking, there is a direct human label that says the passage can be used to answer the query, whereas for training the document ranking task we transfer the same passage-level labels to document-level labels. Participants can also use external corpora for large scale language model pretraining, or adapt algorithms built for one task (e.g. passage ranking) to the other task (e.g. document ranking). This allows participants to study a variety of transfer learning strategies.

#### Document Ranking Task
The first task focuses on document ranking. We have two subtasks related to this: Full ranking and top-100 re-ranking.

In the full ranking (retrieval) subtask, you are expected to rank documents based on their relevance to the query, where documents can be retrieved from the full document collection provided. You can submit up to 100 documents for this task. It models a scenario where you are building an end-to-end retrieval system.

In the re-ranking subtask, we provide you with an initial ranking of 100 documents from a simple IR system, and you are expected to re-rank the documents in terms of their relevance to the question. This is a very common real-world scenario, since many end-to-end systems are implemented as retrieval followed by top-k re-ranking. The re-ranking subtask allows participants to focus on re-ranking only, without needing to implement an end-to-end system. It also makes those re-ranking runs more comparable, because they all start from the same set of 100 candidates.

\begin{table}
	\footnotesize
	\centering
	\begin{tabular}{p{0.3\textwidth}p{0.3\textwidth}p{0.3\textwidth}}
		\hline
		elegxo meaning & what does physical medicine do & feeding rice cereal how many times per day\\
		\hline
		most dependable affordable cars & lithophile definition & what is a flail chest \\
		\hline
		put yourself on child support in texas & what happens in a wrist sprain & what are rhetorical topics \\
		\hline
		what is considered early fall & what causes elevated nitrate levels in aquariums & lyme disease symptoms mood \\
		\hline
		what forms the epineurium & an alpha helix is an example of which protein structure? & aggregate demand curve \\
		\hline
		what county is ackley iowa in & what is adiabatic? & what is a nrr noise reduction rating mean \\
		\hline
		fibroid symptoms after menopause & what are the requirements for neurologist & watts \& browning engineers \\

		\hline
	\end{tabular}
	\caption{Example queries in MS MARCO dataset.}\label{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:tab:example_queries_MSMARCO}
\end{table}

Documents
\begin{table}[H]
	\notsotiny
	\centering
	\begin{tabular}{p{0.15\textwidth}p{0.80\textwidth}}
		\hline
		Query & Document\\
		\hline
		{\scriptsize what is a dogo dog breed} & D3233725       \url{http://www.akc.org/dog-breeds/dogo-argentino/care/}      Dogo Argentino  Dogo Argentino Miscellaneous The Dogo Argentino is a pack-hunting dog, bred for the pursuit of big-game such as wild boar and puma, and possesses the strength, intelligence and quick responsiveness of a serious athlete. His short, plain and smooth coat is completely white, but a dark patch near the eye is permitted as long as it doesn't cover more than 10\% of the head. Dog Breeds Dogo Argentinocompare this breed with any other breed/s of your choosing Personality: Loyal, trustworthy, and, above all else, courageous Energy Level: Somewhat Active; Dogos require vigorous exercise to stay at their physical and temperamental best Good with Children: Better with Supervision Good with other Dogs: With Supervision Shedding: Infrequent Grooming: Occasional Trainability: Responds Well Height: 24-27 inches (male), 23.5-26 inches (female)Weight: 80-100 pounds Life Expectancy: 9-15 years Barking Level: Barks When Necessarymeet the Dogo Argentino Did you know? The Dogo Argentino is also known as the Argentinian Mastiff. How similar are you? Find out!Breed History1995The Dogo Argentino was first recorded in the Foundation Stock Service. Breed Standard Learn more about the Dogo Argentino breed standard. Click Here Colors \& Markings10Learn Moreat a Glance Energy \& Sizemedium ENERGY \& large size NATIONAL BREED CLUBThe Dogo Argentino Club Of America Find Dogo Argentino Puppiesthe Breed Standard Embed the breed standard on your site. Download the complete breed standard or club flier PDFs.01020304050607General Appearance Molossian normal type, mesomorphic and macrothalic, within the desirable proportions without gigantic dimensions. Its aspect is harmonic and vigorous due to its powerful muscles which stand out under the consistent and elastic skin adhered to the body through a not very lax subcutaneous tissue. It walks quietly but firmly, showing its intelligence and quick responsiveness and revealing by means of its movement its permanent happy natural disposition. Of a kind and loving nature, of a striking whiteness, its physical virtues turn it into a real athlete. Dogo Argentino Care Nutrition \& Feeding Good nutrition for Dogo Argentinos is very important from puppyhood to their senior years. Read More Coat \& Grooming The breed's coat needs only weekly brushing. Read More Exercise
		The Dogo Argentina is not a good choice for the novice owner. Read More Health Dogo Argentinos are generally a healthy breed. Read M
		oreget involved in Clubs \& Events National Breed Clubthe Dogo Argentino Club of America Local Clubs Find the Local Clubs in your area
		. Meetups Explore meetups.com and see all of the local Dogo Argentinos in your area.breeds Similar to the Dogo Argentino Cane Corso P
		lott Rhodesian Ridgeback Great Dane Bullmastiffexplore Other Breeds By Dog Breed By Dog Group \\
		{\scriptsize NA}  & D3048094 \url{https://answers.yahoo.com/question/index?qid=20080718121858AAmfk0V}      I have trouble swallowing due to MS, can I crush valium \& other meds to be easier to swallowll? Health Other - Health I have trouble swallowing due to MS, can I crush valium \& other meds to be easier to swallowll? Follow 5 answers Answers Relevance Rating Newest Oldest Best Answer: If you have a problem swallowing, try crushing Valium (or other tablets) between two spoons, and taking them in a teaspoon of your favorite Jelly (raspberry???). 	The jelly helps the crushed meds slide down &nbsp;Anonymous · 10 years ago0 2 Comment Asker's rating Ask your pharmacist if any or all of your meds can be made into syrup form if you have trouble swallowing. Many forms of medication are designed to be swallowed whole and not interferred with. Do not take advice from those people on here who are only guessing at a correct answer. Seek the advice of professionals. Lady spanner · 10 years ago0 0 Comment I'm pretty sure its not a good idea to crush pills. You should definitely ask your doctor before doing anything like that, it might be dangerous.little Wing · 10 years ago0 0 Comment Please ask your doctor! This is	not a question for random people to answer. Medication is not something to mess around with. Look at Heath Ledger. He will be missed by everyone, especially his daughter. Don't make the same mistake.pink · 10 years ago0 1 Comment Your doctor or any pharmacist should be able to tell you. Could vary for each medication. Bosco · 10 years ago0 0 Comment Maybe you would like to learn more about one of these? Glucose Monitoring Devices Considering an online college? Need migraine treatment? VPN options for your computer \\
		{\scriptsize NA} &  D2342771     \url{http://www.marketwatch.com/story/the-4-best-strategies-for-dealing-with-customer-service-2014-08-14}     The 4 best st
		rategies for dealing with customer service Shutterstock.com / wavebreakmedia If you want your cable company, airline or pretty much any other company to resolve your complaints quickly and completely, you may need to change the way you deal with customer service. According to the latest data from the American Customer Service Index, Americans are increasingly dissatisfied with the companies they deal with. In the first quarter of 2014, overall customer satisfaction scores across all industries fell to 76.2 out of 100, which the researchers who compile the ratings say was “one of the largest [drops] in the 20-year history of the Index.” And some industries are particularly hated: Internet (63 out of 100) and cable and subscription TV (65) companies and airlines (69) rank at the bottom when it comes to customer satisfaction. Part of the dissatisfaction may be because we are interacting with customer service the wrong way. A report released Wednesday by customer-service software firm Zendesk, which looked at customer-service interactions from more than 25,000 companies across 140 countries, as well as insight from customer-service experts, found that some consumers are acting in ways that aren’t yielding them good results. Here are four strategies to employ when dealing with customer service. Don’t be too stern Being “overbearing” or “overly stern” is “a common strategy for some customers seeking better service,” the Zendesk survey revealed. “However, the data indicates that customers who are polite tend to submit higher customer satisfaction scores than those who aren’t.”	Indeed, customers who ask the customer service rep to “please” help them with their request, and who said “thank you” for help they received throughout the call reported that they got better customer service than those who did not use these words, the survey showed. Of course, it could be that people who use these words tend to be more satisfied anyway. But there’s a strong chance that saying these platitudes ingratiates the customer service rep to you -- making them more willing to help you quickly and kindly with your request, experts say. “If you can find any excuse at all to praise them and build them up -- ‘you’re very good at what you do,’ ‘I really appreciate the extra effort,’ — people will almost always bend over backwards to maintain that good opinion and to show you just how good they really are,” says Barry Maher, the author of “Filling the Glass,” which focuses in part on getting better customer service and on personal empowerment. “Treat people well who aren’t used to being treated well…and you may well be astonished by the results you get.” Chip Bell, the founder of business consulting firm Chip Bell Group and author of “The 9 1/2 Principles of Innovative Service” adds that “you don’t have to be from the South to show respect with a liberal amount of ‘sir’ and ‘ma’am — it will open doors otherwise closed.”At the very least, “don’t go in guns a’ blazing,” says Shep Hyken, author of “Amaze Every Customer Every Time.” “The moment you lose your cool is the moment you lose control.” And, if that kindness does not work — rather than starting to get nasty — simply ask to speak to a higher authority, says Maher. Take notes Take a lesson from Rich Davis, the Comcast customer who recorded his call with the company and saw it go viral: It often pays to have good records of the antics a company is up to if you want to get your way down the road. “Have you ever heard a recording when you call for assistance that stated, ‘be advised this call may be recorded for quality assurance,’?” says motivational coach and speaker Chantay Bridges. “Flip the script, let the rep know: you are documenting them and keeping a record, that their service is also being noted.” While few customers keep detailed notes or recordings of their interactions, experts say, this will help them should there be a problem down the road, as they’ll have proof of what went on. Don’t keep your problems to yourself Most customers know that they should contact customer service via multiple channels like phone, email and social media to get their complaint resolved ASAP. But few take the logical next step: contacting competitors via these channels too to explain how their current company is not serving their needs. “If you receive a response from a competitor, watch or ask for a counter from the business you originally were contacting,” says John Huehn, CEO of In the Chat, which provides social media and text
		messaging platforms to enhance customer service. This could provide you with the leverage you need to get your issue resolved. It’s OK to keep it brief When it comes to email complaints, longer isn’t better, so don’t waste your breath (you might, after all, need that energy for dealing with the company later on). While experts often advise consumers to give customer service reps as many details as possible so they can effectively fix the customer’s complaint, customers who wrote long emails didn’t report getting any better customer service than those who wrote short ones, the Zendesk survey revealed. Indeed, the customer’s satisfaction levels at the end of
		the exchange were roughly the same whether they wrote 50 words or 200 words. Jason Maynard, product manager and data science lead at Zendesk, says that this doesn’t mean you should not include detail (indeed, that helps) but that you should make this detail as succinct as possible; if you have a complicated problem or request, you may want to consider calling, as most of us aren’t trained as technical writers. \\
		\hline
	\end{tabular}
	\caption{Example queries (may not exist) and relevant documents for document ranking tasks in MS MARCO dataset.}\label{ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:tab:example_documents_MSMARCO}
\end{table}

#### Passage Ranking Task

Similar to the document ranking task, the passage ranking task also has a full ranking and reranking subtasks.

In context of full ranking (retrieval) subtask, given a question, you are expected to rank passages from the full collection in terms of their likelihood of containing an answer to the question. You can submit up to 1,000 passages for this end-to-end retrieval task.

In context of top-1000 reranking subtask, we provide you with an initial ranking of 1000 passages and you are expected to rerank these passages based on their likelihood of containing an answer to the question. In this subtask, we can compare different reranking methods based on the same initial set of 1000 candidates, with the same rationale as described for the document reranking subtask.

One caveat of the MSMARCO collection is that only contains binary annotations for fewer than two positive examples per query, and no explicit annotations for non-relevant passages. During the reranking task, negative examples are generated from the top candidates of a traditional retrieval system. This approach works reasonably well, but accidentally picking relevant passages as negative examples is possible.

\begin{table}
\scriptsize
\centering
\begin{tabular}{p{0.15\textwidth}p{0.4\textwidth}p{0.4\textwidth}}
	\hline
	Query & Relevant Passage & Irrelevant Passage\\
	\hline
	foods that will help lower blood sugar & Lemons are rich in Vitamin C and their acidity helps to lower other foods' glycemic indexes. Oat and rice bran crackers make healthy snacks. Complement with organic nut butter or cheese. Other foods that stabilize blood sugar are cheese, egg yolks, berries and brewer's yeast.
	& Low hemoglobin, high blood pressure, high levels of bad cholesterol and abnormal blood sugar levels are a few factors that influence blood health. Your diet can go a long way in promoting healthy blood, and most foods that are good for the blood also promote healthy
	weight and general well being.n fact, foods that contain monounsaturated and polyunsaturated fat actually lower your bad cholesterol levels while increasing good cholesterol. Foods that promote healthy blood cholesterol levels include plant oils -- except for palm and coconut oil -- as well as fish, nuts and avocados.
	\\
	cancer of the pancreas symptoms & Symptoms of Pancreatic Cancer. Pancreatic cancer may cause only vague unexplained symptoms. Pain (usually in the abdomen or back), weight loss, jaundice (yellowing of the skin and/or eyes) with or without itching, loss of appetite, nausea, change in stool, pancreatitis and recent-onset diabetes are symptoms that may indicate pancreatic cancer. &Pancreatic cancer develops as abnormal pancreatic cells multiply rapidly in the pancreas. These cells don't die, but continue growing to form tumors. As the stages of pancreatic cancer progress in dogs, tissue in the pancreas begins to die. In the later stages of pancreatic cancer, tumors can spread to other organs, causing tissue death and organ dysfunction throughout the body.\\
	is pizza considered fast food & Fast Food Pizza is Unhealthy. Fast food pizza is unhealthy because of its ingredients. Fast food pizza is made on a white crust that is filled with refined carbs. These refined or processed grains are stripped of most of the healthy nutrients in the name of taste. & I have already proven that the mode can be numerical in the sentences above. For an example of categorical data, say I surveyed some people about what their favorite food was and this was the data: Pizza, pizza, pizza, ice cream, ice cream, strawberries, strawberries, oranges, spaghetti. The mode would have been pizza.\\
	cost to install a sump pump & The average cost to install a sump pump ranges from $550 to $1,100 depending on the size of the pump. How do you build a sump pit? A sump pit is literally a hole just large enough to hold the pump with a plastic lining. & Protect machinery from water damage with an elevator sump pump with an oil sensor from Grainger. If water is allowed to collect in an elevator pit it can facilitate the growth of bacteria, mold and mildew. These pumps that are designed and approved for safe operation of pumping, alarming and monitoring of elevator sump pits, transformer vaults and other applications where oil and water must be detected.\\
	\hline
\end{tabular}
\caption{Example queries, relevant passages, and irrelevant passages for passage ranking tasks in MS MARCO dataset.}
\end{table}

### TERC

#### TREC-deep learning track

Deep Learning Track at the Text REtrieval Conferences (TRECs) deep learning track<sup>[^7]</sup>\cite{craswell2020overview} is another large scale dataset used to evaluate retrieval and ranking model through two tasks: Document retrieval and passage retrieval. 

Both tasks use a large human-generated set of training labels, from the MS-MARCO dataset. The document retrieval task has a corpus of 3.2 million documents with 367 thousand training queries, and there are a test set of 43 queries. The passage retrieval task has a corpus of 8.8 million passages with 503 thousand training queries, and there are a test set of 43 queries.

#### TREC-CAR

TREC-CAR (Complex Answer Retrieval) {cite}`dietz2017trec` is a dataset where the input query is the concatenation of a Wikipedia article title with the title of one of its sections. The ground-truth documents are the paragraphs within that section. The corpus consists of all English Wikipedia paragraphs except the abstracts. The released dataset has five predefined folds, and we use the first four as a training set (approx. 3M queries), and the remaining as a validation set (approx. 700k queries). The test set has approx. 2,250 queries.

\iffalse
#### TERC-COVID

TREC-COVID\cite{roberts2020trec} is a data set containing 50 topics (queries) and a corpus of around 190,000 abstracts. Topics from Round 14 and about $90 \%$ of their relevance judgments are used for training, and their remaining relevance judgments and all the relevance judgements for Round 5 topics are held out for evaluation. We use retrieval methods described in [2] to retrieve a subset of candidate abstracts for each topic. Further, we use 1 relevant and 5 random irrelevant abstracts to create ranking lists for training.

\fi

### Natural Question (NQ)
Natural Question (NQ) {cite}`kwiatkowski2019natural` introduces a large dataset for open-domain QA. The original dataset contains more than 300,000 questions collected from Google search logs. In {cite}`karpukhin2020dense`, around 62,000 factoid questions are selected, and all the Wikipedia articles are processed as the collection of passages. There are more than 21 million passages in the corpus. 

## Note on bibliography and software

### Bibliography
For excellent reviews in neural information retrieval, see {cite}`guo2020deep, mitra2018introduction, lin2021pretrained`

For traditional information retrieval, see {cite}`schutze2008introduction, buttcher2016information, robertson2009probabilistic, croft2010search`

### Software

[Faiss](https://github.com/facebookresearch/faiss/wiki/) is a recently developed computational library for efficient similarity search and clustering of dense vectors. 

\printbibliography
\end{refsection}

[^1]: Semantic matching means words (or phrases) have similar meaning despite they are different words

[^2]: This is also known as query agnostic problem {cite}`tang2021improving`.

[^3]: To avoid excessive memory usage, we truncate each document to 400 tokens and queries to 100 tokens.

[^4]: Let $L = \sum_j q_j\log p_j$, where $p_j = \exp(z_j)/\sum_j\exp(z_j)$. Use $\log q_i = z_i - \log (\sum_j \exp(z_j))$, we have
	```{math}
	\begin{align*}
	\frac{\partial L}{\partial z_i} &= \sum_j q_j (\delta_{ij} - p_j) \\
	& =(q_i - p_i)
	\end{align*}
	```

[^5]: \url{https://en.wikipedia.org/wiki/GNU_Aspell}

[^6]: https://en.wikipedia.org/wiki/Hunspell

[^7]: \url{https://microsoft.github.io/msmarco/TREC-Deep-Learning.html}

