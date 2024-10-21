
(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch)=
# Information Retrieval and Text Ranking

```{index} IR, Neural IR, Information retrieval
```



## Overview of Information Retrieval

### Ad-hoc Retrieval

Ad-hoc search and retrieval is a classic **information retrieval (IR)** task consisting of two steps: first, the user specifies his or her information need through a query; second, the information retrieval system fetches documents from a large corpus that are likely to be relevant to the query. Key elements in an ad-hoc retrieval system include
- **Query**, the textual description of information need.
- **Corpus**, a large collection of textual documents to be retrieved.
- **Relevance** is about whether a retrieved document can meet the user's information need.

There has been a long research and product development history on ad-hoc retrieval. Successful products in ad-hoc retrieval include Google search engine [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:adhocretrievaldemogoogle`] and Microsoft Bing [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:adhocretrievaldemobing`].  
One core component within Ad-hoc retrieval is text ranking. The returned documents from a retrieval system or a search engine are typically in the form of an ordered list of texts. These texts (web pages, academic papers, news, tweets, etc.) are ordered with respect to the relevance to the user's query, or the user's information need.

A major characteristic of ad-hoc retrieval is the heterogeneity of the query and the documents [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:querylengthDocLengthmsmarco`]. A user's query often comes with potentially unclear intent and is usually very short, ranging from a few words to a few sentences. On the other hand, documents are typically from a different set of authors with varying writing styles and have longer text length, ranging from multiple sentences to many paragraphs. Such heterogeneity poses significant challenges for vocabulary match and semantic match<sup>[^1]</sup> for ad-hoc retrieval tasks. 


```{figure} ../img/chapter_application_IR/ApplicationIRSearch/DataExploration/query_length_MS_MARCO.png
---
scale: 30%
name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:querylengthDocLengthmsmarco
---
Query length and document length distribution in Ad-hoc retrieval example using MS MARCO dataset.
```

There have been decades' research and engineering efforts on developing ad-hoc retrieval models and system. Traditional IR systems primarily rely on techniques to identify exact term matches between a query and a document and compute final relevance score between various weighting schemes. Such exact matching approach has achieved tremendous success due to scalability and computational efficiency - fetching a handful of relevant document from billions of candidate documents. Unfortunately, exact match often suffers from vocabulary mismatch problem where sentences with similar meaning but in different terms are considered not matched. Recent development of deep neural network approach {cite}`huang2013learning`, particularly Transformer based pre-trained large language models, has made great progress in semantic matching, or inexact match, by incorporating recent success in natural language understanding and generation. Recently, combining exact matching with semantic matching is empowering many IR and search products. 


```{figure} ../img/chapter_application_IR/ApplicationIRSearch/ad_hoc_retrieval_demo_google.png
---
scale: 60%
name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:adhocretrievaldemogoogle
---
Google search engine.
```


```{figure} ../img/chapter_application_IR/ApplicationIRSearch/ad_hoc_retrieval_demo_bing.png
---
scale: 70%
name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:adhocretrievaldemobing
---
Microsoft Bing search engine.
```

### Open-domain Question Answering

Another application closely related IR is **open-domain question answering (OpenQA)**, which has found a widespread adoption in products like search engine, intelligent assistant, and automatic customer service. OpenQA is a task to answer factoid questions that humans might ask, using a large collection of documents (e.g., Wikipedia, Web page, or collected document) as the information source. An  OpenQA example is like

**Q:** *What is the capital of China?*

**A:** *Beijing*.

Contrast to Ad-hoc retrieval, instead of simply returning a
list of relevant documents, the goal of OpenQA is to identify (or extract) a span of text that directly answers the user’s question. Specifically, for *factoid* question answering, the OpenQA system primarily focuses on questions that can be answered with short phrases or named entities such as dates, locations, organizations, etc.

A typical modern OpenQA system adopts a two-stage pipeline {cite}`chen2017reading` [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:open-domainqa`]: (1) A document **retriever** selects a small set of relevant passages that probably contain the answer from a large-scale collection; (2) A document **reader** extracts the answer from relevant documents returned by the document retriever. Similar to ad-hoc search, relevant documents are required to be not only topically related to but also correctly address the question, which requires more semantics understanding beyond exact term matching features. One widely adopted strategy to improve OpenQA system with large corpus is to use an efficient document (or paragraph) retrieval technique to obtain a few relevant documents, and then use an accurate (yet expensive) reader model to read the retrieved documents and find the answer.

Nowadays many web search engines like Google and Bing have been evolving towards higher intelligence by incorporating OpenQA techniques into their search functionalities.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/open-domain_QA.png
:scale: 40%
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

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/traditional_IR_engine.png
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:traditionalirengine
Key steps in a traditional IR system.
```

The rapid progress of deep neural network learning {cite}`goodfellow2016deep` and their profound impact on natural language processing has also reshaped IR systems and brought IR into a deep learning age. Deep neural networks (e.g., Transformers {cite}`devlin2018bert`) have proved their unparalleled capability in semantic understanding over traditional IR margin yet they suffer from high computational cost and latency. This motivates the development of multi-stage retrieval and ranking IR system [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:retrieverankingarch`] in order to better balance trade-offs between effectiveness (i.e., quality and accuracy of final results) and efficiency (i.e., computational cost and latency). 

In this multi-stage pipeline, early stage models consists of simpler but more efficient models to reduce the candidate documents from billions to thousands; later stage models consists of complex models to perform accurate ranking for a handful of documents coming from early stages.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/retrieve_ranking_arch.png
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:retrieverankingarch
The multi-stage architecture of modern information retrieval systems.
```

In modern search engine, traditional IR models, which is based on term matching, serve as good candidates for early stage model due to their efficiency. The core idea of the traditional approach is to count repetitions of query terms in the document. Large counts indicates higher relevance. Different transformation and weighting schemes for those counts lead to a variety of possible TF-IDF ranking features. 

Later stage models are primarily deep learning model. Deep learning models in IR not only provide powerful representations of textual data that capture word and document semantics, allowing a machine to better under queries and documents, but also open doors to multi-modal (e.g., image, video) and multilingual search, ultimately paving the way for developing intelligent search engines that deliver rich contents to users. 

````{prf:remark} Why we need a semantic understanding model 
For web-scale search engines like Google or Bing, typically a very small set of popular pages that can answer a good proportion of queries.{cite:p}`mitra2016dual` The vast majority of queries contain common terms. It is possible to use term matching between key words in URL or title and query terms for text ranking; It is also possible to simply memorize the user clicks between common queries between their ideal URLs. For example, a query *CNN* is always matched to the CNN landing page. These simple methods clearly do not require a semantic understanding on the query and the document content. 

However, for new or tail queries as well as new and tail document, a semantic understanding on queries and documents is crucial. For these cases, there is a lack of click evidence between the queries and the documents, and therefore a model that capture the semantic-level relationship between the query and the document content is necessary for text ranking.
````

### Challenges And Opportunities In IR Systems

#### Query Understanding And Rewriting

A user's query does not always have crystal clear description on the information need of the user. Rather, it often comes with potentially misspellings and unclear intent,  and is usually very short, ranging from a few words to a few sentences [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:tab:example_queries_MSMARCO`]. There are several challenges to understand the user's query.

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

**Query expansion**
Query expansion improves search result retrieval by adding or substituting terms to the user’s query. Essentially, these additional terms aim to minimize the mismatch between the searcher’s query and available documents. For example, the query *italian restaurant*, we can expand *restaurant* to *food* or *cuisine* to search all potential candidates.

**Query relaxation**
The reverse of query expansion is query relaxation, which expand the search scope when the user's query is too restrictive. For example, a search for *good Italian restaurant* can be relaxed to *italian restaurant*.

**Query intent understanding**
This subcomponent aims to figure out the main intent behind the query, e.g., the query *coffee shop* most likely has a local intent (an interest in nearby places) and the query *earthquake* may have a news intent. Later on, this intent will help in selecting and ranking the best documents for the query. 

Given a rewritten query, It is also important to correctly weigh specific terms in a query such that we can narrow down the search scope. Consider the query *NBA news*, a relevant document is expected to be about *NBA* and **news** but have more focus on *NBA*. There are traditional rule-based approach to determine the term importance as well as recent data-driven approach that determines the term importance based on sophisticated natural language and context understanding.

To improve relevance ranking, it is often necessary to incorporate additional context information (e.g., time, location, etc.) into the user's query. For example, when a user types in a query *coffee shop*, retrieve coffee shops by ascending distance to the user's location can generally improve relevance ranking. Still, there are challenges on deciding for which type of query we need to incorporate the context information.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/DataExploration/query_word_cloud.png
:name: fig:querywordcloud
Word cloud visualization for common query words using MS MARCO data. 
```

#### Exact Match And Semantic Match

Traditional IR systems retrieve documents mainly by matching keywords in documents with those in search queries. While in many cases exact term match naturally ensure semantic match, there are cases, exact term matching can be insufficient. 

The first reason is due to the polysemy of words. That is, a word can mean different things depending on context. The meaning of *book* is different in *text book* and *book a hotel room*. Short queries particularly suffer from Polysemy because they are often devoid of context. 

The second reason is due to the fact that a concept is often expressed using different vocabularies and language styles in documents and queries. As a result, such a model would have difficulty in retrieving documents that have none of the query terms but turn out to be relevant.

Modern neural-based IR model enable semantic retrieval by learning latent representations of text from data and enable document retrieval based on semantic similarity. 


```{table} Retrieval results based on exact matching methods and semantic matching methods.

| **Query** | "Weather Related Fatalities" |
|-----------|------------------------------|
| **Information Need** | A relevant document will report a type of weather event which has directly caused at least one fatality in some location. |
| **Lexical Document** | ".. Oklahoma and South Carolina each recorded three fatalities. There were two each in Arizona, Kentucky, Missouri, Utah and Virginia. Recording a single lightning death for the year were Washington, D.C.; Kansas, Montana, North Dakota, .." |
| Semantic Document | .. Closed roads and icy highways took their toll as at least one motorist was killed in a 17-vehicle pileup in Idaho, a tour bus crashed on an icy stretch of Sierra Nevada interstate and 100-car string of accidents occurred near Seattle ... |

```

An IR system solely rely on semantic retrieval is vulnerable to queries that have rare words. This is because rare words are infrequent or even never appear in the training data and learned representation associated with rare words query might be poor due to the nature of data-driven learning. On the other hand, exact matching approach are robust to rare words and can precisely retrieve documents containing rare terms.

Another drawback of semantic retrieval is high false positives: retrieving documents that are only loosely related to the query.

Nowadays, much efforts have been directed to achieve a strong and intelligent modern IR system by combining exact match and semantic match approaches in different ways. Examples include joint optimization of hybrid exact match and semantic match systems, enhancing exact match via semantic based query and document expansion, etc. 

#### Robustness To Document Variations

In response to users' queries and questions, IR systems needs to search a large collection of text documents, typically at the billion-level scale, to retrieve relevant ones. These documents are comprised of mostly unstructured natural language text, as compared to structured data like tables or forms. 

Documents can vary in length, ranging **from sentences** (e.g., searching for similar questions in a community question answering application like Quora) **to lengthy web pages**. A long document might like a short document, covering a similar scope but with more words, which is also known as the **Verbosity hypothesis**. On the other hand, a long document might consist of a number of unrelated short documents concatenated together, which is known as **Scope hypothesis**. The wide variation of document forms lead to different strategies. For example, following the Verbosity hypothesis a long document is represented by a single feature vector. Following the Scope hypothesis, one can break a long document into several semantically distinctive parts and represent each of them as separate feature vectors. We can consider each part as the unit of retrieval or rank the long document by aggregating evidence across its constituent parts.   
For full-text scientific articles, we might choose to only consider article titles and abstracts, and ignoring most of the numerical results and analysis. 

There are also challenges on breaking a long document into semantically distinctive parts and encode each part into meaningful representation. Recent neural network methods extract semantic parts by clustering tokens in the hidden space and represent documents by multi-vector representations{cite:p}`humeau2019poly, tang2021improving, luan2021sparse`.  

#### Computational Efficiency

IR product such as search engines often serve a huge pool of user and need to handle tremendous volume of search requests during peak time (e.g., when there is breaking news events). To provide the best user experience, computational efficiency of IR models often directly affect user perceived latency. A long standing challenge is to achieve high accuracy/relevance in fetched documents yet to maintain a low latency. While traditional IR methods based on exact term match has excellent computational efficiency and scalability, it suffers from low accuracy due to the vocabulary and semantic mismatch problems. Recent progress in deep learning and natural language process are highlighted by complex transformer-based model {cite}`devlin2018bert` that achieved accuracy gain over traditional IR by a large margin yet experienced high latency issues. There are numerous ongoing studies (e.g., {cite}`mitra2017learning, mitra2019updated, gao2021coil`) aiming to bring the benefits from the two sides via hybrid modeling methodology.  

To alleviate the computational bottleneck from deep learning based dense retrieval, state-of-the-art search engines also adopts a multi-stage retrieval pipeline system: an efficient first-stage retriever uses a query to fetch a set of documents from the entire document collection, and subsequently one or more more powerful retriever to refine the results. 

## Text Ranking Evaluation Metrics

Consider a large corpus containing $N$ documents $D=\{d_1,...,d_N\}$. Given a query $q$, suppose the retriever and its subsequent re-ranker (if there is) ultimately produce an ordered list of $k$ relevant document $R_q = \{d_{i_1},...,d_{i_k}\}$, where documents $d_{i_1},...,d_{i_k} $ are ranked by some relevance measure with respect to the query. 

In the following, we discuss several commonly used metrics to evaluate text ranking quality and IR system. 

### Precision And Recall

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

### Normalized Discounted Cumulative Gain (NDCG)

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

````{prf:example}

Consider 5 candidate documents with respect to a query. Let their ground truth relevance scores be

 $$s_1=10, s_2=0,s_3=0,s_4=1,s_5=5,$$

which corresponds to a perfect rank of $$s_1, s_5, s_4, s_2, s_3.$$
Let the predicted scores be $$y_1=0.05, y_2=1.1, y_3=1, y_4=0.5, y_5=0.0,$$
which corresponds to rank $$s_2, s_3, s_4, s_1, s_5.$$

For $k=1,2$, we have

$$DCG@k = 0, NDCG@k=0.$$

For $k=3$, we have

$$DCG@k = \frac{s_4}{\log_2 (3+1)} = 0.5, IDCG@k = 10.0 + \frac{5.0}{\log_2 3} + \frac{1.0}{\log_2 4} = 13.65, NDCG@k=0.0366.$$

For $k=4$, we have

$$DCG@k = \frac{s_4}{\log_2 4} + \frac{s_1}{\log_2 5} = 4.807, IDCG@k = IDCG@3 + 0.0 = 13.65, NDCG@k=0.352.$$ 

````

### Online Metrics

When a text ranking model is deployed to serve user's request, we can also measure the model performance by tracking several online metrics.

**Click-through rate and dwell time** When a user types a query and starts a search session, we can measure the success of a search session on user's reactions. On a per-query level, we can define success via click-through rate.
The Click-through rate (CTR) measures the ratio of clicks to impressions.

$$\operatorname{CTR} = \frac{\text{Number of clicks}}{\text{Number of impressions}},$$

where an impression means a page displayed on the search result page a search engine result page and a click means that the user clicks the page.

One problem with the click-through rate is we cannot simply treat a click as the success of document retrieval and ranking. For example, a click might be immediately followed by a click back as the user quickly realizes the clicked doc is not what he is looking for. We can alleviate this issue by removing clicks that have a short dwell time.

**Time to success**: Click-through rate only considers the search session of a single query. In real application case, a user's search experience might span multiple query sessions until he finds what he needs. For example, the users initially search *action movies* and they do not find that the ideal results and refine the initial query to a more specific one: *action movies by Jackie Chan*. Ideally, we can measure the time spent by the user in identifying the page he wants as a metrics.

## Traditional Sparse IR Fundamentals

### Exact Match Framework

Most traditional approaches to ad-hoc retrieval simply count repetitions of the query terms in the document text and assign proper weights to matched terms to calculate a final matching score. This framework, also known as exact term matching, despite its simplicity, serves as a foundation for many IR systems. A variety of traditional IR methods fall into this framework and they mostly differ in different weighting (e.g., tf-idf) and term normalization (e.g., dogs to dog) schemes.

In the exact term matching, we represent a query and a document by a set of their constituent terms, that is, $q = \{t^{q}_1,...,t^q_M\}$ and $d = \{t^{d}_1,...,t^d_M\}$. The matching score between $q$ and $d$ with respect to a vocabulary $V$ is given by:

$$
S(q, d)= \sum_{t\in V} f(t)\cdot\mathbb{1}(t\in q\cap d) = \sum_{t \in q \cap d} f(t)
$$

where $f$ is some function of a term and its associated statistics, the three most important of which are 
- Term frequency (how many times a term occurs in a document);
- Document frequency (the number of documents that contain at least once instance of the term);
- Document length (the length of the document that the term occurs in).

Exact term match framework estimates document relevance based on the count of only the query terms in the document. The position of these occurrences and relationship with other terms in the document are ignored.

BM25 are based on exact matching of query and document words, which
limits the in- formation available to the ranking model and may lead to problems such
vocabulary mismatch

### TF-IDF Vector Space Model

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

One of the most widely adopted exact matching method is called BM25 (short for Okapi BM25){cite:p}`yang2018anserini, robertson2009probabilistic, croft2010search`. BM25 combines overlapping terms, term-frequency (TF), inverse document frequency (IDF), and document length into following formula

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

## Semantic Dense Retrieval Models

### Motivation

For ad-hoc search, traditional exact-term matching models (e.g., BM25) are playing critical roles in both traditional IR systems [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:traditionalirengine`] and modern multi-stage pipelines [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:retrieverankingarch`]. Unfortunately, exact-term matching inherently suffers from the vocabulary mismatch problem due to the fact that a concept is often expressed using different vocabularies and language styles in documents and queries.

Early latent semantic models such as **latent semantic analysis (LSA)** illustrated the idea of identifying semantically relevant documents for a query when lexical matching is insufficient. However, their effectiveness in addressing the language discrepancy between documents and search queries are limited by their weak modeling capacity (i.e., simple, linear models). Also, these model parameters are typically learned via the unsupervised learning, i.e., by grouping different terms that occur in a similar context into the same semantic cluster. 

The introduction of deep neural networks for semantic modeling and retrieval was pioneered in {cite}`huang2013learning`. Recent deep learning model utilize the neural networks with large learning capacity and user-interaction data for supervised learning, which has led to significance performance gain over LSA.  Similarly in the field of OpenQA {cite}`karpukhin2020dense`, whose first stage is to retrieve relevant passages that might contain the answer, semantic-based retrieval has also demonstrated performance gains over traditional retrieval methods. 

### Two Architecture Paradigms

The current neural  architecture paradigms for IR can be categorized into two classes: **representation-based** and **interaction-based** [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:twoparadigms`]. 

In the representation-based architecture, a query and a document are encoded independently into two embedding vectors, then their relevance is estimated based on a single similarity score between the two embedding vectors. 

Here we would like to make a critical distinction on symmetric vs. asymmetric encoding:
- For **symmetric encoding**, the query and the entries in the corpus are typically of the similar length and have the same amount of content and they are encoded using the same network. Symmetric encoding is used for symmetric semantic search. An example would be searching for similar questions. For instance, the query could be *How to learn Python online?* and the entry that satisfies the search is like *How to learn Python on the web?*. 
- For **asymmetric encoding**, we usually have a short query (like a question or some keywords) and we would like to find a longer paragraph answering the query; they are encoded using two different networks. An example would be information retrieval. The entry is typically a paragraph or a web-page.

In the interaction-based architecture, instead of directly encoding $q$ and $d$ into individual embeddings, term-level interaction features across the query and the document are first constructed. Then a deep neural network is used to extract high-level matching features from the interactions and produce a final relevance score.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/DeepRetrievalModels/Two_paradigms.png
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:twoparadigms
Two common architectural paradigms in semantic retrieval learning: representation-based learning (left) and interaction-based learning (right).
```

These two architectures have different strengths in modeling relevance and final model serving. For example, a representation-based model architecture makes it possible to pre-compute and cache document representations offline, greatly reducing the online computational load per query. However, the pre-computation of query-independent document representations often miss term-level matching features that are critical to construct high-quality retrieval results. On the other hand, interaction-based architectures are often good at capturing the fine-grained matching feature between the query and the document. 

Since interaction-based models can model interactions between word pairs in queries and document, they are effective for re-ranking, but are cost-prohibitive for first-stage retrieval as the expensive document-query interactions must be computed online for all ranked documents.

Representation-based models enable low-latency, full-collection retrieval with a dense index. By representing queries and documents with dense vectors, retrieval is reduced to nearest neighbor search, or a maximum inner product search (MIPS) {cite}`shrivastava2014asymmetric` problem if similarity is represented by an inner product.

In recent years, there has been increasing effort on accelerating maximum inner product and nearest neighbor search, which led to high-quality implementations of libraries for nearest neighbor search such as hnsw {cite}`malkov2018efficient`, FAISS {cite}`johnson2019billion`, and SCaNN {cite}`guo2020accelerating`. 

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:classicRepresentationLearning)=
### Classic Representation-based Learning

#### DSSM

Deep structured semantic model (DSSM) {cite}`huang2013learning` improves the previous latent semantic models in two aspects: 1) DSSM is supervised learning based on labeled data, while latent semantic models are unsupervised learning; 2) DSSM utilize deep neural networks to capture more semantic meanings. 

The high-level architecture of DSSM is illustrated in {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:dssm`. First, we represent a query and a document (only its title) by a sparse vector, respectively. Second, we apply a non-linear projection to map the query and the document sparse vectors to two low-dimensional embedding vectors in a common semantic space. Finally, the relevance of each document given the query is calculated as the cosine similarity between their embedding vectors in that semantic space. 

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/DeepRetrievalModels/DSSM/dssm.png
:scale: 40%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:dssm
The architecture of DSSM. Two MLP encoders with shared parameters are used to encode a query and a document into dense vectors. Query and document are both represented by term vectors. The final relevance score is computed via dot product between the query vector and the document vector.
```

To represent word features in the query and the documents, DSSM adopt a word level sparse term vector representation with letter 3-gram vocabulary, whose size is approximately $30k \approx 30^3$. Here 30 is the approximate number of alphabet letters. This is also known as a letter trigram word hashing technique. In other words, both query and the documents will be represented by sparse vectors with dimensionality of $30k$.

The usage of letter 3-gram vocabulary has multiple benefits compared to the full vocabulary:
- Avoid OOV problem with finite-size vocabulary or term vector dimensionality.
- The use of letter n-gram can capture morphological meanings of words.

One problem of this method is collision, i.e., two different
words could have the same letter n-gram vector representation because this is a bag-of-words representation that does not take into account orders. But the collision probability is rather low.

```{table} Word hashing token size and collision numbers as a function of the vocabulary size and the type of letter ngrams.
|  Word Size  | Letter-Bigram |  | Letter-Trigram |  |
|-------------|---------------|------------------|-----------------|-----------------|
|             | Token Size    | Collision        | Token Size      | Collision       |
| 40k         | 1107          | 18               | 10306           | 2               |
| 500k        | 1607          | 1192             | 30621           | 22              |
```

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


```{table}
| Models    | NDCG@1 | NDCG@3 | NDCG@10 |
|-----------|--------|--------|---------|
| TF-IDF    | 0.319  | 0.382  | 0.462   |
| BM25      | 0.308  | 0.373  | 0.455   |
| LSA       | 0.298  | 0.372  | 0.455   |
| L-WH DNN  | **0.362**  | **0.425**  | **0.498**   |
```

#### CNN-DSSM

DSSM treats a query or a document as a bag of words, the fine-grained contextual structures embedding in the word order are lost. The DSSM-CNN{cite:p}`shen2014latent` [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:cnndssm`] directly represents local contextual features at the word n-gram level; i.e., it projects each raw word n-gram to a low dimensional feature vector where semantically similar word $\mathrm{n}$ grams are projected to vectors that are close to each other in this feature space. 

Moreover, instead of simply summing all local word-n-gram features evenly, the DSSM-CNN performs a max pooling operation to select the highest neuron activation value across all word n-gram features at each dimension. This amounts to extract the sentence-level salient semantic concepts. 

Meanwhile, for any sequence of words, this operation forms a fixed-length sentence level feature vector, with the same dimensionality as that of the local word n-gram features.

Given the letter-trigram based word representation, we represent a word-n-gram by concatenating the letter-trigram vectors of each word, e.g., for the $t$-th word-n-gram at the word-ngram layer, we have:

$$
l_{t}=\left[f_{t-d}^{T}, \ldots, f_{t}^{T}, \ldots, f_{t+d}^{T}\right]^{T}, \quad t=1, \ldots, T
$$

where $f_{t}$ is the letter-trigram representation of the $t$-th word, and $n=2 d+1$ is the size of the contextual window. In our experiment, there are about $30K$ unique letter-trigrams observed in the training set after the data are lower-cased and punctuation removed. Therefore, the letter-trigram layer has a dimensionality of $n \times 30 K$.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/DeepRetrievalModels/CNN_DSSM/CNN_DSSM.png
:scale: 40%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:cnndssm
The architecture of CNN-DSSM. Each term together with its left and right contextual words are encoded together into term vectors. 
```

### Mono-BERT And Duo-BERT

#### Why Transformers?

BERT (Bidirectional Encoder Representations from Transformers) {cite}`devlin2018bert` and its transformer variants {cite}`lin2021survey` represent the state-of-the-art modeling strategies in a broad range of natural language processing tasks. The application of BERT in information retrieval and ranking was pioneered by {cite}`nogueira2019passage, nogueira2019multi`. The fundamental characteristics of BERT architecture is self-attention. By pretraining BERT on large scale text data, BERT encoder can produce contextualized embeddings can better capture semantics of different linguistic units. By adding additional prediction head to the BERT backbone, such BERT encoders can be fine-tuned to retrieval related tasks. In this section, we will go over the application of different BERT-based models in neural information retrieval and ranking tasks. 

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:monoBERT)=
#### Mono-BERT For Point-wise Ranking

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/DeepRetrievalModels/Berts/BERT/mono_bert_arch.png
:scale: 30%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:monobertarch
The architecture of Mono-BERT for document relevance ranking. The input is the concatenation of the query token sequence and the candidate document token sequence. Once the input sequence is passed through the model, we use the [CLS] embedding as input to a single layer neural network to obtain a posterior probability $p_{i}$ of the candidate $d_{i}$ being relevant to query $q$. 
```

The first application of BERT in document retrieval is using BERT as a cross encoder, where the query token sequence and the document token sequence are concatenated via [SEP] token and encoded together. This architecture [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:monobertarch`], called mono-BERT, was first proposed by {cite}`nogueira2019passage, nogueira2019multi`.

To meet the token sequence length constraint of a BERT encoder (e.g., 512), we might need to truncate the query (e.g, not greater than 64 tokens) and the candidate document token sequence such that the total concatenated token sequence have a maximum length of 512 tokens.

Once the input sequence is passed through the model, we use the [CLS] embedding as input to a single layer neural network to obtain a posterior probability $p_{i}$ of the candidate $d_{i}$ being relevant to query $q$. The posterior probability can be used to rank documents.

The training data can be represented by a collections of triplets $(q, J_P^q, J_N^q), q\in Q$, where $Q$ is the set of queries, $J_{P}^q$ is the set of indexes of the relevant candidates associated with query $q$ and $J_{N}^q$ is the set of indexes of the nonrelevant candidates.

The encoder can be fine-tuned using cross-entropy loss:

$$
L_{\text {mono-BERT}}=-\sum_{q\in Q}( \sum_{j \in J_{P}^q} \log \left(s_{j}\right)-\sum_{j \in J_{N}^q} \log \left(1-s_{j}\right) ).
$$

#### Duo-BERT For Pairwise Ranking



Mono-BERT can be characterized as a *pointwise* approach for ranking. Within the *framework of learning to rank*, {cite}`nogueira2019passage, nogueira2019multi` also proposed duo-BERT, which is  a *pairwise* ranking approach. In this pairwise approach, the duo-BERT ranker model estimates the probability $p_{i, j}$ of the candidate $d_{i}$ being more relevant than $d_{j}$ with respect to query $q$.

The duo-BERT architecture [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:duobertarch`] takes the concatenation of the query $q$, the candidate document $d_{i}$, and the candidate document $d_{j}$ as the input. We also need to truncate the query, candidates $d_{i}$ and $d_{j}$ to proper lengths (e.g., 62 , 223 , and 223 tokens, respectively), so the entire sequence will have at most 512 tokens. 

Once the input sequence is passed through the model, we use the [CLS] embedding as input to a single layer neural network to obtain a posterior probability $p_{i,j}$. This posterior probability can be used to rank documents $i$ and $j$ with respect to each other. If there are $k$ candidates for query $q$, there will be $k(k-1)$ passes to compute all the pairwise probabilities. 

The model can be fine-tune using with the following loss per query.
```{math}
\begin{align*}
L_{\text {duo }}=&-\sum_{i \in J_{P}, j \in J_{N}} \log \left(p_{i, j}\right) \\
	&-\sum_{i \in J_{N}, j \in J_{P}} \log \left(1-p_{i, j}\right)
\end{align*}
```

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/DeepRetrievalModels/Berts/BERT/duo_bert_arch.png
:scale: 30%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:duobertarch
The duo-BERT architecture takes the concatenation of the query and two candidate documents as the input. Once the input sequence is passed through the model, we use the [CLS] embedding as input to a single layer neural network to obtain a posterior probability that the first document is more relevant than the second document. 
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

#### Multistage Retrieval And Ranking Pipeline

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/DeepRetrievalModels/Berts/BERT/multistage_retrieval_ranking_bert.png
:scale: 40%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:multistageretrievalrankingbert
Illustration of a three-stage retrieval-ranking architecture using BM25, monoBERT and duoBERT. Image from {cite}`nogueira2019multi`.
```

With BERT variants of different ranking capability, we can construct a multi-stage ranking architecture to select a handful of most relevant document from a large collection of candidate documents given a query. Consider a typical architecture comprising a number of stages from $H_0$ ot $H_N$. $H_0$ is a exact-term matching stage using from an inverted index. $H_0$ stage take billion-scale document as input and output thousands of candidates $R_0$. For stages from $H_1$ to $H_N$, each stage $H_{n}$ receives a ranked list $R_{n-1}$ of candidates from the previous stage and output candidate list $R_n$. Typically $|R_n| \ll |R_{n-1}|$ to enable efficient retrieval. 

An example three-stage retrieval-ranking system is shown in {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:multistageretrievalrankingbert`. In the first stage $H_{0}$, given a query $q$, the top candidate documents $R_{0}$ are retrieved using BM25. In the second stage $H_{1}$, monoBERT produces a relevance score $s_{i}$ for each pair of query $q$ and candidate $d_{i} \in R_{0}.$ The top candidates with respect to these relevance scores are passed to the last stage $H_{2}$, in which duoBERT computes a relevance score $p_{i, j}$ for each triple $\left(q, d_{i}, d_{j}\right)$. The final list of candidates $R_{2}$ is formed by re-ranking the candidates according to these scores .

**Evaluation.** Different multistage architecture configurations are evaluated using the MS MARCO dataset. We have following observations:
- Using a single stage of BM25 yields the worst performance.
- Adding an additional monoBERT significantly improve the performance over the single BM25 stage architecture.
- Adding the third component duoBERT only yields a diminishing gain.

Further, the author found that	employing the technique of Target Corpus Pre-training (TCP)\ gives additional performance gain. Specifically, the BERT backbone will undergo a two-phase pre-training. In the first phase, the model is pre-trained using the original setup, that is Wikipedia (2.5B words) and the Toronto Book corpus ( 0.8B words) for one million iterations. In the second phase, the model is further pre-trained on the MS MARCO corpus.

```{table}
| Method | Dev | Eval |
|--------|-----|------|
| Anserini (BM25) | 18.7 | 19.0 |
| + monoBERT | 37.2 | 36.5 |
| + monoBERT + duoBERT<sub>MAX</sub> | 32.6 | - |
| + monoBERT + duoBERT<sub>MIN</sub> | 37.9 | - |
| + monoBERT + duoBERT<sub>SUM</sub> | 38.2 | 37.0 |
| + monoBERT + duoBERT<sub>BINARY</sub> | 38.3 | - |
| + monoBERT + duoBERT<sub>SUM</sub> + TCP | 39.0 | 37.9 |
```




### DC-BERT



One way to improve the computational efficiency is to employ dual BERT encoders for partial separate encoding and then employ an additional shallow module for cross encoding. One example is the architecture shown in {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:dcbert`, which is called **DC-BERT** and proposed in {cite}`nie2020dc`. The overall architecture of DC-BERT  consists of a dual-BERT component for decoupled encoding, a Transformer component for question-document interactions, and a binary classifier component for document relevance scoring.

The document encoder can be run offline to pre-encodes all documents and caches all term representations. During online inference, we only need to run the BERT query encodes online. Then the obtained contextual term representations are fed into high-layer Transformer interaction layer. 

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/DeepRetrievalModels/Berts/DC_BERT/DC_bert.png
:scale: 30%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:dcbert
The overall architecture of DC-BERT [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:dcbert`] consists of a dual-BERT component for decoupled encoding, a Transformer component for question-document interactions, and a classifier component for document relevance scoring.
```

**Dual-BERT component**. DC-BERT contains two pre-trained BERT models to independently encode the question and each retrieved document. During training, the parameters of both BERT models are fine-tuned to optimize the learning objective.

**Transformer component.** The dual-BERT components produce contextualized embeddings for both the query token sequence and the document token sequence. Then we add global position embeddings $\mathbf{E}_{P_{i}} \in \mathbb{R}^{d}$ and segment embedding again to re-encode the position information and segment information (i.e., query vs document). Both the global position and segment embeddings are initialized from pre-trained BERT, and will be fine-tuned. The number of Transformer layers $K$ is configurable to trade-off between the model capacity and efficiency. The Transformer layers are initialized by the last $K$ layers of pre-trained BERT, and are fine-tuned during the training.

**Classifier component.** The two CLS token output from the Transformer layers will be fed into a linear binary classifier to predict whether the retrieved document is relevant to the query. Following previous work (Das et al., 2019; Htut et al., 2018; Lin et al., 2018), we employ paragraph-level distant supervision to gather labels for training the classifier, where a paragraph that contains the exact ground truth answer span is labeled as a positive example. We parameterize the binary classifier as a MLP layer on top of the Transformer layers:

$$
p\left(q_{i}, d_{j}\right)=\sigma\left(\operatorname{Linear}\left(\left[o_{[C L S]} ; o_{[C L S]}^{\prime}\right]\right)\right)
$$

where $\left(q_{i}, d_{j}\right)$ is a pair of question and retrieved document, and $o_{[C L S]}$ and $o_{[C L S]}^{\prime}$ are the Transformer output encodings of the [CLS] token of the question and the document, respectively. The MLP parameters are updated by minimizing the cross-entropy loss.

DC-BERT uses one Transformer layer for question-document interactions. Quantized BERT is a 8bit-Integer model. DistilBERT is a compact BERT model with 2 Transformer layers.

We first compare the retriever speed. DC-BERT achieves over 10x speedup over the BERT-base retriever, which demonstrates the efficiency of our method. Quantized BERT has the same model architecture as BERT-base, leading to the minimal speedup. DistilBERT achieves about 6x speedup with only 2 Transformer layers, while BERT-base uses 12 Transformer layers.

With a 10x speedup, DC-BERT still achieves similar retrieval performance compared to BERT- base on both datasets. At the cost of little speedup, Quantized BERT also works well in ranking documents. DistilBERT performs significantly worse than BERT-base, which shows the limitation of the distilled BERT model. 

```{table}
| Model | SQuAD | | Natural Questions | |
|-------|-------|-------|-------------------|-------|
| | PTB@10 | Speedup | P@10 | Speedup |
| BERT-base | 71.5 | 1.0x | 65.0 | 1.0x |
| Quantized BERT | 68.0 | 1.1x | 64.3 | 1.1x |
| DistilBERT | 56.4 | 5.7x | 60.6 | 5.7x |
| DC-BERT | 70.1 | 10.3x | 63.5 | 10.3x |
```

To further investigate the impact of our model architecture design, we compare the performance of DC-BERT and its variants, including 1) DC-BERT-Linear, which uses linear layers instead of Transformers for interaction; and 2) DC-BERT-LSTM, which uses LSTM and bi- linear layers for interactions following previous work (Min et al., 2018). We report the results in Table 3. Due to the simplistic architecture of the interaction layers, DC-BERT-Linear achieves the best speedup but has significant performance drop, while DC-BERT-LSTM achieves slightly worse performance and speedup than DC-BERT.

```{table}
| Retriever Model | Retriever P@10 | Retriever Speedup |
|-----------------|----------------|-------------------|
| DC-BERT-Linear  | 57.3           | 43.6x             |
| DC-BERT-LSTM    | 61.5           | 8.2x              |
| DC-BERT         | 63.5           | 10.3x             |
```

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:colBERT)=
### ColBERT

#### Model Architecture And Training

ColBERT {cite}`khattab2020colbert` is another example architecture that consists of an early separate encoding phase and a late interaction phase, as shown in {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:colbert`. ColBERT employs a single BERT model for both query and document encoders but distinguish input sequences that correspond to queries and documents by prepending a special token [Q] to queries and another token [D] to documents.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/DeepRetrievalModels/Berts/Col_BERT/Col_bert.png
:scale: 30%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:colbert
The architecture of ColBERT, which consists of an early separate encoding phase and a late interaction phase.
```


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
Here # refers to the [mask] tokens and $\operatorname{Normalize}$ denotes $L_2$ length normalization.

In the late interaction phase, every query embedding interacts with all document embeddings via a MaxSimilarity operator, which computes maximum similarity (e.g., cosine similarity), and the scalar outputs of these operators are summed across query terms.

Formally, the final similarity score between the $q$ and $d$ is given by

$$
S_{q, d} =\sum_{i \in I_q} \max _{j \in I_d} E_{q_{i}} \cdot E_{d_{j}}^{T},
$$

where $I_q = \{1,...,l\}$, $I_d = \{1, ..., n\}$ are the index sets for query token embeddings and document token embeddings, respectively. 
ColBERT is differentiable end-to-end and we can fine-tune the BERT encoders and train from scratch the additional parameters (i.e., the linear layer and the $[Q]$ and $[D]$ markers' embeddings). Notice that the final aggregation interaction mechanism has no trainable parameters. 

The retrieval performance of ColBERT is evaluated on MS MARCO dataset. Compared with traditional exact term matching retrieval, ColBERT has shortcomings in terms of latency but MRR is significantly better.

```{table}
| Method | MRR@10(Dev) | MRR@10 (Local Eval) | Latency (ms) | Recall@50 |
|--------|-------------|---------------------|--------------|-----------|
| BM25 (official) | 16.7 | - | - | - |
| BM25 (Anserini) | 18.7 | 19.5 | 62 | 59.2 |
| doc2query | 21.5 | 22.8 | 85 | 64.4 |
| DeepCT | 24.3 | - | 62 (est.) | 69[2] |
| docTTTTTquery | 27.7 | 28.4 | 87 | 75.6 |
| ColBERT L2 (re-rank) | 34.8 | 36.4 | - | 75.3 |
| ColBERTL2 (end-to-end) | 36.0 | 36.7 | 458 | 82.9 |
```

Similarly, we can evaluate ColBERT's re-ranking performance against some strong baselines, such as BERT cross encoders {cite}`nogueira2019passage, nogueira2019multi`. ColBERT has demonstrated significant benefits in reducing latency with little cost of re-ranking performance. 

```{table}
| Method | MRR@10 (Dev) | MRR@10 (Eval) | Re-ranking Latency (ms) |
|--------|--------------|---------------|-------------------------|
| BM25 (official) | 16.7 | 16.5 | - |
| KNRM | 19.8 | 19.8 | 3 |
| Duet | 24.3 | 24.5 | 22 |
| fastText+ConvKNRM | 29.0 | 27.7 | 28 |
| BERT base | 34.7 | - | 10,700 |
| BERT large | 36.5 | 35.9 | 32,900 |
| ColBERT (over BERT base) | 34.9 | 34.9 | 61 |
```

## Ranker Training

### Overview



Unlike in classification or regression, the main goal of a ranker{cite:p}`ai2019learning` is not to assign a label or a value to individual items, but to produce an ordering of the items in that list in such a way that the utility of the entire list is maximized. 

In other words, in ranking we are more concerned with the relative ordering items, instead of predicting the numerical value or label of an individual item.

**Pointwise ranking** transforms the ranking problem into a regression problem. Given a certain Query, ranking amounts to 
* Predict the relevance score between the document to the query
* Order the document list based on its relevance score with the query.

**Pairwise Ranking**{cite:p}`burges2010ranknet`, instead of predicting the absolute relevance, learns to predict the relative order of documents. This method is particularly useful in scenarios where the absolute relevance scores are less important than the relative ordering of items,


**Listwise ranking** {cite:p}`cao2007learning` considers the entire list of items simultaneously when training a ranking model. Unlike pairwise or pointwise methods, listwise ranking directly optimizes ranking metrics such as NDCG or MAP, which better aligns the training objective with the evaluation criteria used in information retrieval tasks. This approach can capture more complex relationships between items in a list and often leads to better performance in real-world ranking scenarios, though it may be computationally more expensive than other ranking methods.


<!-- 
\begin{remark}[training loss function and score function]\hfill
- RankNet, LambdaRank are methods with pairwise loss function and univariate score function
- Biencoders with N-pair loss function are methods with list-wise loss function and univariate score function
- Cross-encoder with Binary cross entropy loss are methods with point-wise loss and univariate score function

\end{remark} -->

### Training Data

In a typical model learning setting, we construct training data from user search log, which contains queries issued by users and the documents they clicked after issuing the query. The basic assumption is that a query and a document are relevant if the user clicked the document. 

Model learning in information retrieval typically falls into the category of contrastive learning. The query and the clicked document form a positive example; the query and irrelevant documents form negative examples. For retrieval problems, it is often the case that positive examples are available explicitly, while negative examples are unknown and need to be selected from an extremely large pool. The strategy of selecting negative examples plays an important role in determining quality of the encoders. In the most simple case, we randomly select unclicked documents as irrelevant document, or negative example. We defer the discussion of advanced negative example selecting strategy to {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:negativeSamplingStrategies`.

When there is a shortage of annotation data or click behavior data, we can also leverage weakly supervised data for training {cite}`dehghani2017neural,ram2021learning,haddad2019learning,nie2018multi`. In the weakly supervised data, labels or signals are obtained from an unsupervised ranking model, such as BM25. For example, given a query, relevance scores for all documents can be computed efficiently using BM25. Documents with highest scores can be used as positive examples and documents with lower scores can be used as negatives or hard negatives. 



### Model Training Objective Functions

#### Pointwise Regression Objective

The idea of pointwise regression objective is to model the numerical relevance score for a given query-document. During inference time, the relevance scores between a set of candidates and a given query can be predicted and ranked.  

During training, given a set of query-document pairs $\left(q_{i}, d_{i, j}\right)$ and their corresponding relevance score $y_{i, j} \in [0, 1]$ and their prediction $f(q_i,d_{i,j})$. A pointwise regression objective tries to optimize a model to predict the relevance score via minimization

$$L = -\sum_{i} \sum_{j} (f(q_i,d_{i,j})) - y_{i,j})^2.
$$

Using a regression objective offer flexible for the user to model different levels of relevance between queries and documents. However, such flexibility also comes with** a requirement that the target relevance score should be accurate in absolute scale.**
While human annotated data might provide absolute relevance score, human annotation data is expensive and small scale. On the other hand, absolute relevance scores that are approximated by click data can be noisy and less optimal for regression objective. To make** training robust to label noises**, one can consider **Pairwise ranking objectives.** This particularly important in weak supervision scenario with noisy label. Using the ranking objective alleviates this issue by forcing the 	model to learn a preference function rather than reproduce absolute scores. 

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:pointwise_ranking_loss)=
#### Pointwise Ranking Objective

The idea of pointwise ranking objective is to simplify a ranking problem to a binary classification problem. Specifically, given a set of query-document pairs $\left(q_{i}, d_{i, j}\right)$ and their corresponding relevance label $y_{i, j} \in \{0, 1\}$, where 0 denotes irrelevant and 1 denotes relevant. A pointwise learning objective tries to optimize a model to predict the relevance label. 

A commonly used pointwise loss functions is the binary Cross Entropy loss:

$$
L=-\sum_{i} \sum_{j} y_{i, j} \log \left(p\left(q_{i}, d_{i, j}\right)\right)+\left(1-y_{i, j}\right) \log \left(1-p\left(q_{i}, d_{i, j}\right)\right)
$$

where $p\left(q_{i}, d{i, j}\right)$ is the predicted probability of document $d_{i,j}$ being relevant to query $q_i$.

The advantages of pointwise ranking objectives are two-fold. First, pointwise ranking objectives are computed based on each query-document pair $\left(q_{i}, d_{i, j}\right)$ separately, which makes it simple and easy to scale. Second, the outputs of neural models learned with pointwise loss functions often have real meanings and value in practice. For instance, in sponsored search, the predicted the relevance probability can be used in ad bidding, which is more important than creating a good result list in some application scenarios.

In general, however, pointwise ranking objectives are considered to be less effective in ranking tasks. Because pointwise loss functions consider no document preference or order information, they do not guarantee to produce the best ranking list when the model loss reaches the global minimum. Therefore, better ranking paradigms that directly optimize document ranking based on pairwise loss functions and even listwise loss functions.

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:triplet_loss)=
#### Pairwise Ranking Via Triplet Loss

Pointwise ranking loss aims to optimize the model to directly predict relevance between query and documents on absolute score. From embedding optimization perspective, it train the neural query/document encoders to produce similar embedding vectors for a query and its relevant document and dissimilar embedding vectors for a query and its irrelevant documents. 

On the other hand, pairwise ranking objectives focus on **optimizing the relative preferences between documents rather than predicting their relevance labels.** In contrast to pointwise methods where the final ranking loss is the sum of loss on each document, pairwise loss functions are computed based on the different combination of document pairs.

One of the most common pairwise ranking loss function is the **triplet loss**. Let $\mathcal{D}=\left\{\left\langle q_{i}, d_{i}^{+}, d_{i}^{-}\right\rangle\right\}_{i=1}^{m}$ be the training data organized into $m$ triplets. Each triplet contains one query $q_{i}$ and one relevant document $d_{i}^{+}$, along with one irrelevant (negative) documents $d_{i}^{-}$. Negative documents are typically randomly sampled from a large corpus or are strategically constructed [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:negativeSamplingStrategies`]. 
Visualization of the learning process in the embedding space is shown in {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:triplet`. Triplet loss helps guide the encoder networks to pull relevant query and document closer and push irrelevant query and document away.   

The loss function is given by

$$L =- \sum_{\left\langle q_{i}, d_{i}^{+}, d_{i}^{-}\right\rangle}\max(0, m - \operatorname{Sim}(q_i, d_i^+) + \operatorname{Sim}(q_i, d^-_i))$$

where $\operatorname{Sim}(q, d)$ is the similarity score produced by the network between the query and the document and $m$ is a hyper-parameter adjusting the margin. Clearly, only when $\operatorname{Sim}(q_i, d_i^+) - \operatorname{Sim}(q_i, d^-_i) > m$ there will be no loss incurred. Commonly used $\operatorname{Sim}$ functions include **dot product** or **Cosine similarity** (i.e., length-normalized dot product), which are related to distance calculation in the Euclidean space and hyperspherical surface. 

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/TrainingLoss/triplet.png
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
#### N-pair Loss

Triplet loss optimize the neural by encouraging positive pair $(q_i, d^+_i)$ to be more similar than its negative pair $(q_i, d^+_i)$. One improvement is to encourage $q_i$ to be more similar $d^+_i$ compared to $n$ negative examples $ d_{i, 1}^{-}, \cdots, d_{i, n}^{-}$, instead of just one negative example. This is known as N-pair loss {cite}`sohn2016improved`, and it is typically more robust than triplet loss.

Let $\mathcal{D}=\left\{\left\langle q_{i}, d_{i}^{+}, D_i^-\right\rangle\right\}_{i=1}^{m}$, where $D_i^- = \{d_{i, 1}^{-}, \cdots, d_{i, n}^{-}\}$ are a set of negative examples (i.e., irrelevant document) with respect to query $q_i$,  be the training data that consists of $m$ examples. Each example contains one query $q_{i}$ and one relevant document $d_{i}^{+}$, along with $n$ irrelevant (negative) documents $d_{i, j}^{-}$. The $n$ negative documents are typically randomly sampled from a large corpus or are strategically constructed [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:negativeSamplingStrategies`]. 

Visualization of the learning process in the embedding space is shown in {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:npairloss`. Like triplet loss, N-pair loss helps guide the encoder networks to pull relevant query and document closer and push irrelevant query and document away. Besides that, when there are are negatives are involved in the N-pair loss, their repelling to each other appears to help the learning of generating more uniform embeddings{cite:p}`wang2020understanding`. 

The loss function is given by

$$L =-\sum_{\left\langle q_{i}, d_{i}^{+}, D_{i}^{-}\right\rangle}\log \frac{\exp(\operatorname{Sim}\left(e_{q_{i}}, e_{d_{i}^{+}}\right))}{\exp(\operatorname{Sim}\left(e_{q_{i}}, e_{d_{i}^{+}}\right))+\sum_{d^-_i\in D^-} \exp(\operatorname{Sim}\left(e_{q_{i}}, e_{d_{i}^{-}}\right))}$$

where $\operatorname{Sim}(e_q, e_d)$ is the similarity score function taking query embedding $e_q$ and document embedding $e_d$ as the input. 

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/TrainingLoss/N_pair_loss.png
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:npairloss
The illustration of the learning process (in the embedding space) using N-pair loss.
```

#### N-pair Dual Loss

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/TrainingLoss/N_pair_loss_dual.png
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

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:negativeSamplingStrategies)=
## Training Data Sampling Strategies

### Principles

From the ranking perspective, both retrieval and re-ranking requires the generation of some order on the input samples. For example, given a query $q$ and a set of candidate documents $(d_1,...,d_N)$. We need the model to produce an order list $d_2 \succ d_3 ... \succ d_k$ according to their relevance to the query. 

To train a model to produce the expected results during inference, we should ensure the training data distribution to matched with the inference time data distribution. 
Particularly, the inference time the candidate document distribution and ranking granularity  differ vastly for retrieval tasks and re-ranking tasks [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:retrievalrerankingtask`]. Specifically, 
- For the retrieval task, we typically need to identify top $k (k=1000-10000)$ relevant documents from the entire document corpus. This is achieved by ranking all documents in the corpus with respect to the relevance of the query. 
- For the re-ranking task, we need to identify the top $k (k=10)$ most relevant documents from the relevant documents generated by the retrieval task.  

Clearly, features most useful in the retrieval task (i.e., distinguish relevant from irrelevant) are often not the same as the features most useful in re-ranking task (i.e., distinguish most relevant from less relevant). Therefore, the training samples for retrieval and re-ranking need to be constructed differently.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/TrainingDataSampling/retrieval_reranking_task.png
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:retrievalrerankingtask
Retrieval tasks and re-ranking tasks are faced with different the candidate document distribution and ranking granularity.
```

Constructing the proper training data distribution is more challenging to retrieval stage than the re-ranking stage. In re-ranking stage, data in the training and inference phases are both the documents from previous retrieval stages. In the retrieval stage, we need to construct training examples in a mini-batch fashion in a way that each batch approximates the distribution in the inference phase as close as possible. 

This section will mainly focus on constructing training examples for retrieval model training in an efficient and effective way. Since the number of negative examples (i.e., irrelevant documents) significantly outnumber the number of positive examples. Constructing training examples particularly boil down to constructing proper negative examples. 

### Negative Sampling Methods I: Heuristic Methods

#### Overview

The essence of the negative sampling algorithm is to set or adjust the sampling distribution during negative sampling based on certain methods. According to the way the negative sampling algorithm sets the sampling distribution, we can divide the current negative sampling algorithms into two categories: Heuristic Negative Sampling Algorithms and Model-based Negative Sampling Algorithms.

In {cite}`karpukhin2020dense`, there are three different types of negatives: (1) Random: any random passage from the corpus; (2) BM25: top passages returned by BM25 which don’t contain the answer but match most question tokens; (3) Gold: positive passages paired with other questions which appear in the training set.

One approach to improving the effectiveness of single-vector bi-encoders is hard negative mining, by training with carefully selected negative examples that emphasize discrimination between relevant and non-relevant texts.

both large in-batch negative sampling and asynchronous ANN index updates are computationally demanding.

Compared with the two heuristic algorithms mentioned above, the model-based negative sampling algorithm is easier to pick high-quality negative examples, and it is also the more cutting-edge sampling algorithm at present. Here are several model-based negative sampling algorithms:

(ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:in-batch-negatives)=
#### Random Negatives And In-batch Negatives

Random negative sampling is the most basic negative sampling algorithm. The algorithm uniformly sample documents from the corpus and treat it as a negative. Clearly, random negatives can generate negatives that are too easy for the model. For example, a negative document that is topically different from the query. These easy negatives lower the learning efficiency, that is, each batch produces limited information gain to update the model. Still, random negatives are widely used because of its simplicity.

In practice, random negatives are implemented as in-batch negatives.  In the contrastive learning framework with N-pair loss [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:N_pair_loss`], we construct a mini-batch of query-doc examples like $\{(q_1, d_1^+, d_{1,1}^-, d_{1,M}^-), ..., (q_N, d_N^+, d_{N,1}^-, d_{N,M}^M)\}$, Naively implementing N-pair loss would increase computational cost from constructing sufficient negative documents corresponding to each query. In-batch negatives{cite:p}`karpukhin2020dense` is trick to reuse positive documents associated with other queries as extra negatives [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:inbatchnegatives`]. The critical assumption here is that queries in a mini-batch are vastly different semantically, and positive documents from other queries would be confidently used as negatives. The assumption is largely true since each mini-batch is randomly sampled from the set of all training queries, in-batch negative document are usually true negative although they might not be hard negatives.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/TrainingDataSampling/NegativeSampling/in_batch_negatives.png
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

#### Popularity-based Negative Sampling

Popularity-based negative sampling use document popularity as the sampling weight to sample negative documents. The popularity of a document can be defined as some combination of click, dwell time, quality, etc. Compared to random negative sampling, this algorithm replaces the uniform distribution with a popularity-based sampling distribution, which can be pre-computed offline. 

The major rationale of using popularity-based negative examples is to improve representation learning. Popular negative documents represent a harder negative compared to a unpopular negative since they tend to have to a higher chance of being more relevant; that is, lying closer to query in the embedding space. If the model is trained to distinguish these harder cases, the over learned representations will be likely improved. 

Popularity-based negative sampling is also used in word2vec training {cite}`mikolov2013distributed`. For example, the probability to sample a word $w_i$ is given by:

$$
P\left(w_i\right)=\frac{f\left(w_i\right)^{3 / 4}}{\sum_{j=0}^n\left(f\left(w_j\right)^{3 / 4}\right)},
$$

where $f(w)$ is the frequency of word $w$. This equation, compared to linear popularity, has the tendency to increase the probability for less frequent words and decrease the probability for more frequent words.

#### Topic-aware Negative Sampling

In-batch random negatives would often consist of  topically-different documents, leaving little information gain for the training. To improve the information gain from a single random batch, we can constrain the queries and their relevant document are drawn from a similar topic{cite:p}`hofstatter2021efficiently`.

The procedures are
- Cluster queries using query embeddings produced by basic query encoder.
- Sample queries and their relevant documents from a randomly picked cluster. A relevant document of a query form the negative of the other query.

Since queries are topically similar, the formed in-batch negatives are harder examples than randomly formed in-batch negative, therefore delivering more information gain each batch.  

Note that here we group queries into clusters by their embedding similarity, which allows grouping queries without lexical overlap. We can also consider lexical similarity between queries as additional signals to predict query similarity. 

### Negative Sampling Methods II: Model-based Methods
#### Static Hard Negative Examples

Deep model improves the encoded representation of queries and documents by contrastive learning, in which the model learns to distinguish positive examples and negative examples. A simple random sampling strategy tend to produce a large quantity of easy negative examples since easy negative examples make up the majority of negative examples. Here by easy negative examples, we mean a document that can be easily judged to be irrelevant to the query. For example, the document and the query are in completely different topics.   

The model learning from easy negative example can quickly plateau since easy negative examples produces vanishing gradients to update the model. An improvement strategy is to supply additional hard negatives with randomly sampled negatives. In the simplest case, hard negatives can be selected based on a traditional BM25model {cite}`karpukhin2020dense, nogueira2019passage` or other efficient dense retriever: hard negatives are those have a high relevant score to the query but they are not relevant.  

As illustrated in {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:impacthardnegativeonretrieval`, a model trained only with easy negatives can fail to distinguish fairly relevant documents from irrelevant examples; on the other hand, a model trained with some hard negatives can learn better representations:
- Positive document embeddings are more aligned {cite}`wang2020understanding`; that is, they are lying closer with respect to each other.
- Fairly relevant and irrelevant documents are more separated in the embedding space and thus a better decision boundary for relevant and irrelevant documents. 

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/TrainingDataSampling/Impact_hard_negative_on_retrieval.png
:scale: 30%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:impacthardnegativeonretrieval
Illustration of importance of negative hard examples, which helps learning better representations to distinguish irrelevant and fairly relevant documents.
```

In generating these negative examples, the negative-generation model (e.g., BM25) and the model under training are de-coupled; that is the negative-generation model is not updated during training and the hard examples are static. Despite this simplicity, static hard negative examples introduces two shortcomings:
- Distribution mismatch, the negatives generated by the static model might quickly become less hard since the target model is constantly evolving.
- The generated negatives can have a higher risk of being false negatives to the target model because negative-generation model and the target model are two different models.  

#### Dynamic Hard Negative Examples



Dynamic hard negative mining is a scheme proposed in ANCE{cite:p}`xiong2020approximate`. The core idea is to use the target model at previous checkpoint as the negative-generation model [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:ancenegativesamplingdemo`]. However, this negative mining approach is rather computationally demanding since corpus index need updates at every checkpoint. 

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/TrainingDataSampling/NegativeSampling/ANCE_negative_sampling_demo.png
:scale: 30%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:ancenegativesamplingdemo
Dynamic hard negative sampling from ANCE asynchronous training framework. Negatives are drawn from index produced using models at the previous checkpoint. Image from {cite}`xiong2020approximate`.
```
<!-- \begin{remark rom ANCE]

	Based on our analysis, we propose Approximate nearest neighbor Negative Contrastive Estimation (ANCE), a new contrastive representation learning mechanism for dense retrieval. Instead of random or in-batch local negatives, ANCE constructs global negatives using the beingoptimized DR model to retrieve from the entire corpus. 

	This fundamentally aligns the distribution of negative samples in training and of irrelevant documents to separate in testing. From the variance reduction perspective, these ANCE negatives lift the upper bound of per instance gradient norm, reduce the variance of the
	stochastic gradient estimation, and lead to faster learning convergence.
\end{remark} -->

### Label Denoising
#### False Negatives

Hard negative examples produced from static or dynamic negative mining methods are effective to improve the encoder's performance. However, when selecting hard negatives with a less powerful model (e.g., BM25), we are also running the risk of introduce more false negatives (i.e., negative examples are actually positive) than a random sampling approach. Authors in {cite}`qu2020rocketqa` proposed to utilize a well-trained, complex  model (e.g., a cross-encoder) to determine if an initially retrieved hard-negative is a false negative. Such models are more powerful for capturing semantic similarity among query and documents. Although they are less ideal for deployment and inference purpose due to high computational cost, they are suitable for filtering. From the initially retrieved hard-negative documents, we can filter out documents that are actually relevant to the query. The resulting documents can be used as denoised hard negatives. 

#### False Positives

Because of the noise in the labeling process (e.g., based on click data), it is also possible that a positive labeled document turns out to be irrelevant. To reduce false positive examples, one can develop more robust labeling process and merge labels from multiple sources of signals. 



## Multi-vector Representations

### Introduction

In classic representation-based learning for semantic retrieval [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:classicRepresentationLearning`], we use two encoders (i.e., bi-encoders) to separately encoder a query and a candidate document into two dense vectors in the embedding space, and then a score function, such as cosine similarity, to produce the final relevance score. In this paradigm, there is a single global, static representation for each query and each document. Specifically, the document's embedding remain the same regardless of the document length, the content structure of document (e.g., multiple topics) and the variation of queries that are relevant to the document. It is very common that a document with hundreds of tokens might contain several distinct subtopics, some important semantic information might be easily missed or biased by each other when compressing a document into a dense vector.  As such, this simple bi-encoder structure may cause serious information loss when used to encode documents. <sup>[^2]</sup>

On the other hand,  cross-encoders based on BERT variants [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:classicRepresentationLearning`] utilize multiple self-attention layers not only to extract contextualized features from queries and documents but also capture the interactions between them. Cross-encoders only produce intermediate representations that take a pair of query and document as the joint input. While BERT-based cross-encoders brought significant performance gain,  they are computationally prohibitive and impractical for online inference. 

In this section, we focus on different strategies {cite}`humeau2019poly, tang2021improving, luan2021sparse` to encode documents by multi-vector representations, which enriches the single vector representation produced by a bi-encoder. With additional computational overhead, these strategies can gain much improvement of the encoding quality while retaining the fast retrieval strengths of Bi-encoder.

### Token-level Multi-vector Representation

To enrich the representations of the documents produced by Bi-encoder, some researchers extend the original Bi-encoder by employing more delicate structures like later- interaction

ColBERT [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:colBERT`] can be viewed a token-level multi-vector representation encoder for both queries and documents. Token-level representations for documents can be pre-computed offline. During online inference, late interactions of the query's multi-vectors representation and the document's  multi-vectors representation are used to  improve the robustness of dense retrieval, as compared to inner products of single-vector representations. Specifically,

Formally, given $q= q_{1} \ldots q_{l}$ and $d=d_{1} \ldots d_{n}$ and their token level embeddings $\{E_{q_1},\ldots E_{q_l}\}$ and $\{E_{d_1},...,E_{d_n}\}$ and the final similarity score between the $q$ and $d$ is given by

$$
S_{q, d} =\sum_{i \in I_q} \max _{j \in I_d} E_{q_{i}} \cdot E_{d_{j}}^{T},
$$

where $I_q = \{1,...,l\}$, $I_d = \{1, ..., n\}$ are the index sets for query token embeddings and document token embeddings, respectively. 

While this method has shown signficant improvement over bi-encoder methods, it has a main disadvantage of high storage requirements. For example, ColBERT requires storing all the WordPiece token vectors of each text in the corpus. 

### Semantic Clusters As Pseudo Query Embeddings

The primary limitation of Bi-encoder is information loss when we condense the document into a query agnostic dense vector representation. Authors in {cite}`tang2021improving` proposed the idea of representing a document by its semantic salient fragments [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:pseudoqueryembeddings`]. These semantic fragments can be modeled by token embedding vector clusters in the embedding space. By performing clustering algorithms (e.g., k-means) on token embeddings, the generated centroids can be used as a document's multi-vector presentation. Another interpretation is that these centroids can be viewed as multiple potential queries corresponding to the input document; as such, we can call them *pseudo query embeddings*. 

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/MultivectorRepresentation/pseudo_query_embedding/pseudo_query_embeddings.png
:scale: 30%
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


## Query and Document Expansion

### Overview

Query expansion and document expansion techniques provide two potential solutions to the inherent vocabulary mismatch problem in traditional IR systems. The core idea is to add extra relevant terms to queries and documents respectively to aid relevance matching. 

Consider an ad-hoc search example of *automobile sales per year in the US*. 
- Document expansion can be implemented by appending *car* in documents that contains the term *automobile* but not *car*. Then an exact-match retriever can fetch documents containing either *car* and *automobile*.
- Query expansion can be accomplished by retrieving results from both the query *automobile sales per year in the US* and the query *car sales per year in the US*.

There are many different approaches to coming up suitable terms to expand queries and documents in order to make relevance matching easier. These approaches range from traditional rule-based methods such as synonym expansion to recent learning based approaches by mining user logs. For example, augmented terms for a document can come from queries that are relevant from user click-through logs. 
```{figure} ../img/chapter_application_IR/ApplicationIRSearch/QueryDocExpansion/queryExpansion/query_expansion_arch.png
:scale: 30%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:queryexpansionarch
Query expansion module in an IR system.
```

Both query and document expansion can be fit into typical IR architectures through an de-coupled module. A query expansion module takes an input query and output a (richer) expanded query [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:queryexpansionarch`]. They are also known as query rewriters or expanders. The module might remove terms deemed unnecessary in the user’s query, for example stop words and add extra terms facilitate the engine to retrieve documents with a high recall. 

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/QueryDocExpansion/documentExpansion/doc_expansion_arch.png
:scale: 30%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:docexpansionarch
Document expansion module in an IR system.
```

Similarly, document expansion naturally fits into the retrieval and ranking pipeline [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:docexpansionarch`]. The index will be simply built upon the expanded corpus to provide a richer set of candidate documents for retrieval. The extra computation for document expansion can be all carried out offline. Therefore, it presents the same level of effectiveness like query expansion but at lower latency costs (for example, using less computationally intensive rerankers). 

Query expansion and document expansion have different pros and cons. Main advantages of query expansions include
- Compare to document expansion, query expansion techniques can be quickly implemented and experimented without modifying the entire indexing. On the other hand, experimenting document expansion techniques can be costly and time-consuming, since the entire indexing is affected. 
- Query expansion techniques are generally more flexible. For example, it is easy to switch on or off different features at query time (for example, selectively apply expansion only to certain intents or certain query types). 
	The flexibility of query expansion also allow us insert an expansion module in different stages of the retrieval-ranking pipeline. 

On the other hand, one unique advantage for document expansion is that: documents are typically much longer than queries, and thus offer more context for a model to choose appropriate expansion terms. Neural based natural language generation models, like Transformers {cite}`vaswani2017attention`, can benefit from richer contexts and generate cohesive natural language terms to expand original documents. 

### Document Expansion Via Query Prediction

Authors in {cite}`nogueira2019document` proposed DocT5Query, a document expansion strategy based on a seq-to-seq natural language generation model to enrich each document. 

For each document, the transformer generation model predicts a set of queries that are likely to be relevant to the document. Given a dataset of (query, relevant document) pairs, we use a transformer model is trained to takes the document as input and then to produce the target query<sup>[^3]</sup>.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/QueryDocExpansion/documentExpansion/Doc2query_arch.png
:scale: 60%
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
<!-- 
### Pseudo Relevance Feedback

#### Basics

Pseudo relevance feedback (PRF) is another commonly used technique to boost the performance of traditional IR models and to reduce the effect of query-document vocabulary mismatches and improve the estimate the term weights. The interest of using PRF has been recently expanded into the neural IR models {cite}`li2018nprf`.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/PseudoRelevanceFeedback/PRF_arch.png
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:prfarch
A typical architecture for pseudo relevance feedback implementation.
```

{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:prfarch` shows a typical architecture for pseudo relevance feedback implementation. There are two rounds of retrieval. In the initial retrieval, the retriever fetches a batch of relevant documents based on the original query. We can use the top-$k$ documents to expand and refine the original query. In the second round retrieval, the retriever fetches relevant documents as the final result based on the expanded query. Intuitively, the first-round retrieved documents help identify terms not present in the original query that are discriminative of relevant texts. After the expansion, the expanded query effective mitigate the vocabulary gap between original query and the corpus. 

#### Neural Pseudo Relevance Feedback (NPRF)

**Overview**
Given a query q, NPRF estimates the relevance of a target document $d$ relative to $q$ using following key procedures:
	1. Create initial retrieval result. Given a document corpus $\cD$, a simple ranking method (e.g., BM25) $\operatorname{rel}_{q}(q, d)$ is applied to each $d\in \cD$ to obtain the top-$m$ documents, denoted as $D_{q}$ for $q$.
	2. Compute document-document relevance. We extract the relevance between each $d_{q}\in D_{q}$ and the target $q$, using a neural ranking method $\operatorname{rel}_{d}(d_{q}, d)$.
	3. Compute final relevance.  The relevance scores $\operatorname{rel}_{d}(d_{q}, d)$ from previous step weighted by $rel_{q}(q, d_{q})$ to arrive at $\operatorname{rel}'_{d}\left(d_{q}, d\right)$. The weighting which serves as an estimator for the confidence of the contribution of $d_{q}$ relative to $q$. Finally, we relevance between $q$ and $d$ is given by the aggregation of these adjusted relevance scores, $$\operatorname{rel}_{D}(q, D_{q}, d) = \sum_{d_q\in D_q} \operatorname{rel}'_{d}(d_q, d).$$

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/PseudoRelevanceFeedback/NPRF_arch.png
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
$$ -->
<!-- 
### Contextualized Term Importance

#### Context-aware Term Importance: Deep-CT

In ad-hoc search, queries are mainly short and keyword based without complex grammatical structures. To be able to fetch most relevant results, it is  important to take into account term importance. For example,  given the query *bitcoin news*, a relevant document is expected to be about *bitcoin* and *news*, where the term *bitcoin* is more important than *news* in the sense that a document describing other aspects of bitcoin would be more relevant than a document describing news of other things.

In the traditional IR framework, term importance is calculated using inverse document frequency. A term is less important if it is a common term appearing in a large number of documents. These frequency-based term weights have been a huge success in traditional IR systems due to its simplicity and scalability. The problematic aspect is that Tf-idf determines the term importance solely based on word counts rather than the semantics. High-frequency words  does not necessarily indicate their central role to the meaning of the text, especially for short texts where the word frequency distribution is quite flat. Considering the following two passages returned for the query stomach {cite}`dai2019context`, one is relevant and one is not:
- Relevant: In some cases, an upset stomach is the result of an allergic reaction to a certain type of food. It also may be caused by an irritation. Sometimes this happens from consuming too much alcohol or caffeine. Eating too many fatty foods or too much food in general may also cause an upset stomach.
- Less relevant: All parts ofthe body (muscles , brain, heart, and liver) need energy to work. This energy comes from the food we eat. Our bodies digest the food we eat by mixing it with fluids( acids and enzymes) in the stomach. When the stomach digests food, the carbohydrate (sugars and starches) in the food breaks down into another type of sugar, called glucose.

In both passages, the word *stomach* appear twice; but the second passage is actually off-topic. This example also suggests that the importance of a term depends on its context, which helps understand the role of the word playing in the text. 

Authors in {cite}`dai2019context` proposed DeepCT, which uses the contextual word representations from BERT to estimate term importance to improve the traditional IR approach. Specifically, given a word in a specific text, its contextualized word embedding (e.g., BERT) is used a feature vector that characterizes the word's syntactic and semantic role in the text. Then DeepCT estimates the word's importance score via a weighted summation:
$$
\hat{y}_{t, c}= {w} T_{t, c}+b
$$
where $T_{t, c} \in \mathbb{R}^D$ is token $t$ 's contextualized embedding in the text $c$; and, ${w}\in \mathbb{R}^D$ and $b$ are the weights and bias.

The model parameters of DeepCT are the weight and bias, and they can be estimated from a supervised learning task, per-token regression, given by 
$$
L_{MSE}=\sum_{c} \sum_{t}\left(y_{t, c}-\hat{y}_{t, c}\right)^{2}.
$$

The ground truth term weight $y_{t, c}$ for every word in either the query or the document are estimated in the following manner. 
- The importance of a term in a document $d$ is estimated by the occurrence of the term in relevant queries [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:deepcttermimportancedemo`]. More formally, it is given by
	$$
	QTR(t, d)=\frac{\left|Q_{d, t}\right|}{\left|Q_{d}\right|}
	$$
	$Q_{d}$ is the set of queries that are relevant to document $d$. $Q_{d, t}$ is the subset of $Q_{d}$ that contains term $t$. The intuition is that words that appear in relevant queries are more important than other words in the document.
- The importance of a term in a document $d$ is estimated by the occurrence of the term in relevant queries [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:deepcttermimportancedemo`]. More formally, it is given by
	$$
	TR(t, q)=\frac{\left|D_{q, t}\right|}{\left|D_{q}\right|}
	$$
	$D_{q}$ is the set of documents that are relevant to the query $q$. $D_{q, t}$ is the subset of relevant documents that contains term $t$. The intuition is that a query term is more important if it is mentioned by more relevant documents.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/termImportance/deepCT/deepCT_term_importance_demo
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:deepcttermimportancedemo
Illustration of calculating context-aware term importance for a query (left) and a document (right). Term importance in a query is estimated from relevant documents of the query; term importance in a document is estimated from relevant queries of the document.
```

#### Learnable Context-aware Term Importance: Deep-Impact

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

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/termImportance/deepImpact/deepImpact_term_importance_demo
:name: fig:deepimpacttermimportancedemo
DeepImpact architecture.
```

**Quantization and Query Processing.** In our approach we predict real-valued document-term scores, also called impact scores, that we store in the inverted index. Since storing a floating point value per posting would blow up the space requirements of the inverted index, we decided to store impacts in a quantized form. The quantized impact scores belong to the range of $\left[1,2^{b}-1\right]$, where $b$ is the number of bits used to store each value. We experimented with b = 8 using linear quantization, and did not notice any loss in precision w.r.t. the original scores. Since

### Contextualized Sparse Representation

#### Motivation

An important point to make here is that neural networks, particularly transformers, have not made sparse representations obsolete. Both dense and sparse learned representations clearly exploit transformers-the trick is that the latter class of techniques then "projects" the learned knowledge back into the sparse vocabulary space. This allows us to reuse decades of innovation in inverted indexes (e.g., integer coding techniques to compress inverted lists) and efficient query evaluation algorithms (e.g., smart skipping to reduce query latency): for example, the Lucene index used in our uniCOIL experiments is only $1.3 \mathrm{&nbsp;GB}$, compared to $\sim 40$ GB for COIL-tok, 26 GB for TCTColBERTv2, and 154 GB for ColBERT. We note, however, that with dense retrieval techniques, fixedwidth vectors can be approximated with binary hash codes, yielding far more compact representations with sacrificing much effectiveness (Yamada et al., 2021). Once again, no clear winner emerges at present.

The complete design space of modern information retrieval techniques requires proper accounting of the tradeoffs between output quality (effectiveness), time (query latency), and space (index size). Here, we have only focused on the first aspect. Learned representations for information retrieval are clearly the future, but the advantages and disadvantages of dense vs. sparse approaches along these dimensions are not yet fully understood. It'll be exciting to see what comes next!

#### COIL-token And Uni-COIL

The recently proposed COIL architecture (Gao et al., 2021a) presents an interesting case for this conceptual framework. Where does it belong? The authors themselves describe COIL as "a new exact lexical match retrieval architecture armed with deep LM representations". COIL produces representations for each document token that are then directly stored in the inverted index, where the term frequency usually goes in an inverted list. Although COIL is perhaps best described as the intellectual descendant of ColBERT (Khattab and Zaharia, 2020), another way to think about it within our conceptual framework is that instead of assignmodel assigns each term a vector "weight". Query evaluation in COIL involves accumulating inner products instead of scalar weights.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/ContextualizedSparseEmbedding/COIL_tok_embedding
:name: fig:coiltokembedding

```

In another interesting extension, if we reduce the token dimension of COIL to one, the model degenerates into producing scalar weights, which then becomes directly comparable to DeepCT, row (2a) and the "no-expansion" variant of DeepImpact, row (2c). These comparisons isolate the effects of different term weighting models. We dub this variant of COIL "uniCOIL", on top of which we can also add doc2query-T5, which produces a fair comparison to DeepImpact, row ( $2 \mathrm{&nbsp;d})$. The original formulation of COIL, even with a token dimension of one, is not directly amenable to retrieval using inverted indexes because weights can be negative. To address this issue, we added a ReLU operation on the output term weights of the base COIL model to force the model to generate non-negative weights. -->


## Knowledge Distillation

### Introduction

Knowledge distillation aims to transfer knowledge from a well-trained, high-performing yet cumbersome teacher model to a lightweight student model with significant performance loss. Knowledge distillation has been a widely adopted method to achieve efficient neural network architecture, thus reducing overall inference costs, including memory requirements as well as inference latency. Typically, the teacher model can be an ensemble of separately trained models or a single very large model trained with a very strong regularizer such as dropout. The student model uses the distilled knowledge from the teacher network as additional learning cues. The resulting student model is computationally inexpensive and has accuracy better than directly training it from scratch.

As such, tor retrieval and ranking systems, knowledge distillation is a desirable approach to develop efficient models to meet the high requirement on both accuracy and latency. 

For example, one can distill knowledge from a more powerful cross-encoder (e.g., BERT cross-encoder in {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:monoBERT`) to a computational efficient bi-encoders. Empirically, this two-step procedure might be more effective than directly training a bi-encoder from scratch.

In this section, we first review the principle of knowledge distillation. Then we go over a couple examples to demonstrate the application of knowledge distillation in developing retrieval and ranking models.


### Knowledge Distillation Training Framework

In the classic knowledge distillation framework {cite}`hinton2015distilling, tang2019distilling`[{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:teacherstudentdistillationscheme`], the fundamental principle is that the teacher model produces soft label $q$ for each input feature $x$. Soft label $q$ can be viewed as a softened probability vector distributed over class labels of interest. 
Soft targets contain valuable information on the rich similarity structure over the data. Use MNIST classification as an example, a reasonable soft target will tell that 2 looks more like 3 than 9. These soft targets can be viewed as a strategy to mitigate the over-confidence issue and reduce gradient variance when we train neural networks using one-hot hard labels. Similar mechanism is leveraged in smooth label to improves model generalization. 

Allows the smaller Student model to be trained on much smaller data than the original cumbersome model and with a much higher learning rate

Specifically, the logits $z$ from the techer model are outputted to generate soft labels via

$$
q_{i}^T=\frac{\exp \left(z_{i} / T\right)}{\sum_{j} \exp \left(z_{j} / T\right)},
$$

where $T$ is the temperature parameter controlling softness of the probability vector, and the sum is over the entire label space. When $T=1$, it is equivalent to standard Softmax function. As $T$ grows, $q$ become softer and approaches uniform distribution $T=\infty$. On the other hand, as $T\to 0$, the $q$ approaches a one-hot hard label. 

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/KnowledgeDistllation/teacher_student_distillation_scheme.png
:scale: 30%
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
$$L_{MSE} = ||z^{(T)} - z^{(S)}||^2$$

where $z^{(T)}$ and $z^{(S)}$ are logits from the teacher and the student network, respectively. 

````{prf:remark} connections between MSE loss and KL loss

In {cite}`hinton2015distilling`, given a single sample input feature $x$, the gradient of ${L}_{KL}$ with respect to $z_{k}^{(S)}$ is as follows:

$$
\frac{\partial {L}_{KL}}{\partial {z}_{k}^{s}}=T\left(p_{k}^{T}-{q}_{k}^{T}\right).
$$

When $T$ goes to $\infty$, using the approximation $\exp \left({z}_{k}/ T\right) \approx 1+{z}_{k} / T$, the gradient is simplified to:

$$
\frac{\partial {L}_{KL}}{\partial {z}_{k}^{(S)}} \approx T\left(\frac{1+z_{k}^{(S)} / T}{K+\sum_{j} {z}_{j}^{(S)} / T}-\frac{1+{z}_{k}^{(T)} / T}{K+\sum_{j} {z}_{j}^{(T)} / T}\right)
$$

where $K$ is the number of classes.

Here, by assuming the zero-mean teacher and student logit, i.e., $\sum_{j} {z}_{j}^{(T)}=0$ and $\sum_{j} {z}_{j}^{(S)}=0$, and hence $\frac{\partial {L}_{K L}}{\partial {z}_{k}^{(S)}} \approx \frac{1}{K}\left({z}_{k}^{(S)}-{z}_{k}^{(T)}\right)$. This indicates that minimizing ${L}_{KL}$ is equivalent to minimizing the mean squared error ${L}_{MSE}$, under a sufficiently large temperature $T$ and the zero-mean logit assumption for both the teacher and the student.
````

### Example Distillation Strategies

<!-- #### Single Cross-encoder Teacher Distillation

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/KnowledgeDistllation/cross_encoder_distillation.png
:scale: 30%
:name: fig:crossencoderdistillation

``` -->

#### Single Bi-encoder Teacher Distillation

Authors in {cite}`vakili2020distilling, lu2020twinbert` pioneered the strategy of distilling powerful BERT cross-encoder into BERT bi-encoder to retain the benefits of the two model architectures: the accuracy of cross-encoder and the efficiency of bi-encoder.  

Knowledge distillation follows the classic soft label framework. Bi-encoder student model training can use pointwise ranking loss, which is equivalent to binary relevance classification problem given a query and a candidate document. More formally, given training examples $(q_i, d_i)$ and their labels $y_i\in \{0, 1\}$. The BERT cross-encoder as teacher model to produce soft targets for irrelevance label and relevance label.

Although cross-encoder teacher can offer accurate soft labels, it cannot directly extend to the **in-batch negatives** technique and **N-pair loss** [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:N_pair_loss`] when training the student model. The reason is that query and document embedding cannot be computed separately from a cross-encoder. Implementing in-batch negatives using cross-encoder requires exhaustive computation on all combinations between a query and possible documents, which amount to $|B|^2$ ($|B|$ is the batch size) query-document pairs.

Authors in {cite}`lin2021batch` proposed to leverage bi-encoder variant such as Col-BERT [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:sec:colBERT`] can also be leveraged as a teacher model, which has the advantage that it is more feasible to perform exhaustive comparisons between queries and passages since they are passed through the encoder independently [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch::fig:inbatchdistillation`]. 

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/KnowledgeDistllation/in_batch_distillation.png
:scale: 80%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch::fig:inbatchdistillation
Compared to cross-encoder teacher, bi-encoder teacher computes query and document embeddings independents, which enables the application of the in-batch negative trick. Image from {cite}`lin2021batch`.
```

#### Ensemble Teacher Distillation

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


## Approximate Nearest Neighbor Search

### Overview

Applying dense retrieval in the first-stage of the ad-hoc retrieval system involves performing nearest neighbor search among web-scale documents in the high-dimensional embedding space. Exact nearest neighbor search is inherently expensive due to the curse of dimensionality and the large number of documents. Consider a $D$-dimensional Euclidean space $\mathbb{R}^{D}$, the problem is to find the nearest element $\mathrm{NN}(x)$, in a finite set $\mathcal{Y} \subset \mathbb{R}^{D}$ of $n$ vectors, minimizing the distance to the query vector $x \in \mathbb{R}^{D}$ is given by:

$$
\mathrm{NN}(x)=\arg \min _{y \in \mathcal{Y}} d(x, y).
$$

A brute force exhaustive distance calculation has the complexity of $\mathcal{O}(n D)$. Several multi-dimensional indexing methods, such as the popular KD-tree {cite}`friedman1977algorithm` or other branch and bound techniques, have been proposed to reduce the search time. However, nowadays the dominating approaches are approximate nearest neighbor search via vector quantization, which is the primary focus of this section. 

### Vector Quantization

#### Approximate Representation And Storage
Quantization is a technique widely used to reduce the cardinality of high dimensional representation space, in particular when the input data is real-valued. 

Formally, a **quantizer** is a function $q$ mapping a multi-dimensional vector $x \in \mathbb{R}^{D}$ to a pre-defined centroid $q(x) = c_i$, where $c_i \in \mathcal{C} = \{c_1,...,c_{k}\}$. 
The values $c_i \in \mathbb{R}^D$ are called **centroids**, and the set $\mathcal{C}$ is the *codebook* of size $k$.

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

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/ApproximateNearestNeighbor/codebook_construction.png
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:codebookconstruction
Codecook construction can be achieved using k-means algorithm to compute $K$ centroids from database vectors.
```

The **storage** for $N$ vectors now reduce to storage of their index values plus the centroids in the codebook. Each index value requires $\log_{2} K$ bits. On the other hand, storing the original vectors typically take more than $\log_2(k)$ bits.

Two important benefits to compressing the dataset are that (1) memory access times are generally the limiting factor on processing speed, and (2) sheer memory capacity can be a problem for big datasets.

In {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:codebookmemorysavingdemo`, we illustrate the storage saving by representing a $D$ dimensional vector by a codebook of 256 centroids. We only need 8-bits ($2^8 = 256$) to store a centroid id. Each vector is now replace by a 8-bit integers.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/ApproximateNearestNeighbor/codebook_memory_saving_demo.png
:scale: 30%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:codebookmemorysavingdemo
Illustration of memory saving benefits of vector quantization. A $D$-dimensional float vector is stored as its nearest centroid integer id, which only occupies $\log_2 K$ bit.  
```

#### Approximating Distances Using Quantized Codes

Given the representation choices for the query vector $x$ and the database vector $y$, we define two options in approximating the distance $d(x, y)$ [{numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:distance_compute`]. 

**Symmetric distance computation (SDC)**: both the vectors $x$ and $y$ are represented by their respective centroids $q(x)$ and $q(y)$. The distance $d(x, y)$ is approximated by the distance $\hat{d}(x, y) \triangleq d(q(x), q(y))$.

**Asymmetric distance computation (ADC)**: the database vector $y$ is represented by $q(y)$, but the query $x$ is not encoded. The distance $d(x, y)$ is approximated by the distance $\tilde{d}(x, y) \triangleq d(x, q(y))$

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/ApproximateNearestNeighbor/distance_compute.png
:scale: 30%
:name: ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:distance_compute
Illustration of the symmetric (left) and asymmetric distance (right) computation. The distance $d(x, y)$ is estimated with either the distance $d(q(x), q(y))$ (left) or the distance $d(x, q(y))$ (right). 
```

Suppose now we have a query vector $x$ and we want to find its nearest neighbors among all the $y$ in the database $\mathcal{Y}$.

There are benefits in performing symmetric distance computation. To perform symmetric distance computation, we can pre-compute a $K\times K$ table to cache the Euclidean distance between all centroids. After computing the encoding $q(x)$, we can get $d(q(x), q(y))$ by table lookup.
On the other hand, to perform asymmetric distance computation between $x$ and all $y\in \mathcal{Y}$, we can directly calculate the Euclidean distance between the query vector $x$ and centroid $q(y)$ in the codebook. 

### Product Quantization

#### From Vector Quantization To Product Quantization
Let us consider a quantizer producing 64 bits codes, i.e., which can contain $k=2^{64} \approx 1.8\times 10^{19} $ centroids. It is prohibitive run the k-means algorithm and practically impossible to store the $D \times k$ floating point values representing the $k$ centroids.

Product quantization serves as an efficient solution to address these computation and memory consumption issues in vector quantization. The key idea of product quantization is **grouping and splitting**[ {numref}`ch:neural-network-and-deep-learning:ApplicationNLP_IRSearch:fig:codebookmemorysavingdemoproductquantization`]. The input vector $x$ is split into $m$ distinct subvectors $u_{j}, 1 \leq$ $j \leq m$ of dimension $D^{*}=D / m$, where $D$ is a multiple of $m$. The subvectors are then quantized separately using $m$ distinct quantizers. A given vector $x$ is therefore quantized as follows:

$$\underbrace{x_{1}, \ldots, x_{D^{*}}}_{u_{1}(x)}, \ldots, \underbrace{x_{D-D^{*}+1}, \ldots, x_{D}}_{u_{m}(x)} \rightarrow q_{1}\left(u_{1}(x)\right), \ldots, q_{m}\left(u_{m}(x)\right),$$

where $q_{j}$ is a quantizer used to quantize the $j^{\text {th }}$ subvector using the codebook $\mathcal{C}_{j} = \{c_{j,1},...,c_{j,k^*}\}$. Here we assume that
all subquantizers have the same finite number $k^*$ centroids.

```{figure} ../img/chapter_application_IR/ApplicationIRSearch/ApproximateNearestNeighbor/codebook_memory_saving_demo_product_quantization.png
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
<!-- 
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

\fi -->

#### Approximating Distances Using Quantized Codes

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

### Approximate Non-exhaustive Nearest Neighbor Search

#### Hierarchical Quantization And Inverted File Indexing

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

````{prf:remark} shared product quantizer for residuals
The product quantizer can be learned on a set of residual vectors. Ideally, we can learn a product quantizer for each partition since the residual vectors likely to be dependent on the coarse quantizer. One can further reduce memory cost significantly by using the same product quantizer across all coarse quantizers, although this probably gives inferior results
````

## Benchmark Datasets

### MS MARCO

[MS MARCO](https://microsoft.github.io/msmarco/Datasets) (Microsoft MAchine Reading Comprehension) {cite}`nguyen2016ms` is a large scale dataset widely used to train and evaluate models for the document retrieval and ranking tasks as well as tasks like key phrase extraction for question
answering. MS MARCO dataset is sampled from Bing search engine user logs, with Bing retrieved passages given queries and human annotated relevance labels. There are more than 530,000 questions in the "train" data partition, and the evaluation is usually performed on around 6,800 questions in the "dev" and "eval" data partition. The ground-truth labels for the "eval" partition are not published. The original data set contains more than $8.8$ million passages.

There are two tasks: Passage ranking and document ranking; and two subtasks in each case: full ranking and re-ranking.

Each task uses a large human-generated set of training labels. The two tasks have different sets of test queries. Both tasks use similar form of training data with usually one positive training document/passage per training query. In the case of passage ranking, there is a direct human label that says the passage can be used to answer the query, whereas for training the document ranking task we transfer the same passage-level labels to document-level labels. Participants can also use external corpora for large scale language model pretraining, or adapt algorithms built for one task (e.g. passage ranking) to the other task (e.g. document ranking). This allows participants to study a variety of transfer learning strategies.

#### Document Ranking Task
The first task focuses on document ranking. We have two subtasks related to this: Full ranking and top-100 re-ranking.

In the full ranking (retrieval) subtask, you are expected to rank documents based on their relevance to the query, where documents can be retrieved from the full document collection provided. You can submit up to 100 documents for this task. It models a scenario where you are building an end-to-end retrieval system.

In the re-ranking subtask, we provide you with an initial ranking of 100 documents from a simple IR system, and you are expected to re-rank the documents in terms of their relevance to the question. This is a very common real-world scenario, since many end-to-end systems are implemented as retrieval followed by top-k re-ranking. The re-ranking subtask allows participants to focus on re-ranking only, without needing to implement an end-to-end system. It also makes those re-ranking runs more comparable, because they all start from the same set of 100 candidates.
<!-- 
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
\end{table} -->

#### Passage Ranking Task

Similar to the document ranking task, the passage ranking task also has a full ranking and reranking subtasks.

In context of full ranking (retrieval) subtask, given a question, you are expected to rank passages from the full collection in terms of their likelihood of containing an answer to the question. You can submit up to 1,000 passages for this end-to-end retrieval task.

In context of top-1000 reranking subtask, we provide you with an initial ranking of 1000 passages and you are expected to rerank these passages based on their likelihood of containing an answer to the question. In this subtask, we can compare different reranking methods based on the same initial set of 1000 candidates, with the same rationale as described for the document reranking subtask.

One caveat of the MSMARCO collection is that only contains binary annotations for fewer than two positive examples per query, and no explicit annotations for non-relevant passages. During the reranking task, negative examples are generated from the top candidates of a traditional retrieval system. This approach works reasonably well, but accidentally picking relevant passages as negative examples is possible.
<!-- 
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
\end{table} -->

### TERC

#### TREC-deep Learning Track

Deep Learning Track at the Text REtrieval Conferences (TRECs) deep learning track<sup>[^7]</sup>{cite:p}`craswell2020overview` is another large scale dataset used to evaluate retrieval and ranking model through two tasks: Document retrieval and passage retrieval. 

Both tasks use a large human-generated set of training labels, from the MS-MARCO dataset. The document retrieval task has a corpus of 3.2 million documents with 367 thousand training queries, and there are a test set of 43 queries. The passage retrieval task has a corpus of 8.8 million passages with 503 thousand training queries, and there are a test set of 43 queries.

#### TREC-CAR

TREC-CAR (Complex Answer Retrieval) {cite}`dietz2017trec` is a dataset where the input query is the concatenation of a Wikipedia article title with the title of one of its sections. The ground-truth documents are the paragraphs within that section. The corpus consists of all English Wikipedia paragraphs except the abstracts. The released dataset has five predefined folds, and we use the first four as a training set (approx. 3M queries), and the remaining as a validation set (approx. 700k queries). The test set has approx. 2,250 queries.


### Natural Question (NQ)
Natural Question (NQ) {cite}`kwiatkowski2019natural` introduces a large dataset for open-domain QA. The original dataset contains more than 300,000 questions collected from Google search logs. In {cite}`karpukhin2020dense`, around 62,000 factoid questions are selected, and all the Wikipedia articles are processed as the collection of passages. There are more than 21 million passages in the corpus. 

## Note On Bibliography And Software

### Bibliography
For excellent reviews in neural information retrieval, see {cite}`guo2020deep, mitra2018introduction, lin2021pretrained`

For traditional information retrieval, see {cite}`schutze2008introduction, buttcher2016information, robertson2009probabilistic, croft2010search`

### Software

[Faiss](https://github.com/facebookresearch/faiss/wiki/) is a recently developed computational library for efficient similarity search and clustering of dense vectors. 

<!-- \printbibliography
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

[^7]: \url{https://microsoft.github.io/msmarco/TREC-Deep-Learning.html} -->

