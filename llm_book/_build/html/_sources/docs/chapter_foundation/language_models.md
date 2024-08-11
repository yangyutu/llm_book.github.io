# Language Models

## Motivation 

Natural languages emerge from formal or casual communications between human beings and only have a limited set of formal rules to follow. Linguists have been directing decades' efforts to modeling languages via grammars, rules, and structure of natural language. In NLP, language modeling \cite{goldberg2017neural} tasks involve the the use of various statistical and probabilistic techniques to determine the probability of a given sequence of words forming a sentence that make sense [{numref}`chapter_foundation_fig_languagemodelingtaskdemo`]. 

```{figure} ../img/chapter_foundation/languageModeling/languageModelingTask_demo.png
---
scale: 25%
name: chapter_foundation_fig_languagemodelingtaskdemo
---
Illustrating basic language modeling tasks: assigning probability to a sentence (left); and predict the next word based on preceding context (right).
```

Language modeling is an indispensable components in real-world applications. A language model can be used directly to generate new sequences of text that appear to have come from the corpus. For example, an AI-writer can generate an article with given key words. Moreover, language models are an integral part of many text-related applications, where they ensure output word sequences in these tasks to have a sensible meaning or to appear fluent like genuine human languages. These tasks include
* Optical character Recognition, a task converting images to sentences.
*  Machine translation, where one sentence from one language is translated to a sentence in another language.
*  Image captioning, where a image is used as the input and the output is a descriptive sentence.
*  Text summarization, where a large paragraph or a full article is converted several summarizing sentences.
*  Speech recognition, where audio signal is converted to meaningful words, phases, and sentences.




A closed related language modeling task is predict the next word following a sequence of words.

Predicting the next word in a language system can be much more challenging than predicting the next observation in the many physical system. One perspective is that the evolution of many physical systems can governed by physical laws that are well established; on the other hand, the generation of next word, sentence, and even logically coherent paragraph is closely related to human reasoning and intelligence, which remain poorly understood. In fact, it is widely believed that being able to write up logical and coherent paragraphs is an indication of human-level intelligence {cite:p}`radford2019language`.




In this section, we first consider a classical $n$-gram statistical language approach {cite:p}`manning1999foundations`, where we count the co-occurrence of words and model the language generation using probabilistic models (e.g., Markov chains). $n$-gram models focus on predicting next word based on the superficial co-occurrence counts instead of high-level semantic links between the context and the unknown next word,  It further suffers from the curse of dimensionality since sentences are normally long and language vocabulary size is huge (e.g., $10^6$). Over the past two decades, the neural network based language models have attracted considerable attention because these models are able to reveal deep connections between words as well as to alleviate the dimensionality challenge. Our focus in this section is on early development of neural language models; recent developments of pre-trained language models (e.g., GPT 1-3 {cite:p}`radford2018improving, radford2019language, brown2020language`) will be covered in next section.

## Statistical language models

### $n$-gram language model

The essential goal of language model is to estimate the joint probability of word sequence $w_{1:N} = w_1,...,w_{N}$. By definition, the unbiased probability estimation is given by

$$P\left(w_{1:N}\right) = \frac{\operatorname{count}(w_{1:N})}{\operatorname{count}(\text{all word sequence ever exists in the universe})}.$$

Estimating the probability has the following challenge:
* Given a vocabulary size $V$, the number of events $w_{1:N}$ is exponentially increasing with $N$. As $N$ can take arbitrarily large value, it would not be possible even to store the model.
* Because of the intractably large number of events, getting reliable estimation of the denominator and the nominator requires impractically large data. 
  
Alternatively, one can achieve the joint probability estimation via a **truncated conditional probability framework**. To begin with, the joint probability the sequence $w_{1:N}$ has the following decomposition based on the conditional probability **chain rule**:

$$P(w_{1:N}) = P(w_{N} | w_{1:N-1}) P(w_{N-1} | w_{1:N-2}) \cdots P(w_{3} | w_{1}, w_{2})P(w_{2} | w_{1}) P(w_{1}).$$

The language modeling task involves the estimation $P\left(w_1,...,w_N\right)$. The task is equivalent to estimate the conditional probability of predicting $w_N$ based on $w_1,...,w_{N-1}$ thanks to

$$p(w_N|w_1,...,w_{N-1}) = \frac{w_1,...,w_{N}}{w_1,...,w_{N-1}}.$$

In the truncated-$n$ conditional probability framework, we approximate $P(w_{1:N})$ via

$$P(w_{1:N}) \approx P(w_{N} | w_{N-n:N-1}) P(w_{N-1} | w_{N-1-n:N-2}) \cdots P(w_{3} | w_{1}, w_{2})P(w_{2} | w_{1}) P(w_{1}).$$

The truncated-$n$ conditional probability framework is formally known as $n$-gram statistical language model. The $n$-gram statistical language model is one fundamental model that estimates probabilities by counting the co-occurrence of $n$ consecutive words. More precisely, an $n$-gram is a sequence consisting of $n$ words. For example, given a sentence *The homework is due tomorrow*, it has:
* 2-grams (or bigrams) like *the homework, homework is, is due, due tomorrow*;
* 3-grams (or trigrams) like *the homework is, homework is due, is due tomorrow*;
* 4-grams like *the homework is due, homework is due tomorrow*.

Often we pad the sequence in the front by $n-1$ <SOS> (i.e., $w_{-1}, w_{-2}, w_{-n + 1} = $ <SOS>) such that we can write

$$P(w_{1:N}) \approx \prod_{k=1}^{N} P(w_{k} | w_{k-n+1:k-1}).$$

For example, in the bigram language model, we have

$$
P\left(w_{N} | w_{1}^{N-1}\right) \approx P\left(w_{N} | w_{N-1}\right)
$$

and therefore

$$
P\left(w_{1}^{N}\right) \approx \prod_{k=1}^{N} P\left(w_{k} | w_{k-1}\right).
$$

### Model parameter estimation

For a $n$-gram language model, there are $|V|^n$ the model parameters to be estimated: 

$$P(w_n|w_1,...,w_{n-1}),$$

Here $|V|$ is the vocabulary size.

Given a training corpus, the MLE (maximum likelihood estimator) of $n$-gram conditional probability used above can be simply obtained by counting. Specifically, for a bigram model, we have

$$
P\left(w_{n} | w_{n-1}\right)=\frac{\operatorname{count}\left(w_{n-1}, w_{n}\right)}{\sum_{w} \operatorname{count}\left(w_{n-1}, w\right)} = \frac{\operatorname{count}\left(w_{n-1}, w_{n}\right)}{\operatorname{count}\left(w_{n-1}\right)}
$$

where $\operatorname{count}(x, y)$ is the total count of bigrams $x,y$ in the corpus and $\sum_{w} \operatorname{count}\left(w_{n-1}, w\right)$ is the count of all bigrams starting with $w_{n-1}$. 

For a trigram model, we have

$$P\left(w_{n} | w_{n-1}, w_{n-2}\right)=\frac{\operatorname{count}\left(w_{n-2}, w_{n-1}, w_{n}\right)}{\sum_{w\in |V|}(\operatorname{count}\left(w_{n-2}, w_{n-1}, w\right))}=\frac{\operatorname{count}\left(w_{n-2}, w_{n-1}, w_{n}\right)}{\operatorname{count}\left(w_{n-2}, w_{n-1}\right)}.$$

For example,

$$p(\text { is| artificial intelligence })=\frac{\operatorname{count}(\text { artificial intelligence is})}{\operatorname{count}(\text { artificial intelligence })}.$$


#### $\star$ Deriving the MLE

Consider a sequence of random variables $w_0, w_1, ...,w_N$ representing a sentence composed of a sequence of tokens taken from the vocabulary $|V|$. Let $w_0$ be deterministic and take the value of starting token <SOS>. For a bigram language model, the model is completely defined by the conditional probability matrix $\theta \in \mathcal{R}^{|V|\times |V|}$, with $P(w_n = j|w_{n-1} = i) = \theta_{ij}$. Here $i$ is shorthand for the $i$ token in the vocabulary.
 
To estimate the model parameter, we look at the log likelihood of a sample sequence $w_0, w_1,...,w_N$, which is is given by

$$L = \sum_{n=1}^N \ln P(w_n|w_{n-1}) = \sum_{i \in |V|}\sum_{j \in |V|} \operatorname{count}(i, j) \ln  P(j|i) = \sum_{i \in |V|}\sum_{j \in |V|} \operatorname{count}(i, j) \ln \theta_{i,j},$$

Here $\operatorname{count}(i, j)$ is the count of bigram $i,j$ in the training corpus. 

Because $\theta_{ij}$ is under additional constraints given by

$$\sum_{j\in |V|}\theta_{ij} = 1, i=1,...,|V|,$$

we can use Lagrange multiplier to maximize the log likelihood under constraints. The Lagrange is given by

$$L = \sum_{i \in |V|}\sum_{j \in |V|} \operatorname{count}(i, j) \ln \theta_{i,j} - \sum_{i\in |V|}\lambda_i (\sum_{j\in |V|}\theta_{ij} - 1) = 0.$$ 

Set $\partial L / \partial \theta_{ij} = 0$, we get 

$$\frac{\operatorname{count}(i, j)}{\theta_{i,j}} = \lambda_i \Rightarrow \theta_{i,j} = \frac{\operatorname{count}(i, j)}{\lambda_i}.$$

One can easily show that $$\lambda_i = \sum_{j\in |V|} \operatorname{count}(i, j) = \operatorname{count}(i),$$
therefore, 

$$\theta_{ij} = \frac{\operatorname{count}(i, j)}{\operatorname{count}(i)}.$$



\subsubsection{Special case: unigram language model}

In the unigram language model, we assume 
$$P(w_1, w_2,...,w_N) = \prod_{i=1}^N P(w_i).$$

We can empirically estimate 
$$P(w_i) = \frac{\operatorname{count}(w_i)}{\sum_{j=1}^{|V|}\operatorname{count}(w_j)} = \frac{\operatorname{count}(w_i)}{N_{|V|}},$$

where $\operatorname{count}(w_i)$ is the number of $w_i$ in the corpus, $|V|$ is the vocabulary size, and $N_{|V|}$ is the total number of words in the corpus.

A unigram language model does not capture any transitional relationship between words. When we use a unigram language model to generate sentences, the resulted sentences would mostly consist of high-frequent words and thus hardly make sense. 

### Choices of $n$ and bias-variance trade-off



For a $n$-gram model, when $n$ is too small, say a bigram model, the model can capture synactic structure in the language but can hardly capture the sequential, long distance relationship among words. For example, the syntatic structure like the fact that a noun or an adjective comes after \textit{enjoy} and the fact that a verb in its original form typically comes after \textit{to}. For long-range dependency, consider the following sentences:
* Gorillas always like to groom their friends.
* The computer that's on the 3rd floor of our office building crashed.

In each example, the words written in bold depend on each other: the likelihood of their depends on knowing that gorillas is plural, and the likelihood of crashed depends on knowing that the subject is a computer. If the $n$-grams are not big enough to capture this context, then the resulting language model would offer probabilities that are too low for these sentences, and too high for sentences that fail basic linguistic tests like number agreement.

Typically, the longer the context we condition on in predicting the next word, the more coherent the sentences. But a language model with a large $n$ can overfit to small-sized training corpus as it tends to memorize specific long sequence patterns in the training corpus and to give zero probabilities to those unseen. Specifically, a $n$-gram model has model parameter scales like $O(V^n)$, where $V$ is the vocabulary size. 

The choice of appropriate $n$ is related to bias-variance trade-off. A small $n$-gram size introduces high bias, and a large $n$-gram size introduces high variance. Since human language is full of long-range dependencies, in practice, we tend to keep $n$ large and at the same time use smoothing tricks, as some sort of bias, to achieve low-variance estimates of the model parameters. 

### Out of vocabulary (OOV) words and rare words

Some early speech and language applications might just involve a closed vocabulary, in which the vocabulary is known in advance and the runtime test set will contain words from the vocabulary. Modern natural language application typically involves an open vocabulary in which out of vocabulary words can occur. For those cases, which can simply add a pseudo-word <UNK>, and treat all the OOV words as the <UNK>. 

In applications where we don't have a prior vocabulary in advance and need to create one from corpus, we can limit the size of vocabulary by replacing low-frequency rare words by <UNK> or some other more fine-grained special symbols (i.e., replacing rare organization names by <ORG> and replacing rare people names by <PEOPLE>).

Note that how the vocabulary is constructed and how we map rare words to <UNK> affects how we evaluate the quality of language model using likelihood based metrics (e.g., perplexity). For example, a language model can achieve artificially low perplexity by choosing a small vocabulary and assigning many words to the unknown word.

## Smoothing and discounting techniques
### Add $\alpha$ smoothing and discounting
One fundamental limit of the baseline counting method is the zero probability assigned to $n$-grams that do not appear in the training corpus. This has a significant impact when we use the model to score a sentence. For example,  one zero probability of any unseen bigram $w_{i-1},w_i$ will cause the probability of the sequence $w_1,...,w_i,...,w_n$ to be zero. 

````{prf:example}
Suppose in the training corpus we have only seen following phrases starting with *denied*: *denied the allegations*, *denied the speculation, *denied the rumors*, *denied the report*. Now suppose a sentence in the test set has the 
phrase *denied the offer*. Our model will incorrectly estimate that <span style="color:red">**the probability of the sentence is 0**</span>, which is obviously not true.

````

## Neural language models


## Bibliography

```{bibliography} ../../_bibliography/references.bib
```