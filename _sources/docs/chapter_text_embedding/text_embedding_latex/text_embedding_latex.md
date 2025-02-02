## Sentence embeddings and its applications

### Introduction

There are many NLP tasks involves determining the relationship of two sentences, including semantic similarity, semantic relation reasoning, questioning answering etc. For example, Quora needs to determine if a question asked by a user has a semantically similar duplicate. The GLUE benchmark as an example [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP:sec:BERTDownstreamTasks}], 6 of them are tasks that require learning sentences Inter-relationship. Specifically,

**MRPC**: The Microsoft Research Paraphrase Corpus {cite}`dolan2005automatically` is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.\\

**QQP**: The Quora Question Pairs<sup>[^1]</sup> dataset is a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent. As in MRPC, the class distribution in $\mathrm{QQP}$ is unbalanced $(63 \%$ negative), so we report both accuracy and F1 score. We use the standard test set, for which we obtained private labels from the authors. We observe that the test set has a different label distribution than the training set. \\

**STS-B**: The Semantic Textual Similarity Benchmark {cite}`cer2017semeval` is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data.

**Natural language inference**: Understanding entailment and contradiction is fundamental to understanding natural language, and inference about entailment and contradiction is a valuable testing ground for the development of semantic representations. The semantic concepts of entailment and contradiction are central to all aspects of natural language meaning,
from the lexicon to the content of entire texts.
Natural language inference is the task of determining whether a *hypothesis* is true (entailment), false (contradiction), or undetermined (neutral) given a *premise*.
\cite{bowman2015large}
\begin{table}
	\scriptsize
	\centering
	\begin{tabular}{c|c|c}
		\hline
		Premise & Label & Hypothesis \\ \hline
		\begin{tabular}[c]{@{}l@{}}A man inspects the uniform of a figure\\ in some East Asian country.\end{tabular}
		& Contraction & The man is sleeping. \\
		An older and younger man smiling. &
		Neutral &
		\begin{tabular}[c]{@{}l@{}}Two men are smiling and laughing at \\ the cats playing on the floor.\end{tabular}
		\\	
		A soccer game with multiple males playing. &
		Entailment & Some men are playing a sport. \\ \hline
	\end{tabular}	
	\caption{*Entailment*, *contradiction*, and *neural* examples in a natural language inference task.}
\end{table}

Although BERT model and its variant have achieved new state-of-the-art among many sentence-pair classification and regression tasks. It has many practical challenges in tasks like large-scale semantic similarity comparison, clustering, and information retrieval via semantic search, etc. These tasks require that both sentences are fed into the network, which causes a massive computational overhead for large BERT model. Considering the task of finding the most similar pair among $N$ sentences, then it requires $N^2/2$ forward pass computation of BERT. 

An alternative approach is to derive a semantically meaningful sentence embeddings for each sentence. A sentence embedding is a dense vector representation of a sentence. Sentences with similar semantic meanings are close and sentences with different meanings are apart. With sentence embeddings, similarity search can be realized simply via a distance or similarity metrics, such as cosine similarity. Besides performing similarity comparison between two sentences, the sentence embedding can also be used as generic sentence feature for different NLP tasks.

Early on, there have been efforts directed to derive sentence embedding by aggregating word embeddings. For example, one can average the BERT token embedding output as the sentence embedding. Another common practice is to use the output of the first token (the [CLS] token) as the sentence, which is found to be worse than averaging GloVe embeddings\cite{reimers2019sentence}. Recently, contrastive learning and Siamese type of learning have been successfully applied in deriving sentence embeddings, which will be the focus of our following sections. 

\begin{remark}[Why we need sentence embedding fine tuning]

Pre-trained BERT models do not produce efficient and independent sentence embeddings as they always need to be fine-tuned in an end-to-end supervised setting. This is because we can think of a pre-trained BERT model as an indivisible whole and semantics is spread across all layers, not just the final layer. Without fine-tuning, it may be ineffective to use its internal representations independently. It is also hard to handle unsupervised tasks such 
as clustering, topic modeling, information retrieval, or semantic search. Because we have to evaluate many sentence pairs during clustering tasks, for instance, this causes massive computational overhead.
\end{remark}

### InferSent
InferSent is a sentence embeddings training method invented by Facebook AI\cite{conneau2017supervised}. Sentence encoder is trained on natural language inference data\cite{bowman2015large} and can provide semantic sentence representations generalizing well to many different tasks.

InferSent uses the Siamese type of learning scheme [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP:fig:infersent}]. A sentence pair are encoded by sentence encoders into separate sentence representations. Sentence representations are then concatenated and combined before being fed into a linear Softmax classifier with three labels: entailment, contradiction, and neutral.   

In the inference stage, we can compute and store sentence embedding offline for large scale semantic search tasks. 
```{figure} images/../deepLearning/ApplicationsNLP/textMatching/sentenceEmbedding/inferSent
:name: ch:neural-network-and-deep-learning:ApplicationNLP:fig:infersent
Siamese type of training scheme to learning sentence embedding from natural language inference task. The sentence encoder is bi-directional LSTM architecture with max pooling and it takes 300 dimension of Glove word embeddings of constituents as the input.
```

InferSent represents a new paradigm of solving sentence-pair related tasks: each sentence has an representation derived separately, and the representation is used to solve downstream tasks. This contrasts BERT approach, which is a joint method that uses cross-features
or attention from one sentence to the other.

\iffalse
Sentence level classification tasks
\begin{table}
	\centering
	\scriptsize
	\begin{tabular}{l|r|l|l|l|c} 
		name & $\mathbf{N}$ & task & C & examples & label(s) \\
		\hline MR & $11 \mathrm{k}$ & sentiment (movies) & 2 & "Too slow for a younger crowd, too shallow for an older one." & neg \\
		CR & $4 \mathrm{k}$ & product reviews & 2 & "We tried it out christmas night and it worked great $. "$ & pos \\
		SUBJ & $10 \mathrm{k}$ & subjectivity/objectivity & 2 & "A movie that doesn't aim too high, but doesn't need to." & subj \\
		MPQA & $11 \mathrm{k}$ & opinion polarity & 2 & "don't want", "would like to tell"; & neg, pos \\
		TREC & $6 \mathrm{k}$ & question-type & 6 & "What are the twin cities ?" & LOC:city \\
		SST-2 & $70 \mathrm{k}$ & sentiment (movies) & 2 & "Audrey Tautou has a knack for picking roles that magnify her $[. .] "$ & pos \\
		SST-5 & $12 \mathrm{k}$ & sentiment (movies) & 5 & "nothing about this movie works." & 0
	\end{tabular}
\end{table}

The sentence evaluation tool<sup>[^2]</sup> {cite}`conneau2018senteval`

Our aim is to obtain general-purpose sentence embeddings that capture generic information that is
\fi

### Sentence-BERT

While BERT has demonstrated remarkable performance across NLP tasks, a large disadvantage of the BERT approach is that no independent sentence embeddings are computed, which makes it difficult to derive sentence embeddings from BERT. 

Sentence-BERT enables sentence embedding extraction via Siamese type of learning like InferSent. The major difference between the InferSent and Sentence-BERT is that Sentence-BERT utilizes BERT to encode the sentence. Specifically, three strategies can be used to derived sentence embedding
-  the output of the CLS-token
-  the mean of all output vectors 
-  max-over-time of the output vectors 

```{figure} images/../deepLearning/ApplicationsNLP/textMatching/sentenceEmbedding/sentenceBERT
:name: ch:neural-network-and-deep-learning:ApplicationNLP:fig:sentencebert
(**left**) Sentence-BERT architecture with classification objective function for training. The two BERT networks are pre-trained have tied weights (Siamese network structure). (**right**) Sentence-BERT architecture at inference. The similarity between two input sentences can be computed as the similarity score between two sentence embeddings.
```

### Universal Sentence Encoder

\cite{cer2018universal}

### Contrastive learning

#### Contrastive learning via data augmentation

\cite{shen2020simple}

```{figure} images/../deepLearning/ApplicationsNLP/sentence_representation/data_augmented_contrastive_learning/data_cut_off_demo
:name: fig:datacutoffdemo
Figure 1: Schematic illustration of the proposed cutoff augmentation strategies, including token cutoff, feature
		cutoff and span cutoff, respectively. Blue area indicates that the corresponding elements within the sentence’s
		input embedding matrix are removed and converted to 0. Notably, this is distinct from Dropout, which randomly
		transforms elements to 0 (without considering any underlying structure of the matrix).
```

#### SimCSE

\cite{gao2021simcse}

Simple contrastive learning of sentence embeddings

Unsupervised SimCSE

Positive example pair: passing one sentence passing through the encoder network with different dropout masking

Negative example: in batch negatives

Contrastive learning aims to learn effective representation by pulling semantically close neighbors together and pushing apart non-neighbors (Hadsell et al., 2006). It assumes a set of paired examples $\mathcal{D}=\left\{\left(x_{i}, x_{i}^{+}\right)\right\}_{i=1}^{m}$, where $x_{i}$ and $x_{i}^{+}$are semantically related. We follow the contrastive framework in Chen et al. (2020) and take a cross-entropy objective with in-batch negatives (Chen et al., 2017 ; Henderson et al., 2017): let $\mathbf{h}_{i}$ and $\mathbf{h}_{i}^{+}$denote the representations of $x_{i}$ and $x_{i}^{+}$, the training objective for $\left(x_{i}, x_{i}^{+}\right)$with a mini-batch of $N$ pairs is:
$$
\ell_{i}=-\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{i}^{+}\right) / \tau}}{\sum_{j=1}^{N} e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{+}\right) / \tau}}
$$
where $\tau$ is a temperature hyperparameter and $\operatorname{sim}\left(\mathbf{h}_{1}, \mathbf{h}_{2}\right)$ is the cosine similarity $\frac{\mathbf{h}_{1}^{\top} \mathbf{h}_{2}}{\left\|\mathbf{h}_{1}\right\| \cdot\left\|\mathbf{h}_{2}\right\|} .$ In this work, we encode input sentences using a pre-trained language model such as BERT (Devlin et al., 2019) or RoBERTa (Liu et al., 2019): $\mathbf{h}=f_{\theta}(x)$, and then fine-tune all the parameters using the contrastive learning objective (Eq. 1).

**supervised contrastive learning with hard negatives**

Contradiction as hard negatives. Finally, we further take the advantage of the NLI datasets by using its contradiction pairs as hard negatives ${ }^{6}$. In NLI datasets, given one premise, annotators are required to manually write one sentence that is absolutely true (entailment), one that might be true (neutral), and one that is definitely false (contradiction). Therefore, for each premise and its entailment hypothesis, there is an accompanying contradiction hypothesis ${ }^{7}$ (see Figure 1 for an example).

Formally, we extend $\left(x_{i}, x_{i}^{+}\right)$to $\left(x_{i}, x_{i}^{+}, x_{i}^{-}\right)$, where $x_{i}$ is the premise, $x_{i}^{+}$and $x_{i}^{-}$are entailment and contradiction hypotheses. The training objec-
tive $\ell_{i}$ is then defined by $(N$ is mini-batch size 
$$-\log \frac{e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{i}^{+}\right) / \tau}}{\sum_{j=1}^{N}\left(e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{+}\right) / \tau}+e^{\operatorname{sim}\left(\mathbf{h}_{i}, \mathbf{h}_{j}^{-}\right) / \tau}\right)}$$
As shown in Table 4 , adding hard negatives can further improve performance $(84.9 \rightarrow 86.2)$ and

#### CERT

\cite{fang2020cert}

CERT: Contrastive self-supervised Encoder Representations
from Transformers, which pretrains language representation models using contrastive selfsupervised learning at the sentence level. CERT creates augmentations of original sentences
using back-translation. Then it finetunes a pretrained language encoder (e.g., BERT) by
predicting whether two augmented sentences originate from the same sentence.

CERT takes a pretrained language representation model (e.g., BERT) and finetunes it using
contrastive self-supervised learning on the input data of the target task.

```{figure} images/../deepLearning/ApplicationsNLP/sentence_representation/CERT/CERT_workflow
:name: fig:certworkflow
The workflow of CERT. Given the large-scale input texts (without labels) from
		source tasks, a BERT model is first pretrained on these texts. Then we continue
		to train this pretrained BERT model using CSSL on the input texts (without
		labels) from the target task. We refer to this model as pretrained CERT model.
		Then we finetune the CERT model using the input texts and their associated
		labels in the target task and get the final model that performs the target task.
```

### Benchmark dataset
#### MultiNLI data

\cite{williams2017broad}

The Multi-Genre Natural Language Inference (MultiNLI) corpus <sup>[^3]</sup> is a dataset designed for use in the development and evaluation of machine learning models for sentence understanding. It has over 433,000 examples and is one of the largest datasets available for natural language inference (a.k.a recognizing textual entailment). The dataset is also designed so that existing machine learning models trained on the Stanford NLI corpus can also be evaluated using MultiNLI. 

#### SNLI

\cite{bowman2015large}

#### Standard semantic textual similarity tasks

\cite{ agirre2012semeval, agirre2013sem, agirre2014semeval, agirre2015semeval, agirre2016semeval, cer2017semeval}

\cite{marelli2014sick}

#### QQP

QQP. Quora Question Pairs (QQP)<sup>[^4]</sup>; consists of pairs of potentially duplicate questions collected from Quora, a
question-and-answer website. The binary label of
each question pair indicates redundancy.

## Software

[haystack](https://github.com/deepset-ai/haystack)

[^1]: \url{https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs}

[^2]: \url{https://github.com/facebookresearch/SentEval}

[^3]: \url{https://cims.nyu.edu/&nbsp;sbowman/multinli/}

[^4]: \url{https://www.kaggle.com/sambit7/first-quora-dataset}

