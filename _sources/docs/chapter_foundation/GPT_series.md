# GPT Series


## GPT-1

### Introduction

GPT-1\cite{radford2018improving} and its successors (GPT-2 and GPT-3) are a series Transformer-based model architectures aim to offer generic task-agnostic model architecture and learning schemes to diverse natural language processing (NLP) tasks, such as textual entailment, question answering, semantic similarity assessment. 

At that time, specific NLP tasks requires the collection of a massive amount of task-specific labeled data as well as the design of task-specific architectures that learns best from it. Given the broad range of NLP tasks, this paradigm does not scale well and model learning cannot be shared across different tasks. 

One of the major contributions of the GPT-1 study is the introduction of a two-stage unsupervised pretraining and supervised fine-tuning scheme. They demonstrates that a pre-trained model with fine-tuning can achieve satisfactory results over a range of diverse tasks, not just for a single task. 

GPT-1 model consists of multiple transformer decoder layers [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:fig:gptdecoderarch}]. Since there is no encoder, each decoder layer contains a masked multi-head self-attention layer along with a pointwise feed-forward layer. The pretraining task is generative language modeling, that it, predicting the next word given the preceding word sequence. In the Transformer architecture, the activation in the final transformer block is fed into a Softmax function that produces the word probability distributions over an entire vocabulary of words to predict the next word.

In the stage unsupervised pretraining from unlabeled data, the goal is to learn a universal representation that can be easily adapted to a wide range of tasks. Following the pretraining, the model can be easily fine-tuned to a downstream task by a relatively small amount of task-specific data to achieve effective transfer. This two-stage scheme had a profound impact on the subsequent deep model learning and drew significant interest to pretrained large language models. 


\begin{figure}[H]
	\centering
	\includegraphics[width=0.7\linewidth]{../figures/deepLearning/ApplicationsNLP/pretrainedLM/GPT/GPT_decoder_arch}
	\caption{GPT uses the decoder component in the Transformer for language modeling.}
	\label{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:fig:gptdecoderarch}
\end{figure}

### Pretraining

The pretraining task of GPT-1 is generative language modeling, which predict the next words given preceding word sequence. Given an input sequence $\bm{x} = (x_1,...,x_T)$, generative language model maximize the log likelihood given by

$$\sum_{t=1}^{T} \log p\left(x_{t} \mid \mathbf{x}_{t-k-1:t-1},\theta\right)$$

where $p\left(x_{t} \mid \mathbf{x}_{t-k-1:t-1}\right)$ is the predicted probability distribution for token $x_t$ contextualized over preceding token sequence $\bm{x}_{t-k-1:t-1}$ with a context window size $k$.

The input tokens are first converted to input embeddings by summing up token embedding and position embedding. The input embedding $h_0$ is fed into multiple Transformer layers to obtain contextualized hidden state $h$s a multi-headed self-attention operation over the input context tokens followed by position-wise feedforward layers to produce an output distribution over target tokens:

$$\begin{align}
	h_{0} &=U W_{e}+W_{p} \\
	h_{l} &=\operatorname{TransformerLayer}\left(h_{l-1}\right) \forall \ell \in[1, L] \\
	P(u) &=\operatorname{Softmax}\left(h_{n} W_{e}^{T}\right)
\end{align}
$$

where $U=\left(u_{-k}, \ldots, u_{-1}\right)$ is the context vector of tokens, $n$ is the number of layers, $W_{e}$ is the token embedding matrix, and $W_{p}$ is the position embedding matrix.

GPT-1 uses the BooksCorpus dataset for pretraining. BooksCorpus is a large collection of free novel books (11,038 books), containing around 74M sentences and 1G words in 16 different sub-genres (e.g., Romance, Historical, Adventure, etc.). Pretrained GPT-1 can achieve a very low token level perplexity of 18.4 on this corpus.


### GPT-1 Fine Tuning

To pretrained GPT model can be adopted for different downstream tasks by modifying the inputs format or adding minimal component accordingly.  a task-specific format and then adding minimal component to process the output to get task-specific predictions. As summarized in \autoref{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:fig:gptarch},
\begin{itemize}
	\item For a single-sequence task such as text classification, the input is passed through the network as-is, and the output linear layer takes the last activation to make a classification decision.
	\item For sentence-pair tasks such as textual entailment, the input that is made up of two sequences is marked with a delimiter, which helps the pre-trained model to know which part is premise or hypothesis in the case of textual entailment. Finally, the output linear layer takes the last activation to make a classification decision.
	\item For sentence similarity tasks, we use the model to encode the two differently-ordered sentence pairs separately into two sequence representations, which are added element-wise before being fed into the linear output layer.
	\item For tasks like Question Answering and Commonsense Reasoning, we are given a context document $z$, a question $q$, and a set of possible answers $\left\{a_{k}\right\}$. We concatenate the document context and question with each possible answer. Each of these sequences are processed independently with our model and then normalized via a Softmax layer to produce an output distribution over possible answers.
\end{itemize}

\begin{figure}[H]
	\centering
	\includegraphics[width=1.0\linewidth]{../figures/deepLearning/ApplicationsNLP/pretrainedLM/GPT/GPT_arch}
	\caption{Figure 1: (left) Transformer architecture and training objectives used in this work. (right) Input transformations for fine-tuning on different tasks. We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer. Image from \cite{radford2018improving}.}
	\label{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:fig:gptarch}
\end{figure}

The fine-tuning process involves continuing the model training over a labeled dataset $\mathcal{C}$. Take text classification task as an example. Suppose that each labeled example consists of a sequence of input tokens, $x^{1}, \ldots, x^{m}$ along with a label $y$. The input sequence is first encoded by the pre-trained model to an embedding vector $h_{l}^{m}$ at the position of the last input token. $h_{l}^{m}$ is then fed into an linear layer with Softmax to obtain distribution over class labels. The training loss can simply be the binary cross entropy (BCE) loss. It is also found that including language modeling as an auxiliary task during fine-tuning can improve generalization of the fine-tuned model and speed up convergence. 

## GPT-2

GPT-2 \cite{radford2019language}, a successor to the original GPT-1, is a larger model trained on much more training data, called WebText, than the original one. It achieved state-of-the-art results on seven out of the eight tasks in a zero-shot setting in which there is no fine-tuning applied but had limited success in some tasks. 
The key contribution of GPT-2 is not about further refining the two-stage pretraining-fine-tuning paradigm in GPT-1, but about investigating the capability of zero-shot learning with extensively pretrained language model alone. In other words, it aims to answer whether language modeling is a universal task that can help the model to gain universal knowledge that can accomplish other language tasks without subsequent supervised learning.

The intuition is that a model can be very skilled in the sense that it can learn much of the information about a language during the pre-training phase, there will be no need to learn extra information through fine-tuning phase. Take machine translation in the following box as an example, which contains examples of naturally occurring demonstrations of English to French and French to English translation found throughout the WebText training set. By learning to predict future words in the language modeling task, we expect the model to automatically acquire the ability to translate when we can provide the right prompt (e.g., \textit{translate from English to French}) to the model.

\begin{mdframed}{}
	
	\textit{I'm not the cleverest man in the world}, but like they say in French: \textit{Je ne suis pas un imbecile} [I'm not a fool].
	
	In a now-deleted post from Aug. 16, Soheil Eid, Tory candidate in the riding of Joliette, wrote in French: \textit{Mentez mentez, il en restera toujours quelque chose}, which translates as, \textit{Lie lie and something will always remain}.
	
	I hate the word \textit{perfume}, Burr says. It's somewhat better in French: \textit{parfum.}
	
	If listened carefully at 29:55, a conversation can be heard between two guys in French: \textit{Comment on fait pour aller de l'autre coté? -Quel autre coté?}, which means \textit{How do you get to the other side? - What side?}.
	
	If this sounds like a bit of a stretch, consider this question in French: \textit{As-tu aller au cinéma?}, or \textit{Did you go to the movies?}, which literally translates as Have-you to go to movies/theater?
	
	\textit{Brevet Sans Garantie Du Gouvernement}, translated to English: \textit{Patented without government warranty}.
\end{mdframed}

To probe the learned knowledge in the pretraining stage, we specify the task itself through language. That is, a translation task can be specified via \textit{translate to french: English text}. If corresponding translated French text is produced, it suggests that the pretrained model has acquired the machine translation ability. Likewise, a reading comprehension task can be specified as \textit{answer the question: document}.

One critical difference between GPT-2 and traditional NLP models that is the task in GPT-2 is formulated within the input, and the model is expected to understand the nature of downstream tasks and provide answers accordingly, while in traditional NLP models we engineer task-specific special symbols and components such that we convert the model's output to what we need (e.g., a probability). In GPT-2, learning to perform a single task can be expressed in a probabilistic framework as estimating a conditional distribution $p$ (output|input). Since a general system should be able to perform many different tasks, even for the same input, it should condition not only on the input but also on the task to be performed. That is, it should model $p$ (output|input, task). Here we can view that input and task specification are provided via language itself rather than by changing part of the model parameters or architectures. This is the key to unlock the task-agnostic language understanding ability for large-scaled pretrained language models. 

GPT-2 uses the same Transformer decoder architecture like GPT-1, except that the model size is  has been expanded by more than 10 times from GPT-1. The training corpus used by GPT-1is the BookCorpus dataset, while the training corpus used by GPT-2 is crawled from more than 8 million web pages monolingual data, the amount of data is more than 10 times that of the GPT-1.


## GPT-3

### Introduction
GPT-3 model \cite{brown2020language} has 175 billion parameters, which is 100 times bigger than GPT-2. The architecture of GPT-2 and GPT-3 is similar, with the main differences usually being in the model size and the dataset quantity/quality. As a comparison, GPT-3 has 96 decoder layers with 96 multi-heads attentions and $d_model$ of 12,288. In comparison, GPT-1 only has 12 layers, 12 heads, and $d_model$ given by 768. The training data is further expanded from what is used in GPT-2.

{\centering
\begin{tabular}{lc} 
	Dataset & Quantity (tokens) \\
	\hline Common Crawl (filtered) & 410 billion \\
	WebText2 & 19 billion \\
	Books1 & 12 billion \\
	Books2 & 55 billion \\
	Wikipedia & 3 billion
\end{tabular}
}

The major motivation of GPT-3 is to examine the few-shot learning ability for pretrained language model. This is inspired by the fact that humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. 
By training on the massive amount of data using a model with large number of parameters, GPT-3 achieved better results on many downstream tasks in zero-shot, one-shot, and few-shot $(K=32)$ settings without any gradient-based fine-tuning. 

What is most special about GPT-3 is the ability to perform in-context \textbf{few-shot learning} without any model parameter updates via gradient descent [\autoref{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:fig:gpt3fewshotlearningdemo}]. In a typical few-shot learning setting, the model is given a natura language description of the task plus a few demonstration examples (e.g., input and output pairs) of the task at inference time and the model is asked to generate an output given a new input. Here the input is called \textbf{context} and the output is called a \textbf{completion}. Take English sentence to French translation as an example, $K$ examples of context and completion are presented with one final example of context, and the model is expected to provide the completion. Typically, $K$ is in the range of 10 and 100. Clearly, few-shot learning is close to how human intelligence works and bring great metrics in reducing task-specific data.

In the extreme end of few shot learning, one-shot learning is the case in which only one demonstration is presented. Further, zero-shot is the case where no demonstrations are given except for a natural language instruction describing the task. Zero-shot setting offers the ultimate test of the model's learning capacity, but it can also be unfairly hard due to ambiguity. 

\begin{figure}[H]
	\centering
	\includegraphics[width=0.7\linewidth]{../figures/deepLearning/ApplicationsNLP/pretrainedLM/GPT/GPT3_few_shot_learning_demo}
	\caption{Zero-shot, one-shot and few-shot, contrasted with traditional fine-tuning. The panels above show
		four methods for performing a task with a language model – fine-tuning is the traditional method, whereas zero-, one-,
		and few-shot, which we study in this work, require the model to perform the task with only forward passes at test
		time. We typically present the model with a few dozen examples in the few shot setting}
	\label{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:fig:gpt3fewshotlearningdemo}
\end{figure}

### Performance Overview

In the following, we summarize the probing results of GPT-3 in a broad range of domains.

We first look at the  language modeling benchmark on Penn Tree Bank, which is a fundamental measure on the model's capability on understanding and using natural language. GPT-3 is a clear leader in Language Modelling on  with a perplexity of 20.5.
\begin{figure}[H]
	\centering
	\includegraphics[width=0.7\linewidth]{../figures/deepLearning/ApplicationsNLP/pretrainedLM/GPT/language_modeling_benchmark}
	\caption{Language modeling benchmark task on Penn Tree Bank. Image from \url{https://paperswithcode.com/sota/language-modelling-on-penn-treebank-word}.}
	\label{fig:languagemodelingbenchmark}
\end{figure}


Translation
GPT-3’s training data consists of primarily English (93\% by word count), with an additional 7\% of text in other languages. We expect GPT-3 can learns from a blend of training data that mixes many languages together in a natural way, therefor enabling GPT-3 to perform machine translation in zero-shot and few-shot settings. 

\autoref{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:table:GPT-3-machineTranslation} shows the performance comparison among  supervised SOTA neural machine translation models, unsupervised multi-lingual pretrained language models, and GPT-3. Supervised models are the clear winners in this domain. However, GPT-3 demonstrates its decent performance when performing translation back to English, probably because GPT-3 is a strong English language model. 

The performance GPT-3 also has a noticeable skew depending on language direction. Specifically, GPT-3 significantly outperforms prior unsupervised models when translating into English but under-performs when translating in the other direction.

In general, across all three language models tested, there is a smooth upward trend with model capacity. While in the zero-shot setting, GPT-3 underperforms recent unsupervised model, offering a few demonstrations to GPT-3 can quickly boost the BLEU scores.



\begin{table}[H]
\small
\centering
\begin{tabular}{lcccccc} 
	Setting & $\mathrm{En} \rightarrow \mathrm{Fr}$ & $\mathrm{Fr} \rightarrow \mathrm{En}$ & $\mathrm{En} \rightarrow \mathrm{De}$ & $\mathrm{De} \rightarrow \mathrm{En}$ & $\mathrm{En} \rightarrow \mathrm{Ro}$ & $\mathrm{Ro} \rightarrow \mathrm{En}$ \\
	\hline SOTA (Supervised) & $\mathbf{4 5 . 6}^{a}$ & $35.0^{b}$ & $\mathbf{4 1 . 2}^{\boldsymbol{c}}$ & $40.2^{d}$ & $\mathbf{3 8 . 5}^{e}$ & $\mathbf{3 9 . 9}^{\boldsymbol{e}}$ \\
	\hline XLM & $33.4$ & $33.3$ & $26.4$ & $34.3$ & $33.3$ & $31.8$ \\
	MASS  & $\underline{37.5}$ & $34.9$ & $28.3$ & $35.2$ & $\underline{35.2}$ & $33.1$ \\
	mBART  & $-$ & $-$ & $\underline{29.8}$ & $34.0$ & $35.0$ & $30.5$ \\
	\hline GPT-3 Zero-Shot & $25.2$ & $21.2$ & $24.6$ & $27.2$ & $14.1$ & $19.9$ \\
	GPT-3 One-Shot & $28.3$ & $33.7$ & $26.2$ & $30.4$ & $20.6$ & $38.6$ \\
	GPT-3 Few-Shot & $32.6$ & $39.2$ & $29.7$ & $40.6$ & $21.0$ & $39.5$
\end{tabular}
\caption{Performance comparison among  supervised SOTA neural machine translation models, unsupervised multi-lingual pretrained language models, and GPT-3.}
\label{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:table:GPT-3-machineTranslation}
\end{table}

SuperGLUE

SuperGLUE is a collection of benchmark tests to probe the natural language understanding capacity of a model. \autoref{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:fig:supergluebenchmarkgpt3}

\begin{figure}[H]
	\centering
	\includegraphics[width=0.7\linewidth]{../figures/deepLearning/ApplicationsNLP/pretrainedLM/GPT/SuperGLUE_benchmark_GPT3}
	\caption{Performance on SuperGLUE increases with model size and number of examples in context.}
	\label{ch:neural-network-and-deep-learning:ApplicationNLP:pretrainedLM:fig:supergluebenchmarkgpt3}
\end{figure}


Closed book question answering

In this section we measure GPT-3’s ability to answer questions about broad factual knowledge. Due to the immense
amount of possible queries, this task has normally been approached by using an information retrieval system to find
relevant text in combination with a model which learns to generate an answer given the question and the retrieved
text. Since this setting allows a system to search for and condition on text which potentially contains the answer it
is denoted “open-book”. [RRS20] recently demonstrated that a large language model can perform surprisingly well
directly answering the questions without conditioning on auxilliary information.


… or testing broad factual knowledge with GPT-3. As per the GPT-3 research paper, it was tested on Natural Questions, WebQuestions, and TriviaQA datasets, and the results are the following

\begin{figure}[H]
	\centering
	\includegraphics[width=0.7\linewidth]{../figures/deepLearning/ApplicationsNLP/pretrainedLM/GPT/TrivialQA_benchmark_GPT3}
	\caption{On TriviaQA GPT3’s performance grows smoothly with model size, suggesting that language models
		continue to absorb knowledge as their capacity increases. One-shot and few-shot performance make significant gains
		over zero-shot behavior, matching and exceeding the performance of the SOTA fine-tuned open-domain model}
	\label{fig:trivialqabenchmarkgpt3}
\end{figure}






Common sense reasoning

Next we consider three datasets which attempt to capture physical or scientific reasoning, as distinct from sentence
completion, reading comprehension, or broad knowledge question answering.



As for physical or scientific reasoning, GPT-3 is not outperforming fine-tuned SOTA methods:
\begin{table}
\small
\centering
\begin{tabular}{lllll}
	\hline Setting & PIQA & ARC (Easy) & ARC (Challenge) & OpenBookQA \\
	\hline Fine-tuned SOTA & $79.4$ & $\mathbf{9 2 . 0}\left[\mathrm{KKS}^{+} 20\right]$ & $\mathbf{7 8 . 5}\left[\mathrm{KKS}^{+} 20\right]$ & $\mathbf{8 7 . 2}\left[\mathrm{KKS}^{+} 20\right]$ \\
	GPT-3 Zero-Shot & $\mathbf{8 0 . 5}$ & $68.8$ & $51.4$ & $57.6$ \\
	GPT-3 One-Shot & $\mathbf{8 0 . 5}^{*}$ & $71.2$ & $53.2$ & $58.8$ \\
	GPT-3 Few-Shot & $\mathbf{8 2 . 8}^{*}$ & $70.1$ & $51.5$ & $65.4$ \\
	\hline
\end{tabular}

\end{table}



\textbf{Arithmetic tasks}
GPT-3 is not that good at arithmetic still, since the results are the following

\begin{figure}[H]
	\centering
	\includegraphics[width=0.7\linewidth]{../figures/deepLearning/ApplicationsNLP/pretrainedLM/GPT/arithemetic_task_GPT3}
	\caption{Results on all 10 arithmetic tasks in the few-shot settings for models of different sizes. There is a
		significant jump from the second largest model (GPT-3 13B) to the largest model (GPT-3 175)}
	\label{fig:arithemetictaskgpt3}
\end{figure}


\textbf{News article generation}
The most natural ability for GPT-3 type of generative language model is to generate smooth and nature texts. 

Performance is evaluated by how well humans can detect whether generated text is from a model model generated text. Regarding the test,  25 article titles and subtitles
from the website newser.com are arbitrarily selected, with a mean length of 215 words. We then generated completions of these titles and subtitles
from four language models ranging in size from 125M to 175B (GPT-3)

\begin{table}[H]
\small
\centering
\begin{tabular}{lcc}
	\hline & & $95 \%$ Confidence \\
	& Mean accuracy & Interval (low, hi) \\
	\hline Control (deliberately bad model) & $86 \%$ & $83 \%-90 \%$ \\
	GPT-3 Small & $76 \%$ & $72 \%-80 \%$ \\
	GPT-3 Medium & $61 \%$ & $58 \%-65 \%$ \\
	GPT-3 Large & $68 \%$ & $64 \%-72 \%$ \\
	GPT-3 XL & $62 \%$ & $59 \%-65 \%$ \\
	GPT-3 2.7B & $62 \%$ & $58 \%-65 \%$ \\
	GPT-3 6.7B & $60 \%$ & $56 \%-63 \%$ \\
	GPT-3 13B & $55 \%$ & $52 \%-58 \%$ \\
	GPT-3 175B & $52 \%$ & $49 \%-54 \%$ \\
	\hline
\end{tabular}
\caption{Human accuracy in identifying whether short (around 200 word) news articles are model generated.}
\end{table}

### Limitations

First, despite the strong quantitative and qualitative improvements of GPT-3, particularly compared to its direct
predecessor GPT-2, it still has notable weaknesses in text synthesis and several NLP tasks. On text synthesis, although
the overall quality is high, GPT-3 samples still sometimes repeat themselves semantically at the document level, start to
lose coherence over sufficiently long passages, contradict themselves, and occasionally contain non-sequitur sentences
or paragraphs.

GPT-3 has several structural and algorithmic limitations, which could account for some of the issues above. We focused
on exploring in-context learning behavior in autoregressive language models because it is straightforward to both
sample and compute likelihoods with this model class. As a result our experiments do not include any bidirectional
architectures or other training objectives such as denoising. 

Thus our design decision comes at the cost of potentially worse performance on tasks
which empirically benefit from bidirectionality

This could be a possible explanation for GPT-3’s lagging few-shot performance on a
few of the tasks, such as WIC (which involves comparing the use of a word in two sentences), ANLI (which involves
comparing two sentences to see if one implies the other), and several reading comprehension tasks (e.g. QuAC and
RACE). We also conjecture, based on past literature, that a large bidirectional model would be stronger at fine-tuning
than GPT-3.


Another limitation broadly shared by language models is poor sample efficiency during pre-training. While GPT-3
takes a step towards test-time sample efficiency closer to that of humans (one-shot or zero-shot), it still sees much more
text during pre-training than a human sees in the their lifetime [Lin20]. Improving pre-training sample efficiency is
an important direction for future work, and might come from grounding in the physical world to provide additional
information, or from algorithmic improvements.

## Applications

### Machine Translation

Machine translation is typically implemented via encoder-decoder paradigm, in which encoder takes input in the original language and the decoder outputs words in the target language.


