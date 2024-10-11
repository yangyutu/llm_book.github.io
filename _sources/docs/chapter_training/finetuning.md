# LLM finetuning


% https://stanford-cs324.github.io/winter2022/lectures/adaptation/

## Instruction finetuning

Instruction tuning fine-tuning language models
on a collection of datasets described via instructions—substantially improves zero shot performance on unseen tasks
\cite{wei2021finetuned}

\begin{figure}[H]
	\centering
	\includegraphics[width=0.7\linewidth]{../figures/deepLearning/ApplicationsNLP_LLM/instruction_finetuning/instruction_finetuning_demo}
	\caption{Overview of instruction tuning and FLAN. Instruction tuning finetunes a pretrained language model on a mixture of tasks phrased as instructions. At inference time, we evaluate on
	an unseen task type; for instance, we could evaluate the model on natural language inference (NLI) when no NLI tasks were seen during instruction tuning.}
	\label{fig:instructionfinetuningdemo}
\end{figure}

\cite{chung2022scaling}

\begin{remark}[Pros and cons for instruction fine-tuning]
\begin{itemize}
	\item Simple and straightforward, generalize to unseen tasks
	\item Cons: Collecting demonstrations for so many tasks is expensive
	\item Cons: Mismatch between LM objective and human preferences
\end{itemize}
	

\end{remark}



## Parameter-Efficient FineTuning

### Motivation

The motivation for parameter-efficient finetuning of large language models (LLMs) stems from several factors:

1. Resource constraints: Full finetuning of LLMs requires significant computational resources and time.
2. Overfitting concerns: Traditional finetuning can lead to overfitting on small datasets.
3. Storage efficiency: Storing multiple versions of large models for different tasks is impractical.
4. Adaptability: The need to quickly adapt models to new tasks or domains without extensive retraining.
5. Environmental considerations: Reducing the carbon footprint associated with training large AI models.

To adapt a LLM to a specific downstream task, Full-size fine-tunning the whole LLM is usually not cost effective. For models whose model parameters are at the 1B or above, it is difficult to fine-tune the model using a single consumer grade GPU. 

Fine-tuning large pre-trained models is an effective transfer mechanism in NLP. However, in the
presence of many downstream tasks, fine-tuning
is parameter inefficient: an entire new model is
required for every task. 

It also runs the risk of Catastrophic Forgetting {cite:p}`luo2024empiricalstudycatastrophicforgetting` which means
LLMs forget prior knowledge when learning new data. 


### Adapter Tuning

The key idea of Adapter Tuning {cite:p}`houlsby2019parameterefficienttransferlearningnlp` is to add several additional trainable modules (i.e., layers) to the Transformer that acting as adapting module and at the same time freeze the remaining model weights the original LLM. The intuition is that by these adapter modules can be trained to assist the original LLM to better adapt to downstream tasks. 

```{figure} ../img/chapter_training/finetuning/adapter/adapter_arch.png
---
scale: 30%
name: chapter_training_fig_finetuning_adapter_arch
---
Architecture of the adapter module and its integration with the Transformer. (Left) Adapter module are added twice
to each Transformer layer: after the projection following multihead attention and after the position-wise feedforward layers. (Right) The
adapter consists of a bottleneck first mapping the input to lower dimensions and then mappign the output to higher dimensions. The adapter also contains a skip-connection. During adapter tuning, the green layers are trained on the downstream data, this includes the adapter, the layer normalization parameters, and the
final classification layer. Image from {cite:p}`houlsby2019parameterefficienttransferlearningnlp`.
```

**CHANGE**

Figure 2 shows our adapter architecture, and its application it to the Transformer. Each layer of the Transformer contains two primary sub-layers: an attention layer and a feedforward layer. Both layers are followed immediately by a projection that maps the features size back to the size of layer's input. A skip-connection is applied across each of the sub-layers. The output of each sub-layer is fed into layer normalization. We insert two serial adapters after each of these sub-layers. The adapter is always applied directly to the output of the sub-layer, after the projection back to the input size, but before adding the skip connection back. The output of the adapter is then passed directly into the following layer normalization.

To limit the number of parameters, we propose a bottleneck architecture. The adapters first project the original $d$-dimensional features into a smaller dimension, $m$, apply a nonlinearity, then project back to $d$ dimensions. The total number of parameters added per layer, including biases, is $2 m d+d+m$. By setting $m \ll d$, we limit the number of parameters added per task; in practice, we use around $0.5-8 \%$ of the parameters of the original model. The bottleneck dimension, $m$, provides a simple means to tradeoff performance with parameter efficiency. The adapter

### Prompt Tuning

Prompt tuning is a technique that involves learning a small set of continuous task-specific vectors (soft prompts) while keeping the pretrained model parameters frozen.

Key points:
- Introduces trainable "soft prompt" tokens to the input
- Only updates these soft prompt parameters during finetuning
- Can achieve performance comparable to full finetuning with significantly fewer parameters
- Allows for efficient multi-task learning by using different soft prompts for different tasks

Example: P-Tuning v2 (Liu et al., 2021) showed that prompt tuning can match or outperform full finetuning across various NLP tasks.

### Model Adaptation

Model adaptation techniques focus on modifying specific parts of the model architecture to achieve efficient finetuning.

### LLaMA Adapters

LLaMA Adapters is a method introduced for efficient finetuning of LLaMA models:

- Adds adapter layers after each transformer block
- Uses a prefix-tuning approach by adding trainable tokens to the beginning of the input sequence
- Combines the benefits of adapter-based and prefix-tuning methods
- Achieves competitive performance with only about 1.2M parameters per task

Key advantages:
- Maintains the pretrained model weights unchanged
- Allows for efficient multi-task learning
- Significantly reduces the number of trainable parameters compared to full finetuning

### LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient finetuning technique that approximates weight updates using low-rank decomposition:

- Freezes the pretrained model weights
- Injects trainable rank decomposition matrices into each layer of the Transformer architecture
- Significantly reduces the number of trainable parameters
- Can be applied to various parts of the model (e.g., attention layers, feed-forward layers)

Key points:
- Achieves performance comparable to full finetuning with only a fraction of the trainable parameters
- Allows for easy switching between tasks by changing the LoRA weights
- Can be combined with other techniques like prompt tuning for further improvements

Example: LoRA has been successfully applied to various models, including GPT-3, showing competitive performance with full finetuning while using only 0.01% of the trainable parameters.

These parameter-efficient finetuning techniques represent significant advancements in making large language models more accessible and adaptable. They allow for efficient use of computational resources, enable quick adaptation to new tasks, and contribute to more sustainable AI development practices.


{cite:p}`hu2021loralowrankadaptationlarge`

## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```