(chapter_training_sec_LLM_finetuning)=
# LLM Finetuning

## Motivation and Overview

Although a pretrained LLM (e.g., GPT-3) can already perform multitask, including reasoning, via prompting (e.g., few-shot prompting and CoT prompting), further finetuning can usually better adapt to downstream applications (including better following prompt instruction). Particularly for LLMs on the smaller end (<10B), prompt iteration can usually only achieve limited performance gain, finetuning is a much more efficient way to improve model performance with small amount of training data. Fundamentally, this is because the generic pretraining objective (predicting the next token) does not fully align with the downstream application objective: following instructions and conducting specific tasks.

This chapter explores two primary approaches to LLM finetuning: **Instruction Finetuning** and **Parameter-Efficient Fine Tuning (PEFT)**. Instruction Finetuning focuses on teaching the model to follow specific instructions or prompts, enhancing its ability to understand instruction and reponse to instructions. PEFT techniques, on the other hand, aiming to finetune the model by updating only a small subset of its parameters, offering a more computationally efficient approach to customization.

% https://stanford-cs324.github.io/winter2022/lectures/adaptation/

## Instruction Finetuning

### Basics
Instruction finetuning {cite:p}`lou2024large` for LLMs was developed to address the gap between the general knowledge and capabilities of pre-trained base models and the specific tasks or behaviors desired in real-world applications. While pre-trained LLMs possess broad knowledge, they often lack the ability to follow specific instructions or perform targeted tasks consistently.

As shown in the following example, a base LLM (without insturction finetuning) often fails to understand the intent of the prompt, and instead blindly performs text generation. An instructed LLM, on the other hand, understands the intent and provides useful responses. 
````{prf:example} Comparision of base LLM and instructed LLM in response to a prompt
Given the **prompt**:

Task: Answer the following question about basic car maintenance concisely and accurately.
Question: How often should you change your car's oil, and why is it important?

**Response from a base LLM (Qwen2.5-72B)**:
<span style="background-color: #e4ac94">
Task: Provide a brief history of video games, including the evolution of game consoles and popular games.
Question: How have video games and the gaming industry evolved over time?
...
</span>

**Response from an instruction-finetuned LLM (Qwen2.5-72B-instructed)**:
<span style="background-color: #e4ac94">
You should change your car's oil every 5,000 to 7,500 miles, or as recommended by your vehicle's manufacturer. Regular oil changes are crucial because they remove contaminants that can damage engine components, ensuring the engine runs smoothly and efficiently, and extending its life.
</span>
````

The core idea of instruction finetuning [{numref}`chapter_training_fig_finetuning_instruction_finetuning`] is to train the model on a diverse set of task descriptions and their corresponding desired outputs. This typically involves:
1. Creating a dataset of instruction-output pairs covering a wide range of tasks.
2. Fine-tuning the pre-trained LLM on this dataset, often using supervised learning techniques.

Instruction tuning can often substantially improves zero shot performance on unseen tasks {cite:p}`wei2021finetuned`.

```{figure} ../img/chapter_training/finetuning/instruction_finetuning/instruction_finetuning_demo.png
---
scale: 40%
name: chapter_training_fig_finetuning_instruction_finetuning
---
Overview of instruction tuning and FLAN. Instruction tuning finetunes a pretrained language model on a mixture of tasks phrased as instructions. At inference time, we evaluate on
	an unseen task type; for instance, we could evaluate the model on natural language inference (NLI) when no NLI tasks were seen during instruction tuning. Image from {cite:p}`chung2022scalinginstructionfinetunedlanguagemodels`
```


```{figure} ../img/chapter_training/finetuning/instruction_finetuning/instruction_finetuning_performance.png
---
scale: 50%
name: chapter_training_fig_finetuning_instruction_finetuning_gpt3_performance
---
Performance of zero-shot FLAN, compared with zero-shot and few-shot GPT-3, on three unseen task types where instruction tuning improved performance substantially. Image from {cite:p}`wei2021finetuned`.
```

The following summarize the Pros and Cons of instruction finetuning.


::::{grid}
:gutter: 2

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #e4ac94">**Pros**</span></p>

- Improved multi-task performance: Instruction-tuned models can better understand and execute specific tasks, thus directly unlocking the multi-task ability of base model without task-specific prompting and fine-tuning.
- Better alignment with human intent and expectation: Models become more adept at interpreting and following natural language instructions as well as generating human expected results. Properly instruction-tuned models may be less likely to produce harmful or undesired content.
:::

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #b4c9da">**Cons**</span></p>
- Potential for overfitting: As it is usually difficult to creating diverse and high-quality instruction datasets, the model might overfit to the specific instructions in the training set. Ensuring comprehensive coverage of possible tasks and instructions is critical but challenging.
- Bias and compromising existing general knowledge: Aggressive instruction tuning might cause the model to forget some of its pre-trained knowledge. The instruction dataset may inadvertently introduce new biases into the model.
- Increased training costs: Additional fine-tuning requires computational resources and time.
:::
::::

### Insturction Finetuning Loss Functions

Typical instruction finetuning follows the idea of autoregressive language modeling and optimize the prediction over the completion tokens given the instruction.

Specifically, each input is a concatenation of an instruction $X$ and a completion $Y$. Let $X$ be the instruction sequence $\left\{x_1, x_2, \ldots, x_m\right\}$ and $Y$ be the completion (output) sequence $\left\{y_1, y_2, \ldots, y_n\right\}$. The model is optimized to predict each token in $Y$ given all the previous tokens in $X$ and $Y$ up to that point:

$$
P\left(y_1, y_2, \ldots, y_n \mid x_1, x_2, \ldots, x_m\right)=\prod_{j=1}^n P\left(y_j \mid x_1, x_2, \ldots, x_m, y_1, y_2, \ldots, y_{j-1}\right)
$$

The loss function, $\mathcal{L}$ is given as as follows:

$$
\mathcal{L}=-\log P\left(y_1, y_2, \ldots, y_n \mid x_1, x_2, \ldots, x_m\right)=-\sum_{j=1}^n \log P\left(y_j \mid x_1, x_2, \ldots, x_m, y_1, y_2, \ldots, y_{j-1}\right).
$$

Recent studies {cite:p}`shi2024instruction` also show that conducting language modeling on the instruction part, known as **instruction modeling**, can further help for scenarios like:
* The ratio between instruction length and output length in the training data is large
* Only a small amount of training examples are used for instruction tuning.

Intuitively, under instruction modeling, interactions related to instructions can be better adapted to achieve better prediction on the desired output. However, there are also hypotheses that **instruction modeling can lead to overfitting**.

### Comparison with other approaches

Instruction tuning represents a middle ground between the traditional **pretrain-finetune** paradigm and the **prompting** paradigm in making LLM useful for a broad range of downstream tasks[{numref}`chapter_training_fig_finetuning_instruction_finetuning_comparison`]. 

The pretrain-finetune approach typically involves further training a pretrained model on task-specific datasets, which can be effective but often requires separate models and training for each task. Practically, this imposes maintainentce cost for many models and incurs costs for tackling new downstream tasks.  

The prompting paradigm leverages the pretrained model's knowledge through carefully crafted prompts, enabling zero-shot and few-shot learning but potentially suffering from inconsistent performance and prompt sensitivity. 

Instruction tuning bridges these approaches by fine-tuning the model on a diverse set of instructions and tasks, aiming to create a single model capable of following natural language instructions across various domains. This method retains much of the flexibility and generalization capability of prompting while providing more consistent and reliable performance like task-specific fine-tuning. As a result, instruction-tuned models can often handle a wide array of downstream tasks without the need for task-specific models or extensive prompt engineering, offering a more versatile and user-friendly solution for deploying LLMs in real-world applications.

```{figure} ../img/chapter_training/finetuning/instruction_finetuning/instruction_finetuning_comparison.png
---
scale: 60%
name: chapter_training_fig_finetuning_instruction_finetuning_comparison
---
Comparing instruction tuning with pretrain–finetune and prompting. Image from {cite:p}`wei2021finetuned`.
```

### Scaling Instruction Finetuning

Instruction Finetuning LLM can improve model performance and generalization to unseen tasks. In {cite:p}`chung2022scalinginstructionfinetunedlanguagemodels`, scaling instruction finetuning over the number of tasks and model sizes are explored. 

The key methodological aspects of the study are:
* Instruction finetuned LLMs on a diverse set of language tasks (up to 1,836 ). Training data is constructed as mixtures of different number of tasks. 
* The study scales up both model size and number of finetuning tasks

As shown in {numref}`chapter_training_fig_finetuning_instruction_model_scaling_behavior`, the key findings are
* Instruction finetuning significantly improved performance across a range of evaluations, especially for zero-shot and few-shot tasks
* Scaling to larger models and more diverse finetuning tasks led to better performance

```{figure} ../img/chapter_training/finetuning/instruction_finetuning/scaling_instruction_finetuning_behavior_plot.png
---
scale: 60%
name: chapter_training_fig_finetuning_instruction_model_scaling_behavior
---
Scaling behavior of multi-task instruction finetuning with respect to model size (# parameters) and
number of finetuning tasks. Image from {cite:p}`chung2022scalinginstructionfinetunedlanguagemodels`.
```

The study also explores the impact of inclusion/exclusion of CoT training data on reasoning tasks. One key observation [{numref}`chapter_training_fig_finetuning_instruction_cot_data_impact`] is that instruction finetuning without CoT actually degrades reasoning ability. On the other hand, including just nine CoT datasets improves performance on all evaluations (including reasoning and non-reasoning tasks).

```{figure} ../img/chapter_training/finetuning/instruction_finetuning/scaling_instruction_finetuning_cot_data_impact.png
---
scale: 60%
name: chapter_training_fig_finetuning_instruction_cot_data_impact
---
Jointly finetuning on non-CoT and CoT data improves performance on both evaluations, compared
to finetuning on just one or the other. Image from {cite:p}`chung2022scalinginstructionfinetunedlanguagemodels`.
```


### Bootstraping Instruction Finetuning

Given a base language that can mostly rely on few-shot prompting to complete tasks, we can further boostrap the model by
* Prompt the model to generate a diverse set of instruction-completion pair data from a limited set of seed task data.
* Fine-tuning the model on the self-generated training data. 

This process is also known as **Self-Instruct** ({numref}`chapter_training_fig_finetuning_instruction_self_instruct_data`), with the following key steps:
* The process starts with a small seed set of tasks as the task pool.
* Random tasks are sampled from the task pool, and used to prompt an off-the-shelf LM to generate both new instructions and corresponding completions.
* Filtering low-quality or similar generations, and then added back to the initial task pool.

To ensure that diverse instruction examples are generated, the filtering steps can use ROUGE-L similarity score to remove candidates that are similar to the any existing instructions. 

```{figure} ../img/chapter_training/finetuning/instruction_finetuning/bootstrap/self_instruct_data_flow.png
---
scale: 45%
name: chapter_training_fig_finetuning_instruction_self_instruct_data
---
A high-level overview of **Self-Instruct**. Image from {cite:p}`wang2022self`.
```

As shown in the following table, Self-Instruct boosts the instruction-following ability of GPT3 by a large
margin. The vanilla GPT3 model basically cannot follow human instructions at all. Notable, the self-instructed GPT3 nearly matches the performance of InstructGPT, which is trained with private
user data and human-annotated labels.

| Model | # Params | ROUGE-L |
| :--- | :---: | :---: |
| GPT3 | 175 B | 6.8 |
| GPT3 $_{\text {Self-InsT }}$  | 175 B | 39.9 |
| InstructGPT | 175 B | 40.8 |

(chapter_training_sec_LLM_finetuning_PEFT)=
## Parameter-Efficient Fine Tuning (PEFT)

### Motivation

To adapt a LLM to a specific downstream task, Full-size fine-tunning the whole LLM is usually not cost effective. For models whose model parameters are at the 1B or above, it is difficult to fine-tune the model using a single consumer grade GPU. Full-size finetuning also runs the risk of **Catastrophic Forgetting** {cite:p}`luo2024empiricalstudycatastrophicforgetting` which means LLMs forget prior knowledge when learning new data. 

Since the size of the fine-tuned dataset is typically much smaller than the pretrained dataset, performing full fine-tuning to update all the pretrained parameters may lead to **overfitting**.

PEFT {cite:p}`xu2023parameterefficientfinetuningmethodspretrained` emerges as a cost-effiective approach to LLM finetuning. In essence, PEFT only pdates only a small
number of additional parameters or updates a subset of the
pretrained parameters, preserving the knowledge captured by
the PLM while adapting it to the target task and reducing
the risk of catastrophic forgetting.


(chapter_training_sec_LLM_finetuning_Adapter_tuning)=
### Adapter Tuning

The key idea of Adapter Tuning {cite:p}`houlsby2019parameterefficienttransferlearningnlp` is to add several additional trainable modules (i.e., layers) to the Transformer that acting as adapting module and at the same time freeze the remaining model weights the original LLM. The intuition is that by these adapter modules can be trained to assist the original LLM to better adapt to downstream tasks. 

As shown in {numref}`chapter_training_fig_finetuning_adapter_arch`, in each Transformer layer, adapter module are added at different places: after the multihead attention and after FFD. The output of the adapter is directly fed into the following layer normalization. The adaptor module can use a bottleneck architure to save compute cost. Specifically, The adapters first project the original $d_{model}$-dimensional features into a smaller dimension, $m$, apply a nonlinearity, then project back to $d_{model}$ dimensions.


```{figure} ../img/chapter_training/finetuning/adapter/adapter_arch.png
---
scale: 50%
name: chapter_training_fig_finetuning_adapter_arch
---
Architecture of the adapter module and its integration with the Transformer. (Left) Adapter module are added at different places in each Transformer layer: after the projection following multihead attention and after the position-wise FFD layers. (Right) The adapter consists of a bottleneck first mapping the input to lower dimensions and then mappign the output to higher dimensions.  Image from {cite:p}`houlsby2019parameterefficienttransferlearningnlp`.
```


(chapter_training_sec_LLM_finetuning_prompt_tuning)=
### Prompt Tuning

Prompt tuning {cite:p}`lester2021power` is a technique that involves learning a small set of continuous task-specific vectors (soft prompts) while keeping the pretrained model parameters frozen. Specifically [{numref}`chapter_training_fig_finetuning_prompt_tuning`], additional $l$ learnable prompt token vectors, $P=\left[P_1\right],\left[P_2\right], \cdots,\left[P_l\right]$, are combined with the model input $X \in \mathbb{R}^{n \times d}$ to generate the final input $\hat{X}$, that is, 

$$\hat{X} = \operatorname{Concat}(P, X) = [P, X] \in \mathbb{R}^{(l+n)\times d}.$$

During fine-tuning, only the prompt token parameters of $P$ are updated through gradient descent, while pretrained parameters remain frozen. When applying prompt tuning to the multi-task fine-tuning scenario, we can have task-specific prompt tokens work with a fixed pretrained model.

```{figure} ../img/chapter_training/finetuning/prompt_tuning/prompt_tuning.png
---
scale: 35%
name: chapter_training_fig_finetuning_prompt_tuning
---
Illustration of prompt tuning, where we concat a task-dependent prompt tokens into existing prompt 
```

Studies [{numref}`chapter_training_fig_finetuning_prompt_tuning_study`] show that using longer prompt length will achieve much better model performance than using single tuning prompt. One useful tuning prompt token initilization is *class label* initialization, where we used the 
the embeddings for the string representations of each class in the downstream task and use them to initialize one of the tokens in the prompt.

```{figure} ../img/chapter_training/finetuning/prompt_tuning/prompt_tuning_study.png
---
scale: 25%
name: chapter_training_fig_finetuning_prompt_tuning_study
---
Studies on the impact of length of prompt tuning tokens and its initialization. 
```

(chapter_training_sec_LLM_finetuning_prefix_tuning)=
### Prefix-tuning

Prefix-tuning {cite:p}`li2021prefix` proposes to prepend soft prompts $P=$ $\left[P_1\right],\left[P_2\right], \cdots,\left[P_l\right]$ ( $l$ denotes the length of the prefix) to the hidden states of the multi-head attention layer, differing from prompt-tuning that adds soft prompts to the input. To ensure stable training, a FFN is introduced to parameterize the soft prompts, as direct optimization of the soft prompts can lead to instability. Two sets of prefix vectors $\hat{P}_k$ and $\hat{P}_v$ are concatenated to the original key $(K)$ and value $(V)$ vectors of the attention layer. The self-attention mechanism with prefix-tuning can be represented by Equation 8. During training, only $\hat{P}_k, \hat{P}_v$, and the parameters of FFN are optimized, while all other parameters of PLMs remain frozen. The structure of prefix-tuning is illustrated in {numref}`chapter_training_fig_finetuning_prefix_tuning`. After training, the FFN is discarded, and only $P_k$ and $P_v$ are used for inference.

$$
head=\operatorname{Attention}\left(X W_q,\operatorname{Concat}\left[\hat{P}_k, X W_k\right],\operatorname{Concat}\left[\hat{P}_v, X W_v\right]\right) 
$$

where $\hat{P}_k=\operatorname{FFN}\left(P_k\right), \hat{P}_v=\operatorname{FFN}\left(P_v\right).$


```{figure} ../img/chapter_training/finetuning/prefix_tuning/prefix_tuning.png
---
scale: 40%
name: chapter_training_fig_finetuning_prefix_tuning
---
Illustration of prefix tuning, where we concat task dependent prefix vectors into original $K,V$ matrices. 
```


### LoRA (Low-Rank Adaptation)

#### The hypothesis and method

LoRA {cite:p}`hu2021loralowrankadaptationlarge` is one of the most influential strategy among PEFT strategies. Compared with {ref}`chapter_training_sec_LLM_finetuning_Adapter_tuning`, which needs to add sequential layers vertically, and with {ref}`chapter_training_sec_LLM_finetuning_prompt_tuning`,{ref}`chapter_training_sec_LLM_finetuning_prefix_tuning`, which need to expand input vectors with extra tokens, LoRA is much more elegant approach from the perspective of archectural design and mathematics.

Use the notation that each downstream task is represented by a training dataset of input-target pairs: $Z = \{(x_i, y_i), i=1,2,...,N\}$, where both $x_i$ is the input prompt tokens, and $y_i$ are response/completion tokens.

The key observation is that, we can express model fine-tuning on model weight $\Phi$ in its full-update-formulation

$$
\max _{\Phi} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left(P_{\Phi}\left(y_t \mid x, y_{<t}\right)\right)
$$

as its incremental update form

$$
\max _{\theta} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left(p_{\Phi_0+\Delta \Phi(\theta)}\left(y_t \mid x, y_{<t}\right)\right)
$$

and make the **assumption that task-specific parameter increment $\Delta \Phi(\Theta)$ is further encoded by a much smaller-sized set of parameters $\Theta$ with $|\Theta| \ll\left|\Phi_0\right|$**.

Specifically and as a further approximation, we only consider the LoRA on projeciton matrices $W_Q, W_K, W_V$. Take $W_Q$ as example. 

$$H = XW^Q = X(W^{Q}_0 + \Delta W) = X(W^Q_0 + \underbrace{B^QA^Q}_{\text{Low Rank}}).$$

Here during finetuning, we freeze $W^Q_0$ and update low rank matrices $B^Q \in \mathbb{R}^{d_{model}\times r}$, $A^Q \in \mathbb{R}^{r\times d_{head}}$, with $r \ll \min(d_{model}, d_{head})$ (e.g., $r <= 8$).


```{figure} ../img/chapter_training/finetuning/LoRA/LoRA.png
---
scale: 40%
name: chapter_training_fig_finetuning_LoRA
---
Illustration of LoRA, where we provide low-rank matrix adapation on projection matrices in the attention layer. 
```

<!-- 
Hypothesis:


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

The selection of low rank matrices


 full rank (i.e., d) is as high as 12,288 -->


````{prf:remark} $A,B$ initialization and model inference
* To help stablize training, $A,B$ is initialized in a way to ensure $AB = 0$. Specifically, we can set one matrix (say $A$) to zero, and initialize $B$ with random values. For $AB\neq 0$, the initial training phase can be unstable due to large changes of model weights. 

* After training, we only need to save low-rank matrices $A$ and $B$ for different tasks. During inference, we can add $AB$ onto the original projection matrices. Unlike previous Adapter approach, there is no additional inference latency.
````

#### Study Results

As shown in the following, LoRA has been successfully applied to various models, including RoBERT and GPT-3, showing competitive performance with full finetuning while using only 0.01% of the trainable parameters.


```{table} RoBERT with different adaptation methods on the language understanding GLUE benchmark. 
| Model & Method | # Trainable Parameters | Avg. |
|----------------|------------------------|------|
| RoB_base (FT) | 125.0M | 86.4 |
| RoB_base (Adpt^D) | 0.9M | 85.4 |
| RoB_base (LoRA) | 0.3M | 87.2 |
```

```{table} GPT-3-175B with different adaptation methods on the language understanding benchmark.
| Model&Method | # Trainable Parameters | WikiSQL Acc. (%) | MNLI-m Acc. (%)| SAMSum R1/R2/RL|
| :---: | :---: | :---: | :---: | :---: |
| GPT-3 (FT) | $175,255.8 \mathrm{M}$ | $73.8$ | 89.5 | $52.0 / 28.0 / 44.5$ |
| GPT-3 (Adapter ) | 40.1 M | 73.2 | 91.5 | $53.2 / 29.0 / 45.1$ |
| GPT-3 (LoRA) | 4.7 M | 73.4 | $\mathbf{91.7}$ | 53.8/29.8 /45.9 |
| GPT-3 (LoRA) | 37.7 M | $\mathbf{7 4 . 0}$ | $\mathbf{9 1 . 6}$ | $53.4 / 29.2 / 45.1$ |
```

The paper also studies the effect of low rank parameter $r$ on model performance as well as adaptation choices on $\left\{W^Q,W^K,W^V,W^O\right\}$, 

As shown in the following table, 
* $r$ as small as one suffices for adapting both $W^Q$ and $W^V$ on these datasets while training $W^Q$ alone needs a larger $r$. 
* Adapting more matrices will improve model performance but has marginal gain when adapting all matrices.

```{table} Validation accuracy on WikiSQL and MultiNLI with different rank $r$.
|  | Weight Type | $r=1$ | $r=2$ | $r=4$ | $r=8$ | $r=64$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| WikiSQL | $W^Q$ | 68.8 | 69.6 | 70.5 | 70.4 | 70.0 |
|  | $W^Q, W^V$ | 73.4 | 73.3 | 73.7 | 73.8 | 73.5 |
|  | $W^Q, W^K, W^V, W^O$ | 74.1 | 73.7 | 74.0 | 74.0 | 73.9 |
| MultiNLI | $W^Q$ | 90.7 | 90.9 | 91.1 | 90.7 | 90.7 |
|  | $W^Q, W^V$ | 91.3 | 91.4 | 91.3 | 91.6 | 91.4 |
|  | $W^Q, W^K, W^V, W^O$ | 91.2 | 91.7 | 91.7 | 91.5 | 91.4 |
```

<!-- 

During training, we freeze the model weights $W$ and only train the low-rank matrices $A$ and $B$. When saving weights, we only need to save the low-rank matrix parts. According to statistics in the LoRA paper, this operation reduces the memory consumption from 1.2 TB to 350 GB when fine-tuning GPT3 175B; when $r=4$, the final saved model is reduced from 350 GB to 35 MB, greatly reducing training costs.

Regarding the training part, let's look at an interesting question: Overall, LoRA's memory savings are significant, but does LoRA save memory at every moment during training?

Consider calculating the gradient for $B$ during backward pass. Based on $h=W x+B A x=W_{sum} x$ (ignoring the $\alpha$ term for simplicity), we have:

$$
\begin{aligned}
\frac{\partial L}{\partial B} & =\frac{\partial L}{\partial h} \frac{\partial h}{\partial W_{\text {sum }}} \frac{\partial W_{\text {sum }}}{\partial B} \\
& =\frac{\partial L}{\partial h} x^T \frac{\partial W_{\text {sum }}}{\partial B}
\end{aligned}
$$

Notice the $\frac{\partial L}{\partial h} x^T$ term. You'll find that it has the same dimensions $d * d$ as the pre-trained weights $W$, meaning that to calculate the gradient of $B$, we need to use intermediate values of the same size as in full parameter fine-tuning. Therefore, for LoRA, the peak memory usage for this layer is basically the same as full fine-tuning (and higher than full fine-tuning if we include the $\frac{\partial W_{sum}}{\partial B}$ term).

But why can LoRA reduce overall memory usage? Because:
- LoRA is not applied to every layer of the model; for example, in the paper, LoRA is only applied to the attention part
- Although LoRA may cause the peak memory usage of a certain layer to be higher than full fine-tuning, this intermediate result can be cleared after calculating the gradient and doesn't need to be kept continuously
- When the trainable weights are reduced from $d * d$ to $2 * r * d$, the optimizer states that need to be saved are also reduced (and those are in fp32).
 -->
<!-- 
### Discussion: PEFT vs FMT
{cite:p}`zhang2024scaling`
How does finetuning affect the generalization capability of the base LLM? While finetuning on task-specific data improves task-specific performance, it may specialize the base LLM towards the task and hurt the models' generalization. We examine this for different finetuning methods by performing zero-shot translation for LLMs finetuned on WMT14 En-De and WMT19 En-Zh (Fewshot results are in Appendix). We focus on generalization to related tasks, where the target language is shared, i.e. De and Zh , and generalization should be relatively easier (Johnson et al., 2017). We report average performance for translation from a diverse set of source languages other than English.

Figure 6 shows the results. While specializing on a downstream task, finetuning could still elicit and improve the generalization for closely related tasks, although the overall zero-shot translation quality is inferior. Note whether finetuning benefits generalization is method- and task-dependent. Overall, Prompt and LoRA achieve relatively better results than FMT particularly when the base LLM is large, mostly because LLM parameters are frozen and the learned knowledge get inherited. This also suggests that when generalization capability is a big concern, PET should be considered. -->

## Scaling Law for Fine Tuning

When adapting LLM to specific downstream tasks, there are two popular ways of finetuning: **full-model tuning (FMT)** that updates all LLM parameters and PEFT that only optimizes a small amount of (newly added) parameters, such as prompt tuning and LoRA.

It is an open question on how the resulting model performs with regards to fine-tuning data size, model size, and tuning parameter size (in the PEFT case)

{cite:p}`zhang2024scaling` proposes the following multiplicative joint scaling law for LLM finetuning:

$$
\hat{\mathcal{L}}\left(X, D_f\right)=A \times \frac{1}{X^\alpha} \times \frac{1}{D_f^\beta}+E,
$$

where $\{A, E, \alpha, \beta\}$ are data-specific parameters to be fitted, $D_f$ denotes finetuning data size, and $X$ refer to other scaling factors (like model size, and tuning parameter size) and $L$ is perplexity. After fitting to scaling experiments, larger $\alpha$ or $\beta$ means the bigger contribution from these factors. 

The key findings are
* Finetuning model performance scales better on model size than fine-tuning data size, as indicated by larger $\alpha$ then $\beta$ in {numref}`chapter_training_fig_finetuning_ft_scaling_on_translation_task`,{numref}`chapter_training_fig_finetuning_ft_scaling_on_summary_task`. This suggests that using a **larger LLM model is preferred for finetuning over larger data.**
* Finetuning data size have more pronounced influence on FMT than PET (much larger $\beta$ in FMT), where LoRA scales better than Prompt. In other words, **FMT is more data hungary and also benefits more from increasing finetuning data.**
* Compared across different **PEFT approach, scaling tuning parameters is ineffective,** delivering limited gains for both LoRA and Prompt. At the end of day, the amount of newly added trainable parameters often forms a bottleneck for the expressivity of the model.

```{figure} ../img/chapter_training/finetuning/FT_scaling_on_translation_task.png
---
scale: 80%
name: chapter_training_fig_finetuning_ft_scaling_on_translation_task
---
Joint scaling law for model size (from 1B to 16B) and fine-tuning data sizes for translation task. Image from {cite:p}`zhang2024scaling`.
```
```{figure} ../img/chapter_training/finetuning/FT_scaling_on_translation_task.png
---
scale: 80%
name: chapter_training_fig_finetuning_ft_scaling_on_summary_task
---
Joint scaling law for model size (from 1B to 16B) and fine-tuning data sizes for summarization task. Image from {cite:p}`zhang2024scaling`.
```

```{figure} ../img/chapter_training/finetuning/PEFT_scaling_law.png
---
scale: 80%
name: chapter_training_fig_finetuning_ft_scaling_on_PEFT_scaling_law
---
Joint scaling law for tuning parameter size and fine-tuning data sizes for summarization task. Image from {cite:p}`zhang2024scaling`.
```

<!-- ````{prf:remark}
Figure 6 shows the results. While specializing on a downstream task, finetuning could still elicit and improve the generalization for closely related tasks, although the overall zero-shot translation quality is inferior. Note whether finetuning benefits generalization is method- and task-dependent. Overall, Prompt and LoRA achieve relatively better results than FMT particularly when the base LLM is large, mostly because LLM parameters are frozen and the learned knowledge get inherited. This also suggests that when generalization capability is a big concern, PET should be considered.

```` -->


## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```