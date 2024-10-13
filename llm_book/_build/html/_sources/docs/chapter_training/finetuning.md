# LLM Finetuning


% https://stanford-cs324.github.io/winter2022/lectures/adaptation/

## Instruction Finetuning

### Basics
Instruction finetuning for Large Language Models (LLMs) was developed to address the gap between the general knowledge and capabilities of pre-trained models and the specific tasks or behaviors desired in real-world applications. While pre-trained LLMs possess broad knowledge, they often lack the ability to follow specific instructions or perform targeted tasks consistently.

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


Instruction tuning often substantially improves zero shot performance on unseen tasks {cite:p}`wei2021finetuned`.




```{figure} ../img/chapter_training/finetuning/instruction_finetuning/instruction_finetuning_demo.png
---
scale: 40%
name: chapter_training_fig_finetuning_instruction_finetuning
---
Overview of instruction tuning and FLAN. Instruction tuning finetunes a pretrained language model on a mixture of tasks phrased as instructions. At inference time, we evaluate on
	an unseen task type; for instance, we could evaluate the model on natural language inference (NLI) when no NLI tasks were seen during instruction tuning. Image from {cite:p}`chung2022scalinginstructionfinetunedlanguagemodels`
```


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


## Parameter-Efficient Fine Tuning

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

As shown in {numref}`chapter_training_fig_finetuning_adapter_arch`, in each Transformer layer, adapter module are added at different places: after the multihead attention and after FFD. The output of the adapter is directly fed into the following layer normalization. The adaptor module can use a bottleneck architure to save compute cost. Specifically, The adapters first project the original $d_{model}$-dimensional features into a smaller dimension, $m$, apply a nonlinearity, then project back to $d_{model}$ dimensions.


```{figure} ../img/chapter_training/finetuning/adapter/adapter_arch.png
---
scale: 60%
name: chapter_training_fig_finetuning_adapter_arch
---
Architecture of the adapter module and its integration with the Transformer. (Left) Adapter module are added at different places in each Transformer layer: after the projection following multihead attention and after the position-wise FFD layers. (Right) The adapter consists of a bottleneck first mapping the input to lower dimensions and then mappign the output to higher dimensions.  Image from {cite:p}`houlsby2019parameterefficienttransferlearningnlp`.
```



<!-- ### Prompt Tuning

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
- Significantly reduces the number of trainable parameters compared to full finetuning -->

### LoRA (Low-Rank Adaptation)

#### The hypothesis and method

During full fine-tuning, the model is initialized to pre-trained weights $\Phi_0$ and updated to $\Phi_0+\Delta \Phi$ by repeatedly following the gradient to maximize the conditional language modeling objective:

$$
\max _{\Phi} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left(P_{\Phi}\left(y_t \mid x, y_{<t}\right)\right)
$$


One of the main drawbacks for full fine-tuning is that for each downstream task, we learn a different set of parameters $\Delta \Phi$ whose dimension $|\Delta \Phi|$ equals $\left|\Phi_0\right|$. Thus, if the pre-trained model is large (such as GPT-3 with $\left|\Phi_0\right| \approx 175$ Billion), storing and deploying many independent instances of fine-tuned models can be challenging, if at all feasible.

In this paper, we adopt a more parameter-efficient approach, where the task-specific parameter increment $\Delta \Phi=\Delta \Phi(\Theta)$ is further encoded by a much smaller-sized set of parameters $\Theta$ with $|\Theta| \ll\left|\Phi_0\right|$. The task of finding $\Delta \Phi$ thus becomes optimizing over $\Theta$ :

$$
\max _{\Theta} \sum_{(x, y) \in \mathcal{Z}} \sum_{t=1}^{|y|} \log \left(p_{\Phi_0+\Delta \Phi(\Theta)}\left(y_t \mid x, y_{<t}\right)\right)
$$



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


 full rank (i.e., d) is as high as 12,288


### Training

````{prf:remark} LoRA low rank matrices initialization


````


During training, we frezze the model weights $W$ and only train the low-rank matrices $A$ and $B$. When saving weights, we only need to save the low-rank matrix parts. According to statistics in the LoRA paper, this operation reduces the memory consumption from 1.2 TB to 350 GB when fine-tuning GPT3 175B; when $r=4$, the final saved model is reduced from 350 GB to 35 MB, greatly reducing training costs.

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

### Inference

{cite:p}`hu2021loralowrankadaptationlarge`




Our simple linear design allows us to merge the trainable matrices with the frozen weights
when deployed, introducing no inference latency compared to a fully fine-tuned model, by
construction.

No Additional Inference Latency. When deployed in production, we can explicitly compute and store $W=W_0+B A$ and perform inference as usual. Note that both $W_0$ and $B A$ are in $\mathbb{R}^{d \times k}$. When we need to switch to another downstream task, we can recover $W_0$ by subtracting $B A$ and then adding a different $B^{\prime} A^{\prime}$, a quick operation with very little memory overhead. Critically, this

#### Analysis 

OPTIMAL RANK r FOR LORA

## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```