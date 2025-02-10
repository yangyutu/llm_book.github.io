# DeepSeek Series

## DeepSeek V3


Pretraining corpus for DeepSeek-V3 consists of 14.8T high-quality and diverse tokens in our tokenizer.



### Pretraining





## DeepSeek Coder

{cite:p}`guo2024deepseek`
{cite:p}`zhu2024deepseek`

### Data Curation

The training dataset of DeepSeek-Coder is composed of 87% source code, 10% English coderelated natural language corpus, and 3% code-unrelated Chinese natural language corpus.

### PreTraining Strategy

Two training strategies:
* Next Token Prediction. Like the typical language model next-token-prediction task. In this process, various files are concatenated to form a fixed-length entry. 
* Fill-in-the-Middle. In the code pre-training
scenario, it is often necessary to generate corresponding inserted content based on the given
context and subsequent text. Due to specific dependencies in a programming language, relying
solely on next token prediction is insufficient to learn this fill-in-the-middle capability.


Fill-in-the Middle was proposed in {cite:p}`bavarian2022efficient,li2023starcoder`

This approach involves randomly dividing the text into three parts, then shuffling the order of these parts and connecting them with special characters. This method aims to incorporate a fill-in-the-blank pretraining task during the training process.


Within the FIM methodology, two distinct modes are employed: PSM (Prefix-Suffix-Middle) and SPM
(Suffix-Prefix-Middle). In the PSM mode, the training corpus is organized in the sequence
of prefix, suffix, middle, aligning the text in a way that the middle segment is flanked by the
prefix and suffix. Conversely, the SPM mode arranges the segments as suffix, prefix, middle,
presenting a different structural challenge. These modes are instrumental in enhancing the
model’s capability to handle various structural arrangements in code, providing a robust training
framework for advanced code prediction tasks.


### Long Context Extension

we extend the context length of DeepSeek-Coder-V2 to 128K using
Yarn (Peng et al., 2023). The hyper-parameters of YARN are the same as DeepSeek-V2: the scale
𝑠 to 40, 𝛼 to 1, 𝛽 to 32. We further continue training the model using two stages to enhance
its capability for handling long contexts.

In the first stage, we utilize a sequence length of 32K and a batch size of 1152 for 1000 steps

In the second stage, we train the model for an additional 1000 steps, employing a sequence length of 128K and a batch size of 288 sequences.


### Intruction Tuning

We develop DeepSeek-Coder-Instruct by enhancing the DeepSeek-Coder-Base through instructionbased fine-tuning using high-quality data.


### Reinforcement Learning

Reinforcement learning was used to further enhance the reasoning and instruction-following ability.

This stages involve several key elements:
* Training task prompts collection - this involves collecting prompts related to code and math from various sources, and each code prompt comes with corresponding test cases. After filtering the prompts, there are approximately 40 k data in total.
* Reward modeling:
  * Rewards for math problem is constructed based on ground-truth labels. 
  * Rewards for coding is more complex. Code compiler itself can already provide $0-1$ feedback (whether the code pass all test cases or not) but some coding task prompts have a limited number of test cases for full coverage. Instead, the team trained a reward model on the data provided by the compiler, and use the reward model to provide signal during RL training, which is more robust.
* The reinforcement learning algorithm is GRPO (see {ref}`chapter_training_sec_LLM_alignment_GRPO`).

### Evaluation

```{figure} ../img/chapter_LLM_case_study/deepseek_series/deepseek_code/deepseek_coder_performance.png
---
scale: 65%
name: chapter_training_fig_reasoning_inference_time_method_deepseek_coder_performance
---
The performance of DeepSeek-Coder. Image from {cite:p}`guo2024deepseek`.
```

## DeepSeek Math

### Overview



DeepSeekMath 7B, which continues pretraining DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common
Crawl, together with natural language and code data.




```{figure} ../img/chapter_LLM_case_study/deepseek_series/deepseek_math/deepseek_math_performance.png
---
scale: 75%
name: chapter_training_fig_reasoning_inference_time_method_deepseek_math_performance
---
Top1 accuracy of open-source models on the competition-level MATH benchmark Image from {cite:p}`shao2024deepseekmath`
```

### Data Pipeline




DeepSeekMath Corpus, a large-scale high-quality pre-training corpus comprising 120B math tokens. This
dataset is extracted from the Common Crawl (CC) using a fastText-based classifier (Joulin et al.,
2016). In the initial iteration, the classifier is trained using instances from OpenWebMath (Paster
et al., 2023) as positive examples, while incorporating a diverse selection of other web pages to
serve as negative examples. Subsequently, we employ the classifier to mine additional positive
instances from the CC, which are further refined through human annotation.


By implementing a meticulously designed data selection pipeline, we successfully construct the DeepSeekMath
Corpus, a high-quality dataset of 120B tokens from web pages filtered for mathematical content, which is almost 7 times the size of the math web pages used by Minerva
(Lewkowycz et al., 2022a) and 9 times the size of the recently released OpenWebMath
(Paster et al., 2023).

The DeepSeekMath Corpus is of high quality, covers multilingual mathematical content, and
is the largest in size.
* High-quality
* Multilingual
* Large-scale


```{figure} ../img/chapter_LLM_case_study/deepseek_series/deepseek_math/deepseek_math_data_pipeline.png
---
scale: 55%
name: chapter_training_fig_reasoning_inference_time_method_deepseek_math_data_pipeline
---
An iterative pipeline that collects mathematical web pages from Common Crawl. Image from {cite:p}`shao2024deepseekmath`
```


Data validation

We apply math training to a general pre-trained language model with 1.3B parameters,
We separately train a model on each mathematical corpus for 150B tokens.

evaluated using few-shot chain-of-thought prompting.

| Math Corpus | Size | GSM8K | MATH |
| :---: | :---: | :---: | :---: |
| No Math Training | N/A | 2.9% | 3.0% |
| MathPile | 8.9B | 2.7% | 3.3% |
| OpenWebMath | 13.6B | 11.5% | 8.9% |
| Proof-Pile-2 | 51.9B | 14.3% | 11.2% |
| DeepSeekMath Corpus | 120.2B | 23.8% | 13.6% |

Our pre-trained base model DeepSeekMath-Base 7B achieves comparable performance
with Minerva 540B (Lewkowycz et al., 2022a), indicating the number of parameters is not
the only key factor in mathematical reasoning capability. A smaller model pre-trained on
high-quality data could achieve strong performance as well.


### Pretraining


Our model is initialized with DeepSeek-Coder-Base-v1.5 7B

and trained for 500B tokens. The distribution of the data is as follows: 56%
is from the DeepSeekMath Corpus, 4% from AlgebraicStack, 10% from arXiv, 20% is Github
code, and the remaining 10% is natural language data from Common Crawl in both English and
Chinese.

| Data Type | Percentage |
| :--- | :--- |
| DeepSeekMath Corpus| 56% |
| AlgebraicStack| 4% |
| arXiv| 10% |
| Github code| 20% |
| natural language data| 10% |


assessment
* produce self-contained mathematical solutions without relying on external tools
* solve mathematical problems using tools
*  conduct formal theorem proving

 Models are evaluated with chain-of-thought prompting

| Model | Size | English Benchmarks |  |  |  |  | Chinese Benchmarks |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | GSM8K | MATH | OCW | SAT | MMLU STEM | CMATH | Gaokao MathCloze | Gaokao <br> MathQA |
| Mistral | 7B | 40.3% | 14.3% | 9.2% | 71.9% | 51.1% | 44.9% | 5.1% | 23.4% |
| Llemma | 7B | 37.4% | 18.1% | 6.3% | 59.4% | 43.1% | 43.4% | 11.9% | 23.6% |
| Llemma | 34B | 54.0% | 25.3% | 10.3% | 71.9% | 52.9% | 56.1% | 11.9% | 26.2% |
| DeepSeekMath-Base |  | 64.2% | 36.2% | 15.4% | 84.4% | 56.5% | 71.7% | 20.3% | 35.3% |



Findings from pretraining

Code training benefits program-aided mathematical reasoning in a two-stage training setting where first stage consists of code training and second stage consists of math training. 





### Post-Training: SFT and RL


After pre-training, we apply mathematical instruction tuning to supervised intruction tuning DeepSeekMath-Base with da containing **chain-of-thought**, **program-of-thought**, and **tool-integrated reasoning**. The total number of training examples is 776K. Specifically,
- English mathematical datasets covers diverse fields of mathematics, e.g., algebra, probability, number theory, calculus, and geometry. For example, GSM8K and MATH problems with toolintegrated solutions, and adopt a subset of MathInstruct (Yue et al., 2023) along with the training set of Lila-OOD (Mishra et al., 2022) where problems are solved with CoT or PoT. Our English collection 
- Chinese mathematical datasets include  Chinese K-12 mathematical problems spanning 76 sub-topics such as linear equations, with solutions annotated in both CoT and toolintegrated reasoning format.


We conduct RL based on DeepSeekMath-Instruct 7B. The training data of RL are chain-ofthought-format questions related to GSM8K and MATH from the SFT data, which consists
of around 144K questions.



| Model | Size | English Benchmarks |  | Chinese Benchmarks |  |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | GSM8K | MATH | MGSM-zh | CMATH |
| Gemini Ultra | - | 94.4% | 53.2% | - | - |
| GPT-4 | - | 92.0% | 52.9% | - | 86.0% |
| Qwen | 72B | 78.9% | 35.2% | - | - |
| DeepSeekMath-Instruct | 7B | 82.9% | 46.8% | 73.2% | 84.6% |
| DeepSeekMath-RL | 7B | 88.2% | 51.7% | 79.6% | 88.8% |


DeepSeekMath-RL 7B beats all opensource models from 7B to 70B, as well as the majority of closed-source models. Although
DeepSeekMath-RL 7B is only further trained on chain-of-thought-format instruction tuning data
of GSM8K and MATH, it improves over DeepSeekMath-Instruct 7B on all benchmarks.


## DeepSeek Reasoning Models


### DeepSeek-R1-Zero


DeepSeek Team explore the potential of LLMs to develop reasoning capabilities without any supervised data, focusing on their self-evolution through a pure reinforcement learning process. The resulting model is named as **DeepSeek-R1-Zero**.

Specifically, DeepSeek-R1-Zero starts with DeepSeek-V3-Base as the base model and employ a single-stage GRPO reinforcement learning (see {ref}`chapter_training_sec_LLM_alignment_GRPO`) to improve model's reason ability [{numref}`chapter_training_fig_reasoning_inference_time_method_deepseek_r1_zero_workflow`]. The reward signal consists of two types of rewards:
* Accuracy rewards: The accuracy reward model evaluates whether the response is correct. (for math and coding problems, the result is deterministic)
* Format rewards: Encourage model to put thinking process between `<think>` `</think>`

```{figure} ../img/chapter_LLM_case_study/deepseek_series/deepseek_r1/deepseek_r1_zero_workflow.png
---
scale: 35%
name: chapter_training_fig_reasoning_inference_time_method_deepseek_r1_zero_workflow
---
The training workflow for DeepSeek-R1-Zero, which is based on pure reinforcement learning. Reward signals are based on Accuracy reward and format reward. 
```

```{prf:remark} Advantages Pure RL
Pure-RL is slower upfront (trial and error takes time) — but iteliminates the costly, time-intensive labeling bottleneck. Further, the human labeling CoT data is not necessarily the optimal thought process to solve any problems. Using pure RL could have following profound impact in the long run:
* Without the data labeling bottleneck, it’ll be faster, scalable, and way more efficient for building reasoning models.
* Let's the model's self-evolution process to explore better ways to solve problems, instead of relying on human priors. This is necessary for developing superintelligence.
```

As the training proceeds, the model started to develop sophisticated reasoning behaviors, such as reflection, where the model
revisits and reevaluates its previous steps and then the model explores of alternative approaches to problem-solving.

Such self-evolution of reasoning behavior is not a result of explicit programm but instead incentivized from the model’s interaction with the reinforcement learning environment (i.e., the reward signal). 


```{prf:remark} Reward modeling for reasoning task
Reward modeling for reasoning task is much more straightforward than typical human preference learning tasks, which requires non-trivial efforts to build reward model to approximate complicated human preference. 

The reasoning tasks usually have groundtruth - correct answer or not; as a comparison, human preference is usually hard to quantify. 

```

As shown in {numref}`chapter_training_fig_reasoning_inference_time_method_deepseek_rzero_thinking_time_evolution`, DeepSeek-R1-Zero learns to allocate more thinking time in solving problems
```{figure} ../img/chapter_LLM_case_study/deepseek_series/deepseek_r1/deepseek_rzero_thinking_time_evolution.png
---
scale: 75%
name: chapter_training_fig_reasoning_inference_time_method_deepseek_rzero_thinking_time_evolution
---
The average response length of DeepSeek-R1-Zero on the training set during the RL
process. DeepSeek-R1-Zero naturally learns to solve reasoning tasks with more thinking time. Image from {cite:p}`guo2025deepseek`.
```




| Model | AIME 2024 |  | MATH-500 | GPQA <br> Diamond | LiveCode <br> Bench <br> pass@1 | CodeForces |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|  | pass@1 | cons@64 | pass@1 | pass@1 | pating |  |
| OpenAI-o1-mini | 63.6 | 80.0 | 90.0 | 60.0 | 53.8 | 1820 |
| OpenAI-o1-0912 | 74.4 | 83.3 | 94.8 | 77.3 | 63.4 | 1843 |
| DeepSeek-R1-Zero | 71.0 | 86.7 | 95.9 | 73.3 | 50.0 | 1444 |
| DeepSeek-R1 | 71.0 | 79.8 | 97.3 | 71.5 | 65.9 | 2029 |




There are also some observed drawbacks from training model - the output has issues like **poor readability** and **language mixing**.
However, this is expected:
* The reward signal does not enforce readability and language mixing.
* Enforcing readability and language mixing could interfere with the development of reasoning ability during training process. 





### DeepSeek-R1

On top of encourging results from  DeepSeek-R1-Zero, DeepSeek-R1 has the objectives of
* Further enhancing reasoning performance and accelerate convergence 
* Improving user-friendliness by generating human readable, clear anc coherent chain of throught
* Improving the model's general ability beyond reasoning tasks.

DeepSeek-R1 main technical approaches improve over Zero by using high-quality **cold start** data and a **multi-stage training** pipeline.
* The cold start data consists of a small amount (thousands) of long and high-quality (structred, human friendly) CoT data to fine-tune the model as the initial RL actor.
* The multi-stage training pipeline is shown in {numref}`chapter_training_fig_reasoning_deepseek_r1_workflow`, which includes
  * A supervised fine-tuning stage on cold-start data to get to DeepSeek-R0.25
  * A subsequent reasoning-oriented RL to get to DeepSeek-R0.5
  * Additional SFT and RL process on a combined data consisting of 
    * New CoT SFT data collected from DeepSeek-R0.5 via rejection sampling
    * Existing supervised training data from DeepSeek V3 to enhancing non-reasoning tasks 

In the final RL for all scenarios, reward signals are coming from
* Reasoning tasks rewards adopt the rule-based reward in DeepSeek-R1-Zero.
* Non-reasoning tasks rewards focus on aligning with human preferences, like helpfulness, harmless, and safety.  

```{figure} ../img/chapter_LLM_case_study/deepseek_series/deepseek_r1/deepseek_r1_workflow.png
---
scale: 45%
name: chapter_training_fig_reasoning_deepseek_r1_workflow
---
Summary of DeepSeek-R1 multi-stage training, starting from DeepSeek-V3.
```


The motivation and technical details on each training stage is summarized as follows

| Stage | Resulted Model | Motivation & Technical Details |
| :---  | :--- | :--- |
| SFT on cold start CoT data | DeepSeek-R0.25 |  Prevent early instability in RL training; Incorporate human reasoning priors; Improve output readability     | 
| Pure RL | DeepSeek-R0.5 | Enhance reasoning ability; Add additional language consistency reward       |
| Large scale SFT + RL | DeepSeek-R1 | Enhance reasoning and general capability; Use ~600k reasoning + ~200k general training samples; Use RL to enhance helpfulness and harmless     |



As shown in {numref}`chapter_training_fig_reasoning_inference_time_method_deepseek_r1_benchmark`, DeepSeek-R1 achieves comparable performance to OpenAI-o1-1217 on reasoning tasks.

```{figure} ../img/chapter_LLM_case_study/deepseek_series/deepseek_r1/deepseek_r1_benchmark.png
---
scale: 75%
name: chapter_training_fig_reasoning_inference_time_method_deepseek_r1_benchmark
---
Benchmark performanc of DeepSeek-R1 in reasoning oriented tasks. Image from {cite:p}`guo2025deepseek`.
```



### DeepSeek-R1-Distillation



DeepSeek team explored a straightforward distillation approach:
* Using the reasoning data generated by DeepSeek-R1
* Only SFT is used and does not include an RL stage,

As shown in Table 5, simply distilling DeepSeek-R1’s outputs enables the efficient DeepSeekR1-7B (i.e., DeepSeek-R1-Distill-Qwen-7B, abbreviated similarly below) to outperform nonreasoning models like GPT-4o-0513 across the board. DeepSeek-R1-14B surpasses QwQ-32BPreview on all evaluation metrics,

```{table} Comparison of DeepSeek-R1 distilled models and other comparable models on
reasoning-related benchmarks.
| Model | AIME 2024 |  | MATH-500 | GPQA <br> Diamond | LiveCode <br> Bench | CodeForces |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|  | pass@1 | cons@64 | pass@1 | pass@1 | pass@1 | rating |
| GPT-40-0513 | 9.3 | 13.4 | 74.6 | 49.9 | 32.9 | 759 |
| Claude-3.5-Sonnet-1022 | 16.0 | 26.7 | 78.3 | 65.0 | 38.9 | 717 |
| OpenAI-o1-mini | 63.6 | 80.0 | 90.0 | 60.0 | 53.8 | $\mathbf{1 8 2 0}$ |
| QwQ-32B-Preview | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 | 1316 |
| DeepSeek-R1-Distill-Qwen-1.5B | 28.9 | 52.7 | 83.9 | 33.8 | 16.9 | 954 |
| DeepSeek-R1-Distill-Qwen-7B | 55.5 | 83.3 | 92.8 | 49.1 | 37.6 | 1189 |
| DeepSeek-R1-Distill-Qwen-14B | 69.7 | 80.0 | 93.9 | 59.1 | 53.1 | 1481 |
| DeepSeek-R1-Distill-Qwen-32B | 72.6 | 83.3 | 94.3 | 62.1 | 57.2 | 1691 |
```


DeepSeek team also conducted examperiment to answer the question if one can directly apply large-scale RL training to a student model to achieve performance comparable with distillation method?

The results and implications from the following table:
* Distilling more powerful models into smaller ones gives much better results than its counterparts trained by large-scale RL, despite the latter requires enormous computational power.
* Large-scale RL still need efficiency improvement on computation and samples to achieve on-par results with distillation.


```{table}
| Model | AIME 2024 |  | MATH-500 | GPQA Diamond | LiveCodeBench |
| :--- | :---: | :---: | :---: | :---: | :---: |
|  | pass@1 | cons@64 | pass@1 | pass@1 | pass@1 |
| QwQ-32B-Preview | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 |
| DeepSeek-R1-Zero-Qwen-32B | 47.0 | 60.0 | 91.6 | 55.0 | 40.2 |
| DeepSeek-R1-Distill-Qwen-32B | $\mathbf{7 2 . 6}$ | $\mathbf{8 3 . 3}$ | $\mathbf{9 4 . 3}$ | $\mathbf{6 2 . 1}$ | $\mathbf{5 7 . 2}$ |
```


## Bibliography

### Software

[Unsloth](https://github.com/unslothai/unsloth)

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
