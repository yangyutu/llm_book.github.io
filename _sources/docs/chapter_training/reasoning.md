# LLM Reasoning

## Introduction

**Reasoning** is the process in which 
* a complex problem is decomposed into a sequence of smaller logical steps
* the correct execution of these logical steps leading to final soution of the problem. 
In the context of LLM reasoning, it refers to the process in which LLM is generating intermediate chain of thought steps that lead to the final answer. 

When we ask a question to LLM, the nature of question largely determines if a reasoning is required. For example, 
* *What is the captial of China* is a **factual question** does not require reasoning.
* *A man walks at a speed of 3 mph and how har has he walked for 2 hours?* is a **simple reasoning problem**.
* *Prove there are infinitely many prime numbers* is **complex reasoning problem**. Puzzles, riddles, coding, and math tasks are typically falling into the category of complex reasoning problems.

Most well-trained LLMs are capable of basic reasoning skill that can be triggered by CoT prompting; but they often fall short on complex reasoning benchmark. This chapter is about understanding, measuring, and developping complex reasoning skill for LLMs.

The importance of complex reasoning for LLM are:
Sure! Here’s a simplified version of the key points:  
* **Better Real-World Use** – Strong reasoning helps LLMs solve complex problems in fields like law, medicine, and math, making them more useful.  
* **Fixing Weaknesses** – LLMs struggle with logical thinking and complex tasking in out-of-domain situations. Improving reasoning makes can potentially improve their generalization to new domains.   
* **Step Toward AGI** – Reasoning is key to human intelligence. Enhancing it in LLMs moves us closer to Artificial General Intelligence (AGI).


## Reasoning Benchmarks

Large Language Models (LLMs) are evaluated using a variety of benchmarks designed to assess their reasoning capabilities across multiple domains.hese benchmarks provide standardized tasks that test models on aspects such as logical reasoning, mathematical problem-solving, and domain-specific knowledge.ere are some of the most prominent reasoning benchmarks:

### Math benchmarks
* **GSM8K (Grade School Math)** - This benchmark consists of 8,500 linguistically diverse elementary school math word problems that require two to eight basic arithmetic operations to solve {cite:p}`cobbe2021training`.
* **MathEval** - [MathEval](https://github.com/math-eval/MathEval) is a comprehensive benchmark (~20k problems) that contains 20 other benchmarks, such as GSM8K, MATH, and the math subsection of MMLU. It aims to evaluate the model's math skill from elementary school math to high school competitions.
* **FrontierMath** - This benchmark contains questions from areas of modern math that are difficult for professional mathematicians to solve {cite:p}`glazer2024frontiermath`.


### Coding benchmark
* **HumanEval** - This benchmark consists of programming problems where the solution is always a Python function, often just a few lines long {cite:p}`chen2021evaluating`. This assesses a model's basic ability to generate correct and functional code based on problem descriptions. 
* **SWE-Bench** - The SWE-Bench comprises 2,294 software engineering problems drawn from real GitHub issues and corresponding pull requests across 12 popular Python repositories {cite:p}`jimenez2023swe`. 


### Language, knowledge, and logical thinking benchmark
* **MMLU (Measuring Massive Multitask Language Understanding)** - MMLU consists of approximately 16,000 multiple-choice questions spanning 57 academic subjects, including mathematics, philosophy, law, and medicine {cite:p}`hendrycks2020measuring`. It is widely used to assess a model's breadth of knowledge and reasoning across diverse fields.
* **GPQA (Google-Proof Q&A)** - 448 multiple-choice questions written by domain experts in biology, physics, and chemistry, and requires PhD-level experts to solve.{cite:p}`rein2023gpqa`. It is designed to evaluate a model's proficiency in solving PhD-level complex problems across different domains.
* **AGIEval** - This includes questions from 20 official, public, and high-standard admission and qualification exams, such as the SAT, Gaokao, law school admission tests, math competitions, lawyer qualification tests, and national civil service exams {cite:p}`zhong2023agieval`. It evaluates a model's performance on standardized tests that require advanced reasoning skills.
* **ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence)** - [RC-AGI](https://arcprize.org/) is designed to assess a model's abstract reasoning capabilities.t involves tasks that require pattern recognition and logical inference.

## Inference-Time Scaling Strategies

Inference-time scaling refers to using more compute resources during inference, instead of training stage, to improve LLM's ability in solving complex tasks.

Different ways for scaling test-time computing:
* Best-of-N sampling - sampling $N$ outputs in parallel from a base LLM and take the best one (by a learned verifier or a reward model) as the final output.
* Self-critique prompting - asking the model to self-critique its response and revise its response iteratively.
* Guided search - using a process-based verifier to guide the LLM 

Key findings from {cite:p}`snell2024scaling`
* For easy and intermediate questions, it is more effective to pretrain smaller models with less compute and then apply test-time compute to improve model outputs.
* With the most challenging questions, there is very little benfits from scaling-up test time compute for a smaller model; instead, it is more effective to make progress by scaling up pretraining compute .

### Sampling and Search


```{figure} ../img/chapter_training/reasoning/inference_time_method/inference_time_search_method.png
---
scale: 65%
name: chapter_training_fig_reasoning_inference_time_method_search_method
---
Comparing different PRM search methods. Left: Best-of-N samples N full answers and then selects the best answer according to the PRM final score. Center: Beam search samples N candidates at each step, and selects the top M according to the PRM to continue the search from. Right: lookahead-search extends each step in beam-search to utilize a k-step lookahead while assessing which steps to retain and continue the search from. Thus lookahead-search needs more compute. Image from {cite:p}`snell2024scaling`.
```

**Best-of-N weighted**. We sample N answers independently from the base LLM and then select the best answer according to the PRM's final answer judgement.

**Beam search**. Beam search optimizes the PRM by searching over its per-step predictions. Our implementation is similar to BFS-V $[10,48]$. Concretely, we consider a fixed number of beams $N$ and a beam width $M$. We then run the following steps:
1. sample $N$ initial predictions for the first step in the solution
2. score the generated steps according to the PRM's predicted step-wise reward-to-go estimate (which also corresponds to the total reward from the prefix since the reward is sparse in this setting)
3. filter for only the top $\frac{N}{M}$ highest scoring steps
4. now from each candidate, sample $M$ proposals from the next step, resulting in a total of $N / M \times M$ candidate prefixes again. Then repeat steps 2-4 again.




**Lookahead search.** Lookahead search modifies how beam search evaluates individual steps. It uses lookahead rollouts to improve the accuracy of the PRM's value estimation in each step of the search process. Specifically, 
* at each step in the beam search, rather than using the PRM score at the current step to select the top candidates, lookahead search performs a simulation,
* Simulation involves rolling out up to $k$ steps further while stopping early if the end of solution is reached. (To minimize variance in the simulation rollout, we perform rollouts using temperature 0.)
* The PRM's prediction at the end of this rollout is then used to score the current step in the beam search. 

```{prf:remark} Lookahead search vs beam search vs MCTS
That is, in other words, we can view beam search as a special case of lookahead search with $k=0$. Given an accurate PRM, increasing $k$ should improve the accuracy of the per-step value estimates at the cost of additional compute. 

Also note that this version of lookahead search is a special case of MCTS (Monte carlo Tree Search), wherein the stochastic elements of MCTS, designed to facilitate exploration, are removed since the PRM is already trained and is frozen. These stochastic elements are largely useful for learning the value function (which we've already learned with our PRM), but less useful at test-time when we want to exploit rather than explore. Therefore, lookahead search is largely representative of how MCTS-style methods would be applied at test-time.
``` 

Key findings:
* On the easy questions, the verifier will make mostly correct assessments of correctness. Therefore, by applying beam search guided by PRM, we might overfit to  any spurious features learned by the verifier, causing performance degredation. 
* On the more difficult questions, the base model is much less likely to sample the correct answer in the first place, so beam search can serve to help guide the model towards producing the correct answer more often.
* With a given inference time budget, beam-search is more effective on harder questions
and at lower compute budgets, whereas best-of-N is more effective on easier questions and at higher budgets (i.e., large N).


### Proposal Distribution Refinement

```{figure} ../img/chapter_training/reasoning/inference_time_method/parallel_sampling_vs_sequential_revision.png
---
scale: 75%
name: chapter_training_fig_reasoning_inference_time_method_parallel_sampling_vs_sequential_revision
---
Parallel sampling generates N answers independently in parallel, whereas sequential revisions generates each one in sequence conditioned on previous attempts. Image from {cite:p}`snell2024scaling`.
```

```{figure} ../img/chapter_training/reasoning/inference_time_method/revision_model_with_verifier.png
---
scale: 75%
name: chapter_training_fig_reasoning_inference_time_method_revision_model_with_verifier
---
In both the sequential and parallel cases, we can use the verifier to determine the best-of-N answers (e.g. by applying best-of-N
weighted). We can also allocate some of our budget to parallel and some to sequential, effectively enabling a combination of the
two sampling strategies. In this case, we use the verifier to first select the best answer within each sequential chain and then
select the best answer accross chains. Image from {cite:p}`snell2024scaling`.
```


## DeepSeek Coder

{cite:p}`guo2024deepseek`

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

### Intruction Tuning

We develop DeepSeek-Coder-Instruct by enhancing the DeepSeek-Coder-Base through instructionbased fine-tuning using high-quality data.

### Evaluation

```{figure} ../img/chapter_training/reasoning/deepseek_code/deepseek_coder_performance.png
---
scale: 65%
name: chapter_training_fig_reasoning_inference_time_method_deepseek_coder_performance
---
The Performance of DeepSeek-Coder. Image from {cite:p}`guo2024deepseek`
```

## DeepSeek Math

DeepSeekMath 7B, which continues pretraining DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common
Crawl, together with natural language and code data.




```{figure} ../img/chapter_training/reasoning/deepseek_math/deepseek_math_performance.png
---
scale: 75%
name: chapter_training_fig_reasoning_inference_time_method_deepseek_math_performance
---
The training workflow for DeepSeek-R1-Zero, which is based on pure reinforcement learning. Reward signals are based on Accuracy reward and format reward. Image from {cite:p}`shao2024deepseekmath`
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


```{figure} ../img/chapter_training/reasoning/deepseek_math/deepseek_math_data_pipeline.png
---
scale: 55%
name: chapter_training_fig_reasoning_inference_time_method_deepseek_math_data_pipeline
---
An iterative pipeline that collects mathematical web pages from Common Crawl. Image from {cite:p}`shao2024deepseekmath`
```

### Pretraining

### Supervised Instruction Tuning


After pre-training, we apply mathematical instruction tuning to DeepSeekMath-Base with chain-of-thought (Wei et al., 2022), program-of-thought (Chen et al., 2022; Gao et al., 2023), and tool-integrated reasoning (Gou et al., 2023) data.

We construct a mathematical instruction-tuning dataset covering English and Chinese problems from different mathematical fields and of varying complexity levels: problems are paired with solutions in chain-of-thought (CoT) (Wei et al., 2022), program-of-thought (PoT) (Chen et al., 2022; Gao et al., 2023), and tool-integrated reasoning format (Gou et al., 2023). The total number of training examples is 776 K .
- English mathematical datasets: We annotate GSM8K and MATH problems with toolintegrated solutions, and adopt a subset of MathInstruct (Yue et al., 2023) along with the training set of Lila-OOD (Mishra et al., 2022) where problems are solved with CoT or PoT. Our English collection covers diverse fields of mathematics, e.g., algebra, probability, number theory, calculus, and geometry.
- Chinese mathematical datasets: We collect Chinese K-12 mathematical problems spanning 76 sub-topics such as linear equations, with solutions annotated in both CoT and toolintegrated reasoning format.

### Evaluation


| Math Corpus | Size | English Benchmarks |  |  |  |  | Chinese Benchmarks |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  | GSM8K | MATH | OCW | SAT | MMLU STEM | CMATH | Gaokao <br> MathCloze | Gaokao MathQA |
| No Math Training | N/A | 2.9% | 3.0% | 2.9% | 15.6% | 19.5% | 12.3% | 0.8% | 17.9% |
| MathPile | 8.9B | 2.7% | 3.3% | 2.2% | 12.5% | 15.7% | 1.2% | 0.0% | 2.8% |
| OpenWebMath | 13.6B | 11.5% | 8.9% | 3.7% | 31.3% | 29.6% | 16.8% | 0.0% | 14.2% |
| Proof-Pile-2 | 51.9B | 14.3% | 11.2% | 3.7% | 43.8% | 29.2% | 19.9% | 5.1% | 11.7% |
| DeepSeekMath Corpus | 120.2B | 23.8% | 13.6% | 4.8% | 56.3% | 33.1% | 41.5% | 5.9% | 23.6% |



Our pre-trained base model DeepSeekMath-Base 7B achieves comparable performance
with Minerva 540B (Lewkowycz et al., 2022a), indicating the number of parameters is not
the only key factor in mathematical reasoning capability. A smaller model pre-trained on
high-quality data could achieve strong performance as well.

## DeepSeek Reasoning Models


### DeepSeek-R1-Zero


DeepSeek Team explore the potential of LLMs to develop reasoning capabilities without any supervised data, focusing on their self-evolution through a pure reinforcement learning process. The resulting model is named as **DeepSeek-R1-Zero**.

Specifically, DeepSeek-R1-Zero starts with DeepSeek-V3-Base as the base model and employ a single-stage GRPO reinforcement learning (see {ref}`chapter_training_sec_LLM_alignment_GRPO`) to improve model's reason ability [{numref}`chapter_training_fig_reasoning_inference_time_method_deepseek_r1_zero_workflow`]. The reward signal consists of two types of rewards:
* Accuracy rewards: The accuracy reward model evaluates whether the response is correct. (for math and coding problems, the result is deterministic)
* Format rewards: Encourage model to put thinking process between `<think>` `</think>`

```{figure} ../img/chapter_training/reasoning/deepseek_r1/deepseek_r1_zero_workflow.png
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
```{figure} ../img/chapter_training/reasoning/deepseek_r1/deepseek_rzero_thinking_time_evolution.png
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
    * New CoT SFT data collected from DeepSeek-R0.5
    * Existing supervised training data from DeepSeek V3 to enhancing non-reasoning tasks 

In the final RL for all scenarios, 
* Reasoning tasks rewards adopt the rule-based reward in DeepSeek-R1-Zero.
* Non-reasoning tasks rewards focus on aligning with human preferences, like helpfulness, harmless, and safety.  

```{figure} ../img/chapter_training/reasoning/deepseek_r1/deepseek_r1_workflow.png
---
scale: 45%
name: chapter_training_fig_reasoning_deepseek_r1_workflow
---
Summary of DeepSeek-R1 multi-stage training, starting from DeepSeek-V3.
```

As shown in {numref}`chapter_training_fig_reasoning_inference_time_method_deepseek_r1_benchmark`, DeepSeek-R1 achieves comparable performance to OpenAI-o1-1217 on reasoning tasks.

```{figure} ../img/chapter_training/reasoning/deepseek_r1/deepseek_r1_benchmark.png
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

As shown in the following table:
* Distilling more powerful models into smaller ones gives much better results than its counterparts trained by large-scale RL, despite the latter requires enormous computational power.


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
