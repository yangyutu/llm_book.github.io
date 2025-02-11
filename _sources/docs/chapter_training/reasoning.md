# LLM Reasoning

## Introduction

**Reasoning** is the process in which 
* a complex problem is decomposed into a sequence of smaller logical steps
* the correct execution of these logical steps leading to final soution of the problem. 
In the context of LLM reasoning, it refers to the process in which LLM is generating intermediate chain of thought steps that lead to the final answer. 

When we ask a question to LLM, the nature of question largely determines if a reasoning is required. For example [{numref}`chapter_training_fig_reasoning_simple_reasoning_vs_complex_reasoning`], 
* *What is the captial of China* is a **factual question** does not require reasoning.
* *A man walks at a speed of 3 mph and how har has he walked for 2 hours?* is a **simple reasoning problem**.
* *Prove there are infinitely many prime numbers* is **complex reasoning problem**. Puzzles, riddles, coding, and math tasks are typically falling into the category of complex reasoning problems.

```{figure} ../img/chapter_training/reasoning/introduction/simple_reasoning_vs_complex_reasoning.png
---
scale: 65%
name: chapter_training_fig_reasoning_simple_reasoning_vs_complex_reasoning
---
Comparison between simple reasoning task (left) and complex reasoning task (right).
```

The following is a summary on simple reasoning and complex reasoning tasks from different aspects.
| Aspects | Simple reasoning | Complex Reasoning |
| :--- | :--- | :--- |
| Problem Complexity | Low | High |
| Reasoning Steps | Few, linear | Many, linear and nonlinear, requiring planning and abstraction | 
| Domain Knowledge | Basic or general | Often requires specialized knowledge | 
| Example Problems | Arithmetic, factual QA, simple logic | Multi-step math, coding, and logical; plannning, optimizaiton |



Most well-trained LLMs are capable of basic reasoning skill that can be triggered by CoT prompting; but they often fall short on complex reasoning benchmark. This chapter is about understanding, measuring, and developping complex reasoning skill for LLMs.












The importance of complex reasoning for LLM are:
* **Better Real-World Use** – Strong reasoning helps LLMs solve complex problems in fields like law, medicine, and math, making them more useful.  
* **Fixing Weaknesses** – LLMs struggle with logical thinking and complex tasking in out-of-domain situations. Improving reasoning makes can potentially improve their generalization to new domains.   
* **Step Toward AGI** – Reasoning is key to human intelligence. Enhancing it in LLMs moves us closer to Artificial General Intelligence (AGI).


### Math benchmarks
* **GSM8K (Grade School Math)** - This benchmark consists of 8,500 linguistically diverse elementary school math word problems that require two to eight basic arithmetic operations to solve {cite:p}`cobbe2021training`.
* **MathEval** - [MathEval](https://github.com/math-eval/MathEval) is a comprehensive benchmark (~20k problems) that contains 20 other benchmarks, such as GSM8K, MATH, and the math subsection of MMLU. It aims to evaluate the model's math skill from elementary school math to high school competitions.
* **FrontierMath** - This benchmark contains questions from areas of modern math that are difficult for professional mathematicians to solve {cite:p}`glazer2024frontiermath`.


## Coding benchmark


* **HumanEval** - This benchmark consists of programming problems where the solution is always a Python function, often just a few lines long {cite:p}`chen2021evaluating`. This assesses a model's basic ability to generate correct and functional code based on problem descriptions. 
* **SWE-Bench** - The SWE-Bench comprises 2,294 software engineering problems drawn from real GitHub issues and corresponding pull requests across 12 popular Python repositories {cite:p}`jimenez2023swe`. 


## Language, knowledge, and logical thinking benchmark
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

## Reinfrocement Learning





## Preference Learning

### Iterative RPO

```{figure} ../img/chapter_training/reasoning/preference_learning/iterative_RPO/iterative_DPO_workflow.png
---
scale: 75%
name: chapter_training_fig_reasoning_iterative_RPO_workflow
---
Workflow of Iterative RPO, which consists of two steps: (i) Chain-of-Thought \& Answer Generation: training prompts are used to generate candidate reasoning steps and answers from model $M_t$, and then the answers are evaluated for correctness by a given reward model. (ii) Preference Optimization: preference pairs are selected from the generated data, which are used for training via a DPO+NLL objective, resulting in model $M_{t+1}$. This whole procedure is then iterated resulting in improved reasoning ability on the next iteration, until performance saturates.Image from {cite:p}`pang2024iterative`.
```

## Bibliography

### Software

[Unsloth](https://github.com/unslothai/unsloth)

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
