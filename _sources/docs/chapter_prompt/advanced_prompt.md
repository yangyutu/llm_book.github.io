# Advanced prompt techniques

## Dynamic in-context learning

In the most common form of in-context learning (also known as few-shot learning), the foundation model is prompted with a few demonstrations, and the foundation models quickly adapt to a specific domain and learn to follow the task format. These few-shot examples applied in prompting for a particular task are typically fixed; they are unchanged across test examples. 

To achieve the best testing performance, it is necessary that these few-shot examples selected are broadly representative and relevant to a wide distribution of text examples. 

In the dynamic few-shot prompting setting, we can select can select different few-shot examples for different task inputs. The selection criterion can be based on the simiarlity to the testing case at hand.


## ensemble CoT with self-consistency

{cite:p}`wang2022self`

Chain-of-thought prompting combined with pre-trained large language models has achieved encouraging results on complex reasoning tasks. In this paper, we propose a new decoding strategy, self-consistency, to replace the naive greedy decoding used in chain-of-thought prompting. It first samples a diverse set of reasoning paths instead of only taking the greedy one, and then selects the most consistent answer by marginalizing out the sampled reasoning paths. Self-consistency leverages the intuition that a complex reasoning problem typically admits multiple different ways of thinking leading to its unique correct answer.

Self-consistency is far simpler than prior approaches that either train an additional verifier (Cobbe et al., 2021) or train a re-ranker given additional human annotations to improve generation quality
(Thoppilan et al., 2022). Instead, self-consistency is entirely unsupervised, works off-the-shelf with pre-trained language models, requires no additional human annotation, and avoids any additional training, auxiliary models or fine-tuning.


```{figure} ../img/chapter_prompt/advanced_prompt/cot_self_consistency.png
---
scale: 80%
name: hapter_prompt_fig_advanced_prompt_cot_self_consistency_num_paths
---
CoT with self-consistency.
```

Sampling scheme:
    - for UL2-20B and LaMDA-137B we applied temperature sampling with T = 0.5 and truncated at the top-k (k = 40) tokens with the highest probability, 
    - for PaLM-540B we applied T = 0.7, k = 40, 
    - for GPT-3 we use T = 0.7 without top-k truncation.

```{figure} ../img/chapter_prompt/advanced_prompt/cot_self_consistency_example.png
---
scale: 80%
name: chapter_prompt_fig_advanced_prompt_cot_self_consistency_example
---
CoT with self-consistency examples.
```


Self-consistency (blue) significantly improves accuracy over CoT-prompting with greedy
decoding (orange) across arithmetic and commonsense reasoning tasks, over LaMDA-137B. Sampling
a higher number of diverse reasoning paths consistently improves reasoning accuracy


```{figure} ../img/chapter_prompt/advanced_prompt/cot_self_consistency_num_paths.png
---
scale: 80%
name: chapter_prompt_fig_advanced_prompt_cot_self_consistency_num_paths
---
The effect of the number of sampled reasoning paths.
```

## Self-generated chain of thought

{numref}`chapter_prompt_fig_advanced_prompt_self_cot`


During preprocessing, each question in the training dataset is passed through a lightweight embedding model to generate an embedding vector (Line 4 in Algorithm 1). We employed OpenAI’s text-embedding-ada-002 to create an embedding. For each question, GPT-4 is harnessed to create a chain of thought and a prediction of the final answer (Line 5). If the generated answer is correct and matches the ground truth label, we store the associated question, its embedding vector, the chain of thought, and the answer. Otherwise, we discard the question entirely from our retrieval pool, with the assumption that we cannot trust the reasoning if the model ultimately arrives at the wrong final answer.

At inference time, given a test question, we re-embed the test sample with the same embedding model used during pre-processing, and utilize kNN to retrieve similar examples from the preprocessed pool. These examples, and their corresponding GPT-4 generated reasoning chains, are structured as context for GPT-4. The test question and corresponding answer choices are then appended at the end, which serves as the final prompt (Line 17). The model, following the few shot exemplars, then outputs a chain of thought and a candidate answer. 

<!-- ![Self-generated CoT demonstration. Comparison of expert-crafted and GPT-4-generated chain-of-thought (CoT) prompts.](../img/chapter_prompt/advanced_prompt/medprompt_cot_example.png)
:label:`chapter_prompt_fig_advanced_prompt_self_cot` -->

```{figure} ../img/chapter_prompt/advanced_prompt/medprompt_cot_example.png
---
scale: 60%
name: chapter_prompt_fig_advanced_prompt_self_cot
---
Self-generated CoT demonstration. Comparison of expert-crafted and GPT-4-generated chain-of-thought (CoT) prompts.
```

## Choice Shuffling Ensembling

Combines outputs of multiple model runs to achieve a more robust or accurate result through methods like averaging or majority vote.


Example: Applying self-consistency techniques to produce multiple outputs and identifying a consensus output.
Results: While ensembling enhances performance, it was used with consideration for computational demands. Medprompt used simpler techniques to avoid excessive inference costs


Choise shuffling aims to mitigate the position bias for GPT-4 type of LLM that has exhibited for multiple choices questions.

Self-consistency replaces the naive single-path or greedy decoding with a diverse set of reasoning paths when prompted multiple times at some non-zero temperature, a setting that introduces a degree of randomness in generations. 

With choice shuffling, we shuffle the relative order of the answer choices before generating each reasoning path. We then select the most consistent answer, i.e., the one that is least sensitive to choice shuffling.

Choice shuffling has an additional benefit of increasing the diversity of each reasoning path beyond temperature sampling, thereby also improving the quality of the final ensemble


## Combining Together: the Med prompt 

By using advanced prompting techniques, GPT-4 demonstrated significant capabilities in areas involving significant domain knowledge, such as medicine, which challenges the assumption that it requires intensive domain-specific training to match specialist capabilities​.

In {cite:p}`nori2023can`, authors carried out a systematic exploration of prompt engineering strategies that significantly enhance GPT-4's performance in medical question-answering tasks. Medprompt integrates techniques like in-context learning and chain-of-thought reasoning, leading to a 27% reduction in error rate on the MedQA dataset compared to specialist models.

Medprompt employs dynamic few-shot selection, self-generated chain of thought, and choice shuffle ensembling. These techniques collectively contribute to its high performance in medical benchmarks. 

An ablation study in the following highlighted the relative contributions of Medprompt's components.Each technique incrementally improves the model's performance, with the final accuracy reaching 90.2%. The most significant improvements come from GPT Self-Generated CoT (+3.4%) and the 5x Choice-Shuffle Ensemble Layer (+2.1%).


```{figure} ../img/chapter_prompt/advanced_prompt/med_prompt_abalation_study.png
---
scale: 60%
name: chapter_prompt_fig_advanced_prompt_med_prompt_abalation_study
---
Relative contributions of different components of Medprompt via an ablation study.
```


## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```