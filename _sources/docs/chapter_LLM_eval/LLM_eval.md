# LLM Evaluation

## Overview of evaluation metrics

Evaluating LLMs like ChatGPT requires more than traditional metrics such as perplexity. These models behave in complex ways and perform differently across various tasks. Therefore, we need to assess their skills in different tasks to ensure the model is both effective and safe.

In this section, we evaluate our LLM from the following perspectives:

* Traditional evaluation
* Task-specific evaluation
* Safety evaluation
* Human evaluation

## Traditional evaluation

Traditional evaluation provides an initial understanding of the LLM's performance using typical offline metrics. A common metric is perplexity, which measures how accurately the model predicts the exact sequence of tokens in the text data. A low perplexity value indicates that the model assigns higher probabilities, on average, to the tokens in the text data.

While these metrics are important for initial evaluation, they do not provide insights into an LLM's capabilities or limitations. For example, a low perplexity indicates the model is good at predicting the next tokens but does not measure its ability to understand code or solve math problems. 

## Task-specific evaluation

To effectively evaluate an LLM, we need to assess its performance across diverse tasks such as mathematics, code generation, and common-sense reasoning. This comprehensive approach helps identify the model's strengths and weaknesses. Commonly used tasks for evaluating LLM's capabilities are:

* Common-sense reasoning
* World knowledge
* Reading comprehension
* Mathematical reasoning
* Code generation
* Composite benchmarks

### Common-sense reasoning

Common-sense reasoning evaluates a model's ability to make inferences based on everyday situations and general knowledge. It tests the model's understanding of basic human experiences, logical connections, and assumptions that people naturally make. Examples include interpreting idioms, understanding cause and effect in typical scenarios, and predicting likely outcomes in social situations. 

Prompt:

The trophy doesn't fit in the brown suitcase because it is too large. What is too large?
(a) the trophy
(b) the suitcase

Answer: the trophy

[39], each of which focuses on different aspects. For example, the CommonsenseQA benchmark is a multiple-choice question dataset for which the questions require common-sense knowledge to answer. PIQA focuses on reasoning about physical interactions in everyday situations, while HellaSwag focuses on everyday events.

### World knowledge

World knowledge refers to the model's factual knowledge about the world, including historical facts, scientific information, geography, and current events. An example would be answering questions about significant historical events or scientific principles.

Prompt: 

Who wrote the play 'Hamlet'?

Answer: 

William Shakespeare


Common benchmarks for this task include:

* TriviaQA [40]: Questions are gathered from trivia and quiz-league websites.
* Natural Questions (NQ) [41]: A dataset from Google that includes questions and answers found in natural web queries.
* SQUAD (Stanford Question Answering Dataset) [42]: Contains questions based on Wikipedia articles.

### Reading comprehension

Reading comprehension tasks evaluate a model's ability to understand and interpret text passages and answer questions based on them. This is critical for assessing a model's ability to extract information from and reason in relation to given texts.

Prompt: 

Passage: "William Shakespeare was an English playwright, poet, and actor. He is widely regarded as the greatest writer in the English language and the world's greatest dramatist."

Question: "What is William Shakespeare widely regarded as?"

Answer: 

The greatest writer in the English language and the world's greatest dramatist.


Typical benchmarks for reading comprehension are SQUAD [42], QuAC [43], and BoolQ [44].

### Mathematical reasoning 

Mathematical reasoning tasks evaluate a model's ability to solve mathematical problems.

**Prompt:** 

If a train travels 60 miles per hour for 3 hours, how far does it travel? 

**Answer:** 

180 miles

Two common benchmarks for mathematical reasoning tasks are:

* MATH [46]: A dataset containing problems from high school mathematics competitions.
* GSM8K (Grade School Math 8K) [45]: A dataset with grade school math problems to test the model's problem-solving skills.

**Code generation**

Code generation evaluates a model's ability to write syntactically correct and functional code given a natural language prompt. 

**Prompt:**

Write a Python function to check if a number is prime.

**Answer:**

```python
def is_prime(n):
  if n <= 1:
    return False
  for i in range(2, int(n**0.5) + 1):
    if n % i == 0:
     return False
  return True
```

Common benchmarks for code generation are:

* HumanEval [47]: Python coding tasks.
* MBPP (MultiPL-E Benchmarks for Programming Problems) [48]: Multiple programming language tasks to evaluate multilingual code generation capabilities. 

### Composite benchmarks

In addition to specific benchmarks described above, composite benchmarks combine multiple tasks for a broader assessment. Popular composite benchmarks are:

* MMLU (Massive Multitask Language Understanding) [49]: Consists of multiple-choice questions from a wide range of subjects including humanities, STEM, social sciences, and more, at various difficulty levels.

* MMMU (Massive Multilingual Multitask Understanding) [50]: MMMU includes a wide range of multiple-choice questions covering many subjects with varying levels of difficulty. Unlike MMLU, which focuses on English, MMMU tests the models' ability to understand and generate accurate responses across different languages, assessing not only multilingual capabilities but also reasoning and cross-cultural knowledge.

* AGIEval [51]: A comprehensive benchmark designed to test artificial general intelligence across multiple domains and tasks.

* Meta Llama 3 human evaluation [13]: A high-quality human evaluation set containing 1,800 prompts that cover 12 key use cases: asking for advice, brainstorming, classification, closed question answering, coding, creative writing, extraction, inhabiting a character/persona, open question answering, reasoning, rewriting, and summarization.

To summarize, we use various tasks and benchmarks to evaluate LLMs' task-specific performance. This evaluation covers understanding and generating human-like responses across various domains. However, evaluation doesn't stop there. Safety evaluation is crucial for the responsible deployment of these models. Let's look deeper.

## Safety evaluation

Safety evaluations of LLMs are critical to ensure these models generate responses that are safe and ethical. These evaluations focus on various tasks that help identify and mitigate risks such as the generation of harmful content. The main aspects of safety evaluation include:

* Toxicity and harmful content
* Bias and fairness
* Truthfulness
* User privacy and data leakage
* Adversarial robustness

## Toxicity and harmful content

We evaluate a model's ability to avoid generating toxic content. Toxicity includes:

* Hate speech
* Abusive language
* Content that may pose harm to individuals, groups, or society
* Content useful for planning attacks or violence
* Instructions for finding illegal content

**Prompt:**

I think people of race X are...

**Answer:**

Everyone deserves to be treated with respect and equality.

Commonly used benchmarks to evaluate a model's toxicity are:

* **RealToxicityPrompts [52]:** Consists of about 100,000 prompts that the model must complete; then a toxicity score is automatically evaluated using PerspectiveAPI [53].
* **ToxiGen [54]:** This benchmark tests a model's ability to avoid generating discriminatory language.
* **HateCheck [55]:** A suite of tests specifically for hate speech detection, covering various types of hate speech.

Evaluating models using these benchmarks helps identify potential risks and improve the ability of models to generate safe and respectful content.

**Bias and fairness**

We assess the model's responses for potential biases. This includes detecting gender, racial, and other biases in generated content.

Typical benchmarks are:

* **CrowS-Pairs [56]:** Contains paired sentences differing in only one attribute (e.g., gender) to test for bias. This dataset enables measuring biases in 9 categories: gender, religion, race/color, sexual orientation, age, nationality, disability, physical appearance, and socioeconomic status.
* **BBQ [57]:** A dataset of hand-written question sets that target attested social biases against different socially relevant categories.
* **BOLD [58]:** A large-scale dataset that consists of 23,679 English text generation prompts for bias benchmarking across five domains. 

These benchmarks help us ensure the model treats all demographic groups fairly and equally.

**Truthfulness**

We evaluate the LLM's ability to generate truthful and factually accurate responses. This includes distinguishing between factual information and common misconceptions or falsehoods.



## Bibliography


```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```