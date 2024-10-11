# Basic prompt

## Base LLM vs instructed LLM


The purpose of presenting few-shot examples in the prompt is to explain our intent to the model; in other words, describe the task instruction to the model in the form of demonstrations. However, few-shot can be expensive in terms of token usage and restricts the input length due to limited context length. So, why not just give the instruction directly?

Instructed LM (e.g. InstructGPT, natural instruction) finetunes a pretrained model with high-quality tuples of (task instruction, input, ground truth output) to make LM better understand user intention and follow instruction. RLHF (Reinforcement Learning from Human Feedback) is a common method to do so. The benefit of instruction following style fine-tuning improves the model to be more aligned with human intention and greatly reduces the cost of communication.

When interacting with instruction models, we should describe the task requirement in details, trying to be specific and precise and avoiding say “not do something” but rather specify what to do.

## Zero-shot prompt

A zero-shot prompt for LLM is a technique where the model is asked to perform a task or answer a question without providing examples or any specific training for that particular task. The model relies solely on its pre-existing knowledge and general language understanding to generate a response.

For example, in zero shot text classification (see the following), we provide the model with a prompt and a sequence of text that describes what we want our model to do, in natural language. Zero-shot classification excludes any examples of the desired task being completed. This differs from single or few-shot classification, as these tasks include a single or a few examples of the selected task.

````{prf:example} Zero shot prompt for movie review sentiment classification
**Prompt**: 
Classify the sentiment of the following movie review as positive, negative, or neutral:

**Review**: 'The visuals were stunning and the action sequences kept me on the edge of my seat. However, the plot was thin and the characters lacked depth. Overall, it was an entertaining but forgettable experience.

**Response**: neutral
````

Besides the above demonstrated classification text, LLM can accomplish a wide rangle of straight forward tasks, as shown in the following.

````{prf:example} Zero shot prompt for text extracting
**Prompt**: 
Extract the title, h1, and body text from the following HTML

**Document**:

&lt;head>&lt;title>A simple page&lt;/title>&lt;/head>&lt;body>&lt;h1>Hello
World&lt;/h1>&lt;p>This is some text in a simple html page.&lt;/p>&lt;/
body>&lt;/btml>

**Response**:
Title: A simple page

H1: Hello World

Body text: This is some text in a simple html page.
````


````{prf:example} Zero shot prompt for programming task
**Prompt**: Write a python code to compute the 50th element in the Fibonacci sequence.

**Response**: Certainly! Here's a Python code snippet that computes the 50th element in the Fibonacci sequence:

```
def fibonacci(n):
    if n == 0: return 0
    elif n == 1: return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

# Compute the 50th element in the Fibonacci sequence
fibonacci_50th = fibonacci(50)
print(fibonacci_50th)
```

When you run this code, it will calculate and print the 50th element in the Fibonacci sequence, which is 12,586,269,025.

````

On one hand, zero shot learning offers flexibility to address a wide range of tasks without the need to construct task specific examples.

On the other hand, as LLM's pretrained knowledge is the solely relied on to generate the response, it can have variable performance across the tasks. Specifically, the effectiveness can vary depending on the complexity of the task and how well it aligns with the model's pre-trained knowledge.


## Few-shot and in-context learning

Few-shot learning is a technique where an LLM is given a small number of examples (typically 2-5) to guide its understanding of a task before it's asked to perform that task on new, unseen data. This approach interpolates between zero-shot learning (no examples) and fine-tuning (extensive task-specific training).

In the few-shot example, the model has some context for how to approach the task, potentially leading to more accurate and consistent results compared to the zero-shot approach.

````{prf:example}

**Pormpt**: Classify the sentiment of movie reviews as positive, negative, or neutral.

Examples:
1. "A masterpiece of cinema with stunning visuals and a gripping plot." 
Sentiment: Positive

2. "Terrible acting and a nonsensical story. Avoid at all costs."
Sentiment: Negative

3. "It was okay. Nothing special, but not bad either."
Sentiment: Neutral

Now classify this review:
"The special effects were amazing, but the story was boring."
Sentiment:

Response: Neutral
````

In the following, we summarize the key differences between zero-shot and few-shot learning.

::::{grid}
:gutter: 2

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #e4ac94">**Computational complexity**</span></p>
**LayerNorm** involves both mean and variance calculation for each normalization layer, which brings sizable computational cost for high-dimensional inputs in LLM (e.g., GPT-3 $d_model = 12288$). 
**RMSNorm**, on the other hand, only keeps the variance calculation, reducing the normalization cost by half and increses efficiency
:::

:::{grid-item-card} <p style="text-align: center;"><span style="background-color: #b4c9da">**Gradient propogation**</span></p>
**LayerNorm** stablizes the input distribution between layers through normalization and benefits deep networks training by alleviating the problem of vanishing or exploding gradients. However, LayerNorm can also be affected by noise and input shifts when calculating the mean, potentially leading to unstable gradient propagation.
**RMSNorm**, by using only RMS for normalization, can provide a more robust, smoother gradient flow, especially in deeper networks. It reduces the impact of mean on gradient fluctuations, thereby improving the stability and speed of training.
+++
See {cite:p}`zhang2019rootmeansquarelayer` for math derivation
:::
::::


## CoT

{cite:p}`wei2022chain`


{cite:p}`kojima2022large`



```{figure} ../img/chapter_prompt/prompting/chain_of_thought/chain_of_thought_prompt_demo.png
---
scale: 70%
name: chapter_prompt_fig_basic_prompt_COT_demo
---
COT example.
```

The idea of COT can be generalize to different tasks, as shown in the following.

```{figure} ../img/chapter_prompt/prompting/chain_of_thought/chain_of_thought_prompt_demo2.png
---
scale: 70%
name: chapter_prompt_fig_basic_prompt_COT_demo2
---
More COT examples.
```

## Bibliography

```{bibliography} ../../_bibliography/references.bib
:filter: docname in docnames
```