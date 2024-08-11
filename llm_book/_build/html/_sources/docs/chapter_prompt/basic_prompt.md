# Basic prompt

## un-instructed LLM vs instructed LLM


The purpose of presenting few-shot examples in the prompt is to explain our intent to the model; in other words, describe the task instruction to the model in the form of demonstrations. However, few-shot can be expensive in terms of token usage and restricts the input length due to limited context length. So, why not just give the instruction directly?

Instructed LM (e.g. InstructGPT, natural instruction) finetunes a pretrained model with high-quality tuples of (task instruction, input, ground truth output) to make LM better understand user intention and follow instruction. RLHF (Reinforcement Learning from Human Feedback) is a common method to do so. The benefit of instruction following style fine-tuning improves the model to be more aligned with human intention and greatly reduces the cost of communication.

When interacting with instruction models, we should describe the task requirement in details, trying to be specific and precise and avoiding say “not do something” but rather specify what to do.

## Zero-shot prompt

Zero-shot text classification is a fundamental task in natural language understanding where a model is trained on a set of labeled examples but is then able to classify new examples from previously unseen classes.

Zero Shot Classification is the task of predicting a class that wasn't seen by the model during training. This method, which leverages a pre-trained language model, can be thought of as an instance of transfer learning which generally refers to using a model trained for one task in a different application than what it was originally trained for. This is particularly useful for situations where the amount of labeled data is small.

In zero shot classification, we provide the model with a prompt and a sequence of text that describes what we want our model to do, in natural language. Zero-shot classification excludes any examples of the desired task being completed. This differs from single or few-shot classification, as these tasks include a single or a few examples of the selected task.

```
Prompt: 
Extract the title, h1, and body text from the following HTML
document:

<head><title>A simple page</title></head><body><h1>Hello
World</h1><p>This is some text in a simple html page.</p></
body></btml>

Response:
Title: A simple page
H1: Hello World
Body text: This is some text in a simple html page.
```

Zero shot prompt for programming task.

```
Prompt: Write a python code to compute the 50th element in the Fibonacci sequence.

Response: Certainly! Here's a Python code snippet that computes the 50th element in the Fibonacci sequence:


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


When you run this code, it will calculate and print the 50th element in the Fibonacci sequence, which is 12,586,269,025.

```


## Few-shot and in-context learning

### Few-shot examples for uninstruted LLM

A few-shot prompt provides multiple examples—typically, 10 to 100. Multiple examples can be useful for showing a pattern that GPT-3 should continue. Few-shot prompts and more examples will likely increase the quality of the completion because the prompt provides more for GPT-3 to learn from.


### Few-shot examples for instructed LLM



## CoT


\subsection{Few-shot CoT prompting}

\cite{wei2022chain}

\subsection{Zero-shot CoT prompting}

\cite{kojima2022large}


:bibliography:`../llm_book.bib`