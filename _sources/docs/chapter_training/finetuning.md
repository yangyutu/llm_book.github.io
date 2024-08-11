# LLM finetuning


# Parameter efficient large language model finetuning

https://stanford-cs324.github.io/winter2022/lectures/adaptation/

## Motivation

https://sebastianraschka.com/blog/2023/llm-finetuning-llama-adapter.html

In the rapidly evolving field of artificial intelligence, utilizing large language models in an efficient and effective manner has become increasingly important.

Certainly! I'll help populate the contents for the sections you've outlined on parameter-efficient large language model finetuning. Here's an expanded version:

# Parameter Efficient Large Language Model Finetuning

## Motivation

The motivation for parameter-efficient finetuning of large language models (LLMs) stems from several factors:

1. Resource constraints: Full finetuning of LLMs requires significant computational resources and time.
2. Overfitting concerns: Traditional finetuning can lead to overfitting on small datasets.
3. Storage efficiency: Storing multiple versions of large models for different tasks is impractical.
4. Adaptability: The need to quickly adapt models to new tasks or domains without extensive retraining.
5. Environmental considerations: Reducing the carbon footprint associated with training large AI models.

## Prompt Tuning

Prompt tuning is a technique that involves learning a small set of continuous task-specific vectors (soft prompts) while keeping the pretrained model parameters frozen.

Key points:
- Introduces trainable "soft prompt" tokens to the input
- Only updates these soft prompt parameters during finetuning
- Can achieve performance comparable to full finetuning with significantly fewer parameters
- Allows for efficient multi-task learning by using different soft prompts for different tasks

Example: P-Tuning v2 (Liu et al., 2021) showed that prompt tuning can match or outperform full finetuning across various NLP tasks.

## Model Adaptation

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

## LoRA (Low-Rank Adaptation)

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


:bibliography:`../llm_book.bib`