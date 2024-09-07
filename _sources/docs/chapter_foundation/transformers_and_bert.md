# Transformers and BERT

## Pretrained Language Models

Pretrained language models are a key technology in modern natural language processing (NLP) that leverages large scale of un-labelled text data and computational power to drastically improve language understanding and generation tasks.

At their core, pretrained language models are large neural networks that have been exposed to enormous amounts of text data – including millions of books, articles, and websites. Through this exposure, they learn to recognize patterns in language, grasp context, and even pick up on subtle nuances in meaning.

The key advantage of pretrained language model lies in the fact that it can be universally adapted (i.e., fine-tuning) to all sorts of specific tasks with a small amount of labeled data. As a comparison, training a task-specific model from scratch would require large amount of labeled data, which can be expensive to obtain.

Pretrained language models typically use neural network architectures designed to process sequential data like text. The most prominent architectures in recent years have been based on the Transformer model, but there have been other important designs as well. Let's explore some of the key architectures:
Transformer-based models:
The Transformer architecture, introduced in 2017, has become the foundation for most modern language models. It uses a mechanism called self-attention to process input sequences in parallel, allowing the model to capture long-range dependencies in text more effectively than previous approaches.

BERT (Bidirectional Encoder Representations from Transformers):
BERT uses the encoder portion of the Transformer. It's bidirectional, meaning it looks at context from both sides of each word when processing text. This makes it particularly good at tasks like sentence classification and named entity recognition.
GPT (Generative Pre-trained Transformer):
GPT models use the decoder portion of the Transformer. They process text from left to right, making them well-suited for text generation tasks. Each version (GPT, GPT-2, GPT-3, etc.) has scaled up in size and capability.


Other architectures:

ELMo (Embeddings from Language Models):
Before Transformers, ELMo used bidirectional LSTMs (Long Short-Term Memory networks) to create contextual word embeddings. While less common now, it was an important step in the evolution of language models.

## Transformers

### Overall architecture

Since 2007, Transformer {cite:p}`vaswani2017attention` has emerged as one of most successful architectures in tackling challenging seq2seq NLP tasks like machine translation, text summarization, etc. 

Traditionally, seq2seq tasks heavily use RNN-based encoder-decoder architectures, plus attention mechanisms, to transform one sequence into another sequence. Transformer, on the other hand, does not rely on any recurrent structure and is able to process all tokens in a sequence at the same time. This enables computation efficiency optimization via parallel optimization and address long-range dependency, both of which mitigate the shortages of RNN-based encoder-decoder architectures. 

On a high level, Transformer falls into the category of encoder-decoder architecture [\autoref{ch:neural-network-and-deep-learning:Advanced:NLP:fig:transformer}], where the encoder encodes an input token sequence into low-dimensional embeddings, and the decoder takes the embeddings as input, plus some additional prompts, outputs an output sequence probabilities. The position information among tokens, originally stored in recurrent network structure, is now provided through position encoding added at the entry point of the encoder and decoder modules.  

Attention mechanisms are the most crucial components in the Transformer architecture to learn contextualized embeddings and overcome the limitation of recurrent neural network in learning long-term dependencies (e.g., seq2seq model with attention). 

The encoder module on the left [{ref}`chapter_foundation_fig_pretrained_LM_transformer_arch`] consists of blocks that are stacked on top of each other to obtain the embeddings that retain rich information in the input. Multi-head self-attentions are used in each block to enable the extraction of contextual information into the final embedding.
Similarly, the decoder module on the right also consists of blocks that are stacked on top of each other to obtain the embeddings. Two different types of multi-head attentions are used in each docder block, one is self-attention to capture contextual information among output sequence and one is to encoder-decoder attention to capture the dynamic information between input and output sequence. 

In the following, we will discuss each component of transformer architecture in detail. 


```{figure} ../img/chapter_foundation/pretrainedLM/Transformer_arch/transformer_arch.png
---
scale: 40%
name: chapter_foundation_fig_pretrained_LM_transformer_arch
---
The transformer architecture, which consists of an Encoder (left) and a Decoder (right).
```

