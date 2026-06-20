# Attention Mechanism

This repository is a hands-on walkthrough of attention in transformers. The notebooks show how token embeddings are turned into queries, keys, and values, how attention weights are computed, and why causal masking is necessary for autoregressive models.

## Notebooks

### `Self_attention_without_weights.ipynb`
This notebook introduces a simplified self-attention setup without trainable projection matrices. It visualizes token embeddings in 3D and shows how tokens can attend to one another using similarity scores.

Use this notebook to understand the core idea of attention before adding learned parameters. It is useful as a teaching aid because it makes the attention flow easy to inspect geometrically.

### `self_atten_with_weights.ipynb`
This notebook adds trainable weight matrices for queries, keys, and values. It shows how the input embeddings are projected into a new representation space and how the output dimension can differ from the input dimension.

This is the closest notebook to the attention layer used inside real transformer models. The same idea is used in language models, translation systems, summarization models, and other sequence tasks where the model must learn which tokens matter most.

### `causal_attention.ipynb`
This notebook demonstrates causal attention with a lower-triangular mask. The mask prevents a token from attending to future tokens, and the weights are renormalized after masking.

This is the key mechanism that makes decoder-style transformers work for next-token prediction and text generation, because the model can only use past and current context.

## How This Relates To Transformers

Transformers are built around attention. These notebooks cover the main parts of a transformer attention block:

1. Token embeddings are represented as vectors.
2. Learned linear projections create queries, keys, and values.
3. Attention scores measure how strongly one token relates to another.
4. Softmax turns scores into attention weights.
5. Causal masking blocks access to future tokens in autoregressive decoding.

Together, these steps explain how transformers can model long-range dependencies without recurrence or convolution. Attention lets the model dynamically focus on the most relevant tokens for each position, which is why it works well for tasks such as language modeling, machine translation, text summarization, retrieval-augmented generation, and contextual representation learning.

## Practical Applications

- Language generation and next-token prediction
- Machine translation
- Text summarization
- Question answering
- Context-aware representation learning for NLP

## Summary

If you want to learn attention from the ground up, start with the simplified self-attention notebook, then move to the version with trainable weights, and finally study causal attention to understand how transformer decoders generate text step by step.