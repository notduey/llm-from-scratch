## LLM From Practically Scratch

This is an an ongoing project where I'm actively learning, implementing, benchmarking, and will soon get to training a decoder-only Transformer language model from the ground up using PyTorch.

The goal of this project is not just to build a working language model, but to understand the underlying architecture and training process in enough depth that I explain and go about modifying every major component myself.

So rather than starting from an existing Transformer implementation, an enormous amount of time was spent researching each concept individually, recreating basic versions in notebooks, and then translating my learnings into reusable model implementations.

[My Pride & Joy (Research/Documentation)](https://app.notion.com/p/LLM-From-Scratch-3c7e9b9f3cb180ceab82fce19970a6f1?source=copy_link)

---

### Contents

[Overview](#llm-from-practically-scratch)

[Learning Process](#learning-process)

[Roadmap](#roadmap)

[Current Architecture](#current-architecture)

[Model Benchmarking](#model-benchmarking)

[Current Design Decisions and Limitations](#current-design-decisions-and-limitations)

[Setup](#setup)

[Technologies](#technologies)

---

## Learning Process

A huge part of this project has been dedicated to understanding the theory behind language models before implementation. Everything is documented in [Notion](https://app.notion.com/p/LLM-From-Scratch-3c7e9b9f3cb180ceab82fce19970a6f1?source=copy_link).

Included research concepts so far:
````text
- neural network basics
- tensors, gradients, backpropagation, and optimizers
- language modeling and next-token prediction
- tokenization and byte-level BPE
- embeddings
- self-attention
- multi-head attention
- causal masking
- rotary positional embeddings (RoPE)
- RMSNorm
- activation functions and gated activations (SwiGLU)
- residual connections
- Transformer blocks and layers
- Different Transformer architectures
- parameter scaling and drawbacks
- GPU memory usage and throughput
````

research -> recreate concept -> verify/text -> integrate into model -> eval

---

## Roadmap

In progress...
````text
1. NN basics
2. LM fundamentals
3. Tokenization
4. Transformer Architecture
5. Training <------------------- currently here
6. Generation & Inference
7. Optimizations
8. Final Model Evaluations
````

Inference on CPU is planned albeit in the far future.

---

## Current Architecture

The current model is a decoder-only Transformer. The architecture currently uses:
````text
- token embeddings
- tied input/output embedding weights
- causal multi-head self-attention
- combined QKV projection
- rotary positional embeddings (RoPE)
- RMSNorm
- pre-normalization pipeline
- SwiGLU feed-forward networks
- residual connections
- stacked Transformer decoder blocks
- final RMSNorm
- linear language-model output projection
````

Conceptually, the pipeline is:
````text
       Token IDs
           │
           V
 Token Embeddings (tied)
           │
           V
┌─────────────────────┐
│ Transformer Block   │
│                     │
│ RMSNorm             │
│    ↓                │
│ Causal Attention    │
│    ↓                │
│ Residual            │ × Several Layers of Blocks
│    ↓                │
│ RMSNorm             │
│    ↓                │
│ SwiGLU MLP          │
│    ↓                │
│ Residual            │
└─────────────────────┘
           │
           V
     Final RMSNorm
           │
           V
  LM Projection (tied)
           │
           V
        Logits
````

The model successfully completes an end-to-end forward and backward pass, including gradient propagation through the embeddings, attention layers, MLPs, and normalization layers. \
Individual hyperparameters were inspected and tested to see how they affected parameter counts and memory.

Current candidate configuration:
````python
vocab_size = 32000
d_model = 512
num_heads = 8
head_dim = 64
d_ff = 1408
num_layers = 12
````

This has approximately:
````text
54.9M total parameters
38.5M Transformer block parameters
16.4 tied embedding/output parameters
````

---

## Model Benchmarking

My personal NVIDIA RTX 3080 Ti was used and stressed tested to measure how the candidate model behaves during forward/backward passes.

Useful findings included:
- With a fixed batch size, increasing context length resulted in self-attention parameters scaling quadratically
- Token throughout peaked around length 512-1024 at ~24k tokens/second, using a batch size of 4 for that current finding.
- Smaller sequences did not fully utilize the GPU, while excessively large sequences suffered from memory requirements, slowing the system down to a near-useless pace.
- Sequence lengths of 4096 tokens passed the practical memory threshold and became effectively unusable on the 3080 Ti.
- Every hyperparameter has a "goldilocks zone" where the 3080 Ti is effectively utilized without hitting the memory threshold.

These benchmarks were performed before later optimizations, they serve as a baseline for future improvements. Exact results are in [Notion](https://app.notion.com/p/LLM-From-Scratch-3c7e9b9f3cb180ceab82fce19970a6f1?source=copy_link).

---

## Current Design Decisions and Limitations

Did I mentioned that more detailed implementations and notes can be found in [Notion](https://app.notion.com/p/LLM-From-Scratch-3c7e9b9f3cb180ceab82fce19970a6f1?source=copy_link)?
````text
- SwiGLU is used instead of a traditional two-layer MLP and its 3 projection parameters were appropriately scaled.
- RMSNorm was used over LayerNorm as it was simpler and computationally faster.
- Pre-norm Transformer blocks were chosen over the now deprecated post-norm used in the original Transformer.
- Weight tying for the embedding table and LM head projection share the same matrix/weights.
````

Current limitations:
````text
- Model has not a full language-model training
- Still uses more or less theoretical/experimental model dimensions
- Uses explicit attention matrices instead of attention kernels
- Rebuilds RoPE and causal mask every pass instead of caching
- Doesn't yet use mixed-precision training e.g. BF16/FP32
- Doesn't yet optimized autoregressive generation, KV cache
- Still needs final tokenizer and vocabulary
````
Much of these limitations are expected to be taken care of in the near future and are subject to change :)

---

## Setup

Hold your horses it's coming soon...

---

## Technologies

- [Python](https://www.python.org/): primary language used for this project.
- [PyTorch](https://pytorch.org/): used for tensors, auto-differentiation, nn modules, CUDA execution, model training, etc.
- [CUDA](https://developer.nvidia.com/cuda/toolkit): used for experiments, component implementations, notebooks, etc.
- [Notion](https://www.notion.com/): in-depth research notes, documentations, implementation decisions, and keeping track of everything that goes on in this project.
- [Git/Github](https://git-scm.com/): version control.
