# 🤖 KindaGPT

**A minimal GPT-style language model built from scratch in PyTorch to understand how Transformers learn to predict and generate text.**

---

## 🎯 The Objective

KindaGPT is a small-scale implementation of a **GPT-style Transformer language model**, built from the ground up using PyTorch.

Rather than relying on an existing Transformer implementation, the project breaks down the core components behind modern language models — including **token embeddings, positional embeddings, self-attention, multi-head attention, feed-forward networks, Transformer blocks, and autoregressive text generation**.

The goal was not to build a production-ready language model, but to understand what happens inside a GPT model and how these components work together to learn the structure of text.

---

## 🧠 How It Works

At a high level, KindaGPT follows the same fundamental idea as GPT:

```text
Input Text
    │
    ▼
Character Tokenization
    │
    ▼
Token Embeddings + Positional Embeddings
    │
    ▼
Transformer Blocks
    │
    ├── Multi-Head Self-Attention
    │
    └── Feed-Forward Network
    │
    ▼
Language Model Head
    │
    ▼
Next-Character Probabilities
    │
    ▼
Generated Text
```

The model is trained on sequences of characters and learns to predict the **next character given the previous characters**.

---

## 🔤 Character-Level Tokenization

Instead of using words or subword tokens, KindaGPT uses a simple **character-level tokenizer**.

The vocabulary is constructed from every unique character appearing in the training text:

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)
```

Two mappings are created:

```python
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
```

This allows text to be converted into integer representations:

```text
"hello"
   ↓
[...integer token IDs...]
```

and converted back after generation.

This approach keeps tokenization simple and makes it easier to focus on understanding the Transformer architecture.

---

## 📦 Training Data

The encoded text is converted into a PyTorch tensor and divided into training and validation sets:

```text
90% → Training
10% → Validation
```

The model learns from small sequences of length `block_size = 8`.

For example:

```text
Input:  "The cat"
Target: "he cat?"
```

Conceptually, every character is used to predict the character that follows it.

---

## 👁️ Self-Attention

The core mechanism of KindaGPT is **self-attention**.

Each input representation is transformed into three components:

```text
Query (Q)
Key   (K)
Value (V)
```

The attention scores are calculated using:
$$
\text{Attention}(Q,K,V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$
In the implementation:

```python
wei = q @ k.transpose(-2, -1) * C**-0.5
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ v
```

The triangular mask ensures that a character can only attend to **previous characters**, preventing the model from looking into the future during training.

---

## 🧩 Multi-Head Attention

Rather than using a single attention mechanism, KindaGPT uses **four attention heads**.

```python
Block(n_embd, num_heads=4)
```

Each head independently learns different relationships between tokens.

The outputs are concatenated and passed through a projection layer:

```text
                ┌── Attention Head 1 ──┐
                ├── Attention Head 2 ──┤
Input ──────────┼── Attention Head 3 ──┼──► Concatenate ─► Projection
                └── Attention Head 4 ──┘
```

This allows the model to capture different types of relationships within the same sequence.

---

## 🏗️ Transformer Block

KindaGPT contains **three Transformer blocks**.

Each block consists of:

1. Layer Normalization
    
2. Multi-Head Self-Attention
    
3. Residual Connection
    
4. Layer Normalization
    
5. Feed-Forward Network
    
6. Residual Connection
    

Conceptually:

```text
Input
  │
  ▼
LayerNorm
  │
  ▼
Multi-Head Attention
  │
  ▼
Residual Connection
  │
  ▼
LayerNorm
  │
  ▼
Feed-Forward Network
  │
  ▼
Residual Connection
  │
  ▼
Output
```

The feed-forward network expands the embedding dimension by a factor of four before projecting it back:

```text
32 → 128 → 32
```

---

## 📍 Positional Embeddings

Self-attention alone does not inherently understand the order of tokens.

KindaGPT therefore combines token embeddings with learned positional embeddings:

```python
tok_embed = self.token_embedding_table(idx)
pos_embed = self.position_embedding_table(torch.arange(T, device=device))

x = tok_embed + pos_embed
```

With:

```text
Embedding dimension = 32
Context length      = 8
```

This gives the model information about **where each character occurs within the sequence**.

---

## 🎯 Learning Objective

The model is trained as an autoregressive language model.

For every position in the sequence, it predicts the probability distribution of the next character.

The training objective uses **cross-entropy loss**:

```python
loss = F.cross_entropy(logits, targets)
```

Conceptually:

$$
L=-\sum_i y_i \log(\hat{y}_i)
$$
The model adjusts its parameters to increase the probability assigned to the correct next character.

---

## ⚙️ Model Configuration

The project intentionally uses a small architecture so that the mechanics of the model remain understandable.

| Parameter              |         Value |
| ---------------------- | ------------: |
| Embedding dimension    |            32 |
| Context length         |             8 |
| Attention heads        |             4 |
| Transformer blocks     |             3 |
| Feed-forward expansion |            4× |
| Batch size             |            32 |
| Learning rate          |         0.001 |
| Training iterations    |         6,000 |
| Evaluation interval    |           300 |
| Evaluation iterations  |           200 |
| Optimizer              |         AdamW |
| Loss                   | Cross-Entropy |

The model automatically uses CUDA when available:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

## 🔄 Training Loop

The training process follows the standard language-model training cycle:

```text
Sample batch
    │
    ▼
Forward pass
    │
    ▼
Calculate cross-entropy loss
    │
    ▼
Zero gradients
    │
    ▼
Backpropagation
    │
    ▼
Update parameters
    │
    ▼
Repeat
```

Implemented using:

```python
logits, loss = model(xb, yb)

optimizer.zero_grad(set_to_none=True)

loss.backward()

optimizer.step()
```

Training and validation loss are periodically evaluated to monitor learning.

---

## ✍️ Text Generation

Once training is complete, KindaGPT can generate new text autoregressively.

The generation process starts with an initial context:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
```

The model then:

1. Predicts the next character.
    
2. Converts logits into probabilities using softmax.
    
3. Samples a character from the probability distribution.
    
4. Appends the character to the sequence.
    
5. Uses the expanded sequence to predict the next character.
    
6. Repeats.
    

```text
Context
   │
   ▼
Predict next character
   │
   ▼
Sample from probability distribution
   │
   ▼
Append character
   │
   ▼
Repeat
```

The current implementation generates up to **500 new characters**.

---

## 🛠️ Technical Challenges

### Understanding Attention

One of the main challenges was understanding how queries, keys, and values interact to determine which parts of a sequence should receive attention.

The matrix operations:

```python
q @ k.transpose(-2, -1)
```

produce an attention score for every token against every other token.

---

### Preventing Information Leakage

During training, the model must not be allowed to use future characters when predicting the current character.

A causal mask is therefore applied:

```python
self.tril = torch.tril(
    torch.ones(block_size, block_size)
)
```

This creates the characteristic lower-triangular attention pattern used in autoregressive Transformers.

---

### Understanding Tensor Shapes

Another important part of the implementation was keeping track of the transformations between:

```text
(Batch, Time, Channels)
```

and:

```text
(Batch, Time, Time)
```

for attention scores.

Understanding these dimensions makes the mechanics of self-attention much easier to reason about.

---

## 🧰 Tech Stack

- **Python**
    
- **PyTorch**
    
- **Neural Networks**
    
- **Self-Attention**
    
- **Multi-Head Attention**
    
- **Transformer Architecture**
    
- **Character-Level Language Modeling**
    
- **Autoregressive Text Generation**
    

---

## 🚀 Running the Project

### 1. Install dependencies

```bash
pip install torch
```

### 2. Prepare the dataset

Place the training text inside:

```text
input.txt
```

### 3. Run the model

```bash
python model.py
```

The model will train and then generate text based on what it learned from the input corpus.

---

## 🔮 Future Improvements

There are several directions this project could be extended:

- Increase the context length.
    
- Train on a larger corpus.
    
- Implement dropout.
    
- Add temperature-controlled generation.
    
- Experiment with different embedding dimensions.
    
- Add more Transformer blocks.
    
- Compare character-level and subword tokenization.
    
- Implement a more efficient tokenizer such as BPE.
    
- Add learning-rate scheduling.
    
- Track and visualize training/validation loss.
    
- Save and reload trained model checkpoints.
    
- Experiment with different attention architectures.
    

---

## 💭 Why This Project Matters

KindaGPT was primarily an exercise in **understanding Transformers by implementing their core components rather than treating them as a black box**.

Building the model from scratch helped connect the mathematical concepts behind attention, embeddings, probability distributions, and gradient-based optimization with an actual working language model.

More importantly, it provides a foundation for understanding how larger models such as modern **GPT-style LLMs** are constructed and why components like attention, positional information, residual connections, and autoregressive training are so important.

---

## 🧩 Final Thoughts

KindaGPT is intentionally small, but the underlying architecture demonstrates many of the fundamental ideas behind modern language models.

From converting raw text into tokens to computing attention and repeatedly predicting the next character, the project provides a hands-on look at how a Transformer can learn patterns in sequential data and use those patterns to generate new text.

**The goal wasn't to build the next GPT — it was to understand how GPT works.**