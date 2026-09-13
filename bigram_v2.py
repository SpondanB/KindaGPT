import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 6000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32


# for reproducibility, we set the random seed to a fixed value.
# torch.manual_seed(1337)

# reading the input data.
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# creating the set of unique characters that are present in the text.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# to tokenize the text, we can use the following code:
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# putting the data into a tensor.
data = torch.tensor(encode(text), dtype=torch.long)

# making the training and test data.
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# creating the batches of the data - data loader
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# creating a function to estimate the loss on the train and val sets. 
# it is important to note that we are using the @torch.no_grad() decorator to disable gradient calculation, since we are only interested in the loss values and not in updating the model parameters.
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Head module for the self attention mechanism
class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # complyting the attention scores for each query with all keys
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # multiplying the attention scores with the values to get the final output
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # projection layer for the output of the multi-head attention mechanism

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out) # the out is going through the projection layer to get the final output of the multi-head attention mechanism

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer for the output of the feedforward network
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_heads = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# defining a simple Bigram language model.
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # this is a lookup table that maps each token to a vector of size n_embd. The embedding layer takes in the vocab_size and n_embd as input and creates a matrix of size (vocab_size, n_embd) where each row corresponds to a token and each column corresponds to a dimension of the embedding space.
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # this is a lookup table that maps each position in the input sequence to a vector of size n_embd. The embedding layer takes in the block_size and n_embd as input and creates a matrix of size (block_size, n_embd) where each row corresponds to a position in the input sequence and each column corresponds to a dimension of the embedding space.
        self.blocks = nn.Sequential(
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            Block(n_embd, num_heads=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size) # this is a linear layer that takes in the embedding vector and outputs a vector of size vocab_size. The linear layer takes in the n_embd and vocab_size as input and creates a matrix of size (n_embd, vocab_size) where each row corresponds to a dimension of the embedding space and each column corresponds to a token.

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_embed = self.token_embedding_table(idx) # (Batch ,Time ,Channels)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (Time ,Channels)
        x = tok_embed + pos_embed # (Batch ,Time ,Channels)
        x = self.blocks(x) # (Batch ,Time ,Channels) - passing the input through the transformer blocks
        logits = self.lm_head(x) # (Batch ,Time ,Vocab_Size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets) # basically we want to measure the loss between the predicted logits and the actual targets. We use cross-entropy loss for this purpose.
            # pytorch's cross_entropy function expects the input to be of shape (N, C) where N is the number of samples and C is the number of classes. So we reshape the logits and targets to be of shape (B*T, vocab_size) and (B*T,) respectively.

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # croping the context to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond, targets=None)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# setting up the PyTorch optimizer.
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for iter in range(max_iters):

    # check the loss on train and val sets every eval_interval iterations.
    if iter % eval_interval == 0 :
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist())) # generate 500 new characters from the model