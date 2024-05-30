import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer

# hyperparameters
vocab_size = 10000
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 4
dropout = 0.2
# ------------

torch.manual_seed(1337)

corpus_root = '/Users/slebst/Documents/Multimedia/eBooks/Literature/'
with open(corpus_root+'moliere-oeuvres_completes_1_processed.txt', 'r', encoding='utf-8') as f1:
    raw_moliere_1_lines = f1.readlines()
with open(corpus_root+'moliere-oeuvres_completes_2_processed.txt', 'r', encoding='utf-8') as f2:
    raw_moliere_2_lines = f2.readlines()
with open(corpus_root+'moliere-oeuvres_completes_3_processed.txt', 'r', encoding='utf-8') as f3:
    raw_moliere_3_lines = f3.readlines()

raw_moliere_lines = raw_moliere_1_lines + raw_moliere_2_lines + raw_moliere_3_lines

def get_training_corpus(chunk_size: int=1500):
    raw_moliere_chunked = []
    moliere_n_lines = len(raw_moliere_lines)
    line_iter = 0
    while (line_iter < moliere_n_lines):
        raw_moliere_chunked.append('\n'.join(raw_moliere_lines[line_iter:min(line_iter+chunk_size,moliere_n_lines)]))
        line_iter += chunk_size
    return (raw_moliere_chunked)


bert_wp_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
training_corpus = get_training_corpus()
tokenizer = bert_wp_tokenizer.train_new_from_iterator(training_corpus, vocab_size)

raw_moliere = '\n'.join(raw_moliere_lines)
encoded_moliere = tokenizer(raw_moliere)['input_ids']

n = int(0.9*len(encoded_moliere))
train_data = encoded_moliere[:n]
val_data = encoded_moliere[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.tensor([data[i:i+block_size] for i in ix])
    y = torch.tensor([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))    # register attribute that is NOT learned (ie not parameters)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                                 # (B, T, head_size)
        q = self.query(x)                                               # (B, T, head_size)
        # compute attention scores ~ affinities
        wei = q @ k.transpose(-2, -1) * C**-0.5                         # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)                                    # (B, T, T)
        wei = self.dropout(wei)                                        # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)                                              # (B, T, head_size)
        out = wei @ v                                                   # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Language model with attention heads
class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = LanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))