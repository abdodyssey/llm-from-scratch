import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)

        # Scaling
        wei = wei * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)

        out = wei @ v

        if not hasattr(self, "_printed_scores"):
            print("Attention Scores:", wei.shape)

        if not hasattr(self, "_printed_out"):
            print("Attention Output:", out.shape)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)

        if not hasattr(self, "_printed_multi"):
            print("MultiHead Output:", out.shape)
            self._printed_multi = True

        if not hasattr(self, "_printed_proj"):
            print("Projection Output:", out.shape)
            self._printed_proj = True

        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):

        out = self.net(x)

        if not hasattr(self, "_printed_ffn"):
            print("FeedForward Output:", x.shape)
            self._printed_ffn = True

        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
        )

        self.ffwd = FeedForward(n_embd)

        # Untuk attention
        self.ln1 = nn.LayerNorm(n_embd)
        # Untuk Feed Forward
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.block_size = 8
        self.n_embd = 32

        # Token Embedding
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)

        # Positional Embedding
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)

        self.block = Block(
            n_embd=self.n_embd,
            n_head=4,
            block_size=self.block_size,
        )

        # ===========================
        # SIMPLE NEURAL NETWORK MODEL
        # ===========================

        # Language Model Head
        self.lm_head = nn.Linear(self.n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if not hasattr(self, "_printed_input"):
            print("Input IDs          :", idx.shape)
            self._printed_input = True

        # Token embeddings
        token_embeddings = self.token_embedding_table(idx)

        if not hasattr(self, "_printed_token"):
            print("Token Embedding    :", token_embeddings.shape)
            self._printed_token = True

        # Position embeddings
        positions = torch.arange(T, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)

        if not hasattr(self, "_printed_position"):
            print("Position Embedding :", position_embeddings.shape)
            self._printed_position = True

        x = token_embeddings + position_embeddings

        if not hasattr(self, "_printed_sum"):
            print("Embedding Output   :", x.shape)
            self._printed_sum = True

        # =================
        # TRANSFORMER BLOCK
        # =================
        x = self.block(x)

        if not hasattr(self, "_printed_block"):
            print("Block Output       :", x.shape)
            self._printed_block = True

        logits = self.lm_head(x)

        if not hasattr(self, "_printed_logits"):
            print("Logits             :", logits.shape)
            self._printed_logits = True

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            # =================================================
            # SIMPLE NEURAL NETWORK MODEL / MODEL NEURAL BIGRAM
            # =================================================
            # CROSS ENTROPY LOSS
            loss = F.cross_entropy(logits, targets)

        return logits, loss
