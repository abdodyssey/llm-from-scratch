import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.block_size = 8
        self.n_embd = 32

        # Token Embedding
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            self.n_embd
        )

        # Positional Embedding
        self.position_embedding_table = nn.Embedding(
            self.block_size,
            self.n_embd
        )

        # Language Model Head
        self.lm_head = nn.Linear(
            self.n_embd,
            vocab_size
        )
        
    def forward(self, idx, targets=None):

        B, T = idx.shape

        # Token embeddings
        token_embeddings = self.token_embedding_table(idx)

        # Position embeddings
        positions = torch.arange(T, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)

        # Combine token + position
        x = token_embeddings + position_embeddings

      
        # Project to vocabulary
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
            