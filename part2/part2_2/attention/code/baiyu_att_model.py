from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from selfattention import SelfAttention


class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.attn = SelfAttention(config)
        self.fc = nn.Linear(config.hidden_dim, config.num_labels)

    def forward(self, x):
        batch_size, seq_len = x.shape
        h = self.emb(x)
        attn_score, h = self.attn(h)
        h = F.avg_pool1d(h.permute(0, 2, 1), seq_len, 1)
        h = h.squeeze(-1)
        logits = self.fc(h)
        return attn_score, logits


@dataclass
class Config:
    vocab_size: int = 5000  # 词表大小
    hidden_dim: int = 512
    num_heads: int = 16
    head_dim: int = 32
    dropout: float = 0.1

    num_labels: int = 2  # 标签数量，这里是二分类，所以2个标签

    max_seq_len: int = 512

    num_epochs: int = 10


if __name__ == '__main__':
    config = Config(5000, 512, 16, 32, 0.1, 2)
    model = Model(config)
    x = torch.randint(0, 5000, (3, 30))
    print(x.shape)
    attn, logits = model(x)
    print(attn.shape, logits.shape)
