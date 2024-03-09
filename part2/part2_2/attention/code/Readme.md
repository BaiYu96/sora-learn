## 实践体验

### 环境准备

```
conda create --name sora python=3.10 -y
conda activate sora
pip install torch torchvision torchaudio pandas scikit-learn jieba matplotlib
```

### 模型

baiyu_att_model.py

```python
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
    
    vocab_size: int = 5000 # 词表大小
    hidden_dim: int = 512
    num_heads: int = 16
    head_dim: int = 32
    dropout: float = 0.1
    
    num_labels: int = 2 # 标签数量，这里是二分类，所以2个标签
    
    max_seq_len: int = 512
    
    num_epochs: int = 10

if __name__ = __main__():
    config = Config(5000, 512, 16, 32, 0.1, 2)
    model = Model(config)
    x = torch.randint(0, 5000, (3, 30))
    print(x.shape)
    attn, logits = model(x)
    print(attn.shape, logits.shape)

```

### 数据

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import Tokenizer

file_path = "./data/ChnSentiCorp_htl_all.csv"
df = pd.read_csv(file_path)
df = df.dropna()
print(df.head(), df.shape)

print(df.label.value_counts())

# 重采样数据
df = pd.concat([df[df.label==1].sample(2500), df[df.label==0]])
print(df.shape) # 查看数据集大小
print(df.label.value_counts()) # 分类数据统计

tokenizer = Tokenizer(config.vocab_size, config.max_seq_len)

tokenizer.build_vocab(df.review)

tokenizer(["你好", "你好呀"])

def collate_batch(batch):
    label_list, text_list = [], []
    for v in batch:
        _label = v["label"]
        _text = v["text"]
        label_list.append(_label)
        text_list.append(_text)
    inputs = tokenizer(text_list)
    labels = torch.LongTensor(label_list)
    return inputs, labels

from dataset import Dataset

ds = Dataset()
ds.build(df, "review", "label")

print(len(ds), ds[0])

train_ds, test_ds = train_test_split(ds, test_size=0.2)
train_ds, valid_ds = train_test_split(train_ds, test_size=0.1)
len(train_ds), len(valid_ds), len(test_ds)

from torch.utils.data import DataLoader
BATCH_SIZE = 8
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_batch)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, collate_fn=collate_batch)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_batch)
len(train_dl), len(valid_dl), len(test_dl)


```

## 训练

```python
from trainer import train, test
from baiyu_att_model import Model

NUM_EPOCHS = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = Config(5000, 64, 1, 64, 0.1, 2)
model = Model(config)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
train(model, optimizer, train_dl, valid_dl, config)

test(model, test_dl)
```

