import torch
import torch.nn as nn
import torch.nn.functional as F

class TextLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

def load_text_model(model_path, vocab_size, embed_dim, hidden_dim, num_classes):
    model = TextLSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def build_vocab(texts):
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for text in texts:
        for word in text.lower().split():
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
    return word2idx

def preprocess_text(text, word2idx, max_len=20):
    tokens = text.lower().split()
    ids = [word2idx.get(w, word2idx['<UNK>']) for w in tokens][:max_len]
    ids += [word2idx['<PAD>']] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)

def infer_text(model, text, word2idx):
    input_tensor = preprocess_text(text, word2idx)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        topk = torch.topk(probs, k=5)
        return topk.indices[0].tolist(), topk.values[0].tolist()  # 返回类别索引和概率 