import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output.reshape(-1, self.hidden_dim))
        return output.view(batch_size, -1, output.size(-1))
