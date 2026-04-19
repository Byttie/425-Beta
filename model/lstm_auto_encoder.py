import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=5, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        latent = self.linear(hidden[-1]) 
        return latent

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, output_dim=88, seq_len=64):
        super().__init__()
        self.seq_len = seq_len
        self.linear = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=5, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        lstm_out, _ = self.lstm(x)
        out = torch.sigmoid(self.output_layer(lstm_out)) 
        return out

class MusicAutoencoder(nn.Module):
    def __init__(self, input_dim=88, seq_len=64):
        super().__init__()
        self.encoder = LSTMAutoencoder(input_dim=input_dim)
        self.decoder = LSTMDecoder(output_dim=input_dim, seq_len=seq_len)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction