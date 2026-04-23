import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=88, hidden_dim=256, latent_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        h_last = hidden[-1]
        mu = self.mu_head(h_last)
        logvar = self.logvar_head(h_last)
        return mu, logvar


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, output_dim=88, seq_len=64, num_layers=3, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = self.proj(z)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(x)
        return torch.sigmoid(self.output_layer(out))


class MusicVAE(nn.Module):
    def __init__(self, input_dim=88, seq_len=64, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            seq_len=seq_len,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar
