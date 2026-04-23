import os
import sys
import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.lstm_vae import MusicVAE
from preprocess.loadmidi import preprocess_maestro_file, segment_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_maestro_sequences(
    dataset_root,
    metadata_filename="maestro-v3.0.0.csv",
    split="train",
    fs=4,
    window_size=64,
    max_files=None,
):
    metadata_path = os.path.join(dataset_root, metadata_filename)
    all_segments = []
    loaded_files = 0

    with open(metadata_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["split"] == split:
                midi_rel_path = row["midi_filename"]
                midi_path = os.path.join(dataset_root, midi_rel_path)
                piano_roll = preprocess_maestro_file(midi_path, fs=fs)
                segments = segment_sequences(piano_roll, window_size=window_size)
                all_segments.append(segments)
                loaded_files += 1
                print(f"Loaded {loaded_files}: {midi_rel_path} -> {len(segments)} segments")
                if max_files and loaded_files >= max_files:
                    break

    return np.concatenate(all_segments, axis=0)


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def train_vae(model, dataloader, epochs=100, lr=1e-3, beta=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        for (x,) in dataloader:
            x = x.to(device)
            optimizer.zero_grad()

            x_hat, mu, logvar = model(x)

            recon_loss = F.mse_loss(x_hat, x, reduction="mean")
            kl_loss = kl_divergence(mu, logvar).mean()
            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        n_batches = len(dataloader)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Total: {epoch_total / n_batches:.6f} | "
            f"Recon: {epoch_recon / n_batches:.6f} | "
            f"KL: {epoch_kl / n_batches:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train LSTM-VAE on MAESTRO dataset.")
    parser.add_argument("--dataset_root", default=os.path.join(PROJECT_ROOT, "dataset"))
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--fs", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument(
        "--output_model",
        default=os.path.join(PROJECT_ROOT, "model", "music_vae.pth"),
    )
    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset_root}")
    sequences = load_maestro_sequences(
        dataset_root=args.dataset_root,
        split=args.split,
        fs=args.fs,
        window_size=args.window_size,
        max_files=args.max_files,
    )
    print(f"Total sequences: {len(sequences)}")

    data = torch.tensor(sequences, dtype=torch.float32)
    _, seq_len, input_dim = data.shape
    print(f"Loaded data shape: {tuple(data.shape)}")

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = MusicVAE(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    train_vae(model, dataloader, epochs=args.epochs, lr=args.lr, beta=args.beta)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(model.state_dict(), args.output_model)
    print(f"Saved trained VAE to: {args.output_model}")


if __name__ == "__main__":
    main()
