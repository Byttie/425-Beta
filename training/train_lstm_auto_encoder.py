import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.lstm_auto_encoder import MusicAutoencoder
from preprocess.loadmidi import preprocess_maestro_file, segment_sequences
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_autoencoder(model, dataloader, epochs=50, lr=0.001):
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)
            
            optimizer.zero_grad()
            output = model(batch)
            
            loss = loss_function(output, batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title('LSTM Autoencoder Reconstruction Loss')
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('reconstruction_loss_curve.png')
    plt.show()
    
    return model


def load_maestro_sequences(dataset_root, metadata_filename="maestro-v3.0.0.csv", split="train", fs=4, window_size=64, max_files=None):
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


def main():
    parser = argparse.ArgumentParser(description="Train LSTM autoencoder on MAESTRO dataset.")
    parser.add_argument("--dataset_root", default=os.path.join(PROJECT_ROOT, "dataset"))
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--fs", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001,)
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--output_model", default=os.path.join(PROJECT_ROOT, "model", "music_autoencoder.pth"))
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

    tensor_data = torch.tensor(sequences, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = MusicAutoencoder(input_dim=88, seq_len=args.window_size).to(device)
    model = train_autoencoder(model, dataloader, epochs=args.epochs, lr=args.lr)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    torch.save(model.state_dict(), args.output_model)
    print(f"Saved trained model to: {args.output_model}")


if __name__ == "__main__":
    main()