import os
import sys
import argparse
import pretty_midi
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.lstm_vae import MusicVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def matrix_to_midi(piano_roll_matrix, filename, fs=4, threshold=0.5):
    binary_matrix = (piano_roll_matrix > threshold).astype(int)

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for note_idx in range(88):
        actual_pitch = note_idx + 21
        is_playing = False
        start_time = 0

        for time_step, state in enumerate(binary_matrix[:, note_idx]):
            if state == 1 and not is_playing:
                start_time = time_step / fs
                is_playing = True
            elif state == 0 and is_playing:
                end_time = time_step / fs
                note = pretty_midi.Note(velocity=100, pitch=actual_pitch, start=start_time, end=end_time)
                instrument.notes.append(note)
                is_playing = False

        if is_playing:
            end_time = len(binary_matrix) / fs
            note = pretty_midi.Note(velocity=100, pitch=actual_pitch, start=start_time, end=end_time)
            instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(filename)


def load_trained_model(model_path, seq_len=64, hidden_dim=256, latent_dim=128):
    model = MusicVAE(input_dim=88, seq_len=seq_len, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def generate_samples(model, output_dir, num_samples=5, fs=4, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        latent_dim = model.decoder.proj.in_features
        random_latent = torch.randn(num_samples, latent_dim, device=device)
        generated_sequences = model.decoder(random_latent).cpu().numpy()

        for i in range(num_samples):
            filename = os.path.join(output_dir, f"vae_generated_sample_{i+1}.mid")
            matrix_to_midi(generated_sequences[i], filename, fs=fs, threshold=threshold)
            print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI samples with a trained LSTM-VAE.")
    parser.add_argument(
        "--model_path",
        default=os.path.join(PROJECT_ROOT, "model", "music_vae.pth"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_ROOT, "generation", "outputs_vae"),
    )
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--fs", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    model = load_trained_model(
        model_path=args.model_path,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    )
    generate_samples(
        model=model,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        fs=args.fs,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
