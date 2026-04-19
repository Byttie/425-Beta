import os
import sys
import argparse
import pretty_midi
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.lstm_auto_encoder import MusicAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def matrix_to_midi(piano_roll_matrix, filename, fs=4, threshold=0.5):
    binary_matrix = (piano_roll_matrix > threshold).astype(int) #[64, 88]
    
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0) #program=0 is piano
    
    for note_idx in range(88):
        actual_pitch = note_idx + 21 #0-87 -> 21-108
        
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


def generate_samples(model, output_dir, num_samples=5, fs=4, threshold=0.5):
    model.eval()
    with torch.no_grad():
        latent_dim = model.decoder.linear.in_features # 128
        random_latent = torch.randn(num_samples, latent_dim).to(device) #encoded latent vector
        generated_sequences = model.decoder(random_latent).cpu().numpy() #decoded latent vector
        for i in range(num_samples):
            filename = os.path.join(output_dir, f"generated_sample_{i+1}.mid")
            matrix_to_midi(generated_sequences[i], filename, fs=fs, threshold=threshold)
            print(f"Saved: {filename}")


def load_trained_model(model_path, seq_len=64):
    model = MusicAutoencoder(input_dim=88, seq_len=seq_len).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=os.path.join(PROJECT_ROOT, "model", "music_autoencoder.pth"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(PROJECT_ROOT, "generation", "outputs"),
    )
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--fs", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()


    model = load_trained_model(args.model_path, seq_len=args.seq_len)
    generate_samples(
        model=model,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        fs=args.fs,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    
    main()