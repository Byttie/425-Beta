import numpy as np
import pretty_midi

def preprocess_maestro_file(file_path, fs=4):
    pm = pretty_midi.PrettyMIDI(file_path)
    piano_roll = pm.get_piano_roll(fs=fs)
    print(piano_roll.shape)
    piano_roll = piano_roll[21:109, :]
    piano_roll = (piano_roll > 0).astype(np.float32)
    
    return piano_roll.T

def segment_sequences(data, window_size=64):

    segments = []
    for i in range(0, len(data) - window_size, window_size):
        segments.append(data[i:i + window_size])
    return np.array(segments)