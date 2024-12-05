import os
import pickle
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from music21 import stream, note, instrument, tempo

# Set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Step 1: Define the Model Architecture (same as during training)
def build_model(sequence_length, n_unique_notes):
    model = Sequential()
    model.add(LSTM(512, input_shape=(sequence_length, 5), return_sequences=True))  # Adjusted to 5 input features
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_unique_notes, activation='softmax'))
    return model


# Step 2: Load the Trained Model Weights
sequence_length = 100  # The sequence length you used during training
n_unique_notes = 576  # Adjust this based on your unique notes

# Build the model
model = build_model(sequence_length, n_unique_notes)

# Load the weights from the latest checkpoint
latest_checkpoint = '/Users/cristovalneosasono/AI-FinalProject-Models/LSTM/trained-models/weights-epoch-30-loss-1.7863-acc-0.5304.weights.h5'
model.load_weights(latest_checkpoint)

# Step 4: Load the Mapping Data (int to note)
with open('../Subset_Dataset/note_to_int.pkl', 'rb') as f:
    note_to_int = pickle.load(f)

int_to_note = {number: note for note, number in note_to_int.items()}

# Identify rest indices
rest_indices = [index for index, note in int_to_note.items() if note.lower() == 'rest' or note == 'R']

# Step 5: Load the Data and Concatenate Start Sequence
with open('../Subset_Dataset/input_sequences_notes.pkl', 'rb') as f:
    X_notes = pickle.load(f)
with open('../Subset_Dataset/input_sequences_durations.pkl', 'rb') as f:
    X_durations = pickle.load(f)
with open('../Subset_Dataset/input_sequences_tempos.pkl', 'rb') as f:
    X_tempos = pickle.load(f)
with open('../Subset_Dataset/input_sequences_time_signatures.pkl', 'rb') as f:
    X_time_signatures = pickle.load(f)
with open('../Subset_Dataset/input_sequences_key_signatures.pkl', 'rb') as f:
    X_key_signatures = pickle.load(f)

# Concatenate all features to create the full start sequence
X = np.concatenate([X_notes, X_durations, X_tempos, X_time_signatures, X_key_signatures], axis=-1)

# Choose a random start sequence from the training data
start_sequence = X[np.random.randint(0, len(X))]

# Define the C Major scale notes explicitly (no sharps or flats)
c_major_notes = ['C', 'D', 'E', 'F', 'G', 'A', ]
# c_major_notes = ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'] # F Major rn

def generate_music(model, start_sequence, int_to_note, rest_indices, n_generate=100, temperature=1.0):
    generated_notes = []
    generated_durations = []
    current_sequence = np.reshape(start_sequence, (1, len(start_sequence), 5))  # Reshape seed sequence to match model input

    # Define C Major scale notes and desired MIDI range
    c_major_notes = ['C', 'D', 'E', 'F', 'G', 'A']  # Adjust if you want to include 'B'
    desired_octaves = [4]
    min_midi = 60
    max_midi = 75

    while len(generated_notes) < n_generate:
        # Predict the next note
        prediction = model.predict(current_sequence, verbose=0)[0]

        # Zero out the probabilities for rests
        prediction[rest_indices] = 0

        # Apply temperature scaling
        prediction = prediction ** (1 / temperature)
        prediction = prediction / np.sum(prediction)  # Renormalize

        # Filter the prediction to only valid notes in C major scale and desired octave
        valid_indices = []
        for idx, note_str in int_to_note.items():
            # Skip chords and invalid patterns
            if '.' in note_str or note_str.isdigit():
                continue
            # Get the note properties
            try:
                n = note.Note(note_str)
                note_name = n.pitch.name  # Note name (e.g., 'C')
                note_midi = n.pitch.midi  # MIDI number
                note_octave = n.octave    # Octave number
            except:
                continue  # Skip if the note_str cannot be parsed into a note

            # Check if the note is in C major, desired octave, and within the desired MIDI range
            if (
                note_name in c_major_notes and
                note_octave in desired_octaves and
                min_midi <= note_midi <= max_midi
            ):
                valid_indices.append(idx)

        # If no valid notes remain, fall back to random snapping or handle accordingly
        if len(valid_indices) == 0:
            print(f"No valid notes in scale and range at position {len(generated_notes)}/{n_generate}.")
            # Optionally, select a random note from C major within the desired octave
            valid_note_names = [n + str(o) for n in c_major_notes for o in desired_octaves]
            snapped_note = random.choice(valid_note_names)
            print(f"Snapped to random valid note: {snapped_note}")
            generated_notes.append(snapped_note)
            generated_durations.append(1.0 if len(generated_notes) % 2 == 0 else 0.5)
        else:
            # Create a new prediction array with only valid notes
            adjusted_prediction = np.zeros_like(prediction)
            for idx in valid_indices:
                adjusted_prediction[idx] = prediction[idx]
            adjusted_prediction = adjusted_prediction / np.sum(adjusted_prediction)  # Re-normalize

            # Select the next note based on adjusted AI probabilities
            index = np.random.choice(len(adjusted_prediction), p=adjusted_prediction)
            predicted_pattern = int_to_note[index]

            # Append the predicted note and duration
            generated_notes.append(predicted_pattern)
            generated_durations.append(1.0 if len(generated_notes) % 2 == 0 else 0.5)

            print(f"AI selected valid note: {predicted_pattern} with adjusted probability.")

            # Prepare the next input sequence
            next_features = [index] * 5  # Replace with appropriate features if necessary
            next_features_array = np.array([next_features]).reshape((1, 1, 5))
            current_sequence = np.append(current_sequence, next_features_array, axis=1)
            current_sequence = current_sequence[:, 1:]  # Keep sequence length constant

    return generated_notes, generated_durations


def create_midi(generated_notes, generated_durations, output_file='generated_music_melody.mid', tempo_bpm=120):
    output_stream = stream.Stream()

    # Add tempo to the stream
    output_tempo = tempo.MetronomeMark(number=tempo_bpm)
    output_stream.append(output_tempo)

    for i, note_name in enumerate(generated_notes):
        duration = generated_durations[i]

        # Create a new note object and assign its duration
        new_note = note.Note(note_name)
        new_note.duration.quarterLength = duration  # Set the note's duration
        new_note.storedInstrument = instrument.Piano()  # Ensure it's for piano

        output_stream.append(new_note)  # Add the note to the stream

    # Save the Stream to a MIDI file
    output_stream.write('midi', fp=output_file)


# Step 1: Generate 100 notes without chords (melodies only)
generated_notes, generated_durations = generate_music(
    model,
    start_sequence,
    int_to_note,
    rest_indices,
    n_generate=75,  # Ensure we generate exactly 100 melody notes
    temperature=3
)

# Step 2: Create the MIDI file from the generated notes
create_midi(
    generated_notes,
    generated_durations,
    output_file='/Users/cristovalneosasono/AI-FinalProject-Models/generated_music_result/A_generated_music_melody_E30.mid',
    tempo_bpm=120  # Adjust the tempo here
)

print("Music generation complete. Saved to 'A_generated_music_melody.mid'.")