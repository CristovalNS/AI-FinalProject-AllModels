import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from music21 import stream, note, instrument, tempo
import random


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
latest_checkpoint = '/Users/cristovalneosasono/AI_FinalProject_V2/checkpoints_E30/weights-epoch-05-loss-1.7863-acc-0.5304.weights.h5'
model.load_weights(latest_checkpoint)

# Step 4: Load the Mapping Data (int to note)
with open('Subset_Dataset/note_to_int.pkl', 'rb') as f:
    note_to_int = pickle.load(f)

int_to_note = {number: note for note, number in note_to_int.items()}

# Identify rest indices
rest_indices = [index for index, note in int_to_note.items() if note.lower() == 'rest' or note == 'R']

# Step 5: Load the Data and Concatenate Start Sequence
with open('Subset_Dataset/input_sequences_notes.pkl', 'rb') as f:
    X_notes = pickle.load(f)
with open('Subset_Dataset/input_sequences_durations.pkl', 'rb') as f:
    X_durations = pickle.load(f)
with open('Subset_Dataset/input_sequences_tempos.pkl', 'rb') as f:
    X_tempos = pickle.load(f)
with open('Subset_Dataset/input_sequences_time_signatures.pkl', 'rb') as f:
    X_time_signatures = pickle.load(f)
with open('Subset_Dataset/input_sequences_key_signatures.pkl', 'rb') as f:
    X_key_signatures = pickle.load(f)

# Concatenate all features to create the full start sequence
X = np.concatenate([X_notes, X_durations, X_tempos, X_time_signatures, X_key_signatures], axis=-1)

# Choose a random start sequence from the training data
start_sequence = X[np.random.randint(0, len(X))]

# Define the C Major scale notes explicitly (no sharps or flats)
c_major_notes = ['C', 'D', 'E', 'F', 'G', 'A', ]
# c_major_notes = ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'] # F Major rn

def generate_music(model, start_sequence, int_to_note, rest_indices, n_generate=100, temperature=1.0, max_interval=5):
    generated_notes = []
    generated_durations = []
    current_sequence = np.reshape(start_sequence, (1, len(start_sequence), 5))  # Reshape seed sequence to match model input

    while len(generated_notes) < n_generate:  # Keep generating until we have the desired number of notes
        # Predict the next note
        prediction = model.predict(current_sequence, verbose=0)[0]

        # Zero out the probabilities for rests
        prediction[rest_indices] = 0

        # Apply temperature scaling
        prediction = prediction ** (1 / temperature)
        prediction = prediction / np.sum(prediction)  # Renormalize

        # Filter the prediction to only C major natural notes (no sharps or flats)
        scale_filtered_prediction = np.zeros_like(prediction)
        for idx, note_str in int_to_note.items():
            # Extract the note name without the octave (e.g., C4 -> C) and check if it's in C major
            note_name = note_str[:-1]  # Removes the octave number (e.g., 'C4' -> 'C')
            if note_name in c_major_notes:
                scale_filtered_prediction[idx] = prediction[idx]

        # If no valid notes in the scale, pick a random note from C major
        if np.sum(scale_filtered_prediction) == 0:
            print(f"No valid notes in scale, picking a random valid C major note at {len(generated_notes)}/100.")
            valid_indices = [i for i, note_str in int_to_note.items() if note_str[:-1] in c_major_notes]
            if len(valid_indices) > 0:
                index = np.random.choice(valid_indices)
                print(f"Randomly selected note: {int_to_note[index]}")
            else:
                raise ValueError("No valid notes in C major found in int_to_note.")
        else:
            scale_filtered_prediction = scale_filtered_prediction / np.sum(scale_filtered_prediction)  # Renormalize
            index = np.random.choice(len(scale_filtered_prediction), p=scale_filtered_prediction)
            print(f"AI generated note: {int_to_note[index]} with probability: {scale_filtered_prediction[index]}")

        predicted_pattern = int_to_note[index]

        # Ensure that only single notes (no chords) are selected
        if ('.' in predicted_pattern) or predicted_pattern.isdigit():
            print(f"Skipping chord or invalid pattern: {predicted_pattern}")
            continue  # Skip any chord-like patterns

        # For single notes, ensure they fall within the C major scale (C3 to C5)
        note_midi = note.Note(predicted_pattern).pitch.midi
        if note_midi < 60 or note_midi > 70:
            snapped_note = note.Note(random.choice(c_major_notes)).name
            print(f"Snapped to C Major range: {snapped_note} (original: {predicted_pattern})")
            predicted_pattern = snapped_note

        # Append the predicted note
        generated_notes.append(predicted_pattern)

        # Assign a fixed or predetermined duration pattern
        generated_durations.append(1.0 if len(generated_notes) % 2 == 0 else 0.5)

        print(f"Generated {len(generated_notes)}/{n_generate} notes: {predicted_pattern}")

        # Prepare the next input sequence
        next_features = [index] * 5  # Using the index as a placeholder for the next sequence step
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
    temperature=0.8
)

# Step 2: Create the MIDI file from the generated notes
create_midi(
    generated_notes,
    generated_durations,
    output_file='generated_music_result/A_generated_music_melody_E30.mid',
    tempo_bpm=120  # Adjust the tempo here
)

print("Music generation complete. Saved to 'A_generated_music_melody.mid'.")