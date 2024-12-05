import os
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from music21 import stream, note, instrument, tempo
import random

# Step 1: Define the GRU Model Architecture (same as during training)
def build_gru_model(sequence_length, n_unique_notes):
    model = Sequential()
    model.add(GRU(512, input_shape=(sequence_length, 1), return_sequences=True))  # Adjust input shape to match GRU
    model.add(Dropout(0.3))
    model.add(GRU(512, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_unique_notes, activation='softmax'))
    return model

# Step 2: Load the Trained GRU Model Weights
sequence_length = 50  # The sequence length you used during GRU training
n_unique_notes = 576  # Adjust this based on your unique notes

# Build the GRU model
gru_model = build_gru_model(sequence_length, n_unique_notes)

# Load the weights from the latest GRU checkpoint
latest_gru_checkpoint = '/Users/cristovalneosasono/AI_FinalProject_GRU/checkpoints_GRU_SEED/weights-epoch-10-loss-4.0651-acc-0.2262.weights.h5'
gru_model.load_weights(latest_gru_checkpoint)

# Step 3: Load the Mapping Data (int to note)
with open('Subset_Dataset/note_to_int.pkl', 'rb') as f:
    note_to_int = pickle.load(f)

int_to_note = {number: note for note, number in note_to_int.items()}

# Identify rest indices
rest_indices = [index for index, note in int_to_note.items() if note.lower() == 'rest' or note == 'R']

# Step 4: Prepare the Start Sequence for Generation
with open('Subset_Dataset/input_sequences_notes.pkl', 'rb') as f:
    X_notes = pickle.load(f)

# Choose a random start sequence from the training data
start_sequence = X_notes[np.random.randint(0, len(X_notes))]


def generate_music_gru(model, start_sequence, int_to_note, rest_indices, n_generate=100, temperature=1.0):
    generated_notes = []
    generated_durations = []
    current_sequence = np.reshape(start_sequence, (1, len(start_sequence), 1))  # Reshape to match GRU input

    # Define C Major scale and octave of C4
    # valid_scale_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    valid_scale_notes = ['C', 'D', 'E', 'F', 'G', 'A']
    valid_octave = 4  # C4 to B4

    durations = [0.5, 1.0]  # Example durations

    while len(generated_notes) < n_generate:
        # Predict the next note
        prediction = model.predict(current_sequence, verbose=0)[0]

        # Zero out probabilities for rests
        prediction[rest_indices] = 0

        # Apply temperature scaling
        prediction = prediction ** (1 / temperature)
        prediction = prediction / np.sum(prediction)  # Renormalize

        # Select the next note
        index = np.random.choice(len(prediction), p=prediction)
        predicted_note = int_to_note[index]

        # Skip chords (notes containing a dot, e.g., '11.2') and invalid patterns
        if '.' in predicted_note or predicted_note.isdigit():
            print(f"Skipping chord or invalid note: {predicted_note}")
            continue

        # Check if the predicted note is valid
        if predicted_note[:-1] in valid_scale_notes and predicted_note[-1] == str(valid_octave):
            print(f"Note kept: {predicted_note}")
            snapped_note = predicted_note  # Keep the note as-is
        else:
            print(f"Note {predicted_note} is outside constraints. Snapping...")

            # Filter probabilities to keep only valid scale notes in the C4 octave
            scale_filtered_prediction = np.zeros_like(prediction)
            for idx, note_name in int_to_note.items():
                if note_name[:-1] in valid_scale_notes and note_name[-1] == str(valid_octave):
                    scale_filtered_prediction[idx] = prediction[idx]

            # If no valid notes remain, fall back to random snapping
            if np.sum(scale_filtered_prediction) == 0:
                snapped_note = random.choice(valid_scale_notes) + str(valid_octave)
                print(f"Snapped note NOT using AI probabilities")
            else:
                # Re-normalize probabilities
                scale_filtered_prediction = scale_filtered_prediction / np.sum(scale_filtered_prediction)

                # Select the snapped note based on AI probabilities
                index = np.random.choice(len(scale_filtered_prediction), p=scale_filtered_prediction)
                snapped_note = int_to_note[index]
                # print(scale_filtered_prediction)
                print(f"Snapped note using AI probabilities")

            print(f"Snapped note: {predicted_note} -> {snapped_note}")

        # Append the snapped note and a fixed duration
        generated_notes.append(snapped_note)
        generated_durations.append(random.choice(durations))

        # Print the generated note and its duration
        print(f"Generated note: {snapped_note}, Duration: {generated_durations[-1]}")

        # Update the current sequence with the new prediction
        next_features = np.array([[[index]]])  # Ensure 3D shape (1, 1, 1)
        current_sequence = np.append(current_sequence, next_features, axis=1)
        current_sequence = current_sequence[:, 1:]  # Keep sequence length constant

    return generated_notes, generated_durations


# Step 6: Convert Generated Notes into a MIDI File
def create_midi(generated_notes, generated_durations, output_file='generated_music_gru.mid', tempo_bpm=120):
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

# Step 7: Generate 100 notes with GRU, filtered to C Major scale
# valid_scale_notes = ['C', 'D', 'E', 'F', 'G', 'A']  # Adjust this list for your desired scale
generated_notes, generated_durations = generate_music_gru(
    gru_model,
    start_sequence,
    int_to_note,
    rest_indices,
    n_generate=100,
    temperature=3,
    # valid_scale_notes=valid_scale_notes
)

# Save the generated sequence to a text file for analysis
if not os.path.exists('generated_music_result'):
    os.makedirs('generated_music_result')

with open('generated_music_result/generated_notes_gru.txt', 'w') as f:
    f.write('\n'.join(generated_notes))

# Step 8: Create the MIDI file from the generated notes
create_midi(
    generated_notes,
    generated_durations,
    output_file='generated_music_result/generated_music_gru_E10S.mid',
    tempo_bpm=120  # Adjust the tempo here
)

print("Music generation complete. Saved to 'generated_music_gru_E10S.mid'.")
