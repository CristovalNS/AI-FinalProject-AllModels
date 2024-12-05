import os
import pickle
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import json

# Set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# Step 1: Load Preprocessed Data
with open('Subset_Dataset/input_sequences_notes.pkl', 'rb') as f:
    X_notes = pickle.load(f)

with open('Subset_Dataset/output_labels_notes.pkl', 'rb') as f:
    y_notes = pickle.load(f)

with open('Subset_Dataset/input_sequences_durations.pkl', 'rb') as f:
    X_durations = pickle.load(f)

with open('Subset_Dataset/output_labels_durations.pkl', 'rb') as f:
    y_durations = pickle.load(f)

with open('Subset_Dataset/input_sequences_tempos.pkl', 'rb') as f:
    X_tempos = pickle.load(f)

with open('Subset_Dataset/output_labels_tempos.pkl', 'rb') as f:
    y_tempos = pickle.load(f)

with open('Subset_Dataset/input_sequences_time_signatures.pkl', 'rb') as f:
    X_time_signatures = pickle.load(f)

with open('Subset_Dataset/output_labels_time_signatures.pkl', 'rb') as f:
    y_time_signatures = pickle.load(f)

with open('Subset_Dataset/input_sequences_key_signatures.pkl', 'rb') as f:
    X_key_signatures = pickle.load(f)

with open('Subset_Dataset/output_labels_key_signatures.pkl', 'rb') as f:
    y_key_signatures = pickle.load(f)

# Step 2: Concatenate the inputs into a single array
X = np.concatenate([X_notes, X_durations, X_tempos, X_time_signatures, X_key_signatures], axis=-1)

# Step 3: Build the LSTM Model
sequence_length = X_notes.shape[1]  # The length of each input sequence
n_notes = y_notes.shape[1]  # Number of unique notes

# Define the model
model = Sequential()

# LSTM layers
model.add(LSTM(512, input_shape=(sequence_length, X.shape[2]), return_sequences=True))  # Use concatenated inputs
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.3))

# Dense output layers for each feature
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_notes, activation='softmax'))  # Output layer for notes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Step 4: Load the most recent checkpoint
checkpoint_filepath = '/Users/cristovalneosasono/AI_FinalProject_V2/checkpoints_E25/weights-epoch-05-loss-2.0413-acc-0.4787.weights.h5'

if os.path.exists(checkpoint_filepath):
    print(f"Loading weights from {checkpoint_filepath}")
    model.load_weights(checkpoint_filepath)
else:
    print("Checkpoint not found, starting from scratch.")

# Step 5: Define Checkpoints (save at every epoch)
checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_filepath = checkpoint_dir + "/weights-epoch-{epoch:02d}-loss-{loss:.4f}-acc-{accuracy:.4f}.weights.h5"

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    verbose=1,
    save_best_only=False,  # Save the model after every epoch
    mode='min',
    save_weights_only=True  # Save only the weights, not the full model
)

# Step 6: Resume Training the Model
additional_epochs = 5  # Number of additional epochs you want to train for
batch_size = 64

history = model.fit(X, y_notes, epochs=additional_epochs, batch_size=batch_size, callbacks=[checkpoint_callback])

# Step 7: Save the Training History to JSON
with open('resumed_training_history_X.json', 'w') as f:
    json.dump(history.history, f)

# Optionally save training history to CSV as well
history_df = pd.DataFrame(history.history)
history_df.to_csv('resumed_training_history_X.csv', index=False)

print("Training resumed and history saved to 'resumed_training_history_E20.json' and 'resumed_training_history_E20.csv'.")
