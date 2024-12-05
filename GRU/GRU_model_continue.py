import os
import pickle
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import pandas as pd
import json

# Set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Step 1: Load Preprocessed Data (Notes only)
with open('Subset_Dataset/input_sequences_notes.pkl', 'rb') as f:
    X_notes = pickle.load(f)

with open('Subset_Dataset/output_labels_notes.pkl', 'rb') as f:
    y_notes = pickle.load(f)

# Step 2: Use notes as input
X = X_notes  # Only use notes for GRU training
n_notes = y_notes.shape[1]  # Number of unique notes
sequence_length = X.shape[1]  # Length of each input sequence

# Step 3: Build the GRU Model (same as initial training)
model = Sequential()

# Add GRU layers
model.add(GRU(512, input_shape=(sequence_length, X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(512, return_sequences=False))
model.add(Dropout(0.3))

# Add dense layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(n_notes, activation='softmax'))  # Output layer for notes

# Compile the model (same optimizer and learning rate)
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

# Step 4: Load the saved weights from epoch 10
weights_path = '/Users/cristovalneosasono/AI_FinalProject_GRU/checkpoints_GRU_E15/weights-epoch-15-loss-3.9767-acc-0.2276.weights.h5'
model.load_weights(weights_path)
print(f"Weights loaded from {weights_path}")

# Step 5: Define Checkpoints and Callbacks for continued training
checkpoint_dir = 'checkpoints_GRU_E20/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Define a new checkpoint filepath pattern
checkpoint_filepath = checkpoint_dir + "weights-epoch-{epoch:02d}-loss-{loss:.4f}-acc-{accuracy:.4f}.weights.h5"

# Model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    verbose=1,
    save_best_only=False,
    mode='min',
    save_weights_only=True
)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# TensorBoard callback for visualization
tensorboard_callback = TensorBoard(log_dir='./logs_GRU_continued', histogram_freq=1)

# Step 6: Continue Training the Model
additional_epochs = 5  # Number of additional epochs
total_epochs = 10 + additional_epochs  # Total epochs after continuation
batch_size = 64

history = model.fit(
    X, y_notes,
    validation_split=0.2,  # Reserve 20% of data for validation
    epochs=total_epochs,
    initial_epoch=10,  # Start from epoch 10
    batch_size=batch_size,
    callbacks=[checkpoint_callback, early_stopping, tensorboard_callback]
)

# Step 7: Save the Updated Training History
# Load previous history if it exists
history_path_json = 'training_history_data_E20/training_history_GRU.json'
history_path_csv = 'training_history_data_E20/training_history_GRU.csv'

if os.path.exists(history_path_json):
    with open(history_path_json, 'r') as f:
        previous_history = json.load(f)
else:
    previous_history = {}

# Merge histories
for key in history.history:
    if key in previous_history:
        previous_history[key].extend(history.history[key])
    else:
        previous_history[key] = history.history[key]

# Save the merged history as JSON
with open(history_path_json, 'w') as f:
    json.dump(previous_history, f)

# Save the merged history as CSV
history_df = pd.DataFrame(previous_history)
history_df.to_csv(history_path_csv, index=False)

print("Continued training completed and updated history saved to 'training_history_GRU.json' and 'training_history_GRU.csv'.")
