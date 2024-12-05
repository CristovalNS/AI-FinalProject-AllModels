import os
import pickle
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
)
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
X = np.concatenate(
    [X_notes, X_durations, X_tempos, X_time_signatures, X_key_signatures], axis=-1
)

# Step 3: Build the Transformer Model
sequence_length = X_notes.shape[1]  # The length of each input sequence
feature_dim = X.shape[2]  # Number of features after concatenation
n_notes = y_notes.shape[1]  # Number of unique notes

# Define the positional encoding layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def positional_encoding(self, sequence_length, d_model):
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        pe = np.zeros((sequence_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, ...]
        return tf.cast(pe, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = x + res
    return x

# Model parameters
embedding_dim = 512
num_transformer_blocks = 2
head_size = 64
num_heads = 8
ff_dim = 256
dropout = 0.3

# Input layer
inputs = Input(shape=(sequence_length, feature_dim))

# Embedding layer
x = Dense(embedding_dim)(inputs)
x = PositionalEncoding(sequence_length, embedding_dim)(x)

# Transformer blocks
for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

# Global pooling
x = GlobalAveragePooling1D()(x)
x = Dropout(dropout)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(dropout)(x)

# Output layer
outputs = Dense(n_notes, activation='softmax')(x)

# Define the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Step 4: Define Checkpoints (save at every epoch)
checkpoint_dir = 'checkpoints_E10_transformer/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_filepath = (
    checkpoint_dir
    + "weights-epoch-{epoch:02d}-loss-{loss:.4f}-acc-{accuracy:.4f}.weights.h5"
)

checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    verbose=1,
    save_best_only=False,  # Save the model after every epoch
    mode='min',
    save_weights_only=True,  # Save only the weights, not the full model
)

# Step 5: Train the Model and Save Training History
epochs = 10
batch_size = 64

history = model.fit(
    X, y_notes, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint_callback]
)

# Step 6: Save the Training History to JSON
if not os.path.exists('training_history_data'):
    os.makedirs('training_history_data')

with open('training_history_data/training_history_E10_transformer.json', 'w') as f:
    json.dump(history.history, f)

# Optionally save training history to CSV as well
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history_E10_transformer.csv', index=False)

print(
    "Training completed and history saved to 'training_history_E10_transformer.json' and 'training_history_E10_transformer.csv'."
)
