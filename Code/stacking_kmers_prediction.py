import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# ====== 1. Load Data from CSV ======
file_path = "Dataset.csv"  

df = pd.read_csv(file_path)

# Assuming CSV contains "Sequence" (peptide) and "Label" (binary target: 0 or 1)
peptides = df["Sequence"].values
labels = df["Class"].values.astype(int)  # Chuyển sang int cho classification

# ====== 2. Generate K-mer Representation (K = 3) ======
def generate_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Build K-mer vocabulary
all_kmers = []
for seq in peptides:
    all_kmers.extend(generate_kmers(seq, k=3))

kmer_counts = Counter(all_kmers)
kmer_to_index = {kmer: i + 1 for i, kmer in enumerate(kmer_counts.keys())}  # Start indexing from 1

max_length = max(len(generate_kmers(seq, k=3)) for seq in peptides)

def encode_kmer_sequences(sequences, max_length, kmer_to_index, k=3):
    num_samples = len(sequences)
    encoded_sequences = np.zeros((num_samples, max_length), dtype=np.int32)
    for i, seq in enumerate(sequences):
        kmers = generate_kmers(seq, k=k)
        for j, kmer in enumerate(kmers):
            if j < max_length:
                encoded_sequences[i, j] = kmer_to_index.get(kmer, 0)  # Default to 0 if K-mer not found
    return encoded_sequences

X = encode_kmer_sequences(peptides, max_length, kmer_to_index)
y = np.array(labels)

# ====== 3. Define Classification Models (CNN, BiLSTM, Transformer) ======
embedding_dim = 128
vocab_size = len(kmer_to_index) + 1

def create_cnn(input_length, vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_bilstm(input_length, vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_transformer(input_length, vocab_size):
    inputs = Input(shape=(input_length,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    transformer_block = MultiHeadAttention(num_heads=4, key_dim=embedding_dim)(embedding_layer, embedding_layer)
    transformer_block = LayerNormalization(epsilon=1e-6)(transformer_block)
    transformer_block = GlobalAveragePooling1D()(transformer_block)
    output_layer = Dense(1, activation='sigmoid')(transformer_block)
    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model 

# ====== 4. Train & Evaluate Models ======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cnn_model = create_cnn(max_length, vocab_size)
bilstm_model = create_bilstm(max_length, vocab_size)
transformer_model = create_transformer(max_length, vocab_size)

models = [cnn_model, bilstm_model, transformer_model]
model_preds = []

for i, model in enumerate(models):
    print(f"\nTraining Model {i+1}...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
    y_pred = model.predict(X_test).flatten()
    model_preds.append(y_pred)

# Stack predictions for Meta Model
meta_X = np.column_stack(model_preds)

def create_meta_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

meta_model = create_meta_model(meta_X.shape[1])
meta_model.fit(meta_X, y_test, epochs=50, batch_size=32, verbose=0)

meta_y_pred = meta_model.predict(meta_X).flatten()
meta_y_pred_binary = (meta_y_pred > 0.5).astype(int)
meta_y_pred_label = np.where(meta_y_pred_binary == 1, "Anti-Dengue", "Non-Anti-Dengue")

accuracy = accuracy_score(y_test, meta_y_pred_binary)
precision = precision_score(y_test, meta_y_pred_binary)
recall = recall_score(y_test, meta_y_pred_binary)
f1 = f1_score(y_test, meta_y_pred_binary)

print("\n=== Classification Model Evaluation ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# ====== 5. Predict and Save Results for New Sequences ======
def predict_and_save_from_csv(input_csv, output_csv="Predictions_kmers.csv"):
    new_df = pd.read_csv(input_csv)
    new_sequences = new_df["Sequence"].values
    encoded_new_sequences = encode_kmer_sequences(new_sequences, max_length, kmer_to_index)
    model_preds = [model.predict(encoded_new_sequences).flatten() for model in models]
    meta_X_new = np.column_stack(model_preds)
    final_predictions = meta_model.predict(meta_X_new).flatten()
    predicted_labels = np.where(final_predictions > 0.5, "Anti-Dengue", "Non-Anti-Dengue")
    
    results_df = pd.DataFrame({"Sequence": new_sequences, "Probability": final_predictions, "Predicted_Label": predicted_labels})
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

predict_and_save_from_csv("Predictions_kmers_output.csv")
