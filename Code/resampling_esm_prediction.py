import numpy as np
import pandas as pd
import torch
import esm
import tensorflow as tf
from tensorflow.keras.models import load_model

# ====== 1. Load New Data ======
new_file_path = "Dataset.csv" 
new_df = pd.read_csv(new_file_path)
new_peptides = new_df["Sequence"].values

# ====== 2. Load Pretrained ESM Model ======
print("Loading ESM model...")
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

# ====== 3. Extract Features Using ESM Model ======
def extract_esm_features(sequences):
    """
    Convert sequences into feature vectors using ESM model.
    Returns: NumPy array with shape (num_samples, embedding_dim)
    """
    batch_labels = [(str(i), seq) for i, seq in enumerate(sequences)]
    batch_tokens = batch_converter(batch_labels)[2]

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[6])
        token_representations = results["representations"][6]

    # Compute mean of token embeddings for each sequence
    sequence_embeddings = token_representations.mean(dim=1).cpu().numpy()
    return sequence_embeddings

print("Extracting ESM features...")
X_new = extract_esm_features(new_peptides)

# ====== 4. Load Trained BiLSTM Models ======
print("Loading trained models...")
model1 = load_model("bilstm_model1.h5")  # Thay thế bằng đường dẫn thực tế
model2 = load_model("bilstm_model2.h5")

# Reshape for BiLSTM
X_new = X_new[:, np.newaxis, :]

# ====== 5. Predict Probabilities ======
y_pred_prob1 = model1.predict(X_new)
y_pred_prob2 = model2.predict(X_new)

# Average Predictions
y_pred_prob_avg = (y_pred_prob1 + y_pred_prob2) / 2

# Add predictions to DataFrame
new_df["Predicted_Probability"] = y_pred_prob_avg
new_df["Prediction_Label"] = np.where(y_pred_prob_avg > 0.5, "Anti-Dengue", "Non-Anti-Dengue")

# Save results to CSV
output_file = "Predictions_ESM.csv"
new_df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
