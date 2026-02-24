"""
Regenerate app_artifacts/embedding_data.npz using the fitted VotingClassifier pipeline.
Run after retraining the model in NEW-finetuning.ipynb.
"""
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.manifold import TSNE

# --- Load artifacts ---
print("Loading model and metadata...")
with open("best_model_finetuned.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
with open("split_indices.pkl", "rb") as f:
    split_indices = pickle.load(f)

final_features = list(metadata["features"])
train_idx = split_indices["train_idx"]
LABEL_COL = "GRUP"

# --- Load raw data ---
print("Loading new-data.xlsx...")
df = pd.read_excel("NEW_Miyokardit_08.12.2025.xlsx", sheet_name=0)
df = df.dropna(subset=[LABEL_COL])

# --- Extract training split ---
df_train = df.iloc[train_idx].copy()
X_train = df_train[final_features].copy()
y_train = df_train[LABEL_COL].values

# --- Drop rows with any NaN (same as nested CV preprocessing) ---
mask = ~X_train.isna().any(axis=1)
X_train = X_train[mask]
y_train = y_train[mask]
print(f"Training samples after NaN drop: {len(X_train)}")
print(f"Features: {len(final_features)}")

# --- Transform through fitted pipeline steps ---
print("Transforming through UncertaintyTransformer...")
X_unc = model.named_steps["uncertainty"].transform(X_train)
print("Transforming through StandardScaler...")
X_std = model.named_steps["scaler"].transform(X_unc)

# --- Run t-SNE ---
print("Running t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30,
            max_iter=1500, metric="euclidean")
X_emb = tsne.fit_transform(X_std)
print("t-SNE done.")

# --- Save ---
os.makedirs("app_artifacts", exist_ok=True)
out_path = os.path.join("app_artifacts", "embedding_data.npz")
np.savez(out_path, X_std=X_std, X_emb=X_emb, y=y_train)
print(f"Saved to {out_path}")
print(f"  X_std shape:  {X_std.shape}")
print(f"  X_emb shape:  {X_emb.shape}")
print(f"  y distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
