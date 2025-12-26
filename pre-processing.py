import numpy as np
import pandas as pd
import pickle
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from sklearn.exceptions import ConvergenceWarning

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- CONSTANTS (NOTEBOOK İLE UYUMLU) ---
EPS = 1e-12  # Notebook'taki EPS değeri
N_BINS = 20  # Notebook'taki N_BINS değeri
DATA_FILE = "NEW_Miyokardit_08.12.2025.xlsx"  # Notebook'taki dosya adı

# --- NEW LABELING ---
LABEL_COL = "GRUP"  # THIS IS THE CRITICAL CHANGE
G1, G2 = 1, 2  # We only have two groups now


# --- HELPER FUNCTIONS (NOTEBOOK'TAN) ---
def kl_divergence(p, q, eps=EPS):
    """
    D_KL(p || q) - Notebook'tan
    """
    union_idx = p.index.union(q.index)
    p, q = p.reindex(union_idx, fill_value=0), q.reindex(union_idx, fill_value=0)
    return float(np.sum(p * np.log2((p + eps) / (q + eps))))

def get_distribution(series):
    """PMF of a feature - Notebook'tan"""
    return series.value_counts(normalize=True)

def discretise(series):
    """Discretize feature if needed - Notebook'tan"""
    if series.nunique() > N_BINS * 2:
        return pd.qcut(series, q=N_BINS, duplicates="drop")
    return series

def shannon_entropy(p, eps=EPS):
    """H(p) (base-2) - Notebook'tan. `p` is a pandas Series whose values sum to 1."""
    return float(-np.sum(p * np.log2(p + eps)))

def js_divergence(p, q, eps=EPS):
    """
    Jensen-Shannon divergence - Notebook'tan
    Symmetric, bounded in [0, 1] when log base is 2.
    """
    union = p.index.union(q.index)
    p, q = p.reindex(union, fill_value=0), q.reindex(union, fill_value=0)
    m = 0.5 * (p + q)
    kl_p_m = np.sum(p * np.log2((p + eps) / (m + eps)))
    kl_q_m = np.sum(q * np.log2((q + eps) / (m + eps)))
    return 0.5 * (kl_p_m + kl_q_m)


def class_stats(x, y):
    """Calculates stats for only the two groups we care about - Notebook format."""
    stats = {}
    for c in [G1, G2]:  # Only G1 and G2
        xc = x[y == c]
        mu = np.nanmean(xc)
        sd = np.nanstd(xc, ddof=0) if len(xc) > 1 else EPS  # ddof=0 for population std (notebook)
        stats[c] = {"mu": mu, "std": sd + EPS}  # Notebook format: dict with mu and std
    return stats


def assign_nearest_class_and_z(xp, stats):
    """Z-score hesaplama - Notebook mantığı ile uyumlu"""
    best_c, best_absz, best_z = None, np.inf, None
    for c, stat_dict in stats.items():
        if isinstance(stat_dict, dict):
            mu = stat_dict.get("mu")
            sd = stat_dict.get("std")
        else:
            # Eski format: (mu, sd) tuple
            mu, sd = stat_dict
        if sd is None or np.isnan(sd) or sd == 0 or np.isnan(mu): 
            continue
        z = (xp - mu) / sd
        if abs(z) < best_absz: 
            best_absz, best_z, best_c = abs(z), z, c
    return best_c, best_z


def compute_uncertainty_matrix(df, feature_cols, label_col=LABEL_COL, cat_cols=None, eps=EPS):
    """
    NOTEBOOK'TAN: Uncertainty matrix hesaplama
    """
    y = df[label_col].values
    X_mat = np.zeros((len(df), len(feature_cols)))
    model_artifacts = {}

    # 3-a) Distributions, entropies (per class) and JS divergence (per feature)
    dist = {}       # {feat: (p_class1, p_class2)}
    entropy_dict = {}   # {feat: {1: H1 , 2: H2}}
    js_dict = {}        # {feat: JS}

    for feat in feature_cols:
        col_disc = discretise(df[feat])
        p1 = get_distribution(col_disc[y == G1])
        p2 = get_distribution(col_disc[y == G2])

        dist[feat] = (p1, p2)
        entropy_dict[feat] = {
            1: shannon_entropy(p1),
            2: shannon_entropy(p2)
        }
        js_dict[feat] = max(js_divergence(p1, p2), eps)  # never let it be 0

    # 3-b) Means & stds for z-scores (ddof=0 ⇒ population σ)
    stats = {feat: {
        1: {"mu": df.loc[y == G1, feat].mean(),
            "std": df.loc[y == G1, feat].std(ddof=0) + eps},
        2: {"mu": df.loc[y == G2, feat].mean(),
            "std": df.loc[y == G2, feat].std(ddof=0) + eps}
    } for feat in feature_cols}

    # 4. Build the patient × feature matrix
    rows = []
    for i, (_, patient) in enumerate(df.iterrows()):
        vec = []
        for feat in feature_cols:
            v = patient[feat]

            # z-scores vs. each class
            z1 = (v - stats[feat][1]["mu"]) / stats[feat][1]["std"]
            z2 = (v - stats[feat][2]["mu"]) / stats[feat][2]["std"]

            # pick the *magnitude-wise* smaller one
            if abs(z1) < abs(z2):
                z = z1
                cls = 1
            else:
                z = z2
                cls = 2

            h = entropy_dict[feat][cls]  # Shannon entropy of the chosen class
            js_f = js_dict[feat]         # JS divergence for this feature
            x_f = z * (1.0 / (js_f + eps)) * h

            vec.append(x_f)
        rows.append(vec)
        X_mat[i, :] = vec

    # Save artifacts in notebook format
    for feat in feature_cols:
        model_artifacts[feat] = {
            "js": js_dict[feat],  # JS divergence (S_f)
            "entropy": entropy_dict[feat],  # Entropy per class
            "stats": stats[feat],  # Stats per class
            "class_probvec": {G1: dist[feat][0], G2: dist[feat][1]}  # Distributions
        }

    print(f"Computed artifacts for {len(model_artifacts)} features.")
    return X_mat, model_artifacts


# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":

    # --- 1. Load and Prep Data (NOTEBOOK MANTIĞI İLE) ---
    print(f"Loading data from '{DATA_FILE}'...")
    try:
        df = pd.read_excel(DATA_FILE, sheet_name=0)
    except FileNotFoundError:
        print(f"ERROR: '{DATA_FILE}' not found.")
        exit()
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        exit()

    print("Cleaning data...")
    
    # NOTEBOOK'TAN: Feature selection - sadece belirli kolonları seç
    label_series = df[LABEL_COL]
    
    # B..Z kolonları (1:26) ve AD..AU kolonları (29:47)
    df_part1 = df.iloc[:, 1:26]   # B..Z
    df_part2 = df.iloc[:, 29:47]  # AD..AU
    df = pd.concat([df_part1, df_part2], axis=1)
    
    print(f"After feature selection shape: {df.shape}")

    # Hidden NaN filtering
    df = df.replace([" ", "", "-", "--", "nan", "NaN", "None"], pd.NA)
    df = df.apply(lambda col: col.replace(r'^\s*$', pd.NA, regex=True))

    datetime_cols = df.select_dtypes(include=["datetime"]).columns
    df = df.drop(columns=datetime_cols)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[LABEL_COL] = label_series

    # Remove NaN including rows
    initial_rows = len(df)
    df = df.dropna(axis=0)
    final_rows = len(df)

    print(f"Removed {initial_rows - final_rows} rows due to missing values.")
    print(f"Remaining rows: {final_rows}")

    # Remove NaN including cols
    nan_counts = df.isna().sum()
    cols_to_drop = nan_counts[(nan_counts > 0) & (nan_counts.index != LABEL_COL)].index
    df_clean = df.drop(columns=cols_to_drop)

    print(f"Dropped {len(cols_to_drop)} feature columns due to NaN.")
    print("Remaining shape after column drops:", df_clean.shape)

    # Final feature list - sadece numeric kolonlar
    num_cols = df_clean.select_dtypes(include=["int", "float"]).columns.tolist()
    num_cols = [f for f in num_cols if f != LABEL_COL]

    cat_cols = []
    y_labels = df_clean[LABEL_COL].values  # y_labels are now 1s and 2s

    print(f"Data loaded. Using '{LABEL_COL}' as target.")
    print(f"Found {len(df_clean)} patients and {len(num_cols)} features.")
    print(f"Target distribution:\n{df_clean[LABEL_COL].value_counts()}")

    # --- 2. Run Pipeline & Get Artifacts ---
    print("Calculating NEW 2-Class Uncertainty matrix...")
    X_matrix, model_artifacts = compute_uncertainty_matrix(
        df_clean,  # df_clean kullan
        feature_cols=num_cols,
        label_col=LABEL_COL,
        cat_cols=cat_cols
    )
    print("Uncertainty matrix calculation finished.")

    # --- 3. Imputation & Scaling ---
    print("Fitting StandardScaler...")
    X_imputed = np.nan_to_num(X_matrix)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imputed)
    print("StandardScaler fitted.")

    # --- 4. t-SNE Calculation (NOTEBOOK PARAMETRELERİ) ---
    print("Calculating NEW 2-Class t-SNE coordinates...")
    tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto',
                init='pca', random_state=42)
    X_embedding = tsne.fit_transform(X_std)
    print("t-SNE coordinates calculated.")

    # --- 5. Save Core Artifacts ---
    output_dir = "app_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    path1 = os.path.join(output_dir, "model_artifacts.pkl")
    with open(path1, "wb") as f:
        pickle.dump(model_artifacts, f)
    print(f"1. Model artifacts saved to -> {path1}")

    path2 = os.path.join(output_dir, "scaler.pkl")
    with open(path2, "wb") as f:
        pickle.dump(scaler, f)
    print(f"2. Scaler saved to -> {path2}")

    path3 = os.path.join(output_dir, "embedding_data.npz")
    np.savez(path3,
             X_std=X_std,
             X_emb=X_embedding,
             y=y_labels)  # y will contain 1s and 2s
    print(f"3. Embedding data saved to -> {path3}")

    # --- 6. Calculate and Save Imputation Values ---
    print("Calculating imputation default values...")
    imputation_values = {}

    # This categorical map must match the one in app.py (NOTEBOOK FEATURES ONLY)
    yes_no_map = {"No": 0, "Yes": 1}
    categorical_map = {
        "SEX": {"Female": 0, "Male": 1},  # GENDER değil, SEX
        "Chest Pain Character": {"None": 0, "Stabbing / Localized": 1, "Pressure / Anginal": 2},
        "DM": yes_no_map, "HT": yes_no_map, "HL": yes_no_map, "FH": yes_no_map,
        "SIGARA": yes_no_map, "KBY": yes_no_map, "PRIOR_KAH": yes_no_map,  # PRIOR_CAD değil
        "KOAH": yes_no_map, "Chest Pain": yes_no_map, "Radiation": yes_no_map,
        "Arm Pain": yes_no_map, "Back Pain": yes_no_map, "Epigastric Pain": yes_no_map,
        "Relation with exercise": yes_no_map, "Relation with Position": yes_no_map,
        "Dyspnea": yes_no_map, "Fatigue": yes_no_map, "Nausea": yes_no_map,
        "Çarpıntı": yes_no_map, "Recent Infection(4 hafta)": yes_no_map,
    }

    for feature in num_cols:
        if feature in categorical_map:
            imputation_values[feature] = df[feature].mode()[0]
        else:
            imputation_values[feature] = df[feature].mean()

    path4 = os.path.join(output_dir, "imputation_values.pkl")
    with open(path4, "wb") as f:
        pickle.dump(imputation_values, f)

    print(f"4. Imputation values saved to -> {path4}")

    print(f"\n--- PRE-PROCESSING COMPLETE (2-CLASS MODEL) ---")
    print(f"All 4 artifacts are saved in the '{output_dir}' directory.")