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

# --- CONSTANTS ---
EPS = 1e-6
# LAMBDA is no longer needed as D_in is gone
BINS = 20
SMOOTH = 1e-9
DATA_FILE = "new-data.xlsx"

# --- NEW LABELING ---
LABEL_COL = "GRUP"  # THIS IS THE CRITICAL CHANGE
G1, G2 = 1, 2  # We only have two groups now


# --- HELPER FUNCTIONS (Entropy/JSD - Unchanged) ---
def entropy(p, base=2):
    p = np.clip(p, 0, 1)
    p = p / p.sum()
    nz = p > 0
    logp = np.log(p[nz]) / np.log(base)
    return -np.sum(p[nz] * logp)


def jsd(p, q, base=2):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (kl(p, m, base) + kl(q, m, base))


def kl(p, q, base=2):
    p = p / p.sum();
    q = q / q.sum()
    nz = (p > 0) & (q > 0)
    return np.sum(p[nz] * (np.log(p[nz]) - np.log(q[nz])) / np.log(base))


# --- RE-WRITTEN PIPELINE CORE FUNCTIONS ---

def density_to_probs_with_edges(x, edges):
    hist, _ = np.histogram(x, bins=edges, density=False)
    p = hist.astype(float) + SMOOTH
    p /= p.sum()
    return p


def jsd_between_two_classes(x, y):
    """
    Calculates the JSD (separation) between G1 and G2.
    The complex 'D_top' and 'D_in' are gone.
    """
    mask1 = (y == G1)
    mask2 = (y == G2)
    x1, x2 = x[mask1], x[mask2]

    if len(x1) < 2 or len(x2) < 2:
        return np.nan, None

    # Get common histogram edges
    _, edges = np.histogram(x, bins=BINS)
    p1 = density_to_probs_with_edges(x1, edges)
    p2 = density_to_probs_with_edges(x2, edges)

    # The separation score S_f is now just the JSD
    S_f = jsd(p1, p2)

    return S_f, (p1, p2)


def class_stats(x, y):
    """Calculates stats for only the two groups we care about."""
    stats = {}
    for c in [G1, G2]:  # Only G1 and G2
        xc = x[y == c]
        mu, sd = np.nanmean(xc), np.nanstd(xc, ddof=1) if len(xc) > 1 else (np.nan, np.nan)
        stats[c] = (mu, sd)
    return stats


def assign_nearest_class_and_z(xp, stats):
    """Unchanged, works fine with a 2-class stats dict."""
    best_c, best_absz, best_z = None, np.inf, None
    for c, (mu, sd) in stats.items():
        if sd is None or np.isnan(sd) or sd == 0 or np.isnan(mu): continue
        z = (xp - mu) / sd
        if abs(z) < best_absz: best_absz, best_z, best_c = abs(z), z, c
    return best_c, best_z


def H_of_class_probvec(p_vec):
    """Unchanged."""
    return entropy(p_vec, base=2)


def compute_uncertainty_matrix(df, feature_cols, label_col=LABEL_COL, cat_cols=None, eps=EPS):
    """
    Re-written to use the new 2-class binary separation logic.
    """
    y = df[label_col].values
    X_mat = np.zeros((len(df), len(feature_cols)))
    model_artifacts = {}

    for j, f in enumerate(feature_cols):
        x = df[f].values

        # 1. Calculate new 2-class separation score (S_f)
        S_f, reps = jsd_between_two_classes(x, y)

        if np.isnan(S_f) or reps is None:
            X_mat[:, j] = np.nan
            continue

        S_f = max(eps, S_f)  # Ensure S_f is not zero

        # 2. Get 2-class stats
        stats = class_stats(x, y)
        p1, p2 = reps
        class_probvec = {G1: p1, G2: p2}  # New 2-class prob_vec

        # 3. Calculate uncertainty for each patient
        for i, xp in enumerate(x):
            c_star, zpf = assign_nearest_class_and_z(xp, stats)

            if (c_star is None) or (zpf is None) or np.isnan(zpf):
                X_mat[i, j] = np.nan
                continue

            H_f = H_of_class_probvec(class_probvec[c_star])
            X_mat[i, j] = (H_f * zpf) / (S_f + eps)  # Formula remains the same

        # Save artifacts
        model_artifacts[f] = {
            "S_f": S_f,
            "stats": stats,
            "class_probvec": class_probvec
            # D_top and D_in are gone
        }

    print(f"Computed artifacts for {len(model_artifacts)} features.")
    return X_mat, model_artifacts


# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":

    # --- 1. Load and Prep Data ---
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

    # CRITICAL: Drop rows where GRUP is missing
    df = df.dropna(subset=[LABEL_COL])

    df = df.dropna(axis=1, how="all")
    df = df.loc[:, df.nunique() > 1]
    thresh = int(len(df) * 0.1)
    df = df.dropna(axis=1, thresh=thresh)

    # Identify numerical columns, EXCLUDING 'CONFIRMED DIAGNOSIS'
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in [LABEL_COL, "CONFIRMED DIAGNOSIS"]  # IGNORE BOTH
    ]

    df[num_cols] = df[num_cols].apply(lambda c: c.fillna(c.mean()))

    # Final feature list
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [
        c for c in num_cols
        if c in df.columns and c not in [LABEL_COL, "CONFIRMED DIAGNOSIS"]
    ]

    cat_cols = []
    y_labels = df[LABEL_COL].values  # y_labels are now 1s and 2s

    print(f"Data loaded. Using '{LABEL_COL}' as target.")
    print(f"Found {len(df)} patients and {len(num_cols)} features.")
    print(f"Target distribution:\n{df[LABEL_COL].value_counts()}")

    # --- 2. Run Pipeline & Get Artifacts ---
    print("Calculating NEW 2-Class Uncertainty matrix...")
    X_matrix, model_artifacts = compute_uncertainty_matrix(
        df,
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

    # --- 4. t-SNE Calculation ---
    print("Calculating NEW 2-Class t-SNE coordinates...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                max_iter=1500, metric='euclidean')
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

    # This categorical map must match the one in app.py
    yes_no_map = {"No": 0, "Yes": 1}
    categorical_map = {
        "GENDER": {"Female": 0, "Male": 1},
        "Socioeconomic Status": {"Poor": 0, "Good": 1},
        "Chest Pain Character": {"None": 0, "Stabbing / Localized": 1, "Pressure / Anginal": 2},
        "Infection type": {"URTI": 1, "Diarrhea": 2, "Vaccine": 3, "Other": 4},
        "DM": yes_no_map, "HT": yes_no_map, "HL": yes_no_map, "FH": yes_no_map,
        "SIGARA": yes_no_map, "KBY": yes_no_map, "PRIOR_CAD": yes_no_map,
        "HIPOTIROIDI": yes_no_map, "Chest Pain": yes_no_map, "Radiation": yes_no_map,
        "Arm Pain": yes_no_map, "Back Pain": yes_no_map, "Epigastric Pain": yes_no_map,
        "Relation with exercise": yes_no_map, "Relation with Position": yes_no_map,
        "Dyspnea": yes_no_map, "Fatigue": yes_no_map, "Nausea": yes_no_map,
        "Çarpıntı": yes_no_map, "Recent Infection(4 hafta)": yes_no_map,
        "INHOSPITAL_EX": yes_no_map, "Segmentary Wall Motion Abnormality": yes_no_map,
        "Pericardial Effusion": yes_no_map, "ECG_ST depression": yes_no_map,
        "ECG_T neg": yes_no_map, "ECG_Q waves": yes_no_map, "MRI_T2": yes_no_map,
        "MRI_LGE": yes_no_map, "KANSER_KEMOTERAPI": yes_no_map,
        "TASI_BRADIKARDI": yes_no_map, "MADDE_ILAC_KULLANIMI": yes_no_map,
        "Alcohol": yes_no_map, "KOAH": yes_no_map, "PAH": yes_no_map,
        "HIPERTIROIDI": yes_no_map, "REYNAULD": yes_no_map,
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