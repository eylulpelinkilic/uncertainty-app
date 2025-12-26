import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# --- CONSTANTS ---
# Renk Paleti
COLOR_ORANGE = "#F39C12"  # Grup 2 (AKS)
COLOR_BLUE = "#3498DB"  # Grup 1 (Miyokardit) & Ana Tema
COLOR_BACKGROUND = "#FFFFFF" 
COLOR_TEXT = "#333333" 
COLOR_NEW_PATIENT = "#E74C3C" 

# Artifact Dosya Yolları
ARTIFACT_DIR = "app_artifacts"
MODEL_ARTIFACTS_PATH = os.path.join(ARTIFACT_DIR, "model_artifacts.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
EMBEDDING_PATH = os.path.join(ARTIFACT_DIR, "embedding_data.npz")
TSNE_MODEL_PATH = os.path.join(ARTIFACT_DIR, "tsne_model.pkl")

# --- MODEL SABİTLERİ ---
G1, G2 = 1, 2 # Sadece iki grubumuz var
EPS = 1e-12  # Notebook ile uyumlu (EPS_NB ile aynı)

# --- DİL (ÇEVİRİ) SÖZLÜĞÜ (GÜNCELLENDİ) ---
LANG_STRINGS = {
    "app_title": {
        "ENG": "Clinical Uncertainty Positioning",
        "TR": "Klinik Belirsizlik Konumlandırma"
    },
    "main_title": {
        "ENG": "Clinical Uncertainty Positioning Tool (Myocarditis vs ACS)",
        "TR": "Klinik Belirsizlik Konumlandırma Aracı (Miyokardit vs AKS)"
    },
    "header_input": {
        "ENG": "New Patient Input",
        "TR": "Yeni Hasta Girişi"
    },
    "info_note": {
        "ENG": "NOTE: All fields must be filled. No missing values allowed.",
        "TR": "NOT: Tüm alanlar doldurulmalıdır. Eksik değer kabul edilmez."
    },
    "key_features": {"ENG": "Key Features", "TR": "Anahtar Özellikler"},
    "symptoms": {"ENG": "Symptoms & Presentation", "TR": "Semptomlar ve Başvuru"},
    "history": {"ENG": "Patient History & Comorbidities", "TR": "Hasta Geçmişi ve Komorbiditeler"},
    "labs": {"ENG": "Laboratory Results", "TR": "Laboratuvar Sonuçları"},
    "other_features": {"ENG": "Other Features", "TR": "Diğer Özellikler"},
    "calculate_button": {"ENG": "Calculate Patient Position", "TR": "Hasta Pozisyonunu Hesapla"},
    "header_output": {"ENG": "Patient Position Analysis", "TR": "Hasta Pozisyon Analizi"},
    "error_missing_fields": {
        "ENG": "Error: {num_missing} field(s) are missing. All fields must be filled.",
        "TR": "Hata: {num_missing} alan eksik. Tüm alanlar doldurulmalıdır."
    },
    "error_missing_list": {"ENG": "Missing Fields:", "TR": "Eksik Alanlar:"},
    "plot_title_tsne": {
        "ENG": "Diagnostic Landscape (Myocarditis vs ACS)",
        "TR": "Tanısal Manzara (Miyokardit vs AKS)"
    },
    "plot_title_bar": {"ENG": "Feature-based Uncertainty", "TR": "Özellik Bazlı Belirsizlik"},
    "plot_top20": {
        "ENG": "Top 20 features contributing to uncertainty:",
        "TR": "Belirsizliğe katkıda bulunan ilk 20 özellik:"
    },
    "legend_g1": {"ENG": "Grup 1 (Myocarditis)", "TR": "Grup 1 (Miyokardit)"},
    "legend_g2": {"ENG": "Grup 2 (ACS)", "TR": "Grup 2 (AKS)"},
    "legend_new": {"ENG": "New Patient", "TR": "Yeni Hasta"},
    "legend_uncertain": {"ENG": "Uncertain", "TR": "Belirsiz"},
    "legend_title": {"ENG": "Diagnosis", "TR": "Tanı"},
    "bar_chart_title": {"ENG": "Patient's Uncertainty Vector", "TR": "Hastanın Belirsizlik Vektörü"},
    "bar_xaxis": {"ENG": "Uncertainty Score", "TR": "Belirsizlik Skoru"},
    "bar_yaxis": {"ENG": "Feature", "TR": "Özellik"},
    "welcome_header": {"ENG": "About this Tool", "TR": "Araç Hakkında"},
    "welcome_info": {
        "ENG": "Please enter patient data on the left and click 'Calculate' to begin analysis.",
        "TR": "Lütfen soldaki alana hasta verilerini girin ve analize başlamak için 'Hesapla' butonuna tıklayın."
    },
    "welcome_text": {
        "ENG": """
            This tool positions a new patient within the diagnostic landscape
            of **Grup 1 (Myocarditis) vs. Grup 2 (ACS)**
            based on their clinical features.
            
            1.  **Enter** the patient's data in the form on the left.
            2.  **Click** 'Calculate' to see their position.
            3.  **Analyze** their position on the t-SNE map.
        """,
        "TR": """
            Bu araç, yeni bir hastayı klinik özelliklerine göre
            **Grup 1 (Miyokardit) ve Grup 2 (AKS)**
            tanısal manzarası üzerinde konumlandırır.
            
            1.  Sol taraftaki forma hasta verilerini **girin**.
            2.  Hastanın pozisyonunu görmek için 'Hesapla' butonuna **tıklayın**.
            3.  t-SNE haritası üzerindeki konumunu **analiz edin**.
        """
    },
    "load_error": {
        "ENG": "Application failed to load. Please check the terminal for errors.",
        "TR": "Uygulama yüklenemedi. Lütfen terminali kontrol edin."
    }
}

# Çeviri (Translation) için yardımcı fonksiyon
def T(key):
    """Geçerli dile göre çevrilmiş metni alır."""
    lang = st.session_state.language
    return LANG_STRINGS.get(key, {}).get(lang, f"MISSING_KEY: {key}")

# --- YARDIMCI FONKSİYONLAR (NEW PIPELINE FROM NOTEBOOK) ---
EPS_NB = 1e-12  # Notebook'taki EPS değeri
N_BINS = 20     # Notebook'taki N_BINS değeri

def kl_divergence(p, q, eps=EPS_NB):
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

def shannon_entropy(p, eps=EPS_NB):
    """H(p) (base-2) - Notebook'tan. `p` is a pandas Series whose values sum to 1."""
    return float(-np.sum(p * np.log2(p + eps)))

def js_divergence(p, q, eps=EPS_NB):
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

# --- ARTIFACT YÜKLEME ---
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    """Tüm artifact'ları diskten yükler ve cache'ler."""
    try:
        with open(MODEL_ARTIFACTS_PATH, "rb") as f: model_artifacts = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        embedding_data = np.load(EMBEDDING_PATH)
        with open(TSNE_MODEL_PATH, "rb") as f: tsne_model = pickle.load(f)
        feature_list = list(model_artifacts.keys())
        # Sadece ALL_FEATURES listesindeki feature'ları kullan
        feature_list = [f for f in feature_list if f in ALL_FEATURES]
        return model_artifacts, scaler, embedding_data, tsne_model, feature_list
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.error(f"Please RE-RUN `pre-processing.py` to generate all artifacts.")
        return None

# --- TAHMİNLEME FONKSİYONLARI (NEW PIPELINE FROM NOTEBOOK) ---
def predict_patient_uncertainty(input_data, model_artifacts, feature_list):
    """
    Yeni pipeline mantığı ile uncertainty hesaplama - Notebook'tan
    Formula: x_f = z * (h / (js_f + EPS))
    Hem eski hem yeni artifact yapılarını destekler.
    feature_list zaten ALL_FEATURES ile filtrelenmiş olmalı.
    """
    x_new_vec = np.zeros(len(feature_list))
    for i, feature in enumerate(feature_list):
        if feature not in model_artifacts:
            x_new_vec[i] = np.nan
            continue
            
        artifacts = model_artifacts[feature]
        xp = input_data[feature]
        
        # Z-score hesaplama (en yakın class'a göre)
        if 'stats' not in artifacts:
            x_new_vec[i] = np.nan
            continue
            
        c_star, zpf = assign_nearest_class_and_z(xp, artifacts['stats'])
        if (c_star is None) or (zpf is None) or np.isnan(zpf): 
            x_new_vec[i] = np.nan
            continue
        
        # Entropy hesaplama - yeni pipeline'da direkt entropy var
        h = None
        if 'entropy' in artifacts and isinstance(artifacts['entropy'], dict):
            # Yeni format: direkt entropy dict'i var
            h = artifacts['entropy'].get(c_star)
        elif 'class_probvec' in artifacts:
            # Eski format: class_probvec'den entropy hesapla
            if c_star in artifacts['class_probvec']:
                p_vec = artifacts['class_probvec'][c_star]
                if isinstance(p_vec, pd.Series):
                    h = shannon_entropy(p_vec)
                elif isinstance(p_vec, (list, np.ndarray)):
                    # numpy array veya list ise Series'e çevir
                    p_series = pd.Series(p_vec)
                    h = shannon_entropy(p_series)
        
        if h is None or np.isnan(h):
            x_new_vec[i] = np.nan
            continue
        
        # JS divergence (S_f) - yeni pipeline'da 'js' veya 'S_f' olabilir
        js_f = artifacts.get('js', artifacts.get('S_f', EPS_NB))
        if js_f is None or np.isnan(js_f):
            js_f = EPS_NB
        js_f = max(js_f, EPS_NB)  # Never let it be 0
        
        # Yeni pipeline formülü: z * (h / (js_f + EPS))
        x_new_vec[i] = zpf * (h / (js_f + EPS_NB))
    
    return x_new_vec

def find_tsne_position(x_new_std, tsne_model):
    """
    NEW_case_study mantığı: t-SNE modelini kullanarak yeni hasta pozisyonunu hesapla
    """
    x_new_tsne = tsne_model.transform(x_new_std)
    return x_new_tsne[0, 0], x_new_tsne[0, 1]

# --- PLOT/GRAFİK FONKSİYONLARI (NOTEBOOK'TAN KDE BULUTLU GÖRÜNTÜ) ---

def plot_diagnostic_landscape(X_emb_train, y_train, new_patient_coords=None):
    """
    Notebook'taki gibi KDE tabanlı bulutlu görüntü çizer.
    'new_patient_coords' opsiyoneldir. Eğer verilmezse, sadece "bulut" çizilir.
    """
    labels = y_train
    
    # Padding around extreme points so contours don't touch the frame
    pad = 2.0
    
    xmin, xmax = X_emb_train[:, 0].min() - pad, X_emb_train[:, 0].max() + pad
    ymin, ymax = X_emb_train[:, 1].min() - pad, X_emb_train[:, 1].max() + pad
    
    # Grid resolution (notebook'ta 1000, ama performans için 250 kullanabiliriz)
    resolution = 250
    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack([xx.ravel(), yy.ravel()])
    
    # KDE for each class
    class1 = X_emb_train[labels == G1]
    class2 = X_emb_train[labels == G2]
    
    kde1 = gaussian_kde(class1.T, bw_method="scott")
    kde2 = gaussian_kde(class2.T, bw_method="scott")
    
    z1 = kde1(grid).reshape(xx.shape)
    z2 = kde2(grid).reshape(xx.shape)
    
    # Quantile levels (notebook'ta q=0.6)
    q = 0.6
    level1 = np.quantile(z1, q)
    level2 = np.quantile(z2, q)
    
    overlap = (z1 >= level1) & (z2 >= level2)
    
    # Normalize function
    def normalise(z, clip=0.98):
        """Scale density field to 0–1, clipping the top `clip` quantile."""
        zmax = np.quantile(z, clip)
        return np.clip(z / zmax, 0, 1)
    
    # Alpha masks for pure class regions
    alpha1 = normalise(z1)**0.5
    alpha2 = normalise(z2)**0.5
    alpha1[z1 < level1] = 0.0
    alpha2[z2 < level2] = 0.0
    
    # Identify the overlap and give it its own alpha map
    overlap_mask = (alpha1 > 0) & (alpha2 > 0)
    alpha_overlap = np.maximum(alpha1, alpha2)
    alpha_overlap[~overlap_mask] = 0.0
    
    # Remove the red/blue alphas inside the overlap
    alpha1[overlap_mask] = 0.0
    alpha2[overlap_mask] = 0.0
    
    # Per-pixel colour for the overlap
    eps_overlap = 1e-12
    total = z1 + z2 + eps_overlap
    t = (z1 - z2) / total  # -1 → pure blue side, +1 → pure red side
    
    shift_strength = 0.3
    
    # Base: all grey
    R = np.full_like(t, 0.5)
    G = np.full_like(t, 0.5)
    B = np.full_like(t, 0.5)
    
    # Shift toward red where t > 0
    pos = t > 0
    R[pos] += shift_strength * t[pos]
    G[pos] -= shift_strength * t[pos]
    B[pos] -= shift_strength * t[pos]
    
    # Shift toward blue where t < 0
    neg = t < 0
    B[neg] += shift_strength * (-t[neg])
    R[neg] -= shift_strength * (-t[neg])
    G[neg] -= shift_strength * (-t[neg])
    
    # Build RGBA images
    shape = (*alpha1.shape, 4)
    
    red_img = np.zeros(shape)
    red_img[..., 0] = 1.0  # R channel (red for Myocarditis - Group 1)
    red_img[..., 3] = alpha1  # alpha
    
    blue_img = np.zeros(shape)
    blue_img[..., 2] = 1.0  # B channel (blue for ACS - Group 2)
    blue_img[..., 3] = alpha2  # alpha
    
    # Build the RGBA image for the overlap
    shape_over = (*alpha_overlap.shape, 4)
    over_img = np.zeros(shape_over)
    over_img[..., 0] = R
    over_img[..., 1] = G
    over_img[..., 2] = B
    over_img[..., 3] = alpha_overlap
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Filled RGBA layers (notebook'taki gibi)
    for img in [over_img, red_img, blue_img]:
        ax.imshow(img, extent=(xmin, xmax, ymin, ymax), origin="lower", interpolation="bilinear")
    
    # Hide axes (notebook'taki gibi)
    ax.set_axis_off()
    
    # Add legend (NEW_case_study style)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="red", edgecolor="red", label=T("legend_g1")),
        Patch(facecolor="blue", edgecolor="blue", label=T("legend_g2")),
        Patch(facecolor="gray", edgecolor="gray", label=T("legend_uncertain"))
    ]
    
    # Add new patient if provided (NEW_case_study style: yellow dot)
    if new_patient_coords is not None:
        ax.scatter(new_patient_coords[0], new_patient_coords[1], 
                  c="yellow", s=120, marker="o", 
                  edgecolors="k", linewidths=1.2, 
                  zorder=15, label=T("legend_new"))
        # Update legend
        legend_handles.append(
            Patch(facecolor="yellow", edgecolor="k", label=T("legend_new"))
        )
        ax.legend(handles=legend_handles, loc="upper left", frameon=False)
    else:
        ax.legend(handles=legend_handles, loc="upper left", frameon=False)
    
    plt.tight_layout()
    
    return fig

def plot_uncertainty_vector(x_new_vec_df, lang):
    """Çubuk grafiği çizer."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_new_vec_df['Uncertainty Score'], y=x_new_vec_df['Feature'], orientation='h', marker=dict(color=COLOR_BLUE)))
    
    fig.update_layout(
        title=T("bar_chart_title"), 
        xaxis_title=T("bar_xaxis"), 
        yaxis_title=T("bar_yaxis"), 
        plot_bgcolor=COLOR_BACKGROUND, 
        paper_bgcolor=COLOR_BACKGROUND, 
        font_color=COLOR_TEXT, 
        yaxis=dict(autorange="reversed"), 
        height=max(400, len(x_new_vec_df) * 20)
    )
    return fig

# --- KATEGORİK & GRUP HARİTALARI (NOTEBOOK'TAN GELEN GERÇEK FEATURE'LARA GÖRE) ---
# Notebook'ta sadece B..Z (1:26) ve AD..AU (29:47) kolonları seçiliyor
# EKG sonuçları, MRI sonuçları ve bazı lab değerleri KULLANILMIYOR

yes_no_map = {"No": 0, "Yes": 1}
categorical_map = {
    "SEX": {"Female": 0, "Male": 1},  # GENDER değil, SEX kullanılıyor
    "Chest Pain Character": {"None": 0, "Stabbing / Localized": 1, "Pressure / Anginal": 2},
    "DM": yes_no_map, "HT": yes_no_map, "HL": yes_no_map, "FH": yes_no_map,
    "SIGARA": yes_no_map, "KBY": yes_no_map, "PRIOR_KAH": yes_no_map,  # PRIOR_CAD değil, PRIOR_KAH
    "KOAH": yes_no_map, "Chest Pain": yes_no_map, "Radiation": yes_no_map,
    "Arm Pain": yes_no_map, "Back Pain": yes_no_map, "Epigastric Pain": yes_no_map,
    "Relation with exercise": yes_no_map, "Relation with Position": yes_no_map,
    "Dyspnea": yes_no_map, "Fatigue": yes_no_map, "Nausea": yes_no_map,
    "Çarpıntı": yes_no_map, "Recent Infection(4 hafta)": yes_no_map,
}

# KULLANILAN FEATURE LİSTESİ (42 feature - SADECE BUNLAR):
ALL_FEATURES = [
    'AGE', 'SEX', 'DM', 'HT', 'HL', 'FH', 'SIGARA', 'KBY', 'PRIOR_KAH', 'KOAH',
    'Chest Pain', 'Chest Pain Character', 'Any Previous Pain Attacks', 'Chest Pain Duration(saat)',
    'Radiation', 'Arm Pain', 'Back Pain', 'Epigastric Pain', 'Relation with exercise',
    'Relation with Position', 'Dyspnea', 'Fatigue', 'Nausea', 'Çarpıntı', 'Recent Infection(4 hafta)',
    'PEAK_TROP', 'CK-MB', 'GLUKOZ', 'WBCpik', 'NEUpik', 'LYMPpik', 'EOSpik', 'MONOpik',
    'HB', 'HTC', 'PLT', 'KREATIN', 'AST', 'ALT', 'TOTAL_KOLESTEROL', 'TG', 'LDL', 'HDL'
]

# UI için kategorilere ayırma (sadece yukarıdaki feature'lar kullanılacak)
KEY_FEATURES = [
    "AGE", "SEX", "Chest Pain Character", "PEAK_TROP"
]

SYMPTOM_FEATURES = [
    "Chest Pain", "Chest Pain Duration(saat)", "Radiation", "Arm Pain",
    "Back Pain", "Epigastric Pain", "Relation with exercise", 
    "Relation with Position", "Dyspnea", "Fatigue", "Nausea", "Çarpıntı",
    "Any Previous Pain Attacks"
]

HISTORY_FEATURES = [
    "DM", "HT", "HL", "FH", "SIGARA", "KBY", "PRIOR_KAH", "KOAH",
    "Recent Infection(4 hafta)"
]

LAB_FEATURES = [
    "CK-MB", "GLUKOZ", "WBCpik", "NEUpik", "LYMPpik", "EOSpik", "MONOpik", 
    "HB", "HTC", "PLT", "KREATIN", "AST", "ALT", 
    "TOTAL_KOLESTEROL", "TG", "LDL", "HDL"
]

# --- ANINDA DOĞRULAMA YAPAN WIDGET FONKSİYONU ---
def render_feature_widget(feature, data_dict):
    if feature in categorical_map:
        options_map = categorical_map[feature]
        options_list = list(options_map.keys())
        selected_option = st.selectbox(
            label=feature,
            options=options_list,
            index=None,
            placeholder="Select an option..."
        )
        data_dict[feature] = options_map[selected_option] if selected_option is not None else None
    else:
        min_val = 0.0
        max_val = None
        if feature == "AGE":
            min_val = 0
            max_val = 120
        
        data_dict[feature] = st.number_input(
            feature, 
            value=None, 
            placeholder="Enter value... (Numeric only)",
            format="%.4f",
            min_value=min_val,
            max_value=max_val
        )

# --- ANA UYGULAMA MANTIĞI ---
st.set_page_config(
    page_title=LANG_STRINGS["app_title"]["ENG"],
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon=None
)

# --- CUSTOM CSS FOR BETTER UI ---
st.markdown("""
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #2C3E50;
        border-bottom: 3px solid #3498DB;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #EBF5FB;
        border-left: 4px solid #3498DB;
        padding: 1rem;
        border-radius: 5px;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #FEF9E7;
        border-left: 4px solid #F39C12;
        padding: 1rem;
        border-radius: 5px;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #EAFAF1;
        border-left: 4px solid #27AE60;
        padding: 1rem;
        border-radius: 5px;
    }
    
    /* Error boxes */
    .stError {
        background-color: #FADBD8;
        border-left: 4px solid #E74C3C;
        padding: 1rem;
        border-radius: 5px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #2C3E50;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #3498DB;
        color: white;
        font-weight: 600;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2980B9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #ECF0F1;
    }
    </style>
""", unsafe_allow_html=True)

# --- Dil Durumunu Başlat ---
if 'language' not in st.session_state:
    st.session_state.language = 'TR' # DEĞİŞİKLİK: Varsayılan dil Türkçe

lang = st.session_state.language

# Artifact'ları yükle
artifacts = load_artifacts()

if artifacts is not None:
    model_artifacts, scaler, embedding_data, tsne_model, feature_list = artifacts
    
    # --- BASİT BAŞLIK VE DİL SEÇİMİ (APP BAR İPTAL EDİLDİ) ---
    col_title, col_lang = st.columns([4, 1])
    
    with col_title:
        st.title(T("main_title"))
    
    with col_lang:
        st.markdown("<br>", unsafe_allow_html=True)  # Vertical spacing
        new_lang = st.radio(
            "Language / Dil",
            options=['TR', 'ENG'],
            index=0 if lang == 'TR' else 1,
            key="language",
            horizontal=True,
            label_visibility="visible"
        )
    
    st.divider() # Başlık ve içerik arasına ayırıcı çizgi
    
    # --- Ana Arayüz (2 Sütun) ---
    col1, col2 = st.columns([1, 1]) 

    # --- SÜTUN 1: Veri Girişi ---
    with col1:
        st.header(T("header_input"))
        st.info(T("info_note"))
        
        with st.form("patient_form", clear_on_submit=False):
            patient_data = {}
            processed_features = set()
            
            # Sadece ALL_FEATURES listesindeki feature'ları kullan
            # feature_list'ten gelen feature'ları ALL_FEATURES ile filtrele
            valid_features = [f for f in feature_list if f in ALL_FEATURES]

            with st.expander(T("key_features"), expanded=True): 
                for feature in KEY_FEATURES:
                    if feature in valid_features:
                        render_feature_widget(feature, patient_data)
                        processed_features.add(feature)
            with st.expander(T("symptoms"), expanded=False):
                for feature in SYMPTOM_FEATURES:
                    if feature in valid_features:
                        render_feature_widget(feature, patient_data)
                        processed_features.add(feature)
            with st.expander(T("history"), expanded=False):
                for feature in HISTORY_FEATURES:
                    if feature in valid_features:
                        render_feature_widget(feature, patient_data)
                        processed_features.add(feature)
            with st.expander(T("labs"), expanded=False):
                for feature in LAB_FEATURES:
                    if feature in valid_features:
                        render_feature_widget(feature, patient_data)
                        processed_features.add(feature)
            
            # Diğer feature'lar (sadece ALL_FEATURES içindekiler)
            other_features = [f for f in valid_features if f not in processed_features]
            if other_features:
                with st.expander(T("other_features"), expanded=False):
                    for feature in other_features:
                        render_feature_widget(feature, patient_data)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing before button
            submit_button = st.form_submit_button(
                T("calculate_button"),
                type="primary",
                use_container_width=True
            )

    # --- SÜTUN 2: Çıktılar ve Karşılama Ekranı ---
    with col2:
        if submit_button:
            
            missing_features = [feature for feature, value in patient_data.items() if value is None]
            num_missing = len(missing_features)
            
            # Tüm alanlar dolu olmalı
            if num_missing > 0:
                st.header(T("header_output"))
                st.error(T("error_missing_fields").format(num_missing=num_missing))
                st.subheader(T("error_missing_list"))
                for f in missing_features: 
                    st.write(f"- {f}")
            else:
                # Tüm veriler dolu -> HESAPLA
                st.header(T("header_output"))
                
                # NEW_case_study mantığı ile hesapla
                x_new_vec_raw = predict_patient_uncertainty(patient_data, model_artifacts, feature_list)
                x_new_vec_imputed = np.nan_to_num(x_new_vec_raw).reshape(1, -1)
                x_new_std = scaler.transform(x_new_vec_imputed)
                new_coords_xy = find_tsne_position(x_new_std, tsne_model)

                st.subheader(T("plot_title_tsne"))
                # NEW_case_study'deki gibi görselleştir
                fig_tsne = plot_diagnostic_landscape(embedding_data['X_emb'], embedding_data['y'], new_patient_coords=new_coords_xy)
                st.pyplot(fig_tsne)
                
                st.subheader(T("plot_title_bar"))
                x_new_vec_df = pd.DataFrame({"Feature": feature_list, "Uncertainty Score": x_new_vec_raw}).sort_values(by="Uncertainty Score", ascending=False).head(20)
                st.markdown("**" + T("plot_top20") + "**")
                fig_bar = plot_uncertainty_vector(x_new_vec_df, lang)
                st.plotly_chart(fig_bar, use_container_width=True)
                
        else:
            # --- Karşılama Ekranı "Bulut"u gösterir ---
            
            # 1. ÖNCE "BULUT"U GÖSTER
            st.subheader(T("plot_title_tsne"))
            fig_tsne_initial = plot_diagnostic_landscape(
                embedding_data['X_emb'], 
                embedding_data['y']
                # new_patient_coords gönderilmiyor (None olacak)
            )
            st.pyplot(fig_tsne_initial)
            
            st.divider() # Grafik ve açıklama arasına çizgi
            
            # 2. SONRA "ARAÇ HAKKINDA" BİLGİSİNİ GÖSTER
            st.header(T("welcome_header"))
            st.info(T("welcome_info"))
            st.markdown(T("welcome_text"), unsafe_allow_html=True)
else:
    # Artifact'lar yüklenemezse
    st.error(T("load_error"))