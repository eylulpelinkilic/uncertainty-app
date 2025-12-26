import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

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
IMPUTATION_PATH = os.path.join(ARTIFACT_DIR, "imputation_values.pkl")

# --- MODEL SABİTLERİ ---
G1, G2 = 1, 2 # Sadece iki grubumuz var
EPS = 1e-12  # Notebook ile uyumlu (EPS_NB ile aynı)
IMPUTATION_THRESHOLD = 5 # En fazla izin verilen eksik alan sayısı

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
        "ENG": f"NOTE: You may leave up to {IMPUTATION_THRESHOLD} fields blank. They will be auto-imputed.",
        "TR": f"NOT: En fazla {IMPUTATION_THRESHOLD} alanı boş bırakabilirsiniz. Bu alanlar otomatik olarak doldurulacaktır."
    },
    "key_features": {"ENG": "Key Features", "TR": "Anahtar Özellikler"},
    "symptoms": {"ENG": "Symptoms & Presentation", "TR": "Semptomlar ve Başvuru"},
    "history": {"ENG": "Patient History & Comorbidities", "TR": "Hasta Geçmişi ve Komorbiditeler"},
    "labs": {"ENG": "Laboratory Results", "TR": "Laboratuvar Sonuçları"},
    "other_features": {"ENG": "Other Features", "TR": "Diğer Özellikler"},
    "calculate_button": {"ENG": "Calculate Patient Position", "TR": "Hasta Pozisyonunu Hesapla"},
    "header_output": {"ENG": "Patient Position Analysis", "TR": "Hasta Pozisyon Analizi"},
    "error_missing_max": {
        "ENG": f"Error: {{num_missing}} fields are missing. Max allowed is {IMPUTATION_THRESHOLD}.",
        "TR": f"Hata: {{num_missing}} alan eksik. İzin verilen en fazla eksik alan sayısı {IMPUTATION_THRESHOLD}."
    },
    "error_missing_fields": {"ENG": "Missing Fields:", "TR": "Eksik Alanlar:"},
    "warn_imputed": {
        "ENG": f"**Warning:** {{num_missing}} feature(s) were missing and have been imputed using the population average (mean/mode). Results may be less accurate.",
        "TR": f"**Uyarı:** {{num_missing}} özellik eksikti ve popülasyon ortalaması (mean/mode) kullanılarak otomatik dolduruldu. Sonuçlar daha az güvenilir olabilir."
    },
    "view_imputed": {"ENG": "View Imputed Features", "TR": "Doldurulan Özellikleri Gör"},
    "imputed_as": {"ENG": "imputed as", "TR": "olarak dolduruldu"},
    "critical_error_impute": {
        "ENG": "Critical Error: No imputation value found for {feature}. Calculation cannot proceed.",
        "TR": "Kritik Hata: {feature} için doldurma değeri bulunamadı. Hesaplama devam edemez."
    },
    "success_all_data": {
        "ENG": "All data fields were provided. No imputation was needed.",
        "TR": "Tüm veri alanları sağlandı. Otomatik doldurma gerekmedi."
    },
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
    """Tüm 4 artifact'ı diskten yükler ve cache'ler."""
    try:
        with open(MODEL_ARTIFACTS_PATH, "rb") as f: model_artifacts = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        embedding_data = np.load(EMBEDDING_PATH)
        with open(IMPUTATION_PATH, "rb") as f: imputation_values = pickle.load(f)
        feature_list = list(model_artifacts.keys())
        if not all(f in imputation_values for f in feature_list):
             raise Exception("Imputation map is missing features present in the model.")
        return model_artifacts, scaler, embedding_data, imputation_values, feature_list
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.error(f"Please RE-RUN `pre-processing.py` to generate all 4 artifacts.")
        return None

# --- TAHMİNLEME FONKSİYONLARI (NEW PIPELINE FROM NOTEBOOK) ---
def predict_patient_uncertainty(input_data, model_artifacts, feature_list):
    """
    Yeni pipeline mantığı ile uncertainty hesaplama - Notebook'tan
    Formula: x_f = z * (h / (js_f + EPS))
    Hem eski hem yeni artifact yapılarını destekler.
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

def find_tsne_position(x_new_std, X_std_train, X_emb_train, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean'); nn.fit(X_std_train)
    distances, indices = nn.kneighbors(x_new_std)
    neighbor_coords_2d = X_emb_train[indices.flatten()]
    new_position = np.mean(neighbor_coords_2d, axis=0)
    return new_position[0], new_position[1]

# --- PLOT/GRAFİK FONKSİYONLARI (ÇEVİRİLİ) ---

def plot_diagnostic_landscape(X_emb_train, y_train, lang, new_patient_coords=None):
    """
    2-sınıflı (G1 vs G2) t-SNE grafiğini çizer.
    'new_patient_coords' opsiyoneldir. Eğer verilmezse, sadece "bulut" çizilir.
    """
    df_emb = pd.DataFrame({"x": X_emb_train[:, 0], "y": X_emb_train[:, 1], "label": y_train})
    fig = go.Figure()
    
    # Grup 1 (Miyokardit) - Blue
    df_1 = df_emb[df_emb['label'] == G1]
    fig.add_trace(go.Scatter(x=df_1['x'], y=df_1['y'], mode='markers', marker=dict(color=COLOR_BLUE, size=5, opacity=0.6), name=T("legend_g1")))
    
    # Grup 2 (AKS-KONTROL) - Orange
    df_2 = df_emb[df_emb['label'] == G2]
    fig.add_trace(go.Scatter(x=df_2['x'], y=df_2['y'], mode='markers', marker=dict(color=COLOR_ORANGE, size=5, opacity=0.6), name=T("legend_g2")))
    
    # Sadece 'new_patient_coords' varsa kırmızı yıldızı çiz
    if new_patient_coords is not None:
        fig.add_trace(go.Scatter(
            x=[new_patient_coords[0]], y=[new_patient_coords[1]],
            mode='markers',
            marker=dict(color=COLOR_NEW_PATIENT, size=16, symbol='star', line=dict(color='Black', width=2)),
            name=T("legend_new")
        ))
    
    fig.update_layout(title=T("plot_title_tsne"), xaxis_title="t-SNE Dimension 1", yaxis_title="t-SNE Dimension 2", plot_bgcolor=COLOR_BACKGROUND, paper_bgcolor=COLOR_BACKGROUND, font_color=COLOR_TEXT, legend_title_text=T("legend_title"))
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

# NOTEBOOK'TAN GELEN GERÇEK FEATURE LİSTESİ (42 feature):
# ['AGE', 'SEX', 'DM', 'HT', 'HL', 'FH', 'SIGARA', 'KBY', 'PRIOR_KAH', 'KOAH', 
#  'Chest Pain', 'Chest Pain Character', 'Any Previous Pain Attacks', 'Chest Pain Duration(saat)', 
#  'Radiation', 'Arm Pain', 'Back Pain', 'Epigastric Pain', 'Relation with exercise', 
#  'Relation with Position', 'Dyspnea', 'Fatigue', 'Nausea', 'Çarpıntı', 'Recent Infection(4 hafta)', 
#  'PEAK_TROP', 'CK-MB', 'GLUKOZ', 'WBCpik', 'NEUpik', 'LYMPpik', 'EOSpik', 'MONOpik', 
#  'HB', 'HTC', 'PLT', 'KREATIN', 'AST', 'ALT', 'TOTAL_KOLESTEROL', 'TG', 'LDL', 'HDL']

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
    page_icon="🏥"
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
    model_artifacts, scaler, embedding_data, imputation_values, feature_list = artifacts
    
    # --- BASİT BAŞLIK VE DİL SEÇİMİ (APP BAR İPTAL EDİLDİ) ---
    col_title, col_lang = st.columns([4, 1])
    
    with col_title:
        st.title("🏥 " + T("main_title"))
    
    with col_lang:
        st.markdown("<br>", unsafe_allow_html=True)  # Vertical spacing
        new_lang = st.radio(
            "🌐 Language / Dil",
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
        st.header("📝 " + T("header_input"))
        st.info("💡 " + T("info_note"))
        
        with st.form("patient_form", clear_on_submit=False):
            patient_data = {}
            processed_features = set()

            with st.expander("🔑 " + T("key_features"), expanded=True): 
                for feature in KEY_FEATURES:
                    if feature in feature_list:
                        render_feature_widget(feature, patient_data)
                        processed_features.add(feature)
            with st.expander("🩺 " + T("symptoms"), expanded=False):
                for feature in SYMPTOM_FEATURES:
                    if feature in feature_list:
                        render_feature_widget(feature, patient_data)
                        processed_features.add(feature)
            with st.expander("📋 " + T("history"), expanded=False):
                for feature in HISTORY_FEATURES:
                    if feature in feature_list:
                        render_feature_widget(feature, patient_data)
                        processed_features.add(feature)
            with st.expander("🧪 " + T("labs"), expanded=False):
                for feature in LAB_FEATURES:
                    if feature in feature_list:
                        render_feature_widget(feature, patient_data)
                        processed_features.add(feature)
            
            other_features = [f for f in feature_list if f not in processed_features]
            if other_features:
                with st.expander("📊 " + T("other_features"), expanded=False):
                    for feature in other_features:
                        render_feature_widget(feature, patient_data)
            
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing before button
            submit_button = st.form_submit_button(
                "🚀 " + T("calculate_button"),
                type="primary",
                use_container_width=True
            )

    # --- SÜTUN 2: Çıktılar ve Karşılama Ekranı ---
    with col2:
        if submit_button:
            
            missing_features = [feature for feature, value in patient_data.items() if value is None]
            num_missing = len(missing_features)
            
            # 1. Durum: Çok fazla eksik -> HATA
            if num_missing > IMPUTATION_THRESHOLD:
                st.error(T("error_missing_max").format(num_missing=num_missing))
                st.subheader(T("error_missing_fields"))
                for f in missing_features[:10]: st.write(f"- {f}")
                if len(missing_features) > 10: st.write("...and more.")
            
            # 2. Durum: Az eksik -> DOLDUR VE UYAR
            elif num_missing > 0:
                st.header("📊 " + T("header_output"))
                st.warning("⚠️ " + T("warn_imputed").format(num_missing=num_missing))
                imputed_features_list = []
                imputed_patient_data = patient_data.copy()
                
                for feature in missing_features:
                    imputation_value = imputation_values.get(feature)
                    if imputation_value is not None:
                        imputed_patient_data[feature] = imputation_value
                        imputed_features_list.append(f"- **{feature}** ({T('imputed_as')}: `{imputation_value:.2f}`)")
                    else:
                        st.error(T("critical_error_impute").format(feature=feature))
                        imputed_patient_data = None 
                        break
                
                st.expander(T("view_imputed"), expanded=False).markdown("\n".join(imputed_features_list))
                
                if imputed_patient_data:
                    # Doldurulmuş Veri ile Hesapla
                    x_new_vec_raw = predict_patient_uncertainty(imputed_patient_data, model_artifacts, feature_list)
                    x_new_vec_imputed = np.nan_to_num(x_new_vec_raw).reshape(1, -1)
                    x_new_std = scaler.transform(x_new_vec_imputed)
                    new_coords_xy = find_tsne_position(x_new_std, embedding_data['X_std'], embedding_data['X_emb'], k=5)

                    st.subheader("🗺️ " + T("plot_title_tsne"))
                    # 'new_patient_coords' parametresini gönderiyoruz
                    fig_tsne = plot_diagnostic_landscape(embedding_data['X_emb'], embedding_data['y'], lang, new_patient_coords=new_coords_xy)
                    st.plotly_chart(fig_tsne, use_container_width=True)
                    
                    st.subheader("📈 " + T("plot_title_bar"))
                    x_new_vec_df = pd.DataFrame({"Feature": feature_list, "Uncertainty Score": x_new_vec_raw}).sort_values(by="Uncertainty Score", ascending=False).head(20)
                    st.markdown("**" + T("plot_top20") + "**")
                    fig_bar = plot_uncertainty_vector(x_new_vec_df, lang)
                    st.plotly_chart(fig_bar, use_container_width=True)

            # 3. Durum: Eksik yok -> HESAPLA
            else: 
                st.header("📊 " + T("header_output"))
                st.success("✅ " + T("success_all_data"))
                
                # Tam Veri ile Hesapla
                x_new_vec_raw = predict_patient_uncertainty(patient_data, model_artifacts, feature_list)
                x_new_vec_imputed = np.nan_to_num(x_new_vec_raw).reshape(1, -1)
                x_new_std = scaler.transform(x_new_vec_imputed)
                new_coords_xy = find_tsne_position(x_new_std, embedding_data['X_std'], embedding_data['X_emb'], k=5)

                st.subheader("🗺️ " + T("plot_title_tsne"))
                # 'new_patient_coords' parametresini gönderiyoruz
                fig_tsne = plot_diagnostic_landscape(embedding_data['X_emb'], embedding_data['y'], lang, new_patient_coords=new_coords_xy)
                st.plotly_chart(fig_tsne, use_container_width=True)
                
                st.subheader("📈 " + T("plot_title_bar"))
                x_new_vec_df = pd.DataFrame({"Feature": feature_list, "Uncertainty Score": x_new_vec_raw}).sort_values(by="Uncertainty Score", ascending=False).head(20)
                st.markdown("**" + T("plot_top20") + "**")
                fig_bar = plot_uncertainty_vector(x_new_vec_df, lang)
                st.plotly_chart(fig_bar, use_container_width=True)
                
        else:
            # --- Karşılama Ekranı "Bulut"u gösterir ---
            
            # 1. ÖNCE "BULUT"U GÖSTER
            st.subheader("🗺️ " + T("plot_title_tsne"))
            fig_tsne_initial = plot_diagnostic_landscape(
                embedding_data['X_emb'], 
                embedding_data['y'],
                lang
                # new_patient_coords gönderilmiyor (None olacak)
            )
            st.plotly_chart(fig_tsne_initial, use_container_width=True)
            
            st.divider() # Grafik ve açıklama arasına çizgi
            
            # 2. SONRA "ARAÇ HAKKINDA" BİLGİSİNİ GÖSTER
            st.header("ℹ️ " + T("welcome_header"))
            st.info("💡 " + T("welcome_info"))
            st.markdown(T("welcome_text"), unsafe_allow_html=True)
else:
    # Artifact'lar yüklenemezse
    st.error(T("load_error"))