import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import base64
import warnings
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.neighbors import NearestNeighbors
from uncertainty_utils import EPS
from uncertainty_transformer import UncertaintyTransformer  # required for unpickling

# Suppress scikit-learn version warnings for unpickling
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- CONSTANTS ---
# Renk Paleti
COLOR_ORANGE = "#F39C12"  # Grup 2 (AKS)
COLOR_BLUE = "#3498DB"  # Grup 1 (Miyokardit) & Ana Tema
COLOR_BACKGROUND = "#FFFFFF" 
COLOR_TEXT = "#333333" 
COLOR_NEW_PATIENT = "#E74C3C" 

# Artifact Dosya Yolları (absolute, Streamlit Cloud uyumlu)
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE, "best_model_finetuned.pkl")
METADATA_PATH = os.path.join(_BASE, "model_metadata.pkl")
TEST_METRICS_PATH = os.path.join(_BASE, "final_test_metrics.json")
EMBEDDING_PATH = os.path.join(_BASE, "app_artifacts", "embedding_data.npz")
LANDSCAPE_PATH = os.path.join(_BASE, "app_artifacts", "diagnostic_landscape.png")

# --- MODEL SABİTLERİ ---
G1, G2 = 1, 2 # Sadece iki grubumuz var
# EPS imported from uncertainty_utils (1e-12) for consistency 
IMPUTATION_THRESHOLD = 0 # Hiç eksik alan kabul edilmez - tüm feature'lar zorunlu

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
        "ENG": "NOTE: All fields are required. Please fill in all patient data.",
        "TR": "NOT: Tüm alanlar zorunludur. Lütfen tüm hasta verilerini doldurun."
    },
    "key_features": {"ENG": "Key Features", "TR": "Anahtar Özellikler"},
    "symptoms": {"ENG": "Symptoms & Presentation", "TR": "Semptomlar ve Başvuru"},
    "history": {"ENG": "Patient History & Comorbidities", "TR": "Hasta Geçmişi ve Komorbiditeler"},
    "labs": {"ENG": "Lab, ECG, & Imaging Results", "TR": "Laboratuvar, EKG ve Görüntüleme Sonuçları"},
    "other_features": {"ENG": "Other Features", "TR": "Diğer Özellikler"},
    "calculate_button": {"ENG": "Calculate Patient Position", "TR": "Hasta Pozisyonunu Hesapla"},
    "header_output": {"ENG": "Patient Position Analysis", "TR": "Hasta Pozisyon Analizi"},
    "error_missing_max": {
        "ENG": f"Error: {{num_missing}} fields are missing. Max allowed is {IMPUTATION_THRESHOLD}.",
        "TR": f"Hata: {{num_missing}} alan eksik. İzin verilen en fazla eksik alan sayısı {IMPUTATION_THRESHOLD}."
    },
    "error_missing_fields": {"ENG": "Missing Fields:", "TR": "Eksik Alanlar:"},
    "missing_and_more": {
        "ENG": "...and {count} more field(s) are missing.",
        "TR": "...ve {count} alan daha eksik."
    },
    "warning_all_required": {
        "ENG": "Please complete all missing fields before running the analysis. This tool requires complete patient data to produce a reliable clinical result.",
        "TR": "Analizi çalıştırmadan önce lütfen eksik alanların tamamını doldurun. Bu araç, güvenilir bir klinik sonuç üretebilmek için hastaya ait tüm verilerin eksiksiz girilmesini gerektirmektedir."
    },
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
    },
    "prediction_title": {
        "ENG": "Model Prediction",
        "TR": "Model Tahmini"
    },
    "predicted_class": {
        "ENG": "Predicted Class",
        "TR": "Tahmin Edilen Sınıf"
    },
    "prediction_probability": {
        "ENG": "Prediction Probability",
        "TR": "Tahmin Olasılığı"
    },
    "class_1_name": {
        "ENG": "Myocarditis",
        "TR": "Miyokardit"
    },
    "class_2_name": {
        "ENG": "ACS",
        "TR": "AKS"
    },
    "model_info": {
        "ENG": "Model Information",
        "TR": "Model Bilgisi"
    },
    "cv_score": {
        "ENG": "Cross-Validation F1-score",
        "TR": "Çapraz Doğrulama F1-skoru"
    },
    "test_score": {
        "ENG": "Test Set F1-score",
        "TR": "Test Set F1-skoru"
    },
    "placeholder_number": {
        "ENG": "Enter value...",
        "TR": "Değer girin..."
    },
    "placeholder_select": {
        "ENG": "Select an option...",
        "TR": "Bir seçenek seçin..."
    },
    "demo_selector_label": {
        "ENG": "Quick Demo Patient",
        "TR": "Hızlı Demo Hasta"
    }
}

# Çeviri (Translation) için yardımcı fonksiyon
def T(key):
    """Geçerli dile göre çevrilmiş metni alır."""
    lang = st.session_state.language
    return LANG_STRINGS.get(key, {}).get(lang, f"MISSING_KEY: {key}")

# --- ARTIFACT YÜKLEME ---
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    """Tüm artifact'ları diskten yükler ve cache'ler."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        _raw = np.load(EMBEDDING_PATH)
        embedding_data = {k: np.array(_raw[k]) for k in _raw.files}
        _raw.close()

        test_metrics = {}
        if os.path.exists(TEST_METRICS_PATH):
            with open(TEST_METRICS_PATH) as f:
                test_metrics = json.load(f)

        feature_list = list(metadata["features"])
        return model, metadata, embedding_data, test_metrics, feature_list
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        st.error("Please ensure best_model_finetuned.pkl, model_metadata.pkl, and app_artifacts/embedding_data.npz exist.")
        return None

@st.cache_data
def load_demo_patients():
    """demo_patients.txt dosyasından demo hasta verilerini yükler."""
    path = os.path.join(_BASE, "demo_patients.txt")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("patients", raw) if isinstance(raw, dict) else raw
    except Exception:
        return []

# --- TAHMİNLEME FONKSİYONLARI ---
def find_tsne_position(x_new_std, X_std_train, X_emb_train, k=5):
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean'); nn.fit(X_std_train)
    distances, indices = nn.kneighbors(x_new_std)
    neighbor_coords_2d = X_emb_train[indices.flatten()]
    new_position = np.mean(neighbor_coords_2d, axis=0)
    return new_position[0], new_position[1]

# --- PLOT/GRAFİK FONKSİYONLARI (ÇEVİRİLİ) ---

@st.cache_data
def _load_landscape_as_b64(path):
    """Notebook'tan kaydedilen PNG'yi base64'e çevirir (cache'li)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def plot_diagnostic_landscape(X_emb_train, y_train, lang, new_patient_coords=None):
    """
    Plotly interaktif t-SNE grafiği.
    Arka plan: notebook'tan kaydedilen diagnostic_landscape.png (birebir aynı görsel).
    Yeni hasta varsa yeşil yıldız olarak plotly scatter trace ile eklenir.
    """
    pad = 2.0
    xmin = float(X_emb_train[:, 0].min()) - pad
    xmax = float(X_emb_train[:, 0].max()) + pad
    ymin = float(X_emb_train[:, 1].min()) - pad
    ymax = float(X_emb_train[:, 1].max()) + pad

    b64 = _load_landscape_as_b64(LANDSCAPE_PATH)

    fig = go.Figure()

    # Eksen aralığını sabitleyen görünmez iz
    fig.add_trace(go.Scatter(
        x=[xmin, xmax], y=[ymin, ymax],
        mode="markers", marker=dict(opacity=0, size=0.1),
        showlegend=False, hoverinfo="skip"
    ))

    # Notebook PNG'si arka plan olarak
    fig.add_layout_image(dict(
        source=f"data:image/png;base64,{b64}",
        xref="x", yref="y",
        x=xmin, y=ymax,
        sizex=xmax - xmin, sizey=ymax - ymin,
        sizing="stretch",
        layer="below",
        xanchor="left", yanchor="top",
    ))

    # Legend proxy izleri
    for color, label in [
        ("red",  T("legend_g1")),
        ("blue", T("legend_g2")),
        ("gray", "Uncertain"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=12, color=color),
            name=label, showlegend=True
        ))

    # Yeni hasta yıldızı
    if new_patient_coords is not None:
        fig.add_trace(go.Scatter(
            x=[new_patient_coords[0]], y=[new_patient_coords[1]],
            mode="markers",
            marker=dict(symbol="star", size=20, color="limegreen",
                        line=dict(color="darkgreen", width=2)),
            name=T("legend_new"), showlegend=True
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[xmin, xmax]),
        yaxis=dict(visible=False, range=[ymin, ymax], scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(x=1.0, y=1.0, xanchor="right", yanchor="top",
                    bgcolor="rgba(0,0,0,0)", borderwidth=0),
        height=450,
    )
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

# --- KATEGORİK & GRUP HARİTALARI ---
yes_no_map = {"No": 0, "Yes": 1}
categorical_map = {
    "SEX": {"Female": 0, "Male": 1},
    "Socioeconomic Status": {"Poor": 0, "Good": 1},
    "Chest Pain Character": {"None": 0, "Stabbing / Localized": 1, "Pressure / Anginal": 2},
    "Infection type": {"URTI": 1, "Diarrhea": 2, "Vaccine": 3, "Other": 4},
    "DM": yes_no_map, "HT": yes_no_map, "HL": yes_no_map, "FH": yes_no_map,
    "SIGARA": yes_no_map, "KBY": yes_no_map, "PRIOR_KAH": yes_no_map,
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
KEY_FEATURES = [
    "AGE", "SEX", "Chest Pain Character", "PEAK_TROP",
    "Segmentary Wall Motion Abnormality", "MRI_LGE"
]
SYMPTOM_FEATURES = [
    "Chest Pain", "Chest Pain Duration(saat)", "Radiation", "Arm Pain",
    "Back Pain", "Epigastric Pain", "Relation with exercise",
    "Relation with Position", "Dyspnea", "Fatigue", "Nausea", "Çarpıntı",
    "Any Previous Pain Attacks"
]
HISTORY_FEATURES = [
    "DM", "HT", "HL", "FH", "SIGARA", "KBY", "PRIOR_KAH", "HIPOTIROIDI",
    "Socioeconomic Status", "Recent Infection(4 hafta)", "Infection type",
    "KANSER_KEMOTERAPI", "TASI_BRADIKARDI", "MADDE_ILAC_KULLANIMI",
    "Alcohol", "KOAH", "PAH", "HIPERTIROIDI", "REYNAULD"
]
LAB_ECG_FEATURES = [
    "Troponin_Sonucu", "TROPONIN_CUTOFF_DEGERI", "TROP_KATSAYISI", "CK-MB",
    "GLUKOZ", "WBCpik", "NEUpik", "LYMPpik", "EOSpik", "MONOpik", "HB",
    "HTC", "PLT", "KREATIN", "AST", "ALT", "ALBUMIN", "TOTAL_KOLESTEROL",
    "TG", "LDL", "HDL", "hs-CRP", "hs-CRP_CUTOFF", "hs-CRP_FOLD",
    "Sedimantation", "BNP", "HBAB1C", "D-DIMER", "EF", "Pericardial Effusion",
    "ECG_ST depression", "ECG_Location of ST depression ", "Level of ST-Dep_mm",
    "ECG_T neg", "ECG_Location of T negativity", "Level of T invertion_mm",
    "ECG_Q waves", "MRI_T2", "INHOSPITAL_EX", "EX_TARIHI", "BMI"
]

# --- ANINDA DOĞRULAMA YAPAN WIDGET FONKSİYONU ---
def render_feature_widget(feature, data_dict, prefill=None, key_suffix=""):
    # key_suffix, demo hasta değiştiğinde widget'ların sıfırlanmasını sağlar
    wkey = f"w__{feature}__{key_suffix}"
    if feature in categorical_map:
        options_map = categorical_map[feature]
        options_list = list(options_map.keys())
        default_index = None
        if prefill is not None:
            for i, val in enumerate(options_map.values()):
                if val == prefill:
                    default_index = i
                    break
        selected_option = st.selectbox(
            label=feature,
            options=options_list,
            index=default_index,
            placeholder=T("placeholder_select"),
            key=wkey,
        )
        data_dict[feature] = options_map[selected_option] if selected_option is not None else None
    else:
        min_val = 0.0
        max_val = None
        if feature == "AGE":
            min_val = 0.0
            max_val = 120.0

        data_dict[feature] = st.number_input(
            feature,
            value=float(prefill) if prefill is not None else None,
            placeholder=T("placeholder_number"),
            format="%.4f",
            min_value=min_val,
            max_value=max_val,
            key=wkey,
        )

# --- ANA UYGULAMA MANTIĞI ---
st.set_page_config(
    page_title=LANG_STRINGS["app_title"]["ENG"],
    layout="wide"
)

# --- Dil Durumunu Başlat ---
if 'language' not in st.session_state:
    st.session_state.language = 'TR' # DEĞİŞİKLİK: Varsayılan dil Türkçe

lang = st.session_state.language

# Artifact'ları yükle
artifacts = load_artifacts()

if artifacts is not None:
    model, metadata, embedding_data, test_metrics, feature_list = artifacts
    
    # --- BASİT BAŞLIK VE DİL SEÇİMİ (APP BAR İPTAL EDİLDİ) ---
    st.title(T("main_title"))

    demo_patients = load_demo_patients()
    _NO_SEL = "───"
    _demo_options = [_NO_SEL] + [p["name"] for p in demo_patients]

    _hdr_lang, _hdr_demo = st.columns([1, 2])
    with _hdr_lang:
        st.radio(
            "Language / Dil",
            options=['TR', 'ENG'],
            index=0 if lang == 'TR' else 1,
            key="language",
            horizontal=True,
        )
    with _hdr_demo:
        if demo_patients:
            _selected_demo = st.selectbox(
                T("demo_selector_label"),
                options=_demo_options,
                index=0,
                key="demo_patient_key",
            )
            _demo_idx = _demo_options.index(_selected_demo) - 1
            prefill_data = demo_patients[_demo_idx]["data"] if _demo_idx >= 0 else {}
        else:
            prefill_data = {}

    st.divider()
    
    # --- Ana Arayüz (2 Sütun) ---
    col1, col2 = st.columns([1, 1]) 

    # --- SÜTUN 1: Veri Girişi ---
    with col1:
        st.header(T("header_input"))
        st.info(T("info_note"))
        
        with st.form("patient_form"):
            patient_data = {}
            processed_features = set()

            with st.expander(T("key_features"), expanded=False):
                for feature in KEY_FEATURES:
                    if feature in feature_list:
                        render_feature_widget(feature, patient_data, prefill=prefill_data.get(feature), key_suffix=str(_demo_idx))
                        processed_features.add(feature)
            with st.expander(T("symptoms"), expanded=False):
                for feature in SYMPTOM_FEATURES:
                    if feature in feature_list:
                        render_feature_widget(feature, patient_data, prefill=prefill_data.get(feature), key_suffix=str(_demo_idx))
                        processed_features.add(feature)
            with st.expander(T("history"), expanded=False):
                for feature in HISTORY_FEATURES:
                    if feature in feature_list:
                        render_feature_widget(feature, patient_data, prefill=prefill_data.get(feature), key_suffix=str(_demo_idx))
                        processed_features.add(feature)
            with st.expander(T("labs"), expanded=False):
                for feature in LAB_ECG_FEATURES:
                    if feature in feature_list:
                        render_feature_widget(feature, patient_data, prefill=prefill_data.get(feature), key_suffix=str(_demo_idx))
                        processed_features.add(feature)

            other_features = [f for f in feature_list if f not in processed_features]
            if other_features:
                with st.expander(T("other_features"), expanded=False):
                    for feature in other_features:
                        render_feature_widget(feature, patient_data, prefill=prefill_data.get(feature), key_suffix=str(_demo_idx))
            
            submit_button = st.form_submit_button(
                T("calculate_button"),
                type="primary"
            )

    # --- SÜTUN 2: Çıktılar ve Karşılama Ekranı ---
    with col2:
        if submit_button:
            
            missing_features = [feature for feature, value in patient_data.items() if value is None]
            num_missing = len(missing_features)
            
            # Eksik feature varsa -> HATA (imputation yok)
            if num_missing > 0:
                st.error(T("error_missing_max").format(num_missing=num_missing))
                st.subheader(T("error_missing_fields"))
                for f in missing_features[:10]: st.write(f"- {f}")
                if len(missing_features) > 10:
                    st.write(T("missing_and_more").format(count=len(missing_features) - 10))
                st.warning(T("warning_all_required"))

            # Tüm feature'lar dolu -> HESAPLA
            else: 
                st.header(T("header_output"))
                st.success(T("success_all_data"))
                
                # Transform patient data through the fitted pipeline
                X_new_df = pd.DataFrame([patient_data])[feature_list]

                # Raw uncertainty values (transformer step only) — for bar chart
                x_new_unc = model.named_steps['uncertainty'].transform(X_new_df)
                x_new_vec_raw = x_new_unc[0]

                # Fully scaled representation (uncertainty + scaler) — for t-SNE positioning
                x_new_std = model.named_steps['scaler'].transform(x_new_unc)
                new_coords_xy = find_tsne_position(x_new_std, embedding_data['X_std'], embedding_data['X_emb'], k=5)

                # Full pipeline prediction
                y_pred = model.predict(X_new_df)
                y_prob = model.predict_proba(X_new_df)
                predicted_class = int(y_pred[0])
                clf_classes = list(model.named_steps['clf'].classes_)
                prob_class_1 = float(y_prob[0][clf_classes.index(G1)])
                prob_class_2 = float(y_prob[0][clf_classes.index(G2)])

                st.subheader(T("prediction_title"))
                col_pred1, col_pred2 = st.columns(2)

                with col_pred1:
                    st.metric(
                        T("predicted_class"),
                        T("class_1_name") if predicted_class == G1 else T("class_2_name")
                    )

                with col_pred2:
                    prob_display = prob_class_1 if predicted_class == G1 else prob_class_2
                    st.metric(
                        T("prediction_probability"),
                        f"{prob_display:.1%}"
                    )

                # Show probability breakdown
                prob_df = pd.DataFrame({
                    "Class": [T("class_1_name"), T("class_2_name")],
                    "Probability": [f"{prob_class_1:.1%}", f"{prob_class_2:.1%}"]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

                # Model info
                with st.expander(T("model_info"), expanded=False):
                    st.write(f"**Model:** {metadata['model_name']}")
                    cv = metadata.get('cv_results', {})
                    if cv.get('recall_mean') is not None:
                        st.write(f"**CV Recall (macro):** {cv['recall_mean']:.4f} ± {cv.get('recall_std', 0):.4f}")
                        st.caption("Nested CV estimate (single-model selection procedure per fold).")
                    if test_metrics.get('recall_macro') is not None:
                        st.write(f"**Test Recall (macro):** {test_metrics['recall_macro']:.4f}")
                    if test_metrics.get('f1_macro') is not None:
                        st.write(f"**{T('test_score')}:** {test_metrics['f1_macro']:.4f}")

                st.subheader(T("plot_title_tsne"))
                # 'new_patient_coords' parametresini gönderiyoruz
                fig_tsne = plot_diagnostic_landscape(embedding_data['X_emb'], embedding_data['y'], lang, new_patient_coords=new_coords_xy)
                st.plotly_chart(fig_tsne, use_container_width=True)
                
                st.subheader(T("plot_title_bar"))
                x_new_vec_df = pd.DataFrame({"Feature": feature_list, "Uncertainty Score": x_new_vec_raw}).sort_values(by="Uncertainty Score", ascending=False).head(20)
                st.write(T("plot_top20"))
                fig_bar = plot_uncertainty_vector(x_new_vec_df, lang)
                st.plotly_chart(fig_bar, use_container_width=True)
                
        else:
            # --- Karşılama Ekranı "Bulut"u gösterir ---
            
            # 1. ÖNCE "BULUT"U GÖSTER — notebook'tan kaydedilen PNG
            st.subheader(T("plot_title_tsne"))
            if os.path.exists(LANDSCAPE_PATH):
                with open(LANDSCAPE_PATH, "rb") as _f:
                    st.image(_f.read(), use_container_width=True)
            else:
                st.warning("Landscape PNG bulunamadı. NEW_uncertainty.ipynb'deki Cell 27'yi çalıştırın.")
                st.plotly_chart(
                    plot_diagnostic_landscape(embedding_data['X_emb'], embedding_data['y'], lang),
                    use_container_width=True
                )
            
            st.divider() # Grafik ve açıklama arasına çizgi
            
            # 2. SONRA "ARAÇ HAKKINDA" BİLGİSİNİ GÖSTER
            st.header(T("welcome_header"))
            st.info(T("welcome_info"))
            st.markdown(T("welcome_text"), unsafe_allow_html=True)
else:
    # Artifact'lar yüklenemezse
    st.error(T("load_error"))