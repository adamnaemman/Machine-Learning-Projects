# Use for deploy our project
import os
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# Lazy imports (avoid crashing if one framework is missing)
try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    import tensorflow as tf
    TF_OK = True
except ImportError:
    TF_OK = False

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

# ============================================================
# PAGE CONFIG  (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Tomato Disease Classifier",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — dark organic theme with tomato red accents
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #0f1410;
    color: #e8e4dc;
}

/* ── Hero title ── */
.hero-wrap {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 900;
    color: #e8e4dc;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0;
}
.hero-title span { color: #d94f3b; }
.hero-sub {
    color: #7a8c74;
    font-size: 1rem;
    font-weight: 300;
    margin-top: 0.5rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #2a332a;
    margin: 1rem 0 2rem;
}

/* ── Result cards ── */
.result-card {
    background: #161d16;
    border: 1px solid #2a332a;
    border-left: 4px solid #d94f3b;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
}
.result-card.healthy {
    border-left-color: #4caf79;
}
.result-card h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #e8e4dc;
    margin: 0 0 0.3rem;
}
.result-card .meta {
    color: #7a8c74;
    font-size: 0.88rem;
    margin-bottom: 0.8rem;
}
.result-card .desc-label {
    color: #a0b09a;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.8rem;
    margin-bottom: 0.2rem;
}
.result-card .desc-text {
    color: #c8c4bc;
    font-size: 0.95rem;
    line-height: 1.55;
}

/* ── Compare cards ── */
.compare-card {
    background: #161d16;
    border: 1px solid #2a332a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    height: 100%;
}
.compare-card .model-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #d94f3b;
    margin-bottom: 0.6rem;
}
.compare-card .pred-label {
    font-size: 1rem;
    font-weight: 500;
    color: #e8e4dc;
}
.compare-card .conf-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #4caf79;
    line-height: 1;
    margin: 0.3rem 0;
}
.compare-card .time-val {
    font-size: 0.8rem;
    color: #7a8c74;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    margin-right: 0.3rem;
}
.badge-high   { background: #4a1a18; color: #f08080; }
.badge-medium { background: #3d2f10; color: #f0b060; }
.badge-low    { background: #1a3020; color: #70d090; }
.badge-none   { background: #1a2a1a; color: #70d090; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0c110c;
    border-right: 1px solid #1e261e;
}
section[data-testid="stSidebar"] .stMetric {
    background: #161d16;
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.4rem;
    border: 1px solid #2a332a;
}

/* ── Buttons ── */
.stButton > button {
    background: #d94f3b !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #c0392b !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161d16;
    border: 1px dashed #3a4a3a;
    border-radius: 10px;
    padding: 0.5rem;
}

/* ── Selectbox / Checkbox ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stCheckbox"] {
    background: #161d16 !important;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: #a0b09a;
    border-bottom: 1px solid #2a332a;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

/* ── Plotly chart background fix ── */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

DISEASE_INFO = {
    "Tomato_Bacterial_spot": {
        "display": "Bacterial Spot",
        "desc": "Caused by Xanthomonas bacteria. Produces small, dark, water-soaked spots on leaves and fruits that may turn brown and papery.",
        "treatment": "Apply copper-based bactericides at first sign. Remove and destroy infected plant parts. Avoid overhead irrigation.",
        "severity": "Medium",
    },
    "Tomato_Early_blight": {
        "display": "Early Blight",
        "desc": "Caused by Alternaria solani fungus. Presents as dark, concentric ring spots (target-like) on older, lower leaves first.",
        "treatment": "Apply chlorothalonil or mancozeb fungicide. Stake plants for airflow. Rotate crops annually.",
        "severity": "Medium",
    },
    "Tomato_Late_blight": {
        "display": "Late Blight",
        "desc": "Caused by Phytophthora infestans. Rapid water-soaked lesions that turn brown-black; white mold may appear on leaf undersides.",
        "treatment": "Apply copper fungicides immediately. Remove and bag infected plants. Do not compost. Act fast — spreads quickly.",
        "severity": "High",
    },
    "Tomato_Leaf_Mold": {
        "display": "Leaf Mold",
        "desc": "Caused by Passalora fulva. Pale yellow spots on upper leaf surface with olive-green to grey mold on the underside.",
        "treatment": "Improve greenhouse ventilation. Reduce humidity. Apply approved fungicide if severe.",
        "severity": "Low",
    },
    "Tomato_Septoria_leaf_spot": {
        "display": "Septoria Leaf Spot",
        "desc": "Caused by Septoria lycopersici. Small circular spots with dark brown borders and light grey centres; tiny black dots inside.",
        "treatment": "Remove infected leaves. Apply copper or mancozeb fungicides. Mulch soil to reduce splash dispersal.",
        "severity": "Medium",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display": "Spider Mites",
        "desc": "Tiny two-spotted mites that feed on leaf cells, causing fine stippling, yellowing, and webbing on leaf undersides.",
        "treatment": "Apply miticide or neem oil spray. Introduce predatory mites. Keep plants well-watered — mites thrive in drought.",
        "severity": "Medium",
    },
    "Tomato__Target_Spot": {
        "display": "Target Spot",
        "desc": "Caused by Corynespora cassiicola. Brown circular spots with concentric rings resembling a target; affects leaves, stems, and fruit.",
        "treatment": "Apply fungicides (azoxystrobin, chlorothalonil). Prune lower leaves. Avoid overhead watering.",
        "severity": "Medium",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display": "Yellow Leaf Curl Virus",
        "desc": "Viral disease transmitted by silverleaf whiteflies. Causes severe upward leaf curling, yellowing, and stunted growth.",
        "treatment": "Control whitefly population using insecticides or yellow sticky traps. Remove and destroy infected plants. No cure once infected.",
        "severity": "High",
    },
    "Tomato__Tomato_mosaic_virus": {
        "display": "Mosaic Virus",
        "desc": "Highly contagious virus causing mottled light/dark green pattern on leaves, distortion, and reduced fruit quality.",
        "treatment": "No cure. Remove infected plants immediately. Disinfect tools with bleach. Wash hands between plants.",
        "severity": "High",
    },
    "Tomato_healthy": {
        "display": "Healthy",
        "desc": "No disease detected. The tomato plant appears to be in good health with no visible symptoms.",
        "treatment": "Continue regular care — adequate water, balanced fertiliser, and routine scouting for early pest/disease signs.",
        "severity": "None",
    },
}

SEVERITY_BADGE = {
    "None":   ('<span class="badge badge-none">✔ Healthy</span>', "#4caf79"),
    "Low":    ('<span class="badge badge-low">◎ Low</span>',     "#70d090"),
    "Medium": ('<span class="badge badge-medium">⚠ Medium</span>', "#f0b060"),
    "High":   ('<span class="badge badge-high">✕ High</span>',   "#f08080"),
}

# Model file paths (relative to app.py)
MODEL_PATHS = {
    "YOLOv11n": "models/yolov11n3/weights/best.pt",
    "YOLOv8n":  "models/yolov8n/weights/best.pt",
    "CNN":       "models/tomato_cnn/tomato_cnn.keras",
}

# Reported test accuracies for sidebar display
MODEL_ACCURACY = {
    "YOLOv11n": 99.00,
    "YOLOv8n":  98.50,
    "CNN":       98.30,
}

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_all_models():
    """
    Load all three models once and cache them.
    Returns: dict of {name: model_object}
    """
    loaded = {}

    # ── YOLOv11n ──────────────────────────────────────────────
    if YOLO_OK and os.path.exists(MODEL_PATHS["YOLOv11n"]):
        try:
            loaded["YOLOv11n"] = YOLO(MODEL_PATHS["YOLOv11n"])
        except Exception as e:
            st.warning(f"YOLOv11n failed to load: {e}")

    # ── YOLOv8n ───────────────────────────────────────────────
    if YOLO_OK and os.path.exists(MODEL_PATHS["YOLOv8n"]):
        try:
            loaded["YOLOv8n"] = YOLO(MODEL_PATHS["YOLOv8n"])
        except Exception as e:
            st.warning(f"YOLOv8n failed to load: {e}")

    # ── Keras CNN (TensorFlow) ────────────────────────────────
    # Trained with tf.keras and saved via model.save('tomato_cnn.keras')
    if TF_OK and os.path.exists(MODEL_PATHS["CNN"]):
        try:
            cnn = tf.keras.models.load_model(MODEL_PATHS["CNN"])
            loaded["CNN"] = cnn
        except Exception as e:
            st.warning(f"CNN failed to load: {e}")

    return loaded


# ============================================================
# INFERENCE
# ============================================================
def predict(model_name: str, model, image: Image.Image):
    """
    Run inference for one model.
    Returns: (top1_idx: int, probs: np.ndarray, elapsed_ms: float)
    """
    t0 = time.perf_counter()

    if model_name in ("YOLOv11n", "YOLOv8n"):
        # YOLO classification — handles its own preprocessing internally
        result   = model.predict(image, verbose=False, imgsz=224)
        probs    = result[0].probs.data.cpu().numpy().astype(float)
        top1_idx = int(np.argmax(probs))

    elif model_name == "CNN":
        # TensorFlow / Keras CNN
        # Resize to match training input_shape (224, 224)
        img_array = np.array(image.resize((224, 224)), dtype=np.float32)
        # The model's first layer is Rescaling(1/255) so we feed raw [0,255] pixels
        img_tensor = np.expand_dims(img_array, axis=0)          # (1, 224, 224, 3)
        raw_output = model.predict(img_tensor, verbose=0)        # (1, num_classes)
        probs      = raw_output[0].astype(float)
        top1_idx   = int(np.argmax(probs))

    else:
        raise ValueError(f"Unknown model: {model_name}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return top1_idx, probs, elapsed_ms


# ============================================================
# PLOTLY CHART HELPER
# ============================================================
def top5_chart(probs: np.ndarray) -> go.Figure:
    top5_idx   = np.argsort(probs)[::-1][:5]
    top5_names = [DISEASE_INFO[CLASS_NAMES[i]]["display"] for i in top5_idx]
    top5_conf  = [probs[i] * 100 for i in top5_idx]
    bar_colors = [
        "#4caf79" if CLASS_NAMES[i] == "Tomato_healthy" else "#d94f3b"
        for i in top5_idx
    ]
    fig = go.Figure(go.Bar(
        x            = top5_conf,
        y            = top5_names,
        orientation  = "h",
        marker_color = bar_colors,
        text         = [f"{v:.1f}%" for v in top5_conf],
        textposition = "inside",
        textfont     = dict(color="white", size=13, family="DM Sans"),
    ))
    fig.update_layout(
        xaxis         = dict(title="Confidence (%)", range=[0, 108],
                             gridcolor="#1e261e", tickfont=dict(color="#7a8c74")),
        yaxis         = dict(autorange="reversed", tickfont=dict(color="#c8c4bc", size=12)),
        height        = 270,
        margin        = dict(l=10, r=20, t=10, b=40),
        plot_bgcolor  = "#161d16",
        paper_bgcolor = "#161d16",
        font          = dict(family="DM Sans"),
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        "<h2 style='font-family:Playfair Display,serif;color:#d94f3b;"
        "font-size:1.4rem;margin-bottom:0.2rem'>🍅 About</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#7a8c74;font-size:0.88rem;line-height:1.5'>"
        "Upload a tomato leaf photo to detect one of <strong style='color:#c8c4bc'>10 conditions</strong> "
        "using three deep learning models trained on the PlantVillage dataset.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:#1e261e;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-family:Playfair Display,serif;color:#a0b09a;"
        "font-size:1rem;margin-bottom:0.6rem'>📊 Model Accuracy</p>",
        unsafe_allow_html=True,
    )
    for mname, acc in MODEL_ACCURACY.items():
        st.metric(label=mname, value=f"{acc:.2f}%", delta="Test set")

    st.markdown("<hr style='border-color:#1e261e;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-family:Playfair Display,serif;color:#a0b09a;font-size:1rem;margin-bottom:0.4rem'>"
        "🌿 Disease Severity Guide</p>",
        unsafe_allow_html=True,
    )
    for sev, (badge, _) in SEVERITY_BADGE.items():
        st.markdown(
            f"<div style='margin-bottom:0.3rem'>{badge} {sev}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#1e261e;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#3a4a3a;font-size:0.78rem;text-align:center'>"
        "BICS 4340 / CSCI 4340 · Machine Learning<br>"
        "Semester II 2025/2026</p>",
        unsafe_allow_html=True,
    )

# ============================================================
# MAIN — HERO
# ============================================================
st.markdown("""
<div class="hero-wrap">
    <h1 class="hero-title">🍅 Tomato <span>Disease</span> Classifier</h1>
    <p class="hero-sub">PlantVillage · Deep Learning · 10 Classes · 3 Models</p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS
# ============================================================
with st.spinner("🔄 Loading models — this takes a moment on first run..."):
    models_dict = load_all_models()

if not models_dict:
    st.error(
        "⚠️ **No model files found.** "
        "Please place your model files in the `models/` folder as shown in the file structure comment at the top of `app.py`."
    )
    st.stop()

available_models = list(models_dict.keys())

# ============================================================
# CONTROLS + UPLOAD
# ============================================================
ctrl_col, img_col = st.columns([1, 1], gap="large")

with ctrl_col:
    st.markdown('<p class="section-header">⚙️ Settings</p>', unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Model",
        options=available_models,
        help="Choose which model to run inference with.",
    )
    compare_all = st.checkbox(
        "Compare all models side-by-side",
        value=False,
        help="Run all loaded models and display results together.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📁 Upload Image</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a tomato leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        st.markdown("<br>", unsafe_allow_html=True)
        analyse_btn = st.button("🔍  Analyse Leaf")

with img_col:
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.markdown(
            "<div style='background:#161d16;border:1px dashed #2a332a;border-radius:12px;"
            "height:300px;display:flex;align-items:center;justify-content:center;"
            "color:#3a4a3a;font-size:0.9rem'>Leaf image will appear here</div>",
            unsafe_allow_html=True,
        )

# ============================================================
# PREDICTION
# ============================================================
if uploaded and analyse_btn:
    image = Image.open(uploaded).convert("RGB")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown(
        "<p class='section-header' style='font-size:1.4rem'>🧪 Results</p>",
        unsafe_allow_html=True,
    )

    # ── Compare mode ──────────────────────────────────────────
    if compare_all and len(models_dict) > 1:
        st.markdown(
            "<p style='color:#7a8c74;font-size:0.88rem;margin-bottom:1rem'>"
            "Running all available models on your image…</p>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(models_dict), gap="medium")
        for idx, (mname, mmodel) in enumerate(models_dict.items()):
            with st.spinner(f"Running {mname}…"):
                top1, probs, elapsed = predict(mname, mmodel, image)
            info = DISEASE_INFO[CLASS_NAMES[top1]]
            conf = probs[top1] * 100
            badge_html, _ = SEVERITY_BADGE[info["severity"]]

            with cols[idx]:
                st.markdown(f"""
                <div class="compare-card">
                    <div class="model-name">{mname}</div>
                    <div class="pred-label">{info['display']}</div>
                    <div class="conf-val">{conf:.1f}%</div>
                    <div>{badge_html}</div>
                    <div class="time-val" style="margin-top:0.5rem">{elapsed:.0f} ms</div>
                </div>
                """, unsafe_allow_html=True)

        # Grouped bar chart for comparison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p class='section-header'>📊 Confidence Comparison Across Models</p>",
            unsafe_allow_html=True,
        )

        all_results = {}
        for mname, mmodel in models_dict.items():
            top1, probs, _ = predict(mname, mmodel, image)
            all_results[mname] = probs

        # Union of top-5 classes across all models
        union_idx = set()
        for probs in all_results.values():
            union_idx.update(np.argsort(probs)[::-1][:5])
        union_idx = sorted(union_idx, key=lambda i: -max(r[i] for r in all_results.values()))[:6]
        x_labels  = [DISEASE_INFO[CLASS_NAMES[i]]["display"] for i in union_idx]

        comp_fig = go.Figure()
        bar_palette = ["#d94f3b", "#f0a030", "#4caf79"]
        for j, (mname, probs) in enumerate(all_results.items()):
            comp_fig.add_trace(go.Bar(
                name         = mname,
                x            = x_labels,
                y            = [probs[i] * 100 for i in union_idx],
                marker_color = bar_palette[j % len(bar_palette)],
                text         = [f"{probs[i]*100:.1f}%" for i in union_idx],
                textposition = "outside",
                textfont     = dict(size=10, color="#c8c4bc"),
            ))
        comp_fig.update_layout(
            barmode       = "group",
            height        = 320,
            margin        = dict(l=10, r=10, t=10, b=60),
            plot_bgcolor  = "#161d16",
            paper_bgcolor = "#161d16",
            legend        = dict(font=dict(color="#c8c4bc"), bgcolor="#161d16"),
            xaxis         = dict(tickfont=dict(color="#c8c4bc", size=11), gridcolor="#1e261e"),
            yaxis         = dict(title="Confidence (%)", tickfont=dict(color="#7a8c74"),
                                 gridcolor="#1e261e"),
        )
        st.plotly_chart(comp_fig, use_container_width=True)

    # ── Single model mode ──────────────────────────────────────
    else:
        with st.spinner(f"Analysing with {selected_model}…"):
            top1, probs, elapsed = predict(
                selected_model, models_dict[selected_model], image
            )

        info       = DISEASE_INFO[CLASS_NAMES[top1]]
        conf       = probs[top1] * 100
        is_healthy = CLASS_NAMES[top1] == "Tomato_healthy"
        card_cls   = "result-card healthy" if is_healthy else "result-card"
        icon       = "✅" if is_healthy else "⚠️"
        badge_html, badge_color = SEVERITY_BADGE[info["severity"]]

        st.markdown(f"""
        <div class="{card_cls}">
            <h2>{icon} {info['display']}</h2>
            <div class="meta">
                Model: <strong style="color:#c8c4bc">{selected_model}</strong> &nbsp;·&nbsp;
                Confidence: <strong style="color:#c8c4bc">{conf:.1f}%</strong> &nbsp;·&nbsp;
                Severity: {badge_html} &nbsp;·&nbsp;
                Inference: <strong style="color:#c8c4bc">{elapsed:.0f} ms</strong>
            </div>
            <div class="desc-label">Description</div>
            <div class="desc-text">{info['desc']}</div>
            <div class="desc-label">Recommended Action</div>
            <div class="desc-text">{info['treatment']}</div>
        </div>
        """, unsafe_allow_html=True)

        res_left, res_right = st.columns([1.1, 1], gap="large")

        with res_left:
            st.markdown(
                "<p class='section-header' style='margin-top:1.2rem'>📈 Top-5 Predictions</p>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(top5_chart(probs), use_container_width=True)

        with res_right:
            st.markdown(
                "<p class='section-header' style='margin-top:1.2rem'>📋 All Class Scores</p>",
                unsafe_allow_html=True,
            )
            sorted_idx = np.argsort(probs)[::-1]
            for rank, i in enumerate(sorted_idx):
                label  = DISEASE_INFO[CLASS_NAMES[i]]["display"]
                pct    = probs[i] * 100
                bar_w  = int(pct)
                color  = "#4caf79" if CLASS_NAMES[i] == "Tomato_healthy" else "#d94f3b"
                bold_s = "font-weight:600;color:#e8e4dc" if rank == 0 else "color:#a0b09a"
                st.markdown(f"""
                <div style="margin-bottom:0.45rem">
                    <div style="display:flex;justify-content:space-between;
                                font-size:0.82rem;margin-bottom:2px">
                        <span style="{bold_s}">{label}</span>
                        <span style="color:#7a8c74">{pct:.2f}%</span>
                    </div>
                    <div style="background:#1e261e;border-radius:4px;height:5px">
                        <div style="width:{bar_w}%;background:{color};
                                    height:5px;border-radius:4px"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    "<hr style='border-color:#1e261e;margin-top:3rem'>"
    "<p style='text-align:center;color:#3a4a3a;font-size:0.8rem;padding-bottom:1rem'>"
    "BICS 4340 / CSCI 4340 Machine Learning &nbsp;·&nbsp; PlantVillage Dataset &nbsp;·&nbsp; "
    "Supervised by Prof. Dr. Amelia Ritahani binti Ismail</p>",
    unsafe_allow_html=True,
)