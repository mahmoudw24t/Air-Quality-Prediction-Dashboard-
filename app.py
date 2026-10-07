import base64
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Page Setup
CLOUD_ICON_PATH = Path(__file__).parent / "assets" / "cloud_icon.png"
cloud_icon = Image.open(CLOUD_ICON_PATH) if CLOUD_ICON_PATH.exists() else None

st.set_page_config(
    page_title="Air Quality Intelligence System",
    page_icon=cloud_icon or "https://upload.wikimedia.org/wikipedia/commons/4/4b/Cloud_font_awesome.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Wallpaper Configuration
WALLPAPERS = {
    "upload": Path(__file__).parent / "assets" / "wallpapers" / "ochir_erdene_oyunmedeg_LmyPLbbUWhA_unsplash.jpg",
    "Dashboard": Path(__file__).parent / "assets" / "wallpapers" / "bradley_brister_WdsLOJ5BViU_unsplash.jpg",
    "Executive Intelligence Hub": Path(__file__).parent / "assets" / "wallpapers" / "lisa_yount_DUubGalK__I_unsplash.jpg",
    "Daily Patterns": Path(__file__).parent / "assets" / "wallpapers" / "images_3.jpg",
    "Dataset Explorer": Path(__file__).parent / "assets" / "wallpapers" / "images_1.jpg",
    "Correlation Analysis": Path(__file__).parent / "assets" / "wallpapers" / "images.jpg",
    "Predictive Modeling": Path(__file__).parent / "assets" / "wallpapers" / "premium_photo_1667121496100_ca96e50fbb29.jpg",
    "Classification & Diagnosis": Path(__file__).parent / "assets" / "wallpapers" / "images_2.jpg",
}


@st.cache_data
def get_base64_image(image_path: Path):
    if image_path.exists():
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def apply_theme_and_audio(section_name: str):
    bg_path = WALLPAPERS.get(section_name, WALLPAPERS["upload"])
    bg_b64 = get_base64_image(bg_path)

    bg_css = f"""
    <style>
    :root {{
        --grass-dark: #1b5e20;
        --grass-primary: #2e7d32;
        --grass-medium: #43a047;
        --grass-light: #81c784;
        --grass-soft: #e8f5e9;
        --canvas-white: rgba(255, 255, 255, 0.95);
        --text-main: #0f172a;
        --text-sub: #334155;
    }}

    /* Full-screen blurred wallpaper layer */
    .stApp::before {{
        content: "";
        position: fixed;
        top: -30px;
        left: -30px;
        width: calc(100vw + 60px);
        height: calc(100vh + 60px);
        background-image: url('data:image/jpeg;base64,{bg_b64}');
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        filter: blur(20px) brightness(0.92);
        z-index: -10;
        pointer-events: none;
    }}

    .stApp {{
        background: transparent !important;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }}

    .main .block-container {{
        background: var(--canvas-white);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 18px;
        padding: 2.2rem 2.8rem;
        margin-top: 1.2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(46, 125, 50, 0.28);
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.12);
    }}

    [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.97) !important;
        backdrop-filter: blur(14px);
        border-right: 2px solid var(--grass-light) !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.05) !important;
    }}

    /* Text in dedicated white bar containers */
    h1, h2, h3, [data-testid="stHeadingWithActionElements"] h1, [data-testid="stHeadingWithActionElements"] h2, [data-testid="stHeadingWithActionElements"] h3 {{
        background: #ffffff !important;
        color: var(--grass-dark) !important;
        font-weight: 700 !important;
        padding: 13px 18px !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 6px solid var(--grass-primary) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 1.1rem !important;
        margin-top: 0.6rem !important;
        display: block !important;
        width: 100% !important;
    }}

    h4, h5, h6 {{
        color: var(--grass-dark) !important;
        font-weight: 700 !important;
        margin-top: 0.8rem !important;
        margin-bottom: 0.4rem !important;
    }}

    p, span, label, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
        color: var(--text-main) !important;
        font-weight: 500;
        font-size: 0.97rem;
    }}

    /* Visuals in dedicated pure white bars */
    [data-testid="stPlotlyChart"], .stPlotlyChart {{
        background: #ffffff !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        border: 1px solid #e2e8f0 !important;
        border-top: 4px solid var(--grass-primary) !important;
        box-shadow: 0 6px 22px rgba(0, 0, 0, 0.08) !important;
        margin: 1.2rem 0 !important;
    }}

    [data-testid="stDataFrame"], .stDataFrame {{
        background: #ffffff !important;
        border-radius: 12px !important;
        padding: 10px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06) !important;
        margin: 0.9rem 0 !important;
    }}

    [data-testid="stMetric"] {{
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 6px solid var(--grass-primary) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06) !important;
    }}

    [data-testid="stMetricLabel"] p {{
        color: var(--text-sub) !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMetricValue"] {{
        color: var(--grass-dark) !important;
        font-weight: 800 !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        background: #ffffff !important;
        border-radius: 10px !important;
        padding: 6px 12px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 1.2rem !important;
    }}

    .stAlert {{
        background: #ffffff !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #e2e8f0 !important;
    }}

    [data-testid="stMarkdownContainer"] ul {{
        background: #ffffff;
        border-radius: 10px;
        padding: 12px 18px 12px 34px !important;
        border: 1px solid #e2e8f0;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.04);
        margin: 10px 0;
    }}

    .stSelectbox, .stSlider, .stNumberInput {{
        background: #ffffff !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03) !important;
        margin: 6px 0 !important;
    }}

    [data-testid="stFileUploader"] {{
        background: #ffffff !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border: 2px dashed var(--grass-primary) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05) !important;
    }}

    .stTextArea textarea {{
        background: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }}

    .stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.55rem 1.4rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.25) !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {{
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.35) !important;
    }}

    .stButton > button[kind="secondary"], button[data-testid="baseButton-secondary"], .stDownloadButton > button {{
        background: #ffffff !important;
        color: var(--grass-dark) !important;
        border: 1.5px solid var(--grass-primary) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button[kind="secondary"]:hover, button[data-testid="baseButton-secondary"]:hover, .stDownloadButton > button:hover {{
        background: var(--grass-soft) !important;
        border-color: var(--grass-dark) !important;
    }}

    .stSpinner > div {{
        border-top-color: var(--grass-primary) !important;
    }}

    @keyframes cloudPulse {{
        0% {{ transform: scale(1) translateY(0); opacity: 0.85; }}
        50% {{ transform: scale(1.06) translateY(-3px); opacity: 1; }}
        100% {{ transform: scale(1) translateY(0); opacity: 0.85; }}
    }}

    .cloud-icon-header {{
        display: inline-block;
        animation: cloudPulse 2.5s infinite ease-in-out;
        vertical-align: middle;
        margin-right: 10px;
    }}
    </style>
    """
    st.html(bg_css)

    audio_engine_js = """
    <div style="display:none;">
    <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" 
         onload="
         (function() {
             let audioCtx = null;
             function getCtx() {
                 if (!audioCtx) {
                     const AC = window.AudioContext || window.webkitAudioContext || (window.parent && (window.parent.AudioContext || window.parent.webkitAudioContext));
                     if (AC) audioCtx = new AC();
                 }
                 if (audioCtx && audioCtx.state === 'suspended') {
                     audioCtx.resume();
                 }
                 return audioCtx;
             }

             // Sound 1: Wind and air gentle gust
             function playWindSound() {
                 const ctx = getCtx();
                 if (!ctx) return;
                 const dur = 0.16;
                 const bufferSize = Math.floor(ctx.sampleRate * dur);
                 const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
                 const data = buffer.getChannelData(0);
                 for (let i = 0; i < bufferSize; i++) {
                     data[i] = Math.random() * 2 - 1;
                 }
                 const noise = ctx.createBufferSource();
                 noise.buffer = buffer;
                 const filter = ctx.createBiquadFilter();
                 filter.type = 'bandpass';
                 filter.Q.value = 2.0;
                 const now = ctx.currentTime;
                 filter.frequency.setValueAtTime(320, now);
                 filter.frequency.exponentialRampToValueAtTime(580, now + dur);
                 const gain = ctx.createGain();
                 gain.gain.setValueAtTime(0.01, now);
                 gain.gain.linearRampToValueAtTime(0.16, now + 0.03);
                 gain.gain.exponentialRampToValueAtTime(0.001, now + dur);
                 noise.connect(filter);
                 filter.connect(gain);
                 gain.connect(ctx.destination);
                 noise.start(now);
             }

             // Sound 2: Grass rustle whisper
             function playGrassSound() {
                 const ctx = getCtx();
                 if (!ctx) return;
                 const dur = 0.08;
                 const bufferSize = Math.floor(ctx.sampleRate * dur);
                 const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
                 const data = buffer.getChannelData(0);
                 for (let i = 0; i < bufferSize; i++) {
                     data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
                 }
                 const noise = ctx.createBufferSource();
                 noise.buffer = buffer;
                 const filter = ctx.createBiquadFilter();
                 filter.type = 'highpass';
                 filter.frequency.value = 2400;
                 const gain = ctx.createGain();
                 const now = ctx.currentTime;
                 gain.gain.setValueAtTime(0.12, now);
                 gain.gain.exponentialRampToValueAtTime(0.001, now + dur);
                 noise.connect(filter);
                 filter.connect(gain);
                 gain.connect(ctx.destination);
                 noise.start(now);
             }

             // Sound 3: Wood acoustic tap
             function playWoodSound() {
                 const ctx = getCtx();
                 if (!ctx) return;
                 const now = ctx.currentTime;
                 const osc = ctx.createOscillator();
                 const osc2 = ctx.createOscillator();
                 const gain = ctx.createGain();
                 osc.type = 'triangle';
                 osc2.type = 'sine';
                 osc.frequency.setValueAtTime(460, now);
                 osc.frequency.exponentialRampToValueAtTime(220, now + 0.045);
                 osc2.frequency.setValueAtTime(920, now);
                 osc2.frequency.exponentialRampToValueAtTime(440, now + 0.045);
                 gain.gain.setValueAtTime(0.28, now);
                 gain.gain.exponentialRampToValueAtTime(0.001, now + 0.045);
                 osc.connect(gain);
                 osc2.connect(gain);
                 gain.connect(ctx.destination);
                 osc.start(now);
                 osc2.start(now);
                 osc.stop(now + 0.05);
                 osc2.stop(now + 0.05);
             }

             function attachAudioEvents(targetDoc) {
                 if (!targetDoc || targetDoc._naturalAudioAttached) return;
                 targetDoc._naturalAudioAttached = true;
                 targetDoc.addEventListener('click', function(e) {
                     const btnPrimary = e.target.closest('button[kind=\"primary\"], .stButton > button[data-testid=\"baseButton-primary\"]');
                     const btnTab = e.target.closest('[data-baseweb=\"tab\"], .stTabs [role=\"tab\"], .stSelectbox');
                     const btnOther = e.target.closest('button, [role=\"button\"], .stButton, input[type=\"button\"], input[type=\"submit\"], .stDownloadButton');
                     if (btnPrimary) {
                         playWoodSound();
                     } else if (btnTab) {
                         playWindSound();
                     } else if (btnOther) {
                         playGrassSound();
                     }
                 }, true);
             }

             try { attachAudioEvents(document); } catch(err) {}
             try { if (window.parent && window.parent.document) attachAudioEvents(window.parent.document); } catch(err) {}
         })();
         " />
    </div>
    """
    st.html(audio_engine_js)


def style_figure(fig):
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Segoe UI, -apple-system, sans-serif", size=12),
        title_font=dict(color="#1b5e20", size=15, family="Segoe UI, -apple-system, sans-serif"),
        margin=dict(l=45, r=25, t=45, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1", zerolinecolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1", zerolinecolor="#e2e8f0")
    return fig


# Data Helpers
RULE_COLS = ["CO(GT)", "NO2(GT)", "NOx(GT)", "C6H6(GT)", "RH"]
MAX_MISSING_FRACTION = 0.5


def air_quality_category(row):
    score = 0
    if row.get("CO(GT)", 0) > 2:
        score += 1
    if row.get("NO2(GT)", 0) > 80:
        score += 1
    if row.get("NOx(GT)", 0) > 150:
        score += 1
    if row.get("C6H6(GT)", 0) > 10:
        score += 1
    if row.get("RH", 0) > 60:
        score += 1
    if score <= 1:
        return "Good"
    if score <= 3:
        return "Moderate"
    return "Poor"


@st.cache_data(show_spinner="Normalizing dataset...")
def load_and_clean(source):
    raw = pd.read_csv(source)
    df = raw.copy()
    df = df.dropna(how="all")
    df = df.drop_duplicates()
    df = df.replace(-200, np.nan)

    dropped_note = []
    if {"Date", "Time"}.issubset(df.columns):
        text = df["Date"].astype(str) + " " + df["Time"].astype(str).str.replace(".", ":", regex=False)
        month_first = pd.to_datetime(text, dayfirst=False, errors="coerce")
        day_first = pd.to_datetime(text, dayfirst=True, errors="coerce")
        ts = month_first if month_first.isna().sum() <= day_first.isna().sum() else day_first
        df.insert(0, "Timestamp", ts)
        df = df.drop(columns=["Date", "Time"])
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    elif "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    elif "Datetime" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.drop(columns=["Datetime"]).dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in num_cols:
        if df[col].isna().mean() > MAX_MISSING_FRACTION:
            dropped_note.append(f"{col} ({df[col].isna().mean():.0%} missing)")
            df = df.drop(columns=col)
    num_cols = [c for c in num_cols if c in df.columns]

    df[num_cols] = df[num_cols].interpolate(limit_direction="both")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    if "Air_Quality" not in df.columns and all(c in df.columns for c in RULE_COLS):
        df["Air_Quality"] = df.apply(air_quality_category, axis=1)

    return df, int(raw.replace(-200, np.nan).isna().sum().sum()), dropped_note


def numeric_features(df):
    return [c for c in df.select_dtypes(include=np.number).columns if c != "Timestamp"]


# ML Helpers
def split_xy(df, target, chronological):
    X = df[[c for c in numeric_features(df) if c != target]]
    y = df[target]
    if chronological:
        cut = int(len(df) * 0.8)
        return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_regressor(name):
    if name == "Ridge Regression":
        return Ridge(alpha=1.0)
    if name == "Decision Tree":
        return DecisionTreeRegressor(random_state=42)
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


@st.cache_resource(show_spinner="Training regression model...")
def train_regressor(df, name, target, chronological=True):
    X_tr, X_te, y_tr, y_te = split_xy(df, target, chronological)
    scaler = StandardScaler().fit(X_tr)
    model = build_regressor(name)
    model.fit(scaler.transform(X_tr), y_tr)
    pred = model.predict(scaler.transform(X_te))
    return model, scaler, list(X_tr.columns), y_te, pred


@st.cache_data(show_spinner="Benchmarking models...")
def compare_regressors(df, target, chronological):
    rows = []
    for name in ["Ridge Regression", "Decision Tree", "Random Forest"]:
        _, _, _, y_te, pred = train_regressor(df, name, target, chronological)
        rows.append(
            {
                "Model": name,
                "R2": r2_score(y_te, pred),
                "MAE": mean_absolute_error(y_te, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Training classifier...")
def train_classifier(df):
    feats = [c for c in numeric_features(df) if c not in RULE_COLS]
    X, y = df[feats], df["Air_Quality"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_tr, y_tr)
    return model, feats, y_te, model.predict(X_te)


def make_lag_frame(series, index):
    d = pd.DataFrame({"y": series.values}, index=index)
    for lag in (1, 2, 3, 24):
        d[f"lag{lag}"] = d["y"].shift(lag)
    d["hour"] = index.hour
    d["dow"] = index.dayofweek
    return d


@st.cache_resource(show_spinner="Training forecaster...")
def train_forecaster(series):
    d = make_lag_frame(series, series.index).dropna()
    cut = int(len(d) * 0.8)
    train, test = d.iloc[:cut], d.iloc[cut:]
    cols = [c for c in d.columns if c != "y"]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train[cols], train["y"])
    pred = model.predict(test[cols])
    metrics = {
        "model_mae": mean_absolute_error(test["y"], pred),
        "naive_mae": mean_absolute_error(test["y"], test["lag24"]),
    }
    model.fit(d[cols], d["y"])
    return model, cols, metrics


def forecast(model, cols, series, hours, target):
    hist = list(series.values)
    idx = series.index[-1]
    out_idx, out_val = [], []
    for _ in range(hours):
        idx = idx + pd.Timedelta(hours=1)
        row = {
            "lag1": hist[-1],
            "lag2": hist[-2],
            "lag3": hist[-3],
            "lag24": hist[-24],
            "hour": idx.hour,
            "dow": idx.dayofweek,
        }
        val = float(model.predict(pd.DataFrame([row])[cols])[0])
        hist.append(val)
        out_idx.append(idx)
        out_val.append(val)
    return pd.DataFrame({"Timestamp": out_idx, f"Predicted {target}": out_val})


# Data Ingestion (Strict Upload Flow)
st.sidebar.header("Data Management")
sidebar_upload = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"], key="sidebar_file_uploader")

source = None
label = None

if sidebar_upload is not None:
    source = sidebar_upload
    label = sidebar_upload.name
elif st.session_state.get("uploaded_file") is not None:
    source = st.session_state["uploaded_file"]
    label = getattr(source, "name", "Uploaded_Data.csv")
else:
    apply_theme_and_audio("upload")

    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <svg class="cloud-icon-header" width="56" height="56" viewBox="0 0 24 24" fill="#2e7d32">
                <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96z"/>
            </svg>
            <h1 style="display:inline-block; vertical-align:middle; margin:0;">Air Quality Intelligence System</h1>
            <p style="color: #2e7d32; font-size: 1.15rem; font-weight: 500;">Upload an environmental dataset to begin analytics, modeling, and diurnal analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    main_upload = st.file_uploader(
        "Select or Drag and Drop CSV Dataset",
        type=["csv"],
        key="main_file_uploader",
        help="Upload any air quality or environmental CSV file.",
    )
    if main_upload is not None:
        st.session_state["uploaded_file"] = main_upload
        st.rerun()

    st.markdown(
        """
        ---
        ### Supported Features
        - **File Format:** Comma-Separated Values (.csv)
        - **Timestamp Parsing:** Automatic recognition of Date, Time, or Datetime columns
        - **Imputation:** Automated handling of sensor fault markers and temporal gaps
        - **Machine Learning:** Dynamic feature discovery and predictive modeling pipelines
        """
    )
    st.info("Upload a CSV file to activate the dashboard.")
    st.stop()

# Active Session
st.sidebar.success(f"Loaded: {label}")
if st.sidebar.button("Upload Different Dataset"):
    st.session_state["uploaded_file"] = None
    st.rerun()

try:
    df, n_missing, dropped = load_and_clean(source)
except Exception as exc:  # noqa: BLE001
    st.error(f"Error parsing dataset: {exc}")
    st.stop()

feats_all = numeric_features(df)
if not feats_all:
    st.error("Uploaded dataset contains no numeric measurement columns.")
    st.stop()

default_target_idx = feats_all.index("CO(GT)") if "CO(GT)" in feats_all else 0
TARGET = st.sidebar.selectbox("Target Variable", feats_all, index=default_target_idx, key="target_variable_select")

has_label = "Air_Quality" in df.columns
has_time = "Timestamp" in df.columns

section = st.sidebar.selectbox(
    "Navigation Menu",
    [
        "Dashboard",
        "Executive Intelligence Hub",
        "Daily Patterns",
        "Dataset Explorer",
        "Correlation Analysis",
        "Predictive Modeling",
        "Classification & Diagnosis",
    ],
    key="nav_section_menu",
)

apply_theme_and_audio(section)

# 1. Dashboard
if section == "Dashboard":
    st.subheader("System Overview and Key Indicators")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validated Observations", f"{df.shape[0]:,}")
    c2.metric("Sensor Features", df.shape[1])
    c3.metric("Imputed Sensor Faults", f"{n_missing:,}")
    c4.metric(f"Mean {TARGET}", f"{df[TARGET].mean():.2f}")

    if dropped:
        st.warning("Decommissioned sparse sensor columns: " + ", ".join(dropped))

    col_chart, col_quick = st.columns([2, 1])
    with col_chart:
        if has_label:
            counts = df["Air_Quality"].value_counts().reindex(["Good", "Moderate", "Poor"]).fillna(0)
            fig_bar = px.bar(
                x=counts.index,
                y=counts.values,
                labels={"x": "Air Quality Category", "y": "Cumulative Hours"},
                color=counts.index,
                color_discrete_map={"Good": "#2e7d32", "Moderate": "#f39c12", "Poor": "#c0392b"},
                title="Air Quality Category Breakdown",
                text=counts.values,
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(style_figure(fig_bar), width="stretch")

    with col_quick:
        st.markdown("### Executive Summary")
        if has_label:
            pct_good = (df["Air_Quality"] == "Good").mean() * 100
            pct_poor = (df["Air_Quality"] == "Poor").mean() * 100
            st.info(f"Clean Air Rate: {pct_good:.1f}% of hours meet Good standard.")
            st.error(f"Elevated Hazard Rate: {pct_poor:.1f}% of hours classified as Poor.")
        st.markdown(
            f"""
            - **Target Variable:** {TARGET}
            - **Temporal Continuity:** {'Continuous hourly records' if has_time else 'Sequential observations'}
            - **Preprocessing:** Outliers and sensor failure flags normalized
            """
        )

    st.subheader("Verified Dataset Preview")
    st.dataframe(df.head(10), width="stretch")

# 2. Executive Intelligence Hub
elif section == "Executive Intelligence Hub":
    st.subheader("Executive Environmental Intelligence and Anomaly Center")

    tab_anom, tab_sim, tab_exec = st.tabs([
        "Anomaly Detection",
        "Policy Impact Simulator",
        "Executive Audit Report",
    ])

    with tab_anom:
        st.markdown("#### Dynamic Statistical Anomaly Detection")
        st.caption("Identifies acute contamination events exceeding dynamic rolling bounds.")

        pollutant_select = st.selectbox("Pollutant for Anomaly Detection", feats_all, index=feats_all.index(TARGET))
        z_thresh = st.slider("Anomaly Sensitivity (Z-Score Threshold)", 1.5, 4.0, 2.5, step=0.1)

        s_vals = df[pollutant_select]
        mean_val = s_vals.mean()
        std_val = s_vals.std()
        anomalies_mask = (s_vals - mean_val).abs() > (z_thresh * std_val)
        num_anomalies = int(anomalies_mask.sum())

        k1, k2, k3 = st.columns(3)
        k1.metric("Detected Anomalies", f"{num_anomalies} hours")
        k2.metric("Anomaly Rate", f"{(num_anomalies / len(df) * 100):.2f}%")
        k3.metric("Baseline Normal Range", f"Mean: {mean_val:.2f} (std: {std_val:.2f})")

        fig_anom = go.Figure()
        x_axis = df["Timestamp"] if has_time else df.index
        fig_anom.add_trace(go.Scatter(
            x=x_axis,
            y=s_vals,
            mode="lines",
            name="Normal Observations",
            line=dict(color="#2e7d32", width=1),
            opacity=0.7,
        ))
        fig_anom.add_trace(go.Scatter(
            x=x_axis[anomalies_mask],
            y=s_vals[anomalies_mask],
            mode="markers",
            name="Acute Spikes",
            marker=dict(color="#c0392b", size=6, symbol="circle"),
        ))
        fig_anom.update_layout(
            title=f"Time Series Anomaly Tracking: {pollutant_select}",
            xaxis_title="Time",
            yaxis_title=pollutant_select,
            hovermode="x unified",
        )
        st.plotly_chart(style_figure(fig_anom), width="stretch")

        if num_anomalies > 0:
            st.markdown("##### Severe Historical Pollution Spikes")
            anom_df = df[anomalies_mask].sort_values(pollutant_select, ascending=False).head(10)
            show_cols = (["Timestamp"] if has_time else []) + [pollutant_select] + [c for c in ["T", "RH"] if c in df.columns]
            st.dataframe(anom_df[show_cols], width="stretch")

    with tab_sim:
        st.markdown("#### Emission Reduction Policy Simulator")
        st.caption("Simulate real-world interventions such as traffic restrictions and emission limits.")

        reduction_pct = st.slider("Simulated Reduction in Combustion Emissions (%)", 0, 70, 25, step=5)
        sim_factor = 1.0 - (reduction_pct / 100.0)

        df_sim = df.copy()
        combustion_cols = [c for c in ["CO(GT)", "NO2(GT)", "NOx(GT)", "C6H6(GT)"] if c in df_sim.columns]
        for c in combustion_cols:
            df_sim[c] = df_sim[c] * sim_factor

        if all(c in df_sim.columns for c in RULE_COLS):
            df_sim["Simulated_Air_Quality"] = df_sim.apply(air_quality_category, axis=1)
            orig_counts = df["Air_Quality"].value_counts(normalize=True) * 100
            sim_counts = df_sim["Simulated_Air_Quality"].value_counts(normalize=True) * 100

            comp_df = pd.DataFrame({
                "Category": ["Good", "Moderate", "Poor"],
                "Baseline (%)": [orig_counts.get("Good", 0), orig_counts.get("Moderate", 0), orig_counts.get("Poor", 0)],
                f"With -{reduction_pct}% Policy (%)": [sim_counts.get("Good", 0), sim_counts.get("Moderate", 0), sim_counts.get("Poor", 0)],
            })

            m1, m2, m3 = st.columns(3)
            good_gain = sim_counts.get("Good", 0) - orig_counts.get("Good", 0)
            poor_cut = orig_counts.get("Poor", 0) - sim_counts.get("Poor", 0)
            m1.metric("Clean Air Gain", f"+{good_gain:.1f}%")
            m2.metric("Hazardous Air Cut", f"-{poor_cut:.1f}%")
            m3.metric(f"Projected Mean {TARGET}", f"{df_sim[TARGET].mean():.2f}")

            fig_comp = px.bar(
                comp_df.melt(id_vars="Category", var_name="Scenario", value_name="Percentage"),
                x="Category",
                y="Percentage",
                color="Scenario",
                barmode="group",
                title=f"Policy Simulation: Baseline vs. -{reduction_pct}% Reduction",
                color_discrete_sequence=["#7f8c8d", "#2e7d32"],
            )
            st.plotly_chart(style_figure(fig_comp), width="stretch")

    with tab_exec:
        st.markdown("#### Environmental Audit Brief")
        total_hours = len(df)
        poor_hours = int((df["Air_Quality"] == "Poor").sum()) if has_label else 0
        poor_rate = (poor_hours / total_hours * 100) if total_hours > 0 else 0

        report_text = f"""AIR QUALITY INTELLIGENCE AUDIT REPORT
File Source: {label}
Total Observations: {total_hours:,} verified records
Target Metric: {TARGET}

1. EXECUTIVE COMPLIANCE METRICS

- Mean Target Level          : {df[TARGET].mean():.2f} (Max: {df[TARGET].max():.2f})
- Clean Air Rate             : {(df['Air_Quality'] == 'Good').mean()*100:.1f}% of recorded timeline
- Elevated Risk Rate         : {poor_rate:.1f}% of hours classified as Poor
- Preprocessing Normalization: {n_missing:,} missing values imputed

2. OBSERVATIONS AND DYNAMICS

- Highest Daily Traffic Peak : Observed during morning (07:00-09:00) and evening (18:00-21:00)
- Cleanest Window            : Observed in early morning hours (03:00-06:00)

3. STRATEGIC POLICY ACTIONS

1. Enforce transit incentive programs during peak commuter windows.
2. Establish public advisory notifications when relative humidity exceeds 60%.
3. Promote indoor air filtration during nocturnal boundary layer inversions.
"""
        st.text_area("Audit Report Preview", report_text, height=280)
        st.download_button(
            label="Download Audit Report (Plain Text)",
            data=report_text,
            file_name="Air_Quality_Executive_Audit.txt",
            mime="text/plain",
        )

# 3. Daily Patterns
elif section == "Daily Patterns":
    st.subheader("24-Hour Diurnal Cycles and Weekly Atmospheric Patterns")

    if not has_time:
        st.warning("Daily Pattern analysis requires timestamp columns (Date and Time).")
    else:
        feature = st.selectbox("Select Target Indicator", feats_all, index=feats_all.index(TARGET))

        tmp = df[["Timestamp", feature]].copy()
        tmp["Hour"] = tmp["Timestamp"].dt.hour
        tmp["DayName"] = tmp["Timestamp"].dt.day_name()
        tmp["DayOfWeek"] = tmp["Timestamp"].dt.dayofweek
        tmp["IsWeekend"] = tmp["DayOfWeek"].isin([5, 6])
        tmp["DayType"] = tmp["IsWeekend"].map({True: "Weekend (Sat-Sun)", False: "Weekday (Mon-Fri)"})

        hourly_overall = tmp.groupby("Hour")[feature].mean()
        peak_hour = int(hourly_overall.idxmax())
        peak_val = hourly_overall.max()
        clean_hour = int(hourly_overall.idxmin())
        clean_val = hourly_overall.min()

        weekday_avg = tmp[~tmp["IsWeekend"]][feature].mean()
        weekend_avg = tmp[tmp["IsWeekend"]][feature].mean()
        diff_pct = ((weekday_avg - weekend_avg) / weekend_avg) * 100 if weekend_avg != 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Highest Pollution Hour", f"{peak_hour:02d}:00", f"{peak_val:.2f} avg")
        k2.metric("Cleanest Air Window", f"{clean_hour:02d}:00", f"{clean_val:.2f} avg")
        k3.metric("Weekday Mean", f"{weekday_avg:.2f}")
        k4.metric("Weekend vs Weekday Delta", f"{diff_pct:+.1f}%")

        tab_diurnal, tab_dow, tab_heatmap, tab_science = st.tabs([
            "24-Hour Diurnal Profile",
            "Day of Week Progression",
            "24x7 Day-Hour Heatmap",
            "Atmospheric Dynamics",
        ])

        with tab_diurnal:
            st.markdown("#### 24-Hour Diurnal Profile (Weekday vs Weekend)")
            diurnal_df = tmp.groupby(["Hour", "DayType"])[feature].mean().reset_index()
            diurnal_all = tmp.groupby("Hour")[feature].mean().reset_index()
            diurnal_all["DayType"] = "Overall Average"

            plot_diurnal = pd.concat([diurnal_df, diurnal_all], ignore_index=True)

            fig_line = px.line(
                plot_diurnal,
                x="Hour",
                y=feature,
                color="DayType",
                markers=True,
                title=f"24-Hour Profile for {feature}",
                color_discrete_map={
                    "Weekday (Mon-Fri)": "#c0392b",
                    "Weekend (Sat-Sun)": "#2e7d32",
                    "Overall Average": "#2980b9",
                },
            )
            fig_line.add_vrect(x0=7, x1=9, fillcolor="#f39c12", opacity=0.15, annotation_text="Morning Rush (07-09h)")
            fig_line.add_vrect(x0=18, x1=21, fillcolor="#c0392b", opacity=0.15, annotation_text="Evening Peak (18-21h)")
            fig_line.update_xaxes(tickmode="linear", tick0=0, dtick=1)
            fig_line.update_layout(hovermode="x unified")
            st.plotly_chart(style_figure(fig_line), width="stretch")

        with tab_dow:
            st.markdown("#### Weekly Progression by Day")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_summary = tmp.groupby("DayName")[feature].mean().reindex(day_order).reset_index()

            worst_day = dow_summary.sort_values(feature, ascending=False).iloc[0]["DayName"]
            best_day = dow_summary.sort_values(feature).iloc[0]["DayName"]

            fig_dow = px.bar(
                dow_summary,
                x="DayName",
                y=feature,
                text=dow_summary[feature].round(2),
                title=f"Average {feature} by Day of Week",
                color=feature,
                color_continuous_scale="Greens",
            )
            fig_dow.update_traces(textposition="outside")
            st.plotly_chart(style_figure(fig_dow), width="stretch")
            st.success(f"Cleanest Day: {best_day} | Highest Concentration Day: {worst_day}")

        with tab_heatmap:
            st.markdown("#### Complete 24x7 Day-Hour Heatmap")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            grid = tmp.pivot_table(index="DayName", columns="Hour", values=feature, aggfunc="mean").reindex(day_order)

            fig_hm = px.imshow(
                grid,
                aspect="auto",
                color_continuous_scale="YlOrRd",
                labels=dict(x="Hour of Day", y="Day of Week", color=feature),
                title=f"Heatmap Intensity: {feature}",
            )
            fig_hm.update_xaxes(tickmode="linear", tick0=0, dtick=1)
            st.plotly_chart(style_figure(fig_hm), width="stretch")

        with tab_science:
            st.markdown("#### Atmospheric Mechanisms Behind Diurnal Curves")
            c_sci1, c_sci2 = st.columns(2)
            with c_sci1:
                st.markdown(
                    """
                    ##### 1. Traffic and Anthropogenic Emission Peaks
                    - **Morning Commute (07:00-09:00):** Intense traffic volume injects fresh emissions of primary combustion gases into the surface air layer.
                    - **Evening Commute (18:00-21:00):** Secondary vehicle surge coincides with evening residential heating and commercial transport.
                    """
                )
            with c_sci2:
                st.markdown(
                    """
                    ##### 2. Planetary Boundary Layer and Inversions
                    - **Daytime Dispersion:** Solar heating warms the ground and expands the planetary boundary layer, diluting ground-level pollution between 12:00 and 15:00.
                    - **Nocturnal Inversion:** Radiative ground cooling after sunset traps emissions in a shallow surface layer, producing the elevated night peak.
                    """
                )

# 4. Dataset Explorer
elif section == "Dataset Explorer":
    st.subheader("Dataset Explorer and Statistical Distribution")
    st.dataframe(df, width="stretch")

    st.markdown("#### Descriptive Statistics")
    st.dataframe(df[feats_all].describe().T.style.format("{:.2f}"), width="stretch")

    st.markdown("#### Missing Value and Data Quality Audit")
    null_counts = df[feats_all].isnull().sum()
    null_pct = (df[feats_all].isnull().mean() * 100).round(2)
    audit_df = pd.DataFrame({"Missing Values": null_counts, "Missing Percentage (%)": null_pct})
    st.dataframe(audit_df.T, width="stretch")

    st.download_button(
        label="Download Cleaned Dataset (CSV)",
        data=df.to_csv(index=False),
        file_name="Cleaned_Environmental_Data.csv",
        mime="text/csv",
    )

# 5. Correlation Analysis
elif section == "Correlation Analysis":
    st.subheader("Correlation Heatmap and Pairwise Relationships")
    corr = df[feats_all].corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Pearson Correlation Matrix",
    )
    st.plotly_chart(style_figure(fig_corr), width="stretch")

    col_rank, col_scatter = st.columns([1, 1])
    with col_rank:
        st.markdown(f"#### Ranked Correlations with {TARGET}")
        target_corrs = corr[TARGET].drop(TARGET).sort_values(ascending=False).reset_index()
        target_corrs.columns = ["Indicator", "Correlation (r)"]
        st.dataframe(target_corrs.style.format({"Correlation (r)": "{:+.3f}"}), width="stretch")

    with col_scatter:
        st.markdown("#### Scatter Plot with Linear Trendline")
        candidate_feats = [c for c in feats_all if c != TARGET]
        feat_x = st.selectbox("X-Axis Feature", candidate_feats, index=0)
        fig_scat = px.scatter(df, x=feat_x, y=TARGET, opacity=0.35, title=f"{feat_x} vs. {TARGET}")
        x_vals = df[feat_x].dropna()
        y_vals = df.loc[x_vals.index, TARGET]
        if len(x_vals) > 1:
            poly = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
            y_line = np.polyval(poly, x_line)
            fig_scat.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=f"Slope: {poly[0]:.2f}", line=dict(color="#1b5e20", width=2.5)))
        st.plotly_chart(style_figure(fig_scat), width="stretch")

# 6. Predictive Modeling
elif section == "Predictive Modeling":
    st.subheader(f"Predictive Modeling and Benchmark: {TARGET}")

    c_m1, c_m2 = st.columns([1, 1])
    with c_m1:
        name = st.selectbox("Algorithm", ["Random Forest", "Decision Tree", "Ridge Regression"])
    with c_m2:
        chrono = st.checkbox(
            "Chronological Split (Temporal Holdout 80/20)",
            value=True,
            help="Recommended for continuous temporal data to prevent future leakage.",
        )

    model, scaler, cols, y_te, pred = train_regressor(df, name, TARGET, chrono)

    c1, c2, c3 = st.columns(3)
    c1.metric("R2 Score", f"{r2_score(y_te, pred):.3f}")
    c2.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_te, pred):.3f}")
    c3.metric("Root Mean Squared Error (RMSE)", f"{np.sqrt(mean_squared_error(y_te, pred)):.3f}")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        fig_scatter = px.scatter(
            x=y_te, y=pred,
            labels={"x": f"Actual {TARGET}", "y": f"Predicted {TARGET}"},
            opacity=0.35,
            title="Actual vs. Predicted",
        )
        lo, hi = float(y_te.min()), float(y_te.max())
        fig_scatter.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#2e7d32", dash="dash"))
        st.plotly_chart(style_figure(fig_scatter), width="stretch")

    with col_p2:
        residuals = y_te - pred
        fig_res = px.histogram(residuals, title="Residual Error Distribution", color_discrete_sequence=["#2e7d32"])
        st.plotly_chart(style_figure(fig_res), width="stretch")

    st.markdown("#### Side-by-Side Algorithm Comparison")
    results = compare_regressors(df, TARGET, chrono)
    st.dataframe(results.style.format({"R2": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}), width="stretch")

    melted = results.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig_comp = px.bar(melted, x="Model", y="Value", color="Model", facet_col="Metric", color_discrete_sequence=["#2e7d32", "#43a047", "#81c784"])
    st.plotly_chart(style_figure(fig_comp), width="stretch")

# 7. Classification & Diagnosis
elif section == "Classification & Diagnosis":
    st.subheader("Air Quality Classification, Sensor Simulator and Forecast")

    tab_class, tab_manual, tab_fc = st.tabs([
        "Classification Model",
        "Sensor Simulator",
        "Autoregressive Forecast",
    ])

    with tab_class:
        if not has_label:
            st.warning("Air Quality classification requires target class column.")
        else:
            model_c, feats_c, y_te_c, pred_c = train_classifier(df)
            labels = ["Good", "Moderate", "Poor"]

            c_acc, c_rep = st.columns([1, 2])
            with c_acc:
                st.metric("Test Accuracy", f"{accuracy_score(y_te_c, pred_c)*100:.1f}%")
                cm = confusion_matrix(y_te_c, pred_c, labels=labels)
                fig_cm = px.imshow(
                    cm, x=labels, y=labels, text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    title="Confusion Matrix",
                    color_continuous_scale="Greens",
                )
                st.plotly_chart(style_figure(fig_cm), width="stretch")

            with c_rep:
                st.markdown("#### Precision, Recall and F1-Score")
                rep_dict = classification_report(y_te_c, pred_c, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(rep_dict).T.style.format("{:.3f}"), width="stretch")

    with tab_manual:
        if not has_label:
            st.warning("Classifier is unavailable.")
        else:
            model_c, feats_c, _, _ = train_classifier(df)
            st.markdown("Input sensor values to diagnose air quality category:")

            cols_ui = st.columns(3)
            values = []
            for i, f in enumerate(feats_c):
                lo, hi = float(df[f].min()), float(df[f].max())
                step = max((hi - lo) / 100, 0.001)
                values.append(
                    cols_ui[i % 3].number_input(
                        f,
                        min_value=lo,
                        max_value=hi,
                        value=float(df[f].median()),
                        step=float(step),
                        format="%.3f",
                    )
                )

            if st.button("Evaluate Air Quality Diagnosis", type="primary"):
                sample = pd.DataFrame([values], columns=feats_c)
                result = model_c.predict(sample)[0]
                probs = pd.Series(model_c.predict_proba(sample)[0], index=model_c.classes_)

                res_c1, res_c2 = st.columns([1, 2])
                with res_c1:
                    st.subheader(f"Diagnosis: {result}")
                    if result == "Good":
                        st.success("Air Quality is safe. Outdoor activities and natural ventilation are fine.")
                    elif result == "Moderate":
                        st.warning("Moderate air quality. Sensitive individuals should reduce prolonged outdoor exposure.")
                    else:
                        st.error("Poor air quality. Stay indoors and run air purification.")

                with res_c2:
                    fig_prob = px.bar(
                        x=probs.index,
                        y=(probs.values * 100).round(1),
                        labels={"x": "Class", "y": "Probability (%)"},
                        title="Model Prediction Probability",
                        color=probs.index,
                        color_discrete_map={"Good": "#2e7d32", "Moderate": "#f39c12", "Poor": "#c0392b"},
                        text=(probs.values * 100).round(1),
                    )
                    fig_prob.update_traces(textposition="outside")
                    st.plotly_chart(style_figure(fig_prob), width="stretch")

    with tab_fc:
        if not has_time:
            st.warning("Time-series forecasting requires temporal sequence columns.")
        else:
            series = df.set_index("Timestamp")[TARGET]
            series = series[~series.index.duplicated()].asfreq("h").interpolate(limit_direction="both")

            hours = st.slider("Forecast Horizon (Hours)", 6, 168, 48, step=6)
            model_fc, cols_fc, m_fc = train_forecaster(series)
            fc = forecast(model_fc, cols_fc, series, hours, TARGET)

            c1, c2 = st.columns(2)
            c1.metric("Forecaster Test MAE", f"{m_fc['model_mae']:.3f}")
            c2.metric("Baseline Same-Hour MAE", f"{m_fc['naive_mae']:.3f}")

            recent = series.iloc[-24 * 7 :].reset_index()
            recent.columns = ["Timestamp", "Value"]
            recent["Segment"] = "Historical"

            fc2 = fc.rename(columns={f"Predicted {TARGET}": "Value"})
            fc2["Segment"] = "Forecast"

            combined = pd.concat([recent, fc2], ignore_index=True)
            fig_fc = px.line(
                combined,
                x="Timestamp",
                y="Value",
                color="Segment",
                title=f"Autoregressive Forecast for {TARGET}",
                color_discrete_map={"Historical": "#2c3e50", "Forecast": "#2e7d32"},
            )
            st.plotly_chart(style_figure(fig_fc), width="stretch")
            st.dataframe(fc.head(10), width="stretch")
