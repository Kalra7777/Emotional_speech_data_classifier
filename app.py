import warnings
import tempfile

import librosa
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ── Setup & Styling ────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")
st.set_page_config(
    page_title="✨ Emotion Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom CSS
st.markdown("""
    <style>
    body { background: #1f1f2e; color: #f0f0f5; }
    .stButton>button { background-color: #6c5ce7; color: white; }
    .stSidebar { background: #2d2d44; }
    </style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_model(path="xgb_model.json"):
    m = XGBClassifier()
    m.load_model(path)
    return m

model = get_model()

EMO_TAGS = [
    ("angry", "😡"),
    ("calm", "😌"),
    ("disgust", "🤢"),
    ("fearful", "😱"),
    ("happy", "😃"),
    ("neutral", "😐"),
    ("sad", "😢"),
    ("surprise", "😲"),
]

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_feats(fp, mfcc_n=40, mel_n=128):
    y, sr = librosa.load(fp, sr=None)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    mf = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_n).T, axis=0)
    ch = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    ml = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=mel_n).T, axis=0)
    return np.hstack([mf, ch, ml])

# ── App Layout ────────────────────────────────────────────────────────────────
st.title("🎙️ Emotion Explorer")
st.write("Upload a `.wav` below, visualize its waveform & spectrogram, then hit **Predict!**")

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
        This demo loads an XGBoost model trained on MFCC, Chroma & Mel features.
        1. Upload your audio  
        2. View waveform & spectrogram  
        3. Click **Predict** to see the emotion!
    """)
    st.write("---")
    st.write("Made with ❤️ by Ishit Kalra")

# Two columns: visuals | controls
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("📂 Select WAV file", type="wav")
    if uploaded:
        st.audio(uploaded, format="audio/wav")

        # save temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            path = tmp.name

        # plot waveform & spectrogram
        y, sr = librosa.load(path, sr=16000)
        fig, ax = plt.subplots(2, 1, figsize=(8, 4), tight_layout=True)
        ax[0].plot(np.linspace(0, len(y)/sr, len(y)), y)
        ax[0].set_title("Waveform"); ax[0].set_ylabel("Amplitude")
        spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
        img = ax[1].imshow(spec, origin="lower", aspect="auto")
        ax[1].set_title("Mel Spectrogram"); ax[1].set_ylabel("Mel bins"); ax[1].set_xlabel("Time")
        st.pyplot(fig)

with col2:
    if 'path' in locals():
        if st.button("🚀 Predict Emotion"):
            with st.spinner("Analyzing..."):
                try:
                    feats = extract_feats(path).reshape(1, -1)
                    idx = model.predict(feats)[0]
                    label, emoji = EMO_TAGS[idx]
                    st.metric("Detected Emotion", f"{label.upper()} {emoji}")
                except Exception as e:
                    st.error(f"Failed to predict: {e}")
