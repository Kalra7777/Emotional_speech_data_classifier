import warnings
import tempfile

import streamlit as st
import librosa
import numpy as np
from xgboost import XGBClassifier

# ── Configuration ─────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")

@st.cache_resource
def load_classifier(filepath: str = "xgb_model.json") -> XGBClassifier:
    clf = XGBClassifier()
    clf.load_model(filepath)
    return clf

classifier = load_classifier()

EMO_TAGS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprise"
]

# ── Feature Extraction ────────────────────────────────────────────────────────
def compute_features(file_path: str,
                     mfcc_n: int = 40,
                     mel_n: int = 128) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=None)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=mfcc_n).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=mel_n).T, axis=0)

    return np.hstack((mfcc, chroma, mel))

# ── Prediction ───────────────────────────────────────────────────────────────
def detect_emotion(wav_path: str) -> str:
    feats = compute_features(wav_path).reshape(1, -1)
    idx = classifier.predict(feats)[0]
    return EMO_TAGS[idx]

# ── Streamlit App ────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Emotion Recognizer", layout="centered")
    st.title("🎙️ Audio Emotion Recognizer")
    st.write("Upload a WAV file, and the model will guess the emotion!")

    uploaded = st.file_uploader("Select a .wav file", type="wav")
    if not uploaded:
        return

    st.audio(uploaded, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    with st.spinner("Processing..."):
        try:
            emotion = detect_emotion(path)
            st.success(f"Detected Emotion: **{emotion.upper()}**")
        except Exception as e:
            st.error(f"Error analyzing file: {e}")

if __name__ == "__main__":
    main()
