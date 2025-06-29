# Emotional_speech_data_classifier
# 🎧 Speech Emotion Recognition with XGBoost

This project implements an end-to-end system to classify emotions from voice recordings using machine learning. A simple **Streamlit** interface lets users upload `.wav` files and receive real-time predictions from a trained **XGBoost** model.

---

## 🚀 Highlights

- 🎙️ Upload `.wav` audio clips directly via the app  
- 🧪 Automatically extracts features such as MFCC, Chroma, and Mel spectrograms  
- 🤖 Predicts emotional state using a pre-trained XGBoost classifier  
- 🖼️ Displays results in an interactive and responsive UI  
- ⚡ Fast performance — ideal for showcasing ML in action  

---

## 😃 Emotions Identified

The classifier is capable of detecting the following **eight emotional categories** from speech:

- 😠 **Angry**  
- 😌 **Calm**  
- 🤢 **Disgusted**  
- 😨 **Fearful**  
- 😄 **Happy**  
- 😐 **Neutral**  
- 😢 **Sad**  
- 😲 **Surprised**

---

## 🧠 Model Information

- **Algorithm Used:** Gradient Boosted Trees via `XGBClassifier`  
- **Input Features:**  
  - MFCC (Mel-Frequency Cepstral Coefficients)  
  - Chroma Features  
  - Mel Spectrograms  
- **Required Audio Format:** `.wav`, sampled at 16,000 Hz  
- **Exported Model File:** `xgb_model.json`

---
# 🔗 Live Demo

Try out the Emotion Classification app right in your browser:

(https://emotion-classification-kalra-23125014.streamlit.app/)

> 🎙️ Upload any WAV file and see real-time emotion detection powered by XGBoost!
