# nextword_app.py
# Day 5: Final Streamlit app for Next Word Prediction results & inference

import os
import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt

# -------------------------
# Paths
# -------------------------
BASE_DIR = r"C:/Users/padmavathi/solar_demo"
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "nextword_model.pth")
LOSS_CURVE_PATH = os.path.join(BASE_DIR, "loss_curve.png")

# -------------------------
# Ensure NLTK punkt
# -------------------------
NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
try:
    _ = word_tokenize("check punkt")
except Exception:
    nltk.download("punkt", download_dir=NLTK_DIR)
    nltk.data.path.append(NLTK_DIR)

# -------------------------
# Model (same as training)
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Next Word Prediction Results", layout="wide")
st.title("🔮 Next Word Prediction — Results & Inference ")

# Check files
if not (os.path.exists(VOCAB_PATH) and os.path.exists(MODEL_PATH)):
    st.error("❌ Model or vocabulary not found! Please run training first (Day 4).")
    st.stop()

# Load vocab
with open(VOCAB_PATH, "rb") as f:
    word2idx, idx2word = pickle.load(f)
vocab_size = len(word2idx)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

st.success("✅ Model and vocabulary loaded successfully!")

# -------------------------
# Inference function (FIXED)
# -------------------------
def generate_text(seed, num_words=20, temperature=1.0, sampling=True, seq_len=20):
    words = word_tokenize(seed.lower())
    for _ in range(num_words):
        input_seq = [word2idx.get(w, 1) for w in words[-seq_len:]]
        if len(input_seq) < seq_len:
            input_seq = [0] * (seq_len - len(input_seq)) + input_seq
        x = torch.tensor([input_seq], dtype=torch.long).to(device)
        with torch.no_grad():
            logits, _ = model(x)
        logits = logits.cpu().numpy().flatten()

        if not sampling:
            idx = int(np.argmax(logits))
        else:
            # --- FIX: make probabilities safe ---
            probs = np.exp(logits / temperature)
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() == 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()
            idx = np.random.choice(len(probs), p=probs)

        next_word = idx2word.get(idx, "<unk>")
        words.append(next_word)
    return " ".join(words)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("⚙️ Generation Settings")
seed_text = st.sidebar.text_input("Seed text", "the economy of india")
num_words = st.sidebar.slider("Words to generate", 5, 100, 20)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0)
sampling = st.sidebar.checkbox("Enable sampling (instead of greedy)", True)

# -------------------------
# Run generation
# -------------------------
if st.button("🚀 Generate Next Words"):
    with st.spinner("Generating..."):
        output = generate_text(seed_text, num_words, temperature, sampling)
    st.subheader("📝 Generated Text")
    st.write(output)

# -------------------------
# Show training loss curve
# -------------------------
if os.path.exists(LOSS_CURVE_PATH):
    st.subheader("📉 Training Loss Curve")
    st.image(LOSS_CURVE_PATH, use_column_width=True)
else:
    st.info("Loss curve not available. Train model with saving enabled.")

st.markdown("---")
st.caption("Streamlit app to showcase model results and next word predictions.")
