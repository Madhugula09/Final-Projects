import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
import random

# ----------------------------
# Setup and Page Config
# ----------------------------
st.set_page_config(page_title="🩺 Pneumonia Detection", layout="wide")
st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.markdown("""
Upload a chest X-ray image to detect whether it shows signs of **Pneumonia** or is **Normal**.
""")

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# Load Model
# ----------------------------
class_names = ['NORMAL', 'PNEUMONIA']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Image Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------------
# Upload and Predict
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item() * 100

        st.markdown(f"""
        ### 🧠 Prediction:
        - **Class**: `{predicted_class}`
        - **Confidence**: **{confidence_score:.2f}%**
        """)

        if predicted_class == "PNEUMONIA":
            st.error("⚠️ Pneumonia detected. Please consult a medical professional.")
        else:
            st.success("✅ Lungs appear normal. No pneumonia signs detected.")

        # Class Probabilities Bar Chart
        probs_np = probs.cpu().numpy().flatten()
        st.markdown("### 🔎 Class Probabilities")
        st.bar_chart({class_names[i]: probs_np[i] * 100 for i in range(len(class_names))})

        # Optional: Log prediction
        log_df = pd.DataFrame({
            "filename": [uploaded_file.name],
            "prediction": [predicted_class],
            "confidence": [round(confidence_score, 2)]
        })
        if os.path.exists("pneumonia_predictions.csv"):
            log_df.to_csv("pneumonia_predictions.csv", mode="a", header=False, index=False)
        else:
            log_df.to_csv("pneumonia_predictions.csv", index=False)

    except Exception as e:
        st.error(f"Image Error: {str(e)}")

# ----------------------------
# Sidebar: Evaluation Metrics
# ----------------------------
st.sidebar.markdown("### 📊 Evaluation Metrics")
if st.sidebar.button("Show Evaluation"):
    try:
        test_dir = "C:/Users/padmavathi/solar_demo/chest_xray_extracted/chest_xray/test"
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        acc = accuracy_score(y_true, y_pred)

        st.markdown("### 📈 Evaluation Summary")
        st.write(f"**Accuracy**: {acc * 100:.2f}%")
        st.write(f"**Precision (Macro Avg)**: {report['macro avg']['precision']*100:.2f}%")
        st.write(f"**Recall (Macro Avg)**: {report['macro avg']['recall']*100:.2f}%")
        st.write(f"**F1 Score (Macro Avg)**: {report['macro avg']['f1-score']*100:.2f}%")

        # Confusion Matrix
        st.markdown("### 📉 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Evaluation Error: {str(e)}")

# ----------------------------
# Sidebar: Training Performance
# ----------------------------
st.sidebar.markdown("### 📈 Training Curve")
if st.sidebar.button("Show Training Performance"):
    if os.path.exists("training_performance.png"):
        st.image("training_performance.png", caption="Training vs Validation Accuracy", use_container_width=True)
    else:
        st.warning("No training plot found.")

# ----------------------------
# Sidebar: Prediction History
# ----------------------------
st.sidebar.markdown("### 📜 Prediction Log")
if st.sidebar.button("View Prediction History"):
    if os.path.exists("pneumonia_predictions.csv"):
        df_hist = pd.read_csv("pneumonia_predictions.csv")
        st.dataframe(df_hist.tail(20))
        st.download_button("⬇️ Download CSV", df_hist.to_csv(index=False), file_name="pneumonia_predictions.csv")
    else:
        st.info("No history found.")

# ----------------------------
# Insights & Footer
# ----------------------------
with st.expander("💡 Model Insights"):
    st.markdown("""
    - ✅ **Model Accuracy**: ~86%
    - ✅ **High Recall for Pneumonia**: 99%
    - 📉 Performs well on unseen test images
    - ⚠️ No data augmentation or weighted loss applied (optional improvements)
    """)

with st.expander("📦 Environment Info"):
    st.markdown("""
    - Python 3.9+
    - PyTorch 2.0+
    - torchvision 0.15+
    - Streamlit 1.20+
    """)

st.markdown("""
---
📁 Trained on [Chest X-ray dataset from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
🧠 Model: ResNet18 | Test Accuracy: ~86%
""")
