import streamlit as st
import ssl
import certifi
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
import easyocr
import cv2
import re
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from PIL import Image

# ================= LOAD MODEL =================
MODEL_PATH = "roberta-base"

@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ================= EASY OCR =================
@st.cache_resource
def load_ocr():
    reader = easyocr.Reader(['en'], gpu=False)
    return reader

reader = load_ocr()

def extract_text_from_image(image):
    image_np = np.array(image)
    results = reader.readtext(image_np, detail=0)
    text = " ".join(results)

    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ================= PREDICTION =================
def predict_from_text(text):
    if len(text.split()) < 30:
        return "⚠️ Insufficient job description (Poster-style content)", 0

    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        output = model(**enc)
        probs = torch.softmax(output.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    strong_scam_keywords = [
        "registration fee required",
        "processing fee required",
        "security deposit required",
        "pay before joining",
        "investment required",
        "contact on whatsapp only",
        "telegram for application"
    ]

    text_lower = text.lower()
    rule_triggered = any(phrase in text_lower for phrase in strong_scam_keywords)

    if rule_triggered:
        return "🚨 Fake Job Post ", confidence
    else:
        label = "🚨 Fake Job Post" if pred == 1 else "✅ Real Job Post"
        return label, confidence

# ================= STREAMLIT UI =================
st.title("🛡️ Job Post Fraud Detection System")
st.write("Check job descriptions or posters using AI + OCR")

# -------- TEXT INPUT --------
st.subheader("✍️ Enter Job Description")
user_text = st.text_area("Paste job description")

# -------- IMAGE UPLOAD --------
st.subheader("📄 Upload Job Poster")
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

ocr_text = ""
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    ocr_text = extract_text_from_image(image)
    st.write("### 📄 Extracted Text")
    st.write(ocr_text)

# -------- PREDICT --------
input_text = user_text if user_text.strip() != "" else ocr_text

if st.button("🔍 Predict"):
    if input_text.strip() == "":
        st.warning("Enter description or upload image")
    else:
        result, conf = predict_from_text(input_text)
        st.subheader(result)
        if conf != 0:

            st.write(f"Confidence: {conf:.2f}")


