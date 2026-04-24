import os
import zipfile
import gdown
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

MODEL_PATH = "roberta_fraud_model"

def download_model():
    if not os.path.exists(MODEL_PATH):

        print("⬇️ Downloading model from Google Drive...")

        file_id = "1Z_CFE1farbm6p38MRCgeAki-pRFqribv"
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, "model.zip", quiet=False)

        print("📦 Extracting model...")
        with zipfile.ZipFile("model.zip", 'r') as zip_ref:
            zip_ref.extractall(".")

        print("✅ Model Ready!")

download_model()

# Load model
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

def predict_job(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    fraud_score = float(probs[0][1]) * 100
    real_score = float(probs[0][0]) * 100

    prediction = "🚨 Fraud Job Post" if fraud_score > real_score else "✅ Real Job Post"

    reasons = []
    if fraud_score > 70:
        reasons.append("High fraud probability detected")
    if "whatsapp" in text.lower():
        reasons.append("Contains WhatsApp contact (suspicious)")
    if "urgent hiring" in text.lower():
        reasons.append("Urgency keyword detected")

    return prediction, round(fraud_score,2), round(real_score,2), reasons