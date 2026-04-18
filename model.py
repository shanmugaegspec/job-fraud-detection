import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

MODEL_PATH = "./roberta_fraud_model"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

def predict_job(text):

    fraud_keywords = [
        "whatsapp","telegram","no experience",
        "instant joining","earn money",
        "work from home","daily payment",
        "urgent hiring"
    ]

    text_lower = text.lower()

    # 🔥 RULE SCORE
    rule_score = 0
    detected_keywords = []

    for word in fraud_keywords:
        if word in text_lower:
            rule_score += 10
            detected_keywords.append(word)

    # 🔥 AI MODEL
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    real_prob = float(probs[0][0]) * 100
    fraud_prob = float(probs[0][1]) * 100

    # 🔥 FINAL SINGLE SCORE (USE THIS EVERYWHERE)
    final_fraud_score = (fraud_prob * 0.8) + (rule_score * 0.2)

    final_real_score = 100 - final_fraud_score

    # 🔥 DECISION (SAME SCORE)
    if final_fraud_score >= 50:
        prediction = "⚠ Fraud Job Post"
    else:
        prediction = "✅ Real Job Post"

    # 🔥 EXPLAINABLE AI
    reasons = []

    if detected_keywords:
        reasons.append("Suspicious keywords: " + ", ".join(detected_keywords))

    if fraud_prob > 70:
        reasons.append("AI model confidence high for fraud")

    if rule_score > 20:
        reasons.append("Multiple fraud indicators detected")

    if not reasons:
        reasons.append("Looks like a genuine job post")

    return prediction, round(final_fraud_score,2), round(final_real_score,2), reasons