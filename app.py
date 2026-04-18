from flask import Flask, render_template, request, redirect, session
from model import predict_job
import easyocr
import numpy as np
from PIL import Image

app = Flask(__name__)
app.secret_key = "secret123"

# 🔥 OCR INIT
reader = easyocr.Reader(['en'], gpu=False)

fraud_keywords = [
    "work from home",
    "urgent hiring",
    "no experience",
    "huge salary",
    "instant payment",
    "limited offer",
    "earn money",
    "daily payment",
    "whatsapp",
    "telegram"
]

# ================= OCR =================
def extract_text(image):
    try:
        img = np.array(image)
        result = reader.readtext(img, detail=0)
        return " ".join(result)
    except:
        return ""

# ================= HIGHLIGHT =================
def highlight_keywords(text):
    highlighted = text

    for word in fraud_keywords:
        if word.lower() in highlighted.lower():
            highlighted = highlighted.replace(
                word,
                f"<span class='highlight'>{word}</span>"
            )

    return highlighted

# ================= LOGIN =================
@app.route("/", methods=["GET","POST"])
def login():

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "1234":
            session["user"] = username
            return redirect("/home")
        else:
            return render_template("login.html", error="Invalid Login")

    return render_template("login.html")

# ================= HOME =================
@app.route("/home", methods=["GET","POST"])
def home():

    if "user" not in session:
        return redirect("/")

    prediction = None
    fraud_score = None
    real_score = None
    extracted_text = None
    highlighted_text = None
    reasons = None

    if request.method == "POST":

        text = request.form.get("job_text")
        image = request.files.get("job_image")

        # 🔥 IMAGE OCR (SAFE VERSION)
        if image and image.filename != "":
            try:
                img = Image.open(image).convert("RGB")
                extracted_text = extract_text(img)

                # 👉 OCR text irundha atha use pannum
                if extracted_text.strip() != "":
                    text = extracted_text
            except:
                extracted_text = "⚠ Unable to read image"

        # 🔥 PREDICTION (ONLY ONE SCORE SYSTEM)
        if text and text.strip() != "":
            prediction, fraud_score, real_score, reasons = predict_job(text)

            highlighted_text = highlight_keywords(text)

    return render_template(
        "index.html",
        prediction=prediction,
        fraud_score=fraud_score,
        real_score=real_score,
        extracted_text=extracted_text,
        highlighted_text=highlighted_text,
        reasons=reasons
    )

# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)