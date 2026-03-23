from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_PATH = "crop_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[OK] Model loaded from '{MODEL_PATH}'")
except Exception as e:
    model = None
    print(f"[ERROR] Could not load model: {e}")

# ── Class names — update this list to match your model's output ────────────────
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ── Image preprocessing ────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalize and batch the PIL image for model input."""
    image = image.resize((299, 299))
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0                   # normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)   # add batch dimension
    return img_array

# ── Helper: format class name for display ─────────────────────────────────────
def format_label(raw: str) -> str:
    """Turn 'Tomato___Early_blight' → 'Tomato: Early blight'"""
    parts = raw.split("___")
    if len(parts) == 2:
        crop, disease = parts
        disease = disease.replace("_", " ").capitalize()
        return f"{crop}: {disease}"
    return raw.replace("_", " ")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    prediction  = None
    confidence  = None
    error       = None

    if request.method == "POST":
        # ── Validate file upload ───────────────────────────────────────────────
        file = request.files.get("file")

        if not file or file.filename == "":
            error = "No file uploaded. Please select an image."
            return render_template("index.html", error=error)

        allowed_extensions = {"png", "jpg", "jpeg", "webp", "bmp"}
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in allowed_extensions:
            error = f"Unsupported file type '.{ext}'. Use PNG, JPG or WEBP."
            return render_template("index.html", error=error)

        # ── Model check ───────────────────────────────────────────────────────
        if model is None:
            error = "Model not loaded. Please check that 'crop_model.h5' exists."
            return render_template("index.html", error=error)

        # ── Predict ───────────────────────────────────────────────────────────
        try:
            image     = Image.open(file.stream).convert("RGB")
            img_array = preprocess_image(image)

            preds      = model.predict(img_array, verbose=0)   # shape: (1, num_classes)
            pred_index = int(np.argmax(preds))
            conf_score = float(np.max(preds))

            raw_label  = class_names[pred_index] if pred_index < len(class_names) else f"Class {pred_index}"
            prediction = format_label(raw_label)
            confidence = f"{conf_score * 100:.1f}%"

        except Exception as e:
            error = f"Prediction failed: {str(e)}"
            return render_template("index.html", error=error)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           error=error)

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)