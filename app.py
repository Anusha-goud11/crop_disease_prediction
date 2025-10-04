from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator

# --- Flask app setup ---
app = Flask(__name__)

# Folder to store uploads
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
#translation------
def translate_text(text, target_lang="te"):  # "te" for Telugu, "hi" for Hindi, etc.
    try:
        translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
        return translated
    except Exception as e:
        print("Translation error:", e)
        return text  # fallback to original


def get_disease_info(label, target_lang="te"):
    DISEASE_INFO = {
    "Healthy Rice Leaf": {
        "description": "The plant shows no visible signs of disease and appears healthy.",
        "remedy": "Continue regular monitoring and maintain optimal growing conditions."
    },
    "Brown Spot": {
        "description": "Brown spot appears as small brown lesions on leaves, often due to nutrient deficiency or fungal infection.",
        "remedy": "Apply potassium-rich fertilizer and use fungicide if needed."
    },
    "Bacterial LeafBlight": {
        "description": "Bacterial leaf blight causes water-soaked lesions that turn yellow and spread rapidly.",
        "remedy": "Use copper-based bactericides and avoid overhead irrigation."
    },
    "Leaf Blast": {
        "description": "Leaf blast causes diamond-shaped lesions and can lead to leaf drying and death.",
        "remedy": "Apply tricyclazole and maintain proper field drainage."
    },
    "Leaf scald": {
        "description": "Leaf scald appears as elongated reddish-brown lesions with yellow margins, often due to fungal infection.",
        "remedy": "Use fungicides like benomyl and avoid excessive nitrogen fertilization."
    },
    "Sheath Blight": {
        "description": "Sheath blight affects the lower leaves and stems in humid conditions, forming irregular lesions.",
        "remedy": "Apply validamycin and maintain proper spacing between plants."
    }
}

    info = DISEASE_INFO.get(label, {
        "description": "No info available",
        "remedy": "No remedy available"
    })

    # Translate both fields
    translated_info = {
        "description": translate_text(info["description"], target_lang),
        "remedy": translate_text(info["remedy"], target_lang)
    }

    return translated_info

# --- Load your trained model ---
MODEL_PATH = "public/model_final.h5"  # replace with your model path
model = tf.keras.models.load_model(MODEL_PATH)
print("Model expects input shape:", model.input_shape)

# --- Class labels ---
CLASS_NAMES = ["Healthy Rice Leaf", "Brown Spot", "Bacterial Leaf Blight", "Leaf Blast", "Leaf scald", "Sheath Blight"]  # replace with your actual classes

# --- Image preprocessing & prediction ---
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')  # convert to RGB
    img = img.resize((128, 128))                # MUST match training size
    img_array = np.array(img) / 255.0           # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
    print("DEBUG: img_array.shape =", img_array.shape)  # should be (1, 224, 224, 3)
    
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return CLASS_NAMES[class_index], confidence

# --- Routes ---
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Get selected language from form
        language = request.form.get("language", "en")

        # Step 1: Predict disease
        label, confidence = predict_image(filepath)

        # Step 2: Get disease info
        info = get_disease_info(label, target_lang=language)
        description = info["description"]
        remedy = info["remedy"]

        # Step 3: Translate to selected language
        description_translated = translate_text(description, language)
        remedy_translated = translate_text(remedy, language)

        # Step 4: Render result page
        return render_template("result.html",
                               image_path=filepath,
                               label=label,
                               confidence=confidence,
                               description=description_translated,
                               remedy=remedy_translated)
# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True)
