"""
============================================================
Smart Plant Disease Detection System
predict.py — Prediction Engine
============================================================
Handles:
  - Loading the trained model (cached via @st.cache_resource)
  - Image preprocessing (PIL or file path input)
  - Returning prediction + confidence + all class probabilities
  - Disease metadata lookup (description, symptoms, treatment)
============================================================
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Disease Knowledge Base ───────────────────────────────────────────────────
# Each class maps to rich metadata shown in the Streamlit UI
DISEASE_INFO = {
    "Healthy": {
        "icon": "🌿",
        "severity": "None",
        "severity_color": "green",
        "description": (
            "The leaf appears healthy with no visible signs of disease or stress. "
            "Healthy leaves show uniform green coloration, firm texture, and no "
            "lesions, spots, or discoloration."
        ),
        "symptoms": [
            "Uniform bright green coloration",
            "No lesions or spots",
            "Firm, intact leaf structure",
            "Normal size and shape for the species"
        ],
        "treatment": [
            "✅ No treatment required — continue current care routine.",
            "💧 Maintain consistent watering schedule appropriate for the plant.",
            "🌱 Ensure balanced fertilization (NPK) every 2–4 weeks during growth.",
            "☀️ Provide adequate sunlight and air circulation.",
            "🔍 Inspect regularly (weekly) for early signs of pest or disease."
        ],
        "prevention": [
            "Regular monitoring and prompt removal of dead/damaged leaves.",
            "Avoid overwatering — most fungal diseases thrive in excess moisture.",
            "Practice crop rotation if applicable.",
            "Use disease-resistant varieties when replanting."
        ]
    },
    "Powdery": {
        "icon": "🍂",
        "severity": "Moderate",
        "severity_color": "orange",
        "description": (
            "Powdery mildew is a fungal disease caused by Erysiphales fungi. "
            "It appears as white or gray powdery spots on the leaf surface and "
            "undersides. If untreated, it can spread rapidly and reduce "
            "photosynthesis, weakening the plant significantly."
        ),
        "symptoms": [
            "White or grayish powdery coating on leaf surfaces",
            "Yellowing or browning of affected areas",
            "Distorted or stunted new growth",
            "Premature leaf drop in severe cases",
            "Powdery coating easily rubbed off with fingers"
        ],
        "treatment": [
            "🍃 Remove and dispose of heavily infected leaves immediately.",
            "🧴 Apply neem oil spray (2 tbsp per gallon of water) every 7–10 days.",
            "🧪 Use potassium bicarbonate or baking soda solution as fungicide.",
            "💊 Apply sulfur-based fungicides for severe infections.",
            "🌬️ Improve air circulation around the plant by pruning crowded areas.",
            "💧 Avoid overhead watering — water at soil level to reduce humidity."
        ],
        "prevention": [
            "Plant in well-ventilated areas with good sunlight.",
            "Avoid excess nitrogen fertilization (promotes susceptible growth).",
            "Water early in the morning so leaves dry before evening.",
            "Apply preventive fungicide sprays during high-humidity periods.",
            "Use resistant cultivars when possible."
        ]
    },
    "Rust": {
        "icon": "🍁",
        "severity": "High",
        "severity_color": "red",
        "description": (
            "Rust is caused by obligate parasitic fungi in the order Pucciniales. "
            "It manifests as orange, yellow, or reddish-brown pustules on leaf "
            "undersides with corresponding yellow spots on the upper surface. "
            "Rust spreads through airborne spores and can cause significant "
            "yield loss in crops if left untreated."
        ),
        "symptoms": [
            "Orange, yellow, or rust-colored pustules on leaf undersides",
            "Yellow or pale green spots on the upper leaf surface",
            "Powdery orange spore masses that rub off on fingers",
            "Premature yellowing and leaf drop",
            "In severe cases: plant wilting and reduced yield",
            "Lesions may turn brown/black as disease advances"
        ],
        "treatment": [
            "🚨 Isolate infected plants immediately to prevent spore spread.",
            "✂️ Remove and destroy (burn or bag) all infected plant material.",
            "🧪 Apply copper-based fungicide (copper oxychloride) every 7–14 days.",
            "💊 Use triazole fungicides (tebuconazole, propiconazole) for severe cases.",
            "🌿 Apply sulfur dust or wettable sulfur as an effective organic option.",
            "📋 Follow a 3–4 spray rotation to prevent fungicide resistance.",
            "🌡️ Avoid working among plants when leaves are wet to prevent spread."
        ],
        "prevention": [
            "Select rust-resistant plant varieties whenever possible.",
            "Ensure adequate spacing for air circulation between plants.",
            "Remove and destroy plant debris at end of growing season.",
            "Apply preventive fungicide early in the season.",
            "Monitor plants weekly, especially during warm, humid weather.",
            "Avoid wetting foliage during irrigation."
        ]
    }
}

# ── Model Loader ─────────────────────────────────────────────────────────────
def load_model_and_labels(model_dir: str = "models"):
    """
    Load the trained Keras model and class label mapping.
    Supports both .keras (new format) and .h5 (legacy) files.
    Returns (model, label_map) tuple.
    """
    # Try new Keras format first, fall back to .h5
    model_path_keras = os.path.join(model_dir, "plant_disease_model.keras")
    model_path_h5    = os.path.join(model_dir, "plant_disease_model.h5")

    if os.path.exists(model_path_keras):
        model = tf.keras.models.load_model(model_path_keras)
    elif os.path.exists(model_path_h5):
        model = tf.keras.models.load_model(model_path_h5)
    else:
        raise FileNotFoundError(
            f"No model found in '{model_dir}'. "
            "Run train.py first to generate the model file."
        )

    # Load class label mapping
    labels_path = os.path.join(model_dir, "class_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            # Keys are strings in JSON; convert to int
            raw = json.load(f)
            label_map = {int(k): v for k, v in raw.items()}
    else:
        # Fallback: default ordering from training notebook
        label_map = {0: "Healthy", 1: "Powdery", 2: "Rust"}

    return model, label_map


# ── Image Preprocessor ────────────────────────────────────────────────────────
def preprocess_image(image_input, target_size=(224, 224)) -> np.ndarray:
    """
    Accept either:
      - A PIL.Image object (from Streamlit file uploader)
      - A file path string (for batch/script use)
    Returns a float32 numpy array of shape (1, H, W, 3), normalised [0, 1].
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # Normalise to [0, 1]
    arr = np.expand_dims(arr, axis=0)                # Add batch dimension → (1, H, W, 3)
    return arr


# ── Predictor ────────────────────────────────────────────────────────────────
def predict(model, label_map: dict, image_input, target_size=(224, 224)) -> dict:
    """
    Run inference on an image and return a rich result dict containing:
      - predicted_class: str (e.g. "Rust")
      - confidence: float (0.0 – 1.0)
      - all_probs: dict {class_name: probability}
      - disease_info: dict (description, symptoms, treatment, etc.)
    """
    # Preprocess
    x = preprocess_image(image_input, target_size=target_size)

    # Inference
    probs = model.predict(x, verbose=0)[0]  # Shape: (num_classes,)

    # Map probabilities to class names
    all_probs = {label_map[i]: float(probs[i]) for i in range(len(probs))}

    # Top prediction
    top_idx   = int(np.argmax(probs))
    top_class = label_map[top_idx]
    confidence = float(probs[top_idx])

    # Fetch disease metadata
    info = DISEASE_INFO.get(top_class, {})

    return {
        "predicted_class": top_class,
        "confidence": confidence,
        "all_probs": all_probs,
        "disease_info": info
    }


# ── CLI Quick Test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "Dataset/Test/Test/Rust/82f49a4a7b9585f1.jpg"

    print("Loading model...")
    model, label_map = load_model_and_labels()
    print(f"Classes: {label_map}")

    print(f"Predicting: {image_path}")
    result = predict(model, label_map, image_path)

    print(f"\n{'─'*40}")
    print(f"🌿 Prediction : {result['predicted_class']}")
    print(f"📊 Confidence : {result['confidence']*100:.2f}%")
    print(f"\nAll probabilities:")
    for cls, prob in sorted(result['all_probs'].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:12s}: {prob*100:6.2f}%  {bar}")