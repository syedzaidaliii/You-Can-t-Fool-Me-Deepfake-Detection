# model_test.py
# CLI: python model_test.py /path/to/image.jpg /path/to/model.keras

import sys
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def preprocess_image(path, target_size=(224,224,3)):
    img = Image.open(path).convert("RGB")
    h, w, c = target_size
    img = img.resize((w,h), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def main():
    if len(sys.argv) < 2:
        print("Usage: python model_test.py /path/to/image.jpg [/path/to/model.keras]")
        return
    img_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "D:\Downloads\Deepfake\Deepfake\deepfake_streamlit\deepfake_EF_v1.keras"
    if not os.path.exists(img_path):
        print("Image not found:", img_path); return
    if not os.path.exists(model_path):
        print("Model not found:", model_path); return

    model = keras.models.load_model(model_path)
    # Try to infer input size if possible
    try:
        ishape = model.input_shape
        if isinstance(ishape, list): ishape = ishape[0]
        shape = tuple([d for d in ishape if d is not None])
        if len(shape) >= 3:
            target = (shape[0], shape[1], shape[2])
        else:
            target = (224,224,3)
    except Exception:
        target = (224,224,3)

    x = preprocess_image(img_path, target)
    preds = model.predict(x)
    preds = np.asarray(preds).flatten()
    # Simple heuristics:
    if preds.size == 1:
        p_fake = float(preds[0])
        p_real = 1.0 - p_fake
        print(f"Real: {p_real:.4f}, Fake: {p_fake:.4f}")
    else:
        # Normalize and print
        probs = preds[:2] if preds.size>=2 else preds
        probs = probs / (probs.sum() + 1e-12)
        print("Probabilities:", probs)

if __name__ == "__main__":
    main()
