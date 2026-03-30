# You-Can't-Fool-Me — Deepfake Detection

This project started as a personal challenge: can a fine-tuned EfficientNet model reliably tell apart real faces from AI-generated ones? Short answer — yes, pretty well. This repo contains everything from the training notebook to a working Streamlit app you can run locally in minutes.

The model (`deepfake_EF_v1.keras`) was trained on Kaggle using a frozen EfficientNetV2B3 backbone with a custom head on top. The web interface is clean, fast, and requires zero setup beyond installing a few packages.

---

## What's Inside

```
.
├── df-model-train.ipynb        # Full training pipeline (run on Kaggle)
├── streamlit_deepfake_app.py   # Web app for testing images
├── model_test.py               # Quick CLI prediction script
├── deepfake_EF_v1.keras        # Trained model — add this yourself (see below)
└── README.md
```

The `.keras` file is not committed to the repo because of its size. You'll need to either train it yourself using the notebook or download it separately and drop it in the root folder.

---

## Getting Started

You'll need Python 3.8 or higher. Install the dependencies with:

```bash
pip install tensorflow streamlit pillow numpy scikit-learn matplotlib seaborn
```

A GPU makes training significantly faster but isn't required to run inference or the Streamlit app.

---

## Running the Web App

```bash
streamlit run streamlit_deepfake_app.py
```

Before you run it, make sure `deepfake_EF_v1.keras` is sitting in the same directory as the script — otherwise the app will throw a warning and won't load any predictions.

Once it's running, just upload any `.jpg`, `.jpeg`, or `.png` file and the model will return a confidence score along with a breakdown of how likely the image is real vs. fake. The UI has a dark gradient theme and shows results inline without any page reload.

---

## CLI Option

If you don't want to open the browser, `model_test.py` lets you run a quick prediction from the terminal:

```bash
python model_test.py /path/to/image.jpg /path/to/deepfake_EF_v1.keras
```

Output looks like:

```
Real: 0.9231, Fake: 0.0769
```

The script reads the model's input shape automatically, so you don't need to hardcode anything. If no model path is passed, it falls back to a default local path defined inside the file — just update that if your setup is different.

---

## Model Details

The backbone is EfficientNetV2B3 pretrained on ImageNet, kept fully frozen during training. Only the classification head was trained from scratch.

| Layer                  | Details                          |
|------------------------|----------------------------------|
| Input                  | 224 × 224 × 3                    |
| Backbone               | EfficientNetV2B3 (frozen)        |
| Pooling                | GlobalAveragePooling2D           |
| BatchNorm              | After pooling                    |
| Dropout                | 0.5                              |
| Dense                  | 256 units, ReLU                  |
| Output                 | 1 unit, Sigmoid                  |

Predictions above `0.5` are classified as deepfakes; everything below is considered real. The confidence shown in the app reflects how far the prediction is from the 0.5 boundary, not the raw sigmoid output.

---

## Training Setup

Training was done entirely on Kaggle with GPU enabled. Here's a summary of the config used:

| Parameter        | Value                   |
|------------------|-------------------------|
| Image size       | 224 × 224               |
| Batch size       | 16                      |
| Epochs           | 10                      |
| Learning rate    | 2e-4 (with warmup)      |
| Warmup epochs    | 2                       |
| LR schedule      | Warmup + Cosine Decay   |
| Optimizer        | Adam (clipnorm = 5.0)   |
| Precision        | float32                 |

The learning rate doesn't jump straight to `2e-4` from the start. It warms up from `10%` of that value over the first 2 epochs, then follows a cosine decay for the remainder. This helped avoid instability in the early steps before the head weights settled.

Data augmentation was kept minimal on purpose — just a horizontal flip and pixel rescaling. No heavy transforms like rotations or color jitter, since those can actually hurt binary face classification when the dataset is already reasonably balanced.

---

## Dataset

Dataset Link-https://drive.google.com/drive/folders/18htyBi2YIAS-8-qO8MFtvUorch0aRB9N?usp=sharing

The dataset used for training comes from Kaggle:

- **Source:** `syedzaidalii/deepfake` — `df 40` split
- **Classes:** `fake images` / `real images`
- **Train images:** ~6,364
- **Validation images:** ~3,195

Folder layout expected by the training script:

```
df 40/
├── train/
│   ├── fake images/
│   └── real images/
└── val/
    ├── fake images/
    └── real images/
```

If you're swapping in your own dataset, just match this structure and update `BASEPATH` in the notebook.

---

## Evaluation

The notebook runs a full evaluation pass at the end of training. It generates a confusion matrix (plotted with seaborn), a classification report with per-class precision, recall, and F1-score, and logs overall accuracy. All of this is handled by scikit-learn — no custom evaluation code.

To reproduce results, open `df-model-train.ipynb` on Kaggle, attach the dataset, and run all cells with GPU turned on.

---

## Notes

- The app uses `@st.cache_resource` to load the model once and keep it in memory across sessions. Restarting Streamlit clears the cache.
- The CLI script (`model_test.py`) has a hardcoded fallback model path pointing to a Windows directory. Change it to match your local setup if you're not on that machine.
- Mixed precision was set to `float32` explicitly during training to avoid NaN losses that appeared with `float16` on this particular dataset.

---

## Stack

- TensorFlow / Keras
- EfficientNetV2B3 (via `keras.applications`)
- Streamlit
- scikit-learn
- Pillow
- NumPy
