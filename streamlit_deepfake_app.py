import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
import time

st.set_page_config(
    page_title="You-can't-fool-me",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, rgb(15, 12, 41), rgb(48, 43, 99), rgb(36, 36, 62), rgb(26, 26, 46));
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    color: white;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.main-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 3rem;
    border-radius: 30px;
    box-shadow: 0 8px 32px 0 rgba(0, 255, 255, 0.15),
                0 0 0 1px rgba(255, 255, 255, 0.1);
    margin-top: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.header-section {
    text-align: center;
    margin-bottom: 2.5rem;
}

.main-title {
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, rgb(102, 126, 234) 0%, rgb(118, 75, 162) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}

.sub-title {
    font-size: 1.1rem;
    color: #a0a0a0;
    font-weight: 400;
    margin-bottom: 1rem;
}

.feature-pills {
    display: flex;
    justify-content: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}

.pill {
    background: rgba(102, 126, 234, 0.2);
    padding: 0.5rem 1.2rem;
    border-radius: 20px;
    font-size: 0.85rem;
    border: 1px solid rgba(102, 126, 234, 0.3);
    color: #b8c5ff;
}

.upload-section {
    margin: 2rem 0;
}

.stFileUploader {
    margin-bottom: 2rem;
}

.stFileUploader > div {
    border-radius: 20px !important;
    border: 2px dashed rgba(102, 126, 234, 0.5) !important;
    padding: 2.5rem !important;
    background: rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s ease;
}

.stFileUploader > div:hover {
    border-color: rgba(102, 126, 234, 0.8) !important;
    background: rgba(102, 126, 234, 0.05) !important;
    transform: translateY(-2px);
}

.stImage {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

.result-card {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    padding: 2rem;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    height: 100%;
}

.result-header {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    color: #ffffff;
}

.status-badge {
    display: inline-block;
    padding: 0.75rem 1.5rem;
    border-radius: 30px;
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
    text-align: center;
    width: 100%;
}

.badge-fake {
    background: linear-gradient(135deg, rgb(255, 107, 107) 0%, rgb(238, 90, 111) 100%);
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
}

.badge-real {
    background: linear-gradient(135deg, rgb(81, 207, 102) 0%, rgb(55, 178, 77) 100%);
    box-shadow: 0 4px 15px rgba(81, 207, 102, 0.4);
}

.confidence-score {
    text-align: center;
    margin: 1.5rem 0;
}

.confidence-number {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, rgb(102, 126, 234) 0%, rgb(118, 75, 162) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.confidence-label {
    font-size: 0.9rem;
    color: #a0a0a0;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, rgb(102, 126, 234) 0%, rgb(118, 75, 162) 100%);
    border-radius: 10px;
}

.stProgress > div > div {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

.prob-container {
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.prob-title {
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1rem;
}

.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
    border-left: 3px solid;
}

.prob-row.real {
    border-left-color: rgb(81, 207, 102);
}

.prob-row.fake {
    border-left-color: rgb(255, 107, 107);
}

.prob-label {
    font-weight: 600;
    font-size: 0.95rem;
}

.prob-value {
    font-weight: 700;
    font-size: 1.1rem;
}

.stSpinner > div {
    border-top-color: rgb(102, 126, 234) !important;
}

.info-section {
    margin-top: 3rem;
    padding: 1.5rem;
    background: rgba(102, 126, 234, 0.05);
    border-radius: 15px;
    border-left: 4px solid rgb(102, 126, 234);
}

.info-title {
    font-weight: 700;
    color: rgb(102, 126, 234);
    margin-bottom: 0.5rem;
}

.info-text {
    color: #b0b0b0;
    font-size: 0.9rem;
    line-height: 1.6;
}

@media (max-width: 768px) {
    .main-title {
        font-size: 2.5rem;
    }
    
    .main-container {
        padding: 2rem 1.5rem;
    }
    
    .confidence-number {
        font-size: 2.5rem;
    }
}

.fade-in {
    animation: fadeIn 0.6s ease-in;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    color: rgb(102, 126, 234) !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #a0a0a0 !important;
    font-size: 0.9rem !important;
}

.stSuccess, .stError {
    border-radius: 10px !important;
    padding: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("deepfake_EF_v1.keras")
    except Exception as e:
        st.error(f"⚠️ Model file not found. Please ensure 'deepfake_EF_v1.keras' is in the same directory.")
        return None

model = load_model()


st.markdown('<div class="main-container fade-in">', unsafe_allow_html=True)

# Header Section
st.markdown('''
<div class="header-section">
    <div class="main-title"> You-can't-fool-me </div>
    <div class="sub-title">Advanced Neural Network for Deepfake Detection</div>
    <div class="feature-pills">
        <span class="pill"> EfficientNet Architecture</span>
        <span class="pill"> Real-time Analysis</span>
        <span class="pill"> High Accuracy</span>
    </div>
</div>
''', unsafe_allow_html=True)

# Upload Section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file and model:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📸 Uploaded Image")
        st.image(img, use_container_width=True)
    
    # Preprocess image
    IMG_SIZE = 224
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction with loading animation
    with col2:
        with st.spinner("🔍 Analyzing image with AI..."):
            time.sleep(0.5)
            prediction = model.predict(img_array, verbose=0)[0][0]
        
        fake_prob = float(prediction)
        real_prob = 1 - fake_prob
        
        # Result Card
        st.markdown('<div class="result-card fade-in">', unsafe_allow_html=True)
        
        # Status Badge
        if fake_prob > 0.5:
            confidence = fake_prob
            st.markdown('<div class="status-badge badge-fake">⚠️ DEEPFAKE DETECTED</div>', unsafe_allow_html=True)
        else:
            confidence = real_prob
            st.markdown('<div class="status-badge badge-real">✅ AUTHENTIC IMAGE</div>', unsafe_allow_html=True)
        
        # Confidence Score
        st.markdown(f'''
        <div class="confidence-score">
            <div class="confidence-number">{confidence*100:.1f}%</div>
            <div class="confidence-label">Confidence Score</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Progress Bar
        st.progress(confidence)
        
        # Probability Breakdown
        st.markdown('''
        <div class="prob-container">
            <div class="prob-title">📊 Detailed Analysis</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="prob-row real">
            <span class="prob-label">✅ Authentic</span>
            <span class="prob-value">{real_prob*100:.2f}%</span>
        </div>
        <div class="prob-row fake">
            <span class="prob-label">⚠️ Deepfake</span>
            <span class="prob-value">{fake_prob*100:.2f}%</span>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif not model:
    st.warning("⚠️ Model not loaded. Please check if the model file exists.")

# Info Section
if not uploaded_file:
    st.markdown('''
    <div class="info-section">
        <div class="info-title">ℹ️ How it works</div>
        <div class="info-text">
            You-can't-fool-me uses state-of-the-art EfficientNet deep learning architecture to analyze images 
            and detect potential deepfakes. Upload any image to get started with instant AI-powered analysis.
        </div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('''
<div style="text-align: center; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1);">
    <p style="color: #666; font-size: 0.85rem;">
        Powered by TensorFlow & EfficientNet | Built with Streamlit
    </p>
</div>
''', unsafe_allow_html=True)




