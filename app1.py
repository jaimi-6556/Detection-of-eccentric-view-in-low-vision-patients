import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Clinical Eccentric Viewing Detection",
    layout="wide"
)

# --------------------------------
# CUSTOM CSS
# --------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f4fd, #ffffff);
}

.main-header {
    background: linear-gradient(90deg,#0d47a1,#1976d2);
    padding: 20px;
    border-radius: 16px;
    color: white;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}

.metric-card {
    background: rgba(255,255,255,0.9);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 10px;
}

.footer {
    text-align:center;
    color: gray;
    padding:20px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# HEADER
# --------------------------------
st.markdown("""
<div class='main-header'>
    <h1>👁️ AI-Based Eccentric Viewing Detection</h1>
    <p>Clinical Decision Support System for PRL & Fixation Analysis</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# --------------------------------
# SIDEBAR
# --------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")

    duration = st.slider("Recording Duration (seconds)", 5, 30, 10)
    threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7)
    st.info("Model: CNN + LSTM")
    st.info("Input Size: 64x64")
    st.success("System Ready")

# --------------------------------
# MODEL
# --------------------------------
class GazePRLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=8192,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])
        return x


MODEL_PATH = r"E:\D Drive\SEM-6\project\best_gaze_prl_model.pth"


@st.cache_resource
def load_model():
    model = GazePRLModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()

# --------------------------------
# METRICS FUNCTION
# --------------------------------
def calculate_metrics(prl_points, fovea):
    prl = np.mean(prl_points, axis=0)

    dx, dy = prl[0] - fovea[0], prl[1] - fovea[1]
    dist_px = math.sqrt(dx ** 2 + dy ** 2)

    pixel_to_degree = 0.03
    eccentricity = dist_px * pixel_to_degree
    angle = abs(math.degrees(math.atan2(dy, dx)))

    vertical = "Superior" if dy < 0 else "Inferior"
    horizontal = "Temporal" if dx > 0 else "Nasal"
    position = f"{vertical}-{horizontal}"

    std_x = np.std([p[0] for p in prl_points])
    std_y = np.std([p[1] for p in prl_points])

    stability_score = 1 / (1 + std_x + std_y)

    if stability_score < 0.5:
        stability = "Unstable"
    elif stability_score < 0.75:
        stability = "Moderate"
    else:
        stability = "Stable"

    if eccentricity < 2:
        severity = "Normal"
    elif eccentricity < 5:
        severity = "Mild"
    elif eccentricity < 10:
        severity = "Moderate"
    else:
        severity = "Severe"

    return position, angle, eccentricity, stability, severity


# --------------------------------
# SESSION STATE
# --------------------------------
if "recording" not in st.session_state:
    st.session_state.recording = False

if "points" not in st.session_state:
    st.session_state.points = []

if "start_time" not in st.session_state:
    st.session_state.start_time = None

# --------------------------------
# TABS
# --------------------------------
tab1, tab2, tab3 = st.tabs(["🎥 Live Detection", "📋 Clinical Report", "📊 Analytics"])

# =====================================
# TAB 1 - LIVE DETECTION
# =====================================
with tab1:

    col1, col2 = st.columns([2, 1])

    with col2:
        start_btn = st.button("▶ Start Recording")
        stop_btn = st.button("⏹ Stop Recording")

        status_box = st.empty()
        timer_box = st.empty()
        progress_bar = st.progress(0)

    if start_btn:
        st.session_state.recording = True
        st.session_state.points = []
        st.session_state.start_time = time.time()

    if stop_btn:
        st.session_state.recording = False

    frame_placeholder = col1.empty()

    if st.session_state.recording:
        cap = cv2.VideoCapture(0)

        while st.session_state.recording:
            ret, frame = cap.read()

            if not ret:
                st.error("Camera error")
                break

            frame = cv2.flip(frame, 1)

            elapsed = time.time() - st.session_state.start_time
            progress = min(elapsed / duration, 1.0)

            status_box.success("🟢 Recording in Progress")
            timer_box.info(f"⏱ {elapsed:.1f} sec")
            progress_bar.progress(progress)

            if elapsed >= duration:
                st.session_state.recording = False
                break

            h, w, _ = frame.shape
            fovea = (w // 2, h // 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (64, 64)) / 255.0

            img = np.stack([gray, gray], axis=0)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                pred = model(img)

            gx, gy = pred[0].tolist()

            prl = (int(gx * w), int(gy * h))
            st.session_state.points.append(prl)

            cv2.circle(frame, prl, 6, (0, 255, 0), -1)
            cv2.circle(frame, fovea, 6, (0, 0, 255), -1)

            frame_placeholder.image(frame, channels="BGR")

        cap.release()
        status_box.warning("🟡 Recording Complete")

# =====================================
# TAB 2 - REPORT
# =====================================
with tab2:
    if len(st.session_state.points) > 10:

        fovea = (320, 240)
        position, angle, ecc, stability, severity = calculate_metrics(
            st.session_state.points,
            fovea
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("PRL Position", position)
        c2.metric("Angle", f"{angle:.2f}°")
        c3.metric("Eccentricity", f"{ecc:.2f}°")

        c4, c5 = st.columns(2)
        c4.metric("Fixation Stability", stability)
        c5.metric("Severity", severity)

        report = f"""
Clinical Eccentric Viewing Report

PRL Position: {position}
Angle: {angle:.2f}
Eccentricity: {ecc:.2f}
Fixation Stability: {stability}
Severity: {severity}
"""

        st.download_button(
            "⬇ Download Report",
            report,
            file_name="clinical_report.txt"
        )

    else:
        st.info("No recording available.")

# =====================================
# TAB 3 - ANALYTICS
# =====================================
with tab3:
    if len(st.session_state.points) > 10:

        st.subheader("PRL Heatmap")

        x = [p[0] for p in st.session_state.points]
        y = [p[1] for p in st.session_state.points]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, alpha=0.5)
        ax.set_title("Fixation Distribution")
        ax.invert_yaxis()

        st.pyplot(fig)

        st.success(f"✅ Total gaze samples collected: {len(st.session_state.points)}")


    else:
        st.info("No analytics available.")

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("""
<div class='footer'>
Developed for Low Vision Clinical Assessment 
</div>
""", unsafe_allow_html=True)
