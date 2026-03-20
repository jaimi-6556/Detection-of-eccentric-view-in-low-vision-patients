import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import math

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(layout="wide", page_title="Clinical Eccentric Viewing System")

# -----------------------------
# Medical Theme
# -----------------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#e3f2fd,#f8fbff);}

.header {
    background: linear-gradient(90deg,#0d47a1,#1976d2);
    padding:20px 30px;border-radius:16px;color:white;
    box-shadow:0 8px 24px rgba(0,0,0,0.15);
}

.section-title{
    color:#0d47a1;font-size:26px;font-weight:700;margin:15px 0;
}

.card{
    background:rgba(255,255,255,0.85);
    backdrop-filter:blur(10px);
    border-radius:18px;padding:22px;
    box-shadow:0 8px 24px rgba(0,0,0,0.08);
}

.metric-label{color:#607d8b;font-size:14px;}
.metric-value{color:#0d47a1;font-size:30px;font-weight:700;}

.badge-green{background:#e8f5e9;color:#2e7d32;padding:6px 14px;border-radius:20px;font-weight:600;}
.badge-orange{background:#fff3e0;color:#ef6c00;padding:6px 14px;border-radius:20px;font-weight:600;}
.badge-red{background:#ffebee;color:#c62828;padding:6px 14px;border-radius:20px;font-weight:600;}

.stButton>button{border-radius:12px;padding:10px 22px;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header">
<h1>👁️ AI-Based Eccentric Viewing Detection</h1>
<p>Clinical Decision Support System for PRL & Fixation Analysis</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Tuned CNN + LSTM Model (Model 3)
# -----------------------------
class GazePRLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(
            input_size=8192,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(128,2)

    def forward(self,x):
        b,t,c,h,w = x.shape
        x = x.view(b*t,c,h,w)
        x = self.cnn(x)
        x = x.view(b,t,-1)
        x,_ = self.lstm(x)
        x = self.fc(x[:,-1])
        return x

# -----------------------------
# Load Tuned Model
# -----------------------------
MODEL_PATH = r"E:\D Drive\SEM-6\project\best_gaze_prl_model.pth"

@st.cache_resource
def load_model():
    model = GazePRLModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Metrics
# -----------------------------
def calculate_metrics(prl_points, fovea):
    prl = np.mean(prl_points, axis=0)
    dx, dy = prl[0]-fovea[0], prl[1]-fovea[1]

    dist_px = math.sqrt(dx**2 + dy**2)
    pixel_to_degree = 0.03
    eccentricity = dist_px * pixel_to_degree
    angle = abs(math.degrees(math.atan2(dy, dx)))

    vertical = "Superior" if dy < 0 else "Inferior"
    horizontal = "Temporal" if dx > 0 else "Nasal"
    position = f"{vertical}-{horizontal} Region"

    std_x = np.std([p[0] for p in prl_points])
    std_y = np.std([p[1] for p in prl_points])
    stability_score = 1/(1+std_x+std_y)

    if stability_score < 0.5:
        stability = "Unstable"
    elif stability_score < 0.75:
        stability = "Moderate"
    else:
        stability = "Stable"

    if eccentricity < 2:
        severity = "Normal"
    elif eccentricity < 5:
        severity = "Mild Eccentric Viewing"
    elif eccentricity < 10:
        severity = "Moderate Eccentric Viewing"
    else:
        severity = "Severe Eccentric Viewing"

    return position, angle, eccentricity, stability, severity

# -----------------------------
# Session State
# -----------------------------
if "recording" not in st.session_state:
    st.session_state.recording = False
if "points" not in st.session_state:
    st.session_state.points = []

# -----------------------------
# Controls
# -----------------------------
b1,b2,_ = st.columns([1,1,3])
if b1.button("▶ Start Recording"):
    st.session_state.recording = True
    st.session_state.points = []
if b2.button("⏹ Stop Recording"):
    st.session_state.recording = False

st.write("")
frame_box = st.empty()

# -----------------------------
# Camera Loop
# -----------------------------
if st.session_state.recording:
    cap = cv2.VideoCapture(0)

    while st.session_state.recording:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape
        fovea = (w//2, h//2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(64,64)) / 255.0
        img = np.stack([gray,gray],axis=0)
        img = torch.tensor(img,dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = model(img)

        gx,gy = pred[0].tolist()
        prl = (int(gx*w), int(gy*h))
        st.session_state.points.append(prl)

        cv2.circle(frame, prl,6,(0,255,0),-1)
        cv2.circle(frame, fovea,6,(0,0,255),-1)
        frame_box.image(frame, channels="BGR")

    cap.release()

# -----------------------------
# Report
# -----------------------------
if (not st.session_state.recording) and len(st.session_state.points)>10:
    st.markdown('<div class="section-title">📋 Clinical Fixation Report</div>', unsafe_allow_html=True)

    fovea = (640//2,480//2)
    position, angle, ecc, stability, severity = calculate_metrics(st.session_state.points,fovea)

    stab_badge = "badge-green" if stability=="Stable" else "badge-orange" if stability=="Moderate" else "badge-red"
    sev_badge = "badge-green" if "Normal" in severity else "badge-orange" if "Mild" in severity else "badge-red"

    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="card"><div class="metric-label">PRL Position</div><div class="metric-value">{position}</div></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="metric-label">Angle from Fovea</div><div class="metric-value">{angle:.1f}°</div></div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="metric-label">Eccentricity</div><div class="metric-value">{ecc:.1f}°</div></div>',unsafe_allow_html=True)

    st.write("")
    c4,c5 = st.columns(2)
    c4.markdown(f'<div class="card"><div class="metric-label">Fixation Stability</div><br><span class="{stab_badge}">{stability}</span></div>',unsafe_allow_html=True)
    c5.markdown(f'<div class="card"><div class="metric-label">Severity</div><br><span class="{sev_badge}">{severity}</span></div>',unsafe_allow_html=True)