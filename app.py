import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="ImpactSense AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("best_gradient_boosting_model.pkl")

# ---------------- BACKGROUND ----------------
def set_bg(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
            url("data:image/jpg;base64,{data}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- TOP NAV ----------------
def top_nav():
    st.markdown("""
    <style>
    .nav {
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:15px 40px;
        background:rgba(0,0,0,0.65);
        border-radius:0 0 15px 15px;
    }
    .logo {
        font-size:26px;
        font-weight:700;
        color:#ffb703;
    }
    </style>

    <div class="nav">
        <div class="logo">ImpactSense AI</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- LOGIN ----------------
import streamlit as st

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    st.markdown("<h2 style='text-align:center;'>🔐 Login</h2>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")



# ---------------- HOME ----------------
def home_page():
    st.markdown("""
    <div style='text-align:center;color:white;'>
        <h1 style='font-size:55px;'>AI-Powered Earthquake Impact Analysis</h1>
        <p style='font-size:22px;max-width:800px;margin:auto;'>
        Predict earthquake alert levels using machine learning and visualize seismic risk.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.1);padding:25px;border-radius:15px;color:white;text-align:center;'>
        <h3>⚡ Prediction</h3>
        ML-based alert classification
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.1);padding:25px;border-radius:15px;color:white;text-align:center;'>
        <h3>📊 Visualization</h3>
        Impact-based charts
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.1);padding:25px;border-radius:15px;color:white;text-align:center;'>
        <h3>🌍 Awareness</h3>
        Risk communication
        </div>
        """, unsafe_allow_html=True)

# ---------------- MANUAL PREDICTION ----------------

def manual_page():
    st.markdown("<h2 style='color:white;'>Manual Prediction</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        magnitude = st.number_input("Magnitude", 0.0, 10.0, 5.0)
        depth = st.number_input("Depth (km)", 0.0, 700.0, 10.0)
        cdi = st.number_input("CDI", 0.0, 12.0, 3.0)

    with col2:
        mmi = st.number_input("MMI", 0.0, 12.0, 4.0)
        sig = st.number_input("Significance", 0.0, 1000.0, 200.0)

    if st.button("Predict Alert"):
        intensity_score = magnitude * cdi
        magnitude_depth_ratio = magnitude / (depth + 1e-5)
        feature_names = model.feature_names_in_
        X = pd.DataFrame([[magnitude, depth, cdi, mmi, sig, intensity_score, magnitude_depth_ratio]],
                         columns=feature_names)
        pred = model.predict(X)[0]

        alert_map = {
            0: "GREEN",
            1: "YELLOW",
            2: "ORANGE",
            3: "RED"
        }

        color_map = {
            "GREEN": "green",
            "YELLOW": "yellow",
            "ORANGE": "orange",
            "RED": "red"
        }

        label = alert_map.get(pred, "UNKNOWN")

        st.markdown(f"""
        <div style='margin-top:20px;padding:30px;border-radius:15px;
        background:{color_map[label]};color:black;text-align:center;font-size:30px;font-weight:700;'>
        {label} ALERT
        </div>
        """, unsafe_allow_html=True)

        # Bar chart
        fig, ax = plt.subplots()
        ax.bar(["Magnitude","Depth","CDI","MMI","SIG"],
               [magnitude, depth, cdi, mmi, sig])
        st.pyplot(fig)
        
# ---------------- UPLOAD PAGE ----------------
def upload_page():
    st.markdown("<h2 style='color:white;'>Upload Dataset</h2>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

# ---------------- MAIN ----------------
try:
    set_bg("bg.jpg")
except:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    </style>
    """, unsafe_allow_html=True)

top_nav()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
else:

    page = st.radio("Navigate", ["Home", "Manual Prediction", "Upload Data"],
                    horizontal=True)

    if page == "Home":
        home_page()
    elif page == "Manual Prediction":
        manual_page()
    else:
        upload_page()
