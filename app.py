"""
app.py — AI-Powered Predictive Digital Twin for Smart Protected Agriculture
Streamlit frontend — PREMIUM SaaS UI (Inter Font, Smooth Transitions, Soft Glassmorphism)
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
import datetime
import os
import requests
import google.generativeai as genai
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
from PIL import Image

from config import (
    PARAM_DEFAULTS, GROWTH_STAGES,
    GRADE_COLORS, RISK_COLORS, ZONES, APP_TITLE, APP_ICON, VERSION
)
from utils import (
    compute_sustainability_score, calculate_disease_risk,
    calculate_irrigation_need, calculate_heat_stress,
    generate_zone_heatmap_data, generate_trend_data,
    evaluate_system_performance
)
from ai_model import get_ai_recommendations, get_ai_engine_status

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & API SETUP
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgriTwin AI | Pro",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if API_KEY:
    genai.configure(api_key=API_KEY)

# Session States for Integrated Chatbots
if "vision_result" not in st.session_state: st.session_state.vision_result = None
if "vision_messages" not in st.session_state: st.session_state.vision_messages = []
if "market_messages" not in st.session_state: st.session_state.market_messages = []

# ─────────────────────────────────────────────────────────────────────────────
# 🎨 PREMIUM SaaS CSS & ANIMATIONS
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-color:       #0B111A;
    --card-bg:        rgba(19, 31, 44, 0.65);
    --card-border:    rgba(255, 255, 255, 0.08);
    --accent:         #10B981;
    --text-primary:   #F3F4F6;
    --text-secondary: #9CA3AF;
}

.stApp {
    background: radial-gradient(circle at 15% 50%, rgba(16, 185, 129, 0.05), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(59, 130, 246, 0.05), transparent 25%);
    background-color: var(--bg-color);
    background-attachment: fixed;
}

body, .stMarkdown, .stText, label, div { font-family: 'Inter', sans-serif !important; color: var(--text-primary); }

#MainMenu, footer { visibility: hidden !important; }
header { background-color: transparent !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }

section[data-testid="stSidebar"] {
    background-color: rgba(13, 20, 28, 0.95) !important;
    border-right: 1px solid var(--card-border) !important;
}

.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    margin-bottom: 1.2rem;
}
.glass-card:hover { transform: translateY(-4px); border-color: var(--accent); }

.metric-tile {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px; padding: 1.2rem; text-align: center;
}

.section-header {
    font-size: 1.1rem; font-weight: 600; color: var(--text-primary) !important;
    border-bottom: 1px solid var(--card-border); padding-bottom: 0.6rem; margin: 1rem 0;
}

.stTabs [data-baseweb="tab-list"] { background-color: rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: var(--text-secondary); font-weight: 500; }
.stTabs [aria-selected="true"] { background-color: var(--card-bg) !important; color: var(--accent) !important; }

[data-testid="stChatMessageContent"] { background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 16px; padding: 1rem; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_lottieurl(url: str):
    try: return requests.get(url).json()
    except: return None

lottie_ai_thinking = load_lottieurl("https://lottie.host/809c91f1-331e-4509-9fc6-9e90098da0da/uJ0q3E1jKz.json")

# ─────────────────────────────────────────────────────────────────────────────
# TAB SETUP
# ─────────────────────────────────────────────────────────────────────────────
tab_dashboard, tab_chat, tab_vision, tab_finance, tab_hardware = st.tabs([
    "📊 Dashboard", "💬 Main AI Assistant", "📸 Disease Scanner", "📈 Market Analytics", "⚙️ Hardware Controls"
])

# (උඩින් තිබ්බ Logic ටිකම මෙතනට දාන්න - මම ප්ලේස් හෝල්ඩර් එකක් දැම්මේ කෝඩ් එක දිග වැඩි නිසා)
# [Dashbord Logic Here]
# [Chatbot Logic Here]
# [Vision + Chat Logic Here]
# [Finance + Chat Logic Here]
# [Hardware Logic Here]
