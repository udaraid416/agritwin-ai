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
    page_title="AgriTwin AI | Enterprise",
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
# 🌗 DYNAMIC DAY/NIGHT THEME & SaaS BENTO CSS
# ─────────────────────────────────────────────────────────────────────────────
current_hour = datetime.datetime.now().hour

if 6 <= current_hour < 18:
    theme_vars = """
    --bg-deep:      rgba(11, 22, 19, 0.88);
    --bg-mid:       rgba(16, 32, 26, 0.9);
    --glass-bg:     rgba(255, 255, 255, 0.03);
    --glass-border: rgba(16, 185, 129, 0.2);
    --accent:       #10B981; /* Emerald Green */
    --accent2:      #3B82F6; /* Tech Blue */
    --accent3:      #F59E0B; /* Amber */
    --danger:       #EF4444;
    --warn:         #F59E0B;
    --text-primary: #FFFFFF;
    --text-muted:   rgba(255, 255, 255, 0.65);
    """
    bg_video_url = "https://cdn.pixabay.com/video/2020/05/25/40141-424888806_large.mp4"
else:
    theme_vars = """
    --bg-deep:      rgba(6, 11, 19, 0.9);
    --bg-mid:       rgba(11, 19, 32, 0.92);
    --glass-bg:     rgba(255, 255, 255, 0.02);
    --glass-border: rgba(255, 255, 255, 0.08);
    --accent:       #10B981; /* Emerald Green */
    --accent2:      #3B82F6; /* Tech Blue */
    --accent3:      #6366F1; /* Indigo */
    --danger:       #EF4444;
    --warn:         #F59E0B;
    --text-primary: #F3F4F6;
    --text-muted:   #9CA3AF;
    """
    bg_video_url = "https://cdn.pixabay.com/video/2021/08/11/84687-586749942_large.mp4"

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
    {theme_vars}
}}

/* ── 🎥 Full Bleed Video Background ── */
#video-background {{
    position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%;
    z-index: -2; filter: blur(10px) brightness(0.25); object-fit: cover;
}}

/* ── Reset Containers ── */
.stApp, .main, section[data-testid="stSidebar"] > div {{ background: transparent !important; }}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--bg-deep) 0%, var(--bg-mid) 100%) !important;
    border-right: 1px solid var(--glass-border) !important; backdrop-filter: blur(25px);
}}
body, .stMarkdown, .stText, label, .stSelectbox label, .stSlider label, .stCheckbox label, .stTextInput label, .stNumberInput label {{
    font-family: 'Inter', sans-serif !important; color: var(--text-primary) !important;
}}

#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background-color: transparent !important; }}
button[data-testid="stSidebarCollapse"] {{ visibility: visible !important; }}
.block-container {{ padding-top: 2rem !important; padding-bottom: 2rem !important; }}

/* ── Bento Layout Entry Animation ── */
@keyframes bentoFadeUp {{
    from {{ opacity: 0; transform: translateY(16px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* ── Premium SaaS Bento Modules ── */
.glass-card {{
    background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: 16px; padding: 1.5rem;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); 
    animation: bentoFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both; 
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease; margin-bottom: 1rem;
}}
.glass-card:hover {{ 
    border-color: var(--accent); 
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.3);
}}

.metric-tile {{
    background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: 14px; padding: 1.2rem; text-align: center; backdrop-filter: blur(12px); animation: bentoFadeUp 0.5s ease both; transition: all 0.3s ease;
}}
.metric-tile:hover {{ transform: translateY(-3px); border-color: var(--accent); background: rgba(255,255,255,0.04); }}
.metric-value {{ font-size: 2.2rem; font-weight: 700; line-height: 1.1; letter-spacing: -0.03em; }}
.metric-label {{ font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }}

/* ── Progress Indicators ── */
.progress-container {{ margin: 0.8rem 0; }}
.progress-label {{ display:flex; justify-content:space-between; font-size:0.85rem; color:var(--text-muted); margin-bottom:0.4rem; }}
.progress-track {{ background:rgba(255,255,255,0.06); border-radius:99px; height:8px; overflow:hidden; }}
.progress-fill  {{ height:100%; border-radius:99px; transition: width 0.8s ease-in-out; }}

.ai-card {{ background: rgba(255,255,255,0.02); border: 1px solid var(--glass-border); border-radius: 14px; padding: 1.2rem; margin-bottom: 0.8rem; animation: bentoFadeUp 0.6s ease both; }}
.ai-card h4 {{ font-size: 0.8rem; letter-spacing: 0.06em; color: var(--accent) !important; text-transform: uppercase; margin: 0 0 0.5rem; font-weight: 600; }}
.ai-card p  {{ font-size: 0.95rem; line-height: 1.6; color: var(--text-primary); margin: 0; }}

.section-header {{ font-size: 1.1rem; font-weight: 600; color: var(--text-primary) !important; letter-spacing: -0.01em; border-bottom: 1px solid var(--glass-border); padding-bottom: 0.5rem; margin-bottom: 1.2rem; }}
.action-item {{ background: rgba(255,255,255,0.02); border-left: 4px solid var(--accent); border-radius: 0 8px 8px 0; padding: 0.7rem 1rem; margin-bottom: 0.6rem; font-size: 0.9rem; transition: background 0.2s ease; }}
.action-item:hover {{ background: rgba(255,255,255,0.04); }}
.risk-badge {{ display:inline-block; padding:0.3rem 0.8rem; border-radius:99px; font-size:0.8rem; font-weight:600; letter-spacing: 0.02em; }}
.status-pill {{ display:inline-block; padding:0.3rem 0.8rem; border-radius:99px; font-size:0.8rem; font-weight:600; letter-spacing: 0.02em; }}

/* ── Control Panel System Styling ── */
.stSlider > div > div > div {{ background: var(--accent) !important; }}
.stSelectbox div[data-baseweb], .stTextInput input, .stNumberInput input {{ background: rgba(0,0,0,0.3) !important; border: 1px solid var(--glass-border) !important; border-radius:10px !important; color: var(--text-primary) !important; transition: border-color 0.2s ease; }}
.stSelectbox div[data-baseweb]:hover, .stTextInput input:hover, .stNumberInput input:hover {{ border-color: var(--accent) !important; }}

/* ── Premium Segmented Control Tabs ── */
.stTabs [data-baseweb="tab-list"] {{ background-color: rgba(255,255,255,0.03); border-radius: 12px; padding: 6px; gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ color: var(--text-muted); font-size: 0.9rem; font-weight: 500; border-radius: 8px; transition: all 0.3s ease; padding: 8px 16px; }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; background-color: rgba(255,255,255,0.05) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-bottom: none !important; }}

[data-testid="stChatMessageContent"] {{ background-color: var(--card-bg); border: 1px solid var(--glass-border); border-radius: 16px; padding: 1rem; }}
.live-dot {{ width:8px; height:8px; background:var(--accent); border-radius:50%; display:inline-block; animation:blink 1.4s ease-in-out infinite; box-shadow: 0 0 8px var(--accent); vertical-align:middle; margin-right:6px; }}
.engine-pill {{ background: rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.3); border-radius:99px; padding:0.3rem 0.8rem; font-size:0.75rem; color:#60A5FA !important; }}
@keyframes blink {{ 0%,100%{opacity:1} 50%{opacity:0.3} }}
</style>
<video autoplay muted loop id="video-background"><source src="{bg_video_url}" type="video/mp4"></video>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (LOTTIE & ENTERPRISE PLOTLY CONFIG)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_ai_thinking = load_lottieurl("https://lottie.host/809c91f1-331e-4509-9fc6-9e90098da0da/uJ0q3E1jKz.json")

# Enterprise Chart Palette & Configuration
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#9CA3AF", size=11),
    margin=dict(l=25, r=25, t=35, b=25),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#9CA3AF")),
    hoverlabel=dict(
        bgcolor="rgba(15, 23, 42, 0.9)",
        bordercolor="rgba(255,255,255,0.08)",
        font_size=12,
        font_family="Inter",
        font_color="#F3F4F6"
    )
)

def plotly_dark_axes(fig):
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.05)", tickfont=dict(family="Inter", color="#9CA3AF"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.05)", tickfont=dict(family="Inter", color="#9CA3AF"))
    return fig

def progress_html(label: str, value: float, color: str = "#10B981", max_val: float = 100) -> str:
    pct = min(100, max(0, value / max_val * 100))
    return f'<div class="progress-container"><div class="progress-label"><span>{label}</span><span>{value:.1f}</span></div><div class="progress-track"><div class="progress-fill" style="width:{pct}%;background:{color};"></div></div></div>'

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROL CONSOLE
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="font-size: 1.6rem; font-weight: 700; color: #F3F4F6; margin-bottom: 0.2rem; letter-spacing: -0.02em;">AgriTwin <span style="color: #10B981;">Pro</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 0.8rem; color: #9CA3AF; margin-bottom: 1.8rem;">Digital Twin Infrastructure v{VERSION}</div>', unsafe_allow_html=True)

    crop_type    = st.text_input("🌱 Crop Architecture", value="Lettuce", placeholder="e.g., Lettuce, Strawberry...")
    growth_stage = st.selectbox("📈 Developmental Stage", GROWTH_STAGES, index=1)

    st.markdown('<p class="section-header">🌡 Climate Telemetry</p>', unsafe_allow_html=True)
    temperature      = st.slider("Ambient Temperature (°C)", 5.0,  50.0, PARAM_DEFAULTS["temperature"], 0.5)
    humidity         = st.slider("Relative Humidity (%)",    10.0, 99.0, PARAM_DEFAULTS["humidity"],    0.5)
    co2_level        = st.slider("Carbon Dioxide (ppm)",     300.0,2000.0,PARAM_DEFAULTS["co2_level"],  10.0)

    st.markdown('<p class="section-header">💧 Substrate Analytics</p>', unsafe_allow_html=True)
    soil_moisture    = st.slider("Volumetric Moisture (%)",  5.0,  99.0, PARAM_DEFAULTS["soil_moisture"], 0.5)
    ph_level         = st.slider("Potential Hydrogen (pH)",   4.0,   9.0, PARAM_DEFAULTS["ph_level"],      0.1)
    ec_level         = st.slider("Electrical Conduct. (mS)", 0.5,   5.0, PARAM_DEFAULTS["ec_level"],      0.1)

    st.markdown('<p class="section-header">💡 Quantum Irradiation</p>', unsafe_allow_html=True)
    light_intensity  = st.slider("Luminous Flux (lux)",      100.0,6000.0,PARAM_DEFAULTS["light_intensity"],50.0)
    ventilation_rate = st.slider("Air Exchange Velocity (%)",0.0,  100.0, PARAM_DEFAULTS["ventilation_rate"],1.0)
    leaf_area_index  = st.slider("Leaf Area Index (LAI)",    0.5,   7.0,  PARAM_DEFAULTS["leaf_area_index"],  0.1)

    st.markdown("---")
    st.markdown('<p class="section-header">🧊 Spatial Topology Node</p>', unsafe_allow_html=True)
    components.iframe("https://my.spline.design/miniroom-0b666a0d244958ceef967db0b537c376/", height=240)
    
    st.markdown("---")
    engine_status = get_ai_engine_status()
    st.markdown(f'<div style="text-align: center; margin-bottom: 1rem;"><span class="engine-pill">Engine: {engine_status}</span></div>', unsafe_allow_html=True)
    run_ai = st.button("⚡ Execute Analytical Core", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS ENGINE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
params = {
    "crop_type": crop_type, "growth_stage": growth_stage, "temperature": temperature,
    "humidity": humidity, "soil_moisture": soil_moisture, "co2_level": co2_level,
    "light_intensity": light_intensity, "ventilation_rate": ventilation_rate,
    "ph_level": ph_level, "ec_level": ec_level, "leaf_area_index": leaf_area_index,
}

sus_data   = compute_sustainability_score(params)
dis_data   = calculate_disease_risk(params)
irr_data   = calculate_irrigation_need(params)
heat_data  = calculate_heat_stress(temperature, humidity)
analysis   = {"sustainability": sus_data, "disease": dis_data, "irrigation": irr_data, "heat_stress": heat_data}
perf_data  = evaluate_system_performance(sus_data, dis_data, irr_data)
analysis["performance"] = perf_data

if "ai_result" not in st.session_state: st.session_state["ai_result"] = None

if run_ai:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown("<div style='text-align:center; padding: 2rem;'><h3 style='color: var(--accent); font-weight: 500; letter-spacing:-0.02em;'>Compiling Predictive System Metrics...</h3></div>", unsafe_allow_html=True)
        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=180, key="thinking")
        else: st.spinner("Processing...")
    st.session_state["ai_result"] = get_ai_recommendations(params, analysis)
    placeholder.empty()

# ─────────────────────────────────────────────────────────────────────────────
# APP VIEWPORT INTERFACES
# ─────────────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div style="font-size: 2.2rem; font-weight: 700; letter-spacing: -0.04em; margin-bottom: 0.2rem;">Predictive Telemetry Node</div>', unsafe_allow_html=True)
with col_h2:
    now = datetime.datetime.now()
    st.markdown(f'<div style="text-align:right; padding-top: 0.5rem;"><div style="font-size: 0.85rem; color: var(--text-muted); font-weight:500;">{now.strftime("%B %d, %Y • %H:%M")}</div><div style="font-size: 0.9rem; font-weight: 600; color: var(--accent); margin-top: 0.2rem;">Live Matrix: {crop_type}</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

tab_dashboard, tab_chat, tab_vision, tab_finance, tab_hardware = st.tabs([
    "📊 SYSTEM DASHBOARD", "💬 MAIN AI ASSISTANT", "📸 DISEASE VISION SCANNER", 
    "📈 MARKET ANALYZER", "⚙️ HARDWARE & ALERTS"
])

# =============================================================================
# TAB 1: SYSTEM DASHBOARD (MODULAR BENTO GRID ARCHITECTURE)
# =============================================================================
with tab_dashboard:
    # ── BENTO GRID TIER 1: CORE TELEMETRY MATRIX ──
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    def kpi(col, emoji, value, unit, label, color="var(--accent)"):
        col.markdown(f'<div class="metric-tile"><div style="font-size:1.5rem; margin-bottom: 0.3rem;">{emoji}</div><div class="metric-value" style="color:{color};">{value}{unit}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    kpi(k1, "🌡", f"{temperature:.1f}", "°C", "Temperature", "#EF4444" if temperature > 30 else "var(--accent)")
    kpi(k2, "💧", f"{humidity:.1f}", "%",  "Humidity", "#F59E0B" if humidity > 80 else "#3B82F6")
    kpi(k3, "🌱", f"{soil_moisture:.1f}", "%", "Moisture", "#EF4444" if soil_moisture < 30 else "var(--accent)")
    kpi(k4, "💨", f"{co2_level:.0f}", "", "CO₂ Level", "#F59E0B" if co2_level < 600 else "var(--accent)")
    kpi(k5, "☀️", f"{light_intensity:.0f}", "", "Irradiance (lux)", "#FBBF24")
    kpi(k6, "🌬", f"{ventilation_rate:.0f}", "%", "Air Exchange", "#EF4444" if ventilation_rate < 40 else "var(--accent)")

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # ── BENTO GRID TIER 2: ASYMMETRIC ANALYTICAL HUBS ──
    left_bento, right_bento = st.columns([1.5, 2.5])
    
    with left_bento:
        # Bento Module A: Sustainability Dashboard Matrix
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown('<p class="section-header" style="margin-top:0;">♻️ Sustainability Index</p>', unsafe_allow_html=True)
        gc = sus_data["grade_color"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=sus_data["total"], domain={'x': [0, 1], 'y': [0, 1]},
            number={'font': {'color': gc, 'size': 42, 'family': 'Inter'}},
            gauge={'axis': {'range': [None, 100], 'tickcolor': "rgba(255,255,255,0.1)", 'tickwidth': 1}, 
                   'bar': {'color': gc, 'thickness': 0.22}, 'bgcolor': "rgba(255,255,255,0.02)", 'borderwidth': 0,
                   'steps': [{'range': [0, 40], 'color': "rgba(239, 68, 68, 0.1)"}, 
                             {'range': [40, 75], 'color': "rgb
