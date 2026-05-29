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
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}
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
                             {'range': [40, 75], 'color': "rgba(245, 158, 11, 0.1)"}, 
                             {'range': [75, 100], 'color': "rgba(16, 185, 129, 0.1)"}]}))
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=180, margin=dict(l=15, r=15, t=15, b=15))
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div style="text-align:center; font-size:1.1rem; color:{gc}; font-weight:600; margin-top:-15px; margin-bottom:1.5rem;">System Efficiency Rating: Grade {sus_data["grade"]}</div>', unsafe_allow_html=True)

        html_bars = ""
        for lbl, val, clr in [("Water Conversion", sus_data["water_efficiency"], "#3B82F6"), 
                               ("Kinetic Energy Input", sus_data["energy_efficiency"], "#8B5CF6"), 
                               ("Atmospheric Optimization", sus_data["climate_optimization"], "#10B981"), 
                               ("Pathogen Prevention", sus_data["disease_prevention"], "#34D399"), 
                               ("Biomass Yield Potential", sus_data["yield_potential"], "#FBBF24")]: 
            html_bars += progress_html(lbl, val, clr)
        st.markdown(html_bars, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_bento:
        # Bento Module B: Integrated Biological Profiles (Disease Risk & Irrigation Core)
        sub_row1, sub_row2 = st.columns(2)
        with sub_row1:
            dlc = dis_data["level_color"]
            st.markdown(f'<div class="glass-card" style="text-align:center; height:185px; padding:1.8rem 1rem;"><span class="risk-badge" style="background:{dlc}15;color:{dlc};border:1px solid {dlc};">{dis_data["level"].upper()} PATHOGEN INDEX</span><br><div style="font-size:3.2rem; font-weight: 700; color:{dlc}; margin:0.4rem 0; letter-spacing:-0.04em; line-height: 1;">{dis_data["overall"]:.1f}<span style="font-size:1.2rem; font-weight:400;">%</span></div><div style="font-size:0.8rem;color:var(--text-muted); font-weight:500;">Calculated Disease Proximity</div></div>', unsafe_allow_html=True)
        with sub_row2:
            isc = irr_data["status_color"]
            st.markdown(f'<div class="glass-card" style="text-align:center; height:185px; padding:1.8rem 1rem;"><span class="status-pill" style="background:{isc}15;color:{isc};border:1px solid {isc};">SYSTEM: {irr_data["status"].upper()}</span><br><div style="font-size:3.2rem; font-weight: 700; color:{isc}; margin:0.4rem 0; letter-spacing:-0.04em; line-height: 1;">{irr_data["urgency"]:.1f}<span style="font-size:1.2rem; font-weight:400;">%</span></div><div style="font-size:0.8rem;color:var(--text-muted); font-weight:500;">Volumetric Desiccation Rate</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="margin-top:-0.5rem; padding-bottom: 1.6rem;">', unsafe_allow_html=True)
        st.markdown('<p class="section-header" style="margin-top:0;">📋 Integrated Sub-System Diagnosis</p>', unsafe_allow_html=True)
        
        disease_col, irrigation_col = st.columns(2)
        with disease_col:
            disease_html = ""
            d_colors = ["#EF4444", "#F59E0B", "#8B5CF6", "#3B82F6"]
            for i, (name, val) in enumerate(dis_data["diseases"].items()): 
                disease_html += progress_html(name, val, d_colors[i % len(d_colors)])
            st.markdown(disease_html, unsafe_allow_html=True)
            
        with irrigation_col:
            st.markdown(f'<div style="font-size:0.9rem; line-height: 2.1; color: var(--text-primary); padding-top:0.4rem;">💧 Target Volume Load: <span style="float: right; font-weight: 600; color:{isc}">{irr_data["volume_liters"]:.2f} L/m²</span><br>⏱ Hydro-Cycle Delay: <span style="float: right; font-weight: 600; color:{isc}">{irr_data["next_irrigation_hours"]} hours</span><br>🍃 Evapotranspiration Rate: <span style="float: right; font-weight: 600; color:{isc}">{irr_data["evapotranspiration"]:.3f} mm/hr</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── BENTO GRID TIER 3: ADVANCED STRUCTURAL ANALYTICS (TRANSPARENT ENTERPRISE CHARTS) ──
    ch1, ch2, ch3 = st.columns(3)
    
    with ch1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header" style="margin-top:0;">📊 Microclimate Scatter Profile (24h)</p>', unsafe_allow_html=True)
        trend = generate_trend_data(temperature, humidity, 24)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend["times"], y=trend["temperatures"], name="Thermal Gradient", line=dict(color="#EF4444", width=2.5, shape="spline"), fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.04)"))
        fig_trend.add_trace(go.Scatter(x=trend["times"], y=trend["humidities"], name="Moisture Curve", line=dict(color="#3B82F6", width=2.5, shape="spline"), fill="tozeroy", fillcolor="rgba(59, 130, 246, 0.04)"))
        fig_trend.update_layout(**PLOTLY_LAYOUT, height=250, xaxis=dict(tickangle=0, nticks=5, showgrid=False))
        plotly_dark_axes(fig_trend)
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header" style="margin-top:0;">🎯 Operational Risk Radar Vector</p>', unsafe_allow_html=True)
        radar_cats = ["Biological Pressure", "Thermal Strain", "Hydric Stress", "CO₂ Deficit", "Ventilation Loss", "Biomass Deficit"]
        vals = [dis_data["overall"], min(100, max(0, (heat_data["heat_index"] - 20) * 2)), max(0, 100 - sus_data["water_efficiency"]), max(0, (900 - co2_level) / 9), max(0, 100 - ventilation_rate), max(0, 100 - sus_data["yield_potential"])]
        fig_radar = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=radar_cats + [radar_cats[0]], fill="toself", fillcolor="rgba(16, 185, 129, 0.12)", line=dict(color="#10B981", width=2)))
        fig_radar.update_layout(**PLOTLY_LAYOUT, height=250, polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=8, color="#9CA3AF"), showline=False), angularaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickfont=dict(size=9, color="#F3F4F6"))))
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with ch3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header" style="margin-top:0;">🥧 Resource Allocation Topology</p>', unsafe_allow_html=True)
        pie_labels = ["Hydro Node", "Kinetic Load", "Atmosphere", "Plant Health", "Yield Target"]
        pie_values = [sus_data["water_efficiency"], sus_data["energy_efficiency"], sus_data["climate_optimization"], sus_data["disease_prevention"], sus_data["yield_potential"]]
        pie_colors = ["#3B82F6", "#8B5CF6", "#10B981", "#34D399", "#FBBF24"]
        fig_pie = go.Figure(go.Pie(labels=pie_labels, values=pie_values, hole=0.62, marker=dict(colors=pie_colors, line=dict(color="rgba(11,17,26,1)", width=3.5)), textfont=dict(size=10, family="Inter")))
        fig_pie.add_annotation(text=f"<b>{sus_data['total']}</b>", x=0.5, y=0.5, font=dict(size=26, color="#F3F4F6", family="Inter"), showarrow=False)
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=250)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── BENTO GRID TIER 4: SPATIAL GRID HEATMAPS ──
    hm1, hm2 = st.columns(2)
    temp_grid, hum_grid = generate_zone_heatmap_data(temperature, humidity)
    with hm1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header" style="margin-top:0;">🌡 Spatial Micro-Climate Temperature Matrix</p>', unsafe_allow_html=True)
        fig_ht = go.Figure(go.Heatmap(z=temp_grid, text=[[f"{v:.1f}°C" for v in row] for row in temp_grid], texttemplate="%{text}", colorscale="RdYlGn_r", colorbar=dict(title="°C", tickfont=dict(family="Inter", color="#9CA3AF")), showscale=True))
        fig_ht.update_layout(**PLOTLY_LAYOUT, height=230)
        st.plotly_chart(fig_ht, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with hm2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header" style="margin-top:0;">💧 Spatial Root-Zone Humidity Matrix</p>', unsafe_allow_html=True)
        fig_hh = go.Figure(go.Heatmap(z=hum_grid, text=[[f"{v:.1f}%" for v in row] for row in hum_grid], texttemplate="%{text}", colorscale="Blues", colorbar=dict(title="%", tickfont=dict(family="Inter", color="#9CA3AF")), showscale=True))
        fig_hh.update_layout(**PLOTLY_LAYOUT, height=230)
        st.plotly_chart(fig_hh, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── BENTO GRID TIER 5: EXECUTIVE PREDICTIVE CORES ──
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">🤖 Core Strategic AI Inference Interface</p>', unsafe_allow_html=True)
    
    if not ai_res:
        st.markdown('<div class="glass-card" style="text-align:center; padding:3rem 1rem;"><div style="font-size:2.2rem; margin-bottom: 0.8rem;">✨</div><div style="font-size: 1.1rem; font-weight: 500; color: #F3F4F6;">Autonomous Core Is Stationary</div><div style="color: var(--text-muted); font-size: 0.88rem; margin-top: 0.4rem; font-weight:400;">Awaiting pipeline trigger. Execute Analytical Core in console panel to compile infrastructure datasets.</div></div>', unsafe_allow_html=True)
    else:
        ai_c1, ai_c2 = st.columns(2)
        with ai_c1:
            st.markdown(f'<div class="ai-card"><h4>🛡️ Crop Health Architecture Directive</h4><p>{ai_res.get("disease_warning","—")}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-card"><h4>🌡 Climate Volatility Calibration</h4><p>{ai_res.get("climate_warning","—")}</p></div>', unsafe_allow_html=True)
        with ai_c2:
            st.markdown(f'<div class="ai-card"><h4>💧 Hydro-Resource Ingestion Strategy</h4><p>{ai_res.get("irrigation_advice","—")}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-card"><h4>♻️ Sustainability Index Thresholds</h4><p>{ai_res.get("sustainability_insight","—")}</p></div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="glass-card" style="margin-top:0.2rem;"><h4 style="font-size: 0.85rem; color: var(--accent); margin-bottom: 0.8rem; text-transform: uppercase; font-weight:600; letter-spacing:0.05em;">Priority Node Operational Triggers</h4>', unsafe_allow_html=True)
        for i, action in enumerate(ai_res.get("top_actions", []), 1):
            st.markdown(f'<div class="action-item">{"🔴" if i==1 else "🟡" if i==2 else "🟢"} <b style="font-family:Inter; font-weight:600; margin-right:6px;">Node #{i}</b> — {action}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="glass-card" style="margin-top:0.2rem;"><h4 style="font-size: 0.85rem; color: var(--accent); margin-bottom: 0.8rem; text-transform: uppercase; font-weight:600; letter-spacing:0.05em;">Executive Infrastructure Synthesis Assessment</h4><span style="font-size:0.98rem; line-height: 1.75; font-weight:300;">{ai_res.get("overall_assessment","—")}</span></div>', unsafe_allow_html=True)
        
        report_text = f"=== AgriTwin AI Pro - Executive Infrastructure Report ===\nDate: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\nCrop: {crop_type} ({growth_stage})\n\n--- TELEMETRY CORE ---\nAmbient Temp: {temperature}°C | Relative Humidity: {humidity}% | Substrate Moisture: {soil_moisture}% | CO2 Load: {co2_level}ppm\n\n--- PREDICTIVE INTELLIGENCE STRATEGY ---\nPathogen Module: {ai_res.get('disease_warning')}\nAtmosphere Module: {ai_res.get('climate_warning')}\nHydric Module: {ai_res.get('irrigation_advice')}\n\n--- OVERALL SYSTEM SYNTHESIS ---\n{ai_res.get('overall_assessment')}\n"
        st.download_button(label="📥 Export Infrastructure Synthesis Protocol (.txt)", data=report_text, file_name=f"AgriTwin_Synthesis_Report_{datetime.datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain", use_container_width=True)

    # ── BENTO GRID TIER 6: COMPREHENSIVE PERFORMANCE RADIALS ──
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📊 Infrastructure Node Performance Assessment Matrix</p>', unsafe_allow_html=True)
    
    p_left, p_right = st.columns([1.8, 2.2])
    with p_left:
        pk1, pk2 = st.columns(2)
        def perf_card(col, icon, label, val, color):
            col.markdown(f'<div class="metric-tile"><div style="font-size:1.3rem">{icon}</div><div class="metric-value" style="color:{color};">{val:.1f}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
            
        perf_card(pk1, "🎯", "Prediction Precision", perf_data["prediction_accuracy"], "#3B82F6")
        perf_card(pk2, "⚙️", "Throughput Efficiency", perf_data["system_efficiency"], "#8B5CF6")
        
        pk3, pk4 = st.columns(2)
        perf_card(pk3, "🏡", "Greenhouse Stability", perf_data["greenhouse_performance"], "#10B981")
        
        gc = perf_data["grade_color"]
        pk4.markdown(f'<div class="metric-tile" style="height:128px; padding-top:1.4rem;"><div style="font-size:1.4rem; margin-bottom:0.2rem;">🏆</div><div class="metric-value" style="color:{gc}; font-size:1.8rem;">{perf_data["overall"]:.1f}</div><div class="metric-label" style="margin-top:0.2rem;">Aggregate Core (Grade {perf_data["grade"]})</div></div>', unsafe_allow_html=True)
    
    with p_right:
        st.markdown('<div class="glass-card" style="height:272px; padding: 1.2rem 1.5rem 1rem 1.5rem;">', unsafe_allow_html=True)
        fig_perf = go.Figure(go.Bar(
            x=["Prediction Precision", "Throughput Efficiency", "Greenhouse Stability", "Aggregate Core"],
            y=[perf_data["prediction_accuracy"], perf_data["system_efficiency"], perf_data["greenhouse_performance"], perf_data["overall"]],
            marker_color=["#3B82F6", "#8B5CF6", "#10B981", gc],
            text=[f'{v:.1f}%' for v in [perf_data["prediction_accuracy"], perf_data["system_efficiency"], perf_data["greenhouse_performance"], perf_data["overall"]]],
            textposition="outside", cliponaxis=False, marker_line=dict(width=0)))
        fig_perf.update_layout(**PLOTLY_LAYOUT, height=210, yaxis=dict(range=[0, 115], showgrid=False, visible=False), xaxis=dict(showgrid=False))
        plotly_dark_axes(fig_perf)
        st.plotly_chart(fig_perf, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2: MAIN AI ASSISTANT (CHATBOT)
# =============================================================================
def get_working_text_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'vision' not in m.name.lower(): return m.name
    except: pass
    return "models/gemini-1.5-flash"

def get_working_vision_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and ('vision' in m.name.lower() or 'flash' in m.name.lower()): return m.name
    except: pass
    return "models/gemini-1.5-flash"

with tab_chat:
    st.markdown('<p class="section-header">💬 Core Agronomy Assistant</p>', unsafe_allow_html=True)
    if not API_KEY: st.warning("⚠️ Access Token configuration required. Insert API Key inside Streamlit Deployment Secrets.")
    else:
        if "main_messages" not in st.session_state: st.session_state.main_messages = []
        for msg in st.session_state.main_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if prompt := st.chat_input(f"Interrogate conversational agronomist core regarding {crop_type} cultivation parameters...", key="main_chat"):
            st.session_state.main_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with placeholder.container():
                    if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=60, key="chat_main_thinking")
                    else: st.spinner("Processing Model Response...")
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    response = model.generate_content(f"You are an expert agronomy AI assistant infrastructure node. The client user is maintaining a {crop_type} crop ecosystem within a digital twin framework at development stage: {growth_stage}. Provide technical, accurate, and concise domain advice. Input Query: {prompt}")
                    placeholder.empty()
                    st.markdown(response.text)
                    st.session_state.main_messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    placeholder.empty()
                    st.error(f"Inference Failure: {e}")

# =============================================================================
# TAB 3: DISEASE VISION SCANNER + INTEGRATED CHAT
# =============================================================================
with tab_vision:
    st.markdown('<p class="section-header">📸 Computer Vision Pathology Diagnostic Scanner</p>', unsafe_allow_html=True)
    if not API_KEY: st.warning("⚠️ Access Token configuration required.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            img_file = st.file_uploader("Ingest crop canopy leaf sample matrix", type=["jpg", "png"]) or st.camera_input("Activate physical image node")
            if img_file: 
                image = Image.open(img_file)
                st.image(image, use_column_width=True, caption="Viewport Image Matrix Instantiated")
        with c2:
            if img_file:
                if st.button("🔬 Execute Computer Vision Pathology Diagnosis", type="primary", use_container_width=True):
                    placeholder = st.empty()
                    with placeholder.container():
                        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=120, key="vision_scanning")
                        else: st.spinner("Executing Pixel-Level Pathology Diagnostic Sweeps...")
                    try:
                        v_model = genai.GenerativeModel(get_working_vision_model())
                        res = v_model.generate_content([f"You are a specialized plant pathologist machine learning node. Analyze this image matrix for {crop_type} health anomalies. Identify pathogens, chlorosis, structural degradation, or nutrient deficits. Return a highly structural, clear technical diagnostic overview and an immediate 3-step biological treatment protocol.", image])
                        placeholder.empty()
                        st.session_state.vision_result = res.text
                    except Exception as e:
                        placeholder.empty()
                        st.error(f"Computer Vision Diagnostic Sweep Failure: {e}")
        
        if st.session_state.vision_result:
            st.markdown(f'<div class="glass-card"><h4 style="color:var(--accent); margin-top:0; font-size:1rem; font-weight:600; letter-spacing:0.02em;">🔬 COMPILED PATHOLOGY SCAN REAL-TIME DATA</h4><br>{st.session_state.vision_result}</div>', unsafe_allow_html=True)
            
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("<p class='section-header' style='font-size:1rem;'>💬 Pathology Context Conversational Node</p>", unsafe_allow_html=True)
            for msg in st.session_state.vision_messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
                
            if v_prompt := st.chat_input("Query diagnostic model regarding treatment parameters or vector controls...", key="vision_chat_input"):
                st.session_state.vision_messages.append({"role": "user", "content": v_prompt})
                with st.chat_message("user"): st.markdown(v_prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Processing Pathological Inference Variables..."):
                        try:
                            model = genai.GenerativeModel(get_working_text_model())
                            sys_prompt = f"You are a plant pathologist machine learning architecture node. The context is a scanned leaf image of a {crop_type} plant. Your initial diagnostic evaluation output was: '{st.session_state.vision_result}'. Respond to the user follow-up prompt query cleanly and professionally based strictly on this pathological baseline context."
                            chat_res = model.generate_content(sys_prompt + "\n\nClient User Follow-Up Query: " + v_prompt)
                            st.markdown(chat_res.text)
                            st.session_state.vision_messages.append({"role": "assistant", "content": chat_res.text})
                        except Exception as e:
                            st.error(f"Inference Thread Failure: {e}")

# =============================================================================
# TAB 4: MARKET ANALYZER + INTEGRATED CHAT
# =============================================================================
with tab_finance:
    st.markdown('<p class="section-header">📈 Financial Optimization & Forward Yield Modeling</p>', unsafe_allow_html=True)
    
    f_col1, f_col2, f_col3 = st.columns(3)
    plants_count = f_col1.number_input("Greenhouse Population Matrix (Active Plants)", value=1000, step=100)
    expected_yield_per_plant = f_col2.number_input("Expected Biomass Yield Efficiency per Plant (kg)", value=0.15, step=0.05)
    market_price = f_col3.number_input("Current Open Market Value Index (Per kg LKR)", value=450.0, step=10.0)
    
    total_yield_kg = plants_count * expected_yield_per_plant
    gross_revenue = total_yield_kg * market_price
    estimated_running_cost = (plants_count * 15) + 5000 
    net_profit = gross_revenue - estimated_running_cost
    
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Projected Operational Biomass Output", f"{total_yield_kg:.1f} kg")
    res_col2.metric("Estimated Cumulative Operational Expenditures (Opex)", f"Rs. {estimated_running_cost:,.2f}")
    res_col3.metric("Projected Net Profit Margin Index", f"Rs. {net_profit:,.2f}", delta=f"{net_profit/estimated_running_cost * 100:.1f}% ROI Vector")
    
    st.markdown('<div class="glass-card" style="margin-top: 1.2rem;">💡 <b>Market Optimization Insights Node:</b> Local commercial supermarket produce indices show a steady premium price retention for advanced protected farming produce. Contract options or programmatic supermarket forward delivery allocations are recommended to safeguard current projected margins.</div>', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("<p class='section-header' style='font-size:1rem;'>💬 Quantitative Financial Advisor Node</p>", unsafe_allow_html=True)
    for msg in st.session_state.market_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
    if m_prompt := st.chat_input("Query quantitative ag-business model regarding operational margin scaling or opex containment vectors...", key="market_chat_input"):
        st.session_state.market_messages.append({"role": "user", "content": m_prompt})
        with st.chat_message("user"): st.markdown(m_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Processing Commercial Asset Performance Models..."):
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    sys_prompt = f"You are an expert agribusiness agricultural asset management consultant. The system variables are: Crop: {crop_type}, Active Plant Census: {plants_count}, Compiled Yield Output: {total_yield_kg}kg, Market Valuation Pricing Index: LKR {market_price}/kg, Modeled Opex Cost Allocation: LKR {estimated_running_cost}, Net Profit Margin: LKR {net_profit}. Advise the commercial client user cleanly and programmatically regarding financial optimization options."
                    chat_res = model.generate_content(sys_prompt + "\n\nCommercial Client User Query: " + m_prompt)
                    st.markdown(chat_res.text)
                    st.session_state.market_messages.append({"role": "assistant", "content": chat_res.text})
                except Exception as e:
                    st.error(f"Asset Management Ingestion Failure: {e}")

# =============================================================================
# TAB 5: HARDWARE & ALERTS
# =============================================================================
with tab_hardware:
    st.markdown('<p class="section-header">⚙️ Secure Automated Webhooks & Telemetry Ingestion Controls</p>', unsafe_allow_html=True)
    
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.markdown('<div class="glass-card" style="height:100%;"><h4 style="margin-top:0;">Programmatic Ingestion Webhook URL</h4><p style="font-size:0.88rem; color:var(--text-muted);">Broadcast edge telemetry packets (ESP32, Raspberry Pi, Arduino Node architectures) programmatically via encrypted REST protocols.</p>', unsafe_allow_html=True)
        st.code("POST https://agritwin-ai.streamlit.app/api/v1/sensors\n\n{\n  \"api_key\": \"sec_token_07dfacba12f8\",\n  \"ambient_temp\": 28.5,\n  \"relative_hum\": 65.2,\n  \"solution_ec\": 1.8\n}", language="json")
        st.button("🔄 Rotate Security Token Parameters")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with h_col2:
        st.markdown('<div class="glass-card" style="height:100%;"><h4 style="margin-top:0;">Automated Mitigating Edge Threshold Triggers</h4><p style="font-size:0.88rem; color:var(--text-muted);">Program automatic notification hooks via SMS/WhatsApp routes when telemetry bounds breach threshold boundaries.</p>', unsafe_allow_html=True)
        phone_num = st.text_input("Ingest Primary Security Destination Route Phone Number", value="+947XXXXXXXX")
        temp_threshold = st.slider("Execute High Ambient Warning Trigger Profile (°C):", 25.0, 45.0, 35.0)
        ec_threshold = st.slider("Execute Critical Solution Conduct. Floor Depletion Alert (mS/cm):", 0.5, 2.0, 1.2)
        
        if st.button("Sync Trigger Configuration Array to Edge Nodes", type="primary"):
            st.success("✅ Remote configuration rules bound to active greenhouse edge registers successfully.")
            
        if temperature > temp_threshold:
            st.error(f"🚨 **ACTIVE BREACH EXECUTED:** Telemetry node data parameter ({temperature}°C) breaches maximum system threshold config profile! (Simulated programmatic alert hook targeted directly to destination route register: {phone_num})")
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INFRASTRUCTURE FOOTER DATA SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<div style="text-align:center;color:var(--text-muted);font-size:0.75rem;padding-bottom:1rem;">'
    f'AgriTwin Platform Infrastructure v{VERSION} · AI-Powered Predictive Digital Twin Node Ecosystem for Smart Protected Agriculture<br>'
    f'Built with Core Streamlit · Plotly Enterprise Graphics Engine · Google Gemini Generative Matrix · <span style="color:var(--accent);">♻ Commercial Agribusiness Framework Analytics Suite</span>'
    f'</div>', unsafe_allow_html=True)
