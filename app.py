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
/* ── Premium Inter Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-color:       #0B111A; /* Soft Dark Blue-Grey */
    --card-bg:        rgba(19, 31, 44, 0.65);
    --card-border:    rgba(255, 255, 255, 0.08);
    --accent:         #10B981; /* Emerald Green */
    --accent-hover:   #059669;
    --accent-light:   rgba(16, 185, 129, 0.15);
    --text-primary:   #F3F4F6;
    --text-secondary: #9CA3AF;
    --danger:         #EF4444;
    --warning:        #F59E0B;
}

/* ── Smooth Animated Background ── */
.stApp {
    background: radial-gradient(circle at 15% 50%, rgba(16, 185, 129, 0.05), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(59, 130, 246, 0.05), transparent 25%);
    background-color: var(--bg-color);
    background-attachment: fixed;
}

/* ── Typography & Globals ── */
body, .stMarkdown, .stText, p, span, label, div {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary);
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}

#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }

/* ── Sidebar Styling ── */
section[data-testid="stSidebar"] {
    background-color: rgba(13, 20, 28, 0.95) !important;
    border-right: 1px solid var(--card-border) !important;
    backdrop-filter: blur(20px);
}
.stSlider > div > div > div { background: var(--accent) !important; }
.stSelectbox div[data-baseweb], .stTextInput input, .stNumberInput input { 
    background: rgba(255,255,255,0.03) !important; 
    border: 1px solid var(--card-border) !important; 
    border-radius: 10px !important; 
    color: var(--text-primary) !important;
    transition: all 0.3s ease;
}
.stSelectbox div[data-baseweb]:hover, .stTextInput input:hover, .stNumberInput input:hover {
    border-color: var(--accent) !important;
}

/* ── Staggered Fade-Up Animation ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Premium Glass Cards ── */
.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    animation: fadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    margin-bottom: 1.2rem;
}
.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(16, 185, 129, 0.3);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
}

/* ── KPI Tiles ── */
.metric-tile {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    animation: fadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
    transition: all 0.3s ease;
}
.metric-tile:hover { 
    transform: translateY(-5px); 
    background: rgba(255, 255, 255, 0.03);
    border-color: var(--accent);
}
.metric-value { font-size: 2.2rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.03em; }
.metric-label { font-size: 0.75rem; font-weight: 500; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.05em; margin-top: 0.4rem; }

/* ── Badges & Buttons ── */
.risk-badge {
    display: inline-block; padding: 0.3rem 1rem; border-radius: 99px;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em;
}
.status-pill {
    display: inline-block; padding: 0.3rem 1rem; border-radius: 99px; 
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em;
}
.engine-pill {
    background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 99px; padding: 0.3rem 0.8rem; font-size: 0.75rem; color: #60A5FA !important;
}

/* ── AI Recommendation Cards ── */
.ai-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--card-border);
    border-radius: 16px; padding: 1.2rem;
    animation: fadeUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.ai-card h4 { font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; color: var(--accent) !important; text-transform: uppercase; margin: 0 0 0.6rem; }
.ai-card p { font-size: 0.95rem; line-height: 1.6; color: var(--text-primary); margin: 0; font-weight: 300;}

/* ── Progress Bars ── */
.progress-container { margin: 0.8rem 0; }
.progress-label { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.4rem; font-weight: 400;}
.progress-track { background: rgba(255,255,255,0.05); border-radius: 99px; height: 8px; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 99px; transition: width 1s ease-in-out; }

/* ── Section Headers ── */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: var(--text-primary) !important;
    letter-spacing: -0.01em; border-bottom: 1px solid var(--card-border); 
    padding-bottom: 0.6rem; margin-bottom: 1.2rem; margin-top: 1rem;
}

/* ── Streamlit Tabs Styling (Apple Segmented Control Style) ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary);
    font-weight: 500;
    border-radius: 8px;
    padding: 8px 16px;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background-color: var(--card-bg) !important;
    color: var(--accent) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-bottom: none !important;
}

/* ── Chat Messages ── */
.stChatMessage {
    background-color: transparent !important;
}
[data-testid="stChatMessageContent"] {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1rem;
}

/* ── Plotly Overrides ── */
.js-plotly-plot { border-radius: 16px !important; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS (LOTTIE & PLOTLY)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_ai_thinking = load_lottieurl("https://lottie.host/809c91f1-331e-4509-9fc6-9e90098da0da/uJ0q3E1jKz.json")

# Updated Plotly to use Inter font and softer grid lines
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", 
    plot_bgcolor="rgba(0,0,0,0)", 
    font=dict(family="Inter", color="#9CA3AF", size=12), 
    margin=dict(l=20, r=20, t=30, b=20), 
    showlegend=True, 
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11))
)

def plotly_dark_axes(fig):
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)")
    return fig

def progress_html(label: str, value: float, color: str = "#10B981", max_val: float = 100) -> str:
    pct = min(100, max(0, value / max_val * 100))
    return f'<div class="progress-container"><div class="progress-label"><span>{label}</span><span>{value:.1f}</span></div><div class="progress-track"><div class="progress-fill" style="width:{pct}%;background:{color};"></div></div></div>'

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="font-size: 1.8rem; font-weight: 700; color: #F3F4F6; margin-bottom: 0.2rem;">{APP_ICON} AgriTwin<span style="color: #10B981;">Pro</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 0.8rem; color: #9CA3AF; margin-bottom: 2rem;">Farm Management Platform v{VERSION}</div>', unsafe_allow_html=True)

    crop_type    = st.text_input("🌱 Crop Type", value="Lettuce", placeholder="e.g., Lettuce, Tomato...")
    growth_stage = st.selectbox("📈 Growth Stage", GROWTH_STAGES, index=1)

    st.markdown('<div class="section-header">🌡 Environment</div>', unsafe_allow_html=True)
    temperature      = st.slider("Temperature (°C)",     5.0,  50.0, PARAM_DEFAULTS["temperature"], 0.5)
    humidity         = st.slider("Humidity (%)",          10.0, 99.0, PARAM_DEFAULTS["humidity"],    0.5)
    co2_level        = st.slider("CO₂ (ppm)",            300.0,2000.0,PARAM_DEFAULTS["co2_level"],  10.0)

    st.markdown('<div class="section-header">💧 Substrate / Solution</div>', unsafe_allow_html=True)
    soil_moisture    = st.slider("Moisture (%)",    5.0,  99.0, PARAM_DEFAULTS["soil_moisture"], 0.5)
    ph_level         = st.slider("pH Level",             4.0,   9.0, PARAM_DEFAULTS["ph_level"],      0.1)
    ec_level         = st.slider("EC Level (mS/cm)",     0.5,   5.0, PARAM_DEFAULTS["ec_level"],      0.1)

    st.markdown('<div class="section-header">💡 Lighting & Air</div>', unsafe_allow_html=True)
    light_intensity  = st.slider("Light Intensity (lux)",100.0,6000.0,PARAM_DEFAULTS["light_intensity"],50.0)
    ventilation_rate = st.slider("Ventilation Rate (%)", 0.0,  100.0, PARAM_DEFAULTS["ventilation_rate"],1.0)
    leaf_area_index  = st.slider("Leaf Area Index",      0.5,   7.0,  PARAM_DEFAULTS["leaf_area_index"],  0.1)

    st.markdown("---")
    st.markdown('<div class="section-header">🧊 Digital Twin Model</div>', unsafe_allow_html=True)
    components.iframe("https://my.spline.design/miniroom-0b666a0d244958ceef967db0b537c376/", height=220)
    
    st.markdown("---")
    engine_status = get_ai_engine_status()
    st.markdown(f'<div style="text-align: center; margin-bottom: 1rem;"><span class="engine-pill">System Status: {engine_status}</span></div>', unsafe_allow_html=True)
    run_ai = st.button("Generate AI Insights", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE PARAMS & RUN ANALYTICS
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
        st.markdown("<div style='text-align:center; padding: 2rem;'><h3 style='color: #10B981; font-weight: 500;'>AI is compiling insights...</h3></div>", unsafe_allow_html=True)
        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=150, key="thinking")
        else: st.spinner("Processing...")
    st.session_state["ai_result"] = get_ai_recommendations(params, analysis)
    placeholder.empty()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div style="font-size: 2.2rem; font-weight: 600; letter-spacing: -0.03em;">Workspace Overview</div>', unsafe_allow_html=True)
with col_h2:
    now = datetime.datetime.now()
    st.markdown(f'<div style="text-align:right; padding-top: 0.5rem;"><div style="font-size: 0.85rem; color: #9CA3AF;">{now.strftime("%B %d, %Y • %H:%M")}</div><div style="font-size: 0.9rem; font-weight: 500; color: #10B981; margin-top: 0.2rem;">Currently Tracking: {crop_type}</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5 TABS SETUP 
# ─────────────────────────────────────────────────────────────────────────────
tab_dashboard, tab_chat, tab_vision, tab_finance, tab_hardware = st.tabs([
    "Dashboard", "AI Assistant", "Vision Scanner", "Market Analytics", "Hardware Controls"
])

# =============================================================================
# TAB 1: SYSTEM DASHBOARD
# =============================================================================
with tab_dashboard:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    def kpi(col, emoji, value, unit, label, color="#10B981"):
        col.markdown(f'<div class="metric-tile"><div style="font-size:1.6rem; margin-bottom: 0.5rem;">{emoji}</div><div class="metric-value" style="color:{color};">{value}{unit}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    kpi(k1, "🌡", f"{temperature:.1f}", "°C", "Temperature", "#EF4444" if temperature > 30 else "#10B981")
    kpi(k2, "💧", f"{humidity:.1f}", "%",  "Humidity", "#F59E0B" if humidity > 80 else "#3B82F6")
    kpi(k3, "🌱", f"{soil_moisture:.1f}", "%", "Moisture", "#EF4444" if soil_moisture < 30 else "#10B981")
    kpi(k4, "💨", f"{co2_level:.0f}", "", "CO₂ Level", "#F59E0B" if co2_level < 600 else "#10B981")
    kpi(k5, "☀️", f"{light_intensity:.0f}", "", "Light (lux)", "#FBBF24")
    kpi(k6, "🌬", f"{ventilation_rate:.0f}", "%", "Airflow", "#EF4444" if ventilation_rate < 40 else "#10B981")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    c_sus, c_dis, c_irr = st.columns([1.2, 1.2, 1.2])
    
    with c_sus:
        st.markdown('<div class="section-header">Sustainability Index</div>', unsafe_allow_html=True)
        gc = sus_data["grade_color"]
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=sus_data["total"], domain={'x': [0, 1], 'y': [0, 1]}, number={'font': {'color': gc, 'size': 40, 'family': 'Inter'}}, gauge={'axis': {'range': [None, 100], 'tickcolor': "rgba(255,255,255,0.1)"}, 'bar': {'color': gc, 'thickness': 0.25}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 0, 'steps': [{'range': [0, 40], 'color': "rgba(239, 68, 68, 0.15)"}, {'range': [40, 75], 'color': "rgba(245, 158, 11, 0.15)"}, {'range': [75, 100], 'color': "rgba(16, 185, 129, 0.15)"}]}))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(l=20, r=20, t=20, b=20))
        st.markdown(f'<div class="glass-card" style="text-align:center">', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div style="font-size:1.1rem; color:{gc}; font-weight:600; margin-top:-10px;">Grade {sus_data["grade"]}</div></div>', unsafe_allow_html=True)

        html_bars = ""
        for lbl, val, clr in [("Water Efficiency", sus_data["water_efficiency"], "#3B82F6"), ("Energy Efficiency", sus_data["energy_efficiency"], "#8B5CF6"), ("Climate Control", sus_data["climate_optimization"], "#10B981"), ("Disease Prevention", sus_data["disease_prevention"], "#34D399"), ("Yield Potential", sus_data["yield_potential"], "#FBBF24")]: html_bars += progress_html(lbl, val, clr)
        st.markdown(f'<div class="glass-card">{html_bars}</div>', unsafe_allow_html=True)

    with c_dis:
        st.markdown('<div class="section-header">Disease Risk Profile</div>', unsafe_allow_html=True)
        dlc = dis_data["level_color"]
        st.markdown(f'<div class="glass-card" style="text-align:center; padding: 2rem 1.5rem;"><span class="risk-badge" style="background:{dlc}15;color:{dlc};border:1px solid {dlc};">{dis_data["level"].upper()} RISK</span><br><div style="font-size:3rem; font-weight: 700; color:{dlc}; margin:0.5rem 0; line-height: 1;">{dis_data["overall"]:.1f}<span style="font-size:1.2rem;">%</span></div><div style="font-size:0.85rem;color:var(--text-secondary);">Cumulative Probability</div></div>', unsafe_allow_html=True)
        disease_html = ""
        d_colors = ["#EF4444", "#F59E0B", "#8B5CF6", "#3B82F6"]
        for i, (name, val) in enumerate(dis_data["diseases"].items()): disease_html += progress_html(name, val, d_colors[i % len(d_colors)])
        st.markdown(f'<div class="glass-card">{disease_html}</div>', unsafe_allow_html=True)

    with c_irr:
        st.markdown('<div class="section-header">Irrigation Protocol</div>', unsafe_allow_html=True)
        isc = irr_data["status_color"]
        st.markdown(f'<div class="glass-card" style="text-align:center; padding: 2rem 1.5rem;"><span class="status-pill" style="background:{isc}15;color:{isc};border:1px solid {isc};">STATUS: {irr_data["status"].upper()}</span><br><div style="font-size:3rem; font-weight: 700; color:{isc}; margin:0.5rem 0; line-height: 1;">{irr_data["urgency"]:.1f}<span style="font-size:1.2rem;">%</span></div><div style="font-size:0.85rem;color:var(--text-secondary);">Urgency Index</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card">' + progress_html("Urgency Level", irr_data["urgency"], isc) + f'<div style="margin-top:1.2rem;font-size:0.9rem; line-height: 1.8; color: var(--text-primary);">📦 Required Volume: <span style="float: right; font-weight: 600; color:{isc}">{irr_data["volume_liters"]:.2f} L/m²</span><br>⏰ Next Cycle In: <span style="float: right; font-weight: 600; color:{isc}">{irr_data["next_irrigation_hours"]} hours</span><br>🌿 Evapotranspiration: <span style="float: right; font-weight: 600; color:{isc}">{irr_data["evapotranspiration"]:.3f} mm/hr</span></div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.markdown('<div class="section-header">Microclimate Trends (24h)</div>', unsafe_allow_html=True)
        trend = generate_trend_data(temperature, humidity, 24)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend["times"], y=trend["temperatures"], name="Temp °C", line=dict(color="#EF4444", width=2), fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.1)"))
        fig_trend.add_trace(go.Scatter(x=trend["times"], y=trend["humidities"], name="Humidity %", line=dict(color="#3B82F6", width=2), fill="tozeroy", fillcolor="rgba(59, 130, 246, 0.1)"))
        fig_trend.update_layout(**PLOTLY_LAYOUT, height=280, xaxis=dict(tickangle=45, nticks=6))
        plotly_dark_axes(fig_trend)
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        st.markdown('<div class="section-header">Operational Risk Radar</div>', unsafe_allow_html=True)
        radar_cats = ["Disease", "Heat Stress", "Water Stress", "CO₂ Deficit", "Airflow", "Yield Risk"]
        vals = [dis_data["overall"], min(100, max(0, (heat_data["heat_index"] - 20) * 2)), max(0, 100 - sus_data["water_efficiency"]), max(0, (900 - co2_level) / 9), max(0, 100 - ventilation_rate), max(0, 100 - sus_data["yield_potential"])]
        fig_radar = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=radar_cats + [radar_cats[0]], fill="toself", fillcolor="rgba(239, 68, 68, 0.2)", line=dict(color="#EF4444", width=2)))
        fig_radar.update_layout(**PLOTLY_LAYOUT, height=280, polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=9, color="#9CA3AF"), showline=False), angularaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=10, color="#F3F4F6"))))
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    with ch3:
        st.markdown('<div class="section-header">Resource Allocation</div>', unsafe_allow_html=True)
        pie_labels = ["Water", "Energy", "Climate", "Health", "Yield"]
        pie_values = [sus_data["water_efficiency"], sus_data["energy_efficiency"], sus_data["climate_optimization"], sus_data["disease_prevention"], sus_data["yield_potential"]]
        pie_colors = ["#3B82F6", "#8B5CF6", "#10B981", "#34D399", "#FBBF24"]
        fig_pie = go.Figure(go.Pie(labels=pie_labels, values=pie_values, hole=0.6, marker=dict(colors=pie_colors, line=dict(color="var(--bg-color)", width=3)), textfont=dict(size=11)))
        fig_pie.add_annotation(text=f"<b>{sus_data['total']}</b>", x=0.5, y=0.5, font=dict(size=28, color="#F3F4F6", family="Inter"), showarrow=False)
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">AI Strategic Insights</div>', unsafe_allow_html=True)
    ai_res = st.session_state.get("ai_result")
    
    if not ai_res:
        st.markdown('<div class="glass-card" style="text-align:center;padding:3rem 1rem;"><div style="font-size:2rem; margin-bottom: 1rem;">✨</div><div style="font-size: 1.1rem; font-weight: 500; color: #F3F4F6;">AI Engine is standing by</div><div style="color: #9CA3AF; font-size: 0.9rem; margin-top: 0.5rem;">Click "Generate AI Insights" in the sidebar to process current telemetry.</div></div>', unsafe_allow_html=True)
    else:
        ai_c1, ai_c2 = st.columns(2)
        with ai_c1:
            st.markdown(f'<div class="ai-card"><h4>🛡️ Crop Health Directive</h4><p>{ai_res.get("disease_warning","—")}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-card"><h4>🌡 Climate Optimization</h4><p>{ai_res.get("climate_warning","—")}</p></div>', unsafe_allow_html=True)
        with ai_c2:
            st.markdown(f'<div class="ai-card"><h4>💧 Resource Management</h4><p>{ai_res.get("irrigation_advice","—")}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-card"><h4>♻️ Sustainability Target</h4><p>{ai_res.get("sustainability_insight","—")}</p></div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="glass-card" style="margin-top:0.5rem"><h4 style="font-size: 0.9rem; color: #10B981; margin-bottom: 0.8rem; text-transform: uppercase;">Executive Summary</h4><span style="font-size:0.95rem; line-height: 1.6;">{ai_res.get("overall_assessment","—")}</span></div>', unsafe_allow_html=True)
        
        report_text = f"=== AgriTwin AI Pro - Executive Report ===\nDate: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\nCrop: {crop_type} ({growth_stage})\n\n--- TELEMETRY ---\nTemp: {temperature}°C | Humidity: {humidity}% | Moisture: {soil_moisture}% | CO2: {co2_level}ppm\n\n--- STRATEGIC DIRECTIVES ---\nHealth: {ai_res.get('disease_warning')}\nClimate: {ai_res.get('climate_warning')}\nIrrigation: {ai_res.get('irrigation_advice')}\n\n--- SUMMARY ---\n{ai_res.get('overall_assessment')}\n"
        st.download_button(label="📥 Export Executive Report", data=report_text, file_name=f"AgriTwin_Report_{datetime.datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain", use_container_width=True)

# =============================================================================
# Helper Functions for Models
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

# =============================================================================
# TAB 2: MAIN AI ASSISTANT (CHATBOT)
# =============================================================================
with tab_chat:
    st.markdown('<div class="section-header">Agronomy Assistant</div>', unsafe_allow_html=True)
    if not API_KEY: st.warning("⚠️ API Key required. Configure in Streamlit Secrets.")
    else:
        if "main_messages" not in st.session_state: st.session_state.main_messages = []
        for msg in st.session_state.main_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        if prompt := st.chat_input(f"Type your question regarding {crop_type} cultivation...", key="main_chat"):
            st.session_state.main_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with placeholder.container():
                    if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=60, key="chat_main_thinking")
                    else: st.spinner("Analyzing...")
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    response = model.generate_content(f"You are an expert agronomy AI. The user is growing {crop_type} at {growth_stage}. Question: {prompt}")
                    placeholder.empty()
                    st.markdown(response.text)
                    st.session_state.main_messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    placeholder.empty()
                    st.error(f"Error: {e}")

# =============================================================================
# TAB 3: DISEASE VISION SCANNER + INTEGRATED CHAT
# =============================================================================
with tab_vision:
    st.markdown('<div class="section-header">Pathology Scanner</div>', unsafe_allow_html=True)
    if not API_KEY: st.warning("⚠️ API Key required.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            img_file = st.file_uploader("Upload leaf sample", type=["jpg", "png"]) or st.camera_input("Take photo")
            if img_file: 
                image = Image.open(img_file)
                st.image(image, use_column_width=True, caption="Sample Loaded")
        with c2:
            if img_file:
                if st.button("Run Diagnostic Scan", type="primary", use_container_width=True):
                    placeholder = st.empty()
                    with placeholder.container():
                        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=100, key="vision_scanning")
                        else: st.spinner("Processing visual data...")
                    try:
                        v_model = genai.GenerativeModel(get_working_vision_model())
                        res = v_model.generate_content([f"Identify diseases on this {crop_type}. Provide concise diagnosis and treatment.", image])
                        placeholder.empty()
                        st.session_state.vision_result = res.text
                    except Exception as e:
                        placeholder.empty()
                        st.error(f"Error: {e}")
        
        if st.session_state.vision_result:
            st.markdown(f'<div class="glass-card"><h4 style="color:#10B981; margin-top:0;">Diagnostic Report</h4>{st.session_state.vision_result}</div>', unsafe_allow_html=True)
            st.markdown("<div class='section-header' style='font-size: 1rem; margin-top: 2rem;'>Consult Pathologist</div>", unsafe_allow_html=True)
            for msg in st.session_state.vision_messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
                
            if v_prompt := st.chat_input("Ask for clarification or treatment details...", key="vision_chat_input"):
                st.session_state.vision_messages.append({"role": "user", "content": v_prompt})
                with st.chat_message("user"): st.markdown(v_prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Evaluating..."):
                        try:
                            model = genai.GenerativeModel(get_working_text_model())
                            sys_prompt = f"Act as a pathologist. Context: {crop_type} crop. Scan result: '{st.session_state.vision_result}'."
                            chat_res = model.generate_content(sys_prompt + " User asks: " + v_prompt)
                            st.markdown(chat_res.text)
                            st.session_state.vision_messages.append({"role": "assistant", "content": chat_res.text})
                        except Exception as e:
                            st.error(f"Error: {e}")

# =============================================================================
# TAB 4: MARKET ANALYZER + INTEGRATED CHAT
# =============================================================================
with tab_finance:
    st.markdown('<div class="section-header">Financial Modeling</div>', unsafe_allow_html=True)
    
    f_col1, f_col2, f_col3 = st.columns(3)
    plants_count = f_col1.number_input("Active Plants", value=1000, step=100)
    expected_yield_per_plant = f_col2.number_input("Yield/Plant (kg)", value=0.15, step=0.05)
    market_price = f_col3.number_input("Market Price (LKR/kg)", value=450.0, step=10.0)
    
    total_yield_kg = plants_count * expected_yield_per_plant
    gross_revenue = total_yield_kg * market_price
    estimated_running_cost = (plants_count * 15) + 5000 
    net_profit = gross_revenue - estimated_running_cost
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Projected Yield", f"{total_yield_kg:.1f} kg")
    res_col2.metric("Opex Estimate", f"LKR {estimated_running_cost:,.2f}")
    res_col3.metric("Net Profit Margin", f"LKR {net_profit:,.2f}", delta=f"{net_profit/estimated_running_cost * 100:.1f}% ROI")
    
    st.markdown('<div class="glass-card" style="margin-top: 1rem;">💡 <b>Market Intelligence:</b> Demand index for locally sourced greenhouse produce remains strong. Consider forward contracts to lock in current price margins.</div>', unsafe_allow_html=True)
    
    st.markdown("<div class='section-header' style='font-size: 1rem; margin-top: 2rem;'>Financial Advisor</div>", unsafe_allow_html=True)
    for msg in st.session_state.market_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
    if m_prompt := st.chat_input("Ask about ROI optimization...", key="market_chat_input"):
        st.session_state.market_messages.append({"role": "user", "content": m_prompt})
        with st.chat_message("user"): st.markdown(m_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing markets..."):
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    sys_prompt = f"Act as an Ag-Tech financial advisor. Crop: {crop_type}. Yield: {total_yield_kg}kg, Revenue: {gross_revenue}, Cost: {estimated_running_cost}."
                    chat_res = model.generate_content(sys_prompt + " Query: " + m_prompt)
                    st.markdown(chat_res.text)
                    st.session_state.market_messages.append({"role": "assistant", "content": chat_res.text})
                except Exception as e:
                    st.error(f"Error: {e}")

# =============================================================================
# TAB 5: HARDWARE & ALERTS
# =============================================================================
with tab_hardware:
    st.markdown('<div class="section-header">Device Integration & Webhooks</div>', unsafe_allow_html=True)
    
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.markdown('<div class="glass-card"><h4 style="margin-top:0;">API Telemetry Endpoint</h4><p style="font-size:0.9rem; color:var(--text-secondary);">Push sensor data via HTTP POST.</p>', unsafe_allow_html=True)
        st.code("POST https://api.agritwin.com/v1/ingestn{\n  \"device_id\": \"GH-Node-01\",\n  \"temp\": 24.5,\n  \"hum\": 68.0\n}", language="json")
        st.button("Generate Secure Token")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with h_col2:
        st.markdown('<div class="glass-card"><h4 style="margin-top:0;">Automated Triggers</h4><p style="font-size:0.9rem; color:var(--text-secondary);">Set bounds for emergency SMS/WhatsApp alerts.</p>', unsafe_allow_html=True)
        phone_num = st.text_input("Alert Number", value="+94 7X XXX XXXX")
        temp_threshold = st.slider("Max Temp Threshold (°C):", 25.0, 45.0, 32.0)
        
        if st.button("Deploy Rules", type="primary"):
            st.success("Configuration synced to edge nodes.")
            
        if temperature > temp_threshold:
            st.error(f"⚠️ THRESHOLD BREACH: Temperature ({temperature}°C) exceeds safety limit. Alert dispatched.")
        st.markdown('</div>', unsafe_allow_html=True)
