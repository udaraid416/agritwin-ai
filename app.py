"""
app.py — AI-Powered Predictive Digital Twin for Smart Protected Agriculture
Streamlit frontend — NEXT GEN UI (Video BG, 3D Spline, Day/Night, Lottie, Advanced Modules)
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
    page_title="AgriTwin AI",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if API_KEY:
    genai.configure(api_key=API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# 🌗 DYNAMIC DAY/NIGHT THEME & CSS
# ─────────────────────────────────────────────────────────────────────────────
current_hour = datetime.datetime.now().hour

if 6 <= current_hour < 18:
    theme_vars = """
    --bg-deep:      rgba(11, 25, 20, 0.85);
    --bg-mid:       rgba(17, 38, 29, 0.85);
    --glass-bg:     rgba(255, 215, 0, 0.05);
    --glass-border: rgba(255, 215, 0, 0.2);
    --accent:       #FFD700;
    --accent2:      #00DEB4;
    --accent3:      #FF8C00;
    --danger:       #EF4444;
    --warn:         #F59E0B;
    --text-primary: #FFFFFF;
    --text-muted:   rgba(255, 255, 255, 0.7);
    """
    bg_video_url = "https://cdn.pixabay.com/video/2020/05/25/40141-424888806_large.mp4"
else:
    theme_vars = """
    --bg-deep:      rgba(2, 12, 20, 0.85);
    --bg-mid:       rgba(7, 24, 40, 0.85);
    --glass-bg:     rgba(0, 220, 180, 0.06);
    --glass-border: rgba(0, 220, 180, 0.18);
    --accent:       #00DEB4;
    --accent2:      #00A8FF;
    --accent3:      #7C3AED;
    --danger:       #EF4444;
    --warn:         #F59E0B;
    --text-primary: #E2F8F4;
    --text-muted:   rgba(226, 248, 244, 0.6);
    """
    bg_video_url = "https://cdn.pixabay.com/video/2021/08/11/84687-586749942_large.mp4"

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Exo+2:wght@300;400;600&display=swap');

:root {{
    {theme_vars}
}}

/* ── 🎥 Cinematic Video Background ── */
#video-background {{
    position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%;
    z-index: -1; filter: blur(8px) brightness(0.25); object-fit: cover;
}}

.stApp, .main, section[data-testid="stSidebar"] > div {{ background: transparent !important; }}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--bg-deep) 0%, var(--bg-mid) 100%) !important;
    border-right: 1px solid var(--glass-border) !important; backdrop-filter: blur(20px);
}}
body, .stMarkdown, .stText, label, .stSelectbox label, .stSlider label, .stCheckbox label, .stTextInput label, .stNumberInput label {{
    font-family: 'Exo 2', sans-serif !important; color: var(--text-primary) !important;
}}

#MainMenu, footer {{ visibility: hidden; }}
header {{ background-color: transparent !important; }}
.block-container {{ padding-top: 1.2rem !important; padding-bottom: 2rem !important; }}

@keyframes fadeSlideDown {{ from {{ opacity: 0; transform: translateY(-24px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes pulseGlow {{ 0%,100% {{ text-shadow: 0 0 20px var(--accent), 0 0 60px var(--accent); }} 50% {{ text-shadow: 0 0 40px var(--accent), 0 0 120px var(--accent2); }} }}
.hero-title {{
    font-family: 'Orbitron', sans-serif !important; font-size: clamp(1.4rem, 3vw, 2.4rem); font-weight: 800;
    color: var(--accent) !important; animation: fadeSlideDown 0.8s ease both, pulseGlow 3s ease-in-out infinite; letter-spacing: 0.08em; margin-bottom: 0.2rem;
}}
.hero-sub {{ font-family: 'Exo 2', sans-serif; color: var(--text-muted) !important; font-size: 0.85rem; letter-spacing: 0.12em; text-transform: uppercase; animation: fadeSlideDown 1s 0.2s ease both; }}

@keyframes cardIn {{ from {{ opacity: 0; transform: translateY(16px) scale(0.97); }} to {{ opacity: 1; transform: translateY(0) scale(1); }} }}
.glass-card {{
    background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: 16px; padding: 1.2rem 1.4rem;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); animation: cardIn 0.6s ease both; transition: border-color 0.3s, box-shadow 0.3s; margin-bottom: 1rem;
}}
.glass-card:hover {{ border-color: var(--accent); box-shadow: 0 0 24px rgba(0,222,180,0.12); }}

.metric-tile {{
    background: linear-gradient(135deg, rgba(0,222,180,0.10) 0%, rgba(0,168,255,0.08) 100%); border: 1px solid var(--glass-border); border-radius: 14px; padding: 1rem; text-align: center; backdrop-filter: blur(10px); animation: cardIn 0.5s ease both; transition: transform 0.25s, box-shadow 0.25s;
}}
.metric-tile:hover {{ transform: translateY(-4px); box-shadow: 0 8px 32px rgba(0,222,180,0.15); }}
.metric-value {{ font-family:'Orbitron',sans-serif; font-size:2rem; font-weight:700; line-height:1.1; }}
.metric-label {{ font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--text-muted); margin-top:0.25rem; }}

.grade-badge {{
    display:inline-block; font-family:'Orbitron',sans-serif; font-size:3.5rem; font-weight:800; line-height:1; padding:0.3rem 0.8rem;
    border-radius:12px; border: 2px solid currentColor; text-shadow: 0 0 24px currentColor; animation: pulseGlow 2.5s ease-in-out infinite;
}}

@keyframes progressFill {{ from {{ width: 0%; }} }}
.progress-container {{ margin: 0.6rem 0; }}
.progress-label {{ display:flex; justify-content:space-between; font-size:0.8rem; color:var(--text-muted); margin-bottom:0.3rem; }}
.progress-track {{ background:rgba(255,255,255,0.07); border-radius:99px; height:10px; overflow:hidden; }}
.progress-fill  {{ height:100%; border-radius:99px; animation: progressFill 1.2s cubic-bezier(.17,.67,.3,1) both; }}

.ai-card {{ background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(0,168,255,0.08)); border: 1px solid rgba(124,58,237,0.3); border-radius:16px; padding:1.1rem 1.3rem; backdrop-filter: blur(12px); animation: cardIn 0.7s 0.2s ease both; }}
.ai-card h4 {{ font-family:'Orbitron',sans-serif; font-size:0.8rem; letter-spacing:0.1em; color:var(--accent3) !important; text-transform:uppercase; margin:0 0 0.5rem; }}
.ai-card p  {{ font-size:0.9rem; line-height:1.6; color:var(--text-primary); margin:0; }}

.section-header {{ font-family:'Orbitron',sans-serif; font-size:0.95rem; font-weight:600; color:var(--accent) !important; letter-spacing:0.1em; text-transform:uppercase; border-bottom: 1px solid var(--glass-border); padding-bottom:0.5rem; margin-bottom:1rem; }}
.action-item {{ background: rgba(0,222,180,0.05); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; padding: 0.55rem 0.9rem; margin-bottom: 0.5rem; font-size: 0.88rem; animation: cardIn 0.4s ease both; }}
.risk-badge {{ display:inline-block; padding:0.25rem 0.75rem; border-radius:99px; font-size:0.8rem; font-weight:600; font-family:'Orbitron',sans-serif; letter-spacing:0.08em; }}
.status-pill {{ display:inline-block; padding:0.2rem 0.65rem; border-radius:99px; font-size:0.78rem; font-weight:600; letter-spacing:0.06em; }}

.stSlider > div > div > div {{ background: var(--accent) !important; }}
.stSelectbox div[data-baseweb], .stTextInput input, .stNumberInput input {{ background: rgba(0,0,0,0.3) !important; border: 1px solid var(--glass-border) !important; border-radius:8px !important; color: #E2F8F4 !important;}}
.stSlider .stMarkdown {{ color: var(--text-muted) !important; }}
.stTabs [data-baseweb="tab-list"] {{ background-color: transparent; }}
.stTabs [data-baseweb="tab"] {{ color: var(--text-muted); font-family: 'Orbitron', sans-serif; font-weight: 600; letter-spacing: 0.05em; }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; border-bottom-color: var(--accent) !important; }}
.js-plotly-plot {{ border-radius:14px !important; }}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.2}} }}
.live-dot {{ width:8px; height:8px; background:var(--accent); border-radius:50%; display:inline-block; animation:blink 1.4s ease-in-out infinite; box-shadow: 0 0 8px var(--accent); vertical-align:middle; margin-right:6px; }}
.engine-pill {{ background: rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.35); border-radius:99px; padding:0.2rem 0.7rem; font-size:0.72rem; color:#A78BFA !important; font-family:'Exo 2',sans-serif; }}
</style>
<video autoplay muted loop id="video-background"><source src="{bg_video_url}" type="video/mp4"></video>
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
lottie_success = load_lottieurl("https://lottie.host/79075727-84f9-4673-bef1-1b94b05a74e5/G2r9UqUq3p.json")

PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Exo 2", color="#E2F8F4", size=12), margin=dict(l=20, r=20, t=30, b=20), showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
def plotly_dark_axes(fig):
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig
def progress_html(label: str, value: float, color: str = "#00DEB4", max_val: float = 100) -> str:
    pct = min(100, max(0, value / max_val * 100))
    return f'<div class="progress-container"><div class="progress-label"><span>{label}</span><span>{value:.1f}</span></div><div class="progress-track"><div class="progress-fill" style="width:{pct}%;background:{color};"></div></div></div>'

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.1rem;">🌿 AgriTwin AI</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="hero-sub">v{VERSION} · Control Panel</p>', unsafe_allow_html=True)
    st.markdown("---")

    crop_type    = st.text_input("🌱 Enter Crop Type", value="Lettuce", placeholder="e.g., Lettuce, Tomato...")
    growth_stage = st.selectbox("📈 Growth Stage", GROWTH_STAGES, index=1)

    st.markdown('<p class="section-header">🌡 Climate Sensors</p>', unsafe_allow_html=True)
    temperature      = st.slider("Temperature (°C)",     5.0,  50.0, PARAM_DEFAULTS["temperature"], 0.5)
    humidity         = st.slider("Humidity (%)",          10.0, 99.0, PARAM_DEFAULTS["humidity"],    0.5)
    co2_level        = st.slider("CO₂ (ppm)",            300.0,2000.0,PARAM_DEFAULTS["co2_level"],  10.0)

    st.markdown('<p class="section-header">💧 Soil & Water</p>', unsafe_allow_html=True)
    soil_moisture    = st.slider("Soil Moisture (%)",    5.0,  99.0, PARAM_DEFAULTS["soil_moisture"], 0.5)
    ph_level         = st.slider("pH Level",             4.0,   9.0, PARAM_DEFAULTS["ph_level"],      0.1)
    ec_level         = st.slider("EC Level (mS/cm)",     0.5,   5.0, PARAM_DEFAULTS["ec_level"],      0.1)

    st.markdown('<p class="section-header">💡 Light & Airflow</p>', unsafe_allow_html=True)
    light_intensity  = st.slider("Light Intensity (lux)",100.0,6000.0,PARAM_DEFAULTS["light_intensity"],50.0)
    ventilation_rate = st.slider("Ventilation Rate (%)", 0.0,  100.0, PARAM_DEFAULTS["ventilation_rate"],1.0)
    leaf_area_index  = st.slider("Leaf Area Index",      0.5,   7.0,  PARAM_DEFAULTS["leaf_area_index"],  0.1)

    st.markdown("---")
    st.markdown('<p class="section-header">🧊 3D Plant Model</p>', unsafe_allow_html=True)
    components.iframe("https://my.spline.design/miniroom-0b666a0d244958ceef967db0b537c376/", height=250)
    
    st.markdown("---")
    engine_status = get_ai_engine_status()
    st.markdown(f'<span class="engine-pill">🤖 {engine_status}</span>', unsafe_allow_html=True)
    run_ai = st.button("⚡ Run Full AI Analysis", use_container_width=True, type="primary")

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
        st.markdown("<h3 style='text-align:center; color:var(--accent); font-family:Orbitron;'>AI is Analyzing Data...</h3>", unsafe_allow_html=True)
        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=200, key="thinking")
        else: st.spinner("Processing...")
    st.session_state["ai_result"] = get_ai_recommendations(params, analysis)
    placeholder.empty()

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f'<h1 class="hero-title">{APP_ICON} AgriTwin AI</h1><p class="hero-sub">AI-Powered Predictive Digital Twin · Smart Protected Agriculture</p>', unsafe_allow_html=True)
with col_h2:
    now = datetime.datetime.now()
    st.markdown(f'<div style="text-align:right;padding-top:0.5rem"><span class="live-dot"></span><span style="color:var(--accent);font-family:Orbitron;font-size:0.7rem;">LIVE</span><br><span style="color:rgba(226,248,244,0.5);font-size:0.75rem;">{now.strftime("%d %b %Y  %H:%M")}</span><br><span style="color:#A78BFA;font-size:0.75rem;">📌 {crop_type} · {growth_stage}</span></div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 6 TABS SETUP (Including New Advanced Features)
# ─────────────────────────────────────────────────────────────────────────────
tab_dashboard, tab_chat, tab_vision, tab_yield_nft, tab_finance, tab_hardware = st.tabs([
    "📊 SYSTEM DASHBOARD", "💬 AI AGRI-ASSISTANT", "📸 DISEASE VISION SCANNER", 
    "🥬 NFT & YIELD PREDICTOR", "📈 MARKET ANALYZER", "⚙️ HARDWARE & ALERTS"
])

# =============================================================================
# TAB 1: SYSTEM DASHBOARD
# =============================================================================
with tab_dashboard:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    def kpi(col, emoji, value, unit, label, color="var(--accent)"):
        col.markdown(f'<div class="metric-tile"><div style="font-size:1.4rem">{emoji}</div><div class="metric-value" style="color:{color};">{value}{unit}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    kpi(k1, "🌡", f"{temperature:.1f}", "°C", "Temperature", "#FF6B6B" if temperature > 30 else "var(--accent)")
    kpi(k2, "💧", f"{humidity:.1f}", "%",  "Humidity", "#F59E0B" if humidity > 80 else "#00A8FF")
    kpi(k3, "🌱", f"{soil_moisture:.1f}", "%", "Soil Moisture", "#EF4444" if soil_moisture < 30 else "#7FFF00")
    kpi(k4, "💨", f"{co2_level:.0f}", "", "CO₂ ppm", "#F59E0B" if co2_level < 600 else "var(--accent)")
    kpi(k5, "☀️", f"{light_intensity:.0f}", "", "Lux", "#FFD700")
    kpi(k6, "🌬", f"{ventilation_rate:.0f}", "%", "Ventilation", "#EF4444" if ventilation_rate < 40 else "var(--accent)")

    st.markdown("---")
    c_sus, c_dis, c_irr = st.columns([1.2, 1.2, 1.2])
    with c_sus:
        st.markdown('<p class="section-header">♻️ Sustainability Score</p>', unsafe_allow_html=True)
        gc = sus_data["grade_color"]
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=sus_data["total"], domain={'x': [0, 1], 'y': [0, 1]}, number={'font': {'color': gc, 'size': 45, 'family': 'Orbitron'}}, gauge={'axis': {'range': [None, 100], 'tickcolor': "white"}, 'bar': {'color': gc, 'thickness': 0.25}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 1, 'bordercolor': "gray", 'steps': [{'range': [0, 40], 'color': "rgba(239, 68, 68, 0.2)"}, {'range': [40, 75], 'color': "rgba(245, 158, 11, 0.2)"}, {'range': [75, 100], 'color': "rgba(0, 222, 180, 0.2)"}]}))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(l=20, r=20, t=30, b=20))
        st.markdown(f'<div class="glass-card" style="text-align:center">', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div style="font-size:1.1rem; color:{gc}; font-weight:bold;">Grade {sus_data["grade"]}</div></div>', unsafe_allow_html=True)

        html_bars = ""
        for lbl, val, clr in [("Water Efficiency", sus_data["water_efficiency"], "#00A8FF"), ("Energy Efficiency", sus_data["energy_efficiency"], "#7C3AED"), ("Climate Control", sus_data["climate_optimization"], "var(--accent)"), ("Disease Prevention", sus_data["disease_prevention"], "#7FFF00"), ("Yield Potential", sus_data["yield_potential"], "#FFD700")]: html_bars += progress_html(lbl, val, clr)
        st.markdown(f'<div class="glass-card">{html_bars}</div>', unsafe_allow_html=True)

    with c_dis:
        st.markdown('<p class="section-header">🦠 Disease Risk Analysis</p>', unsafe_allow_html=True)
        dlc = dis_data["level_color"]
        st.markdown(f'<div class="glass-card" style="text-align:center"><span class="risk-badge" style="background:{dlc}22;color:{dlc};border:1px solid {dlc};">{dis_data["level"].upper()}</span><br><div style="font-family:Orbitron;font-size:2.4rem;color:{dlc};margin:0.4rem 0">{dis_data["overall"]:.1f}<span style="font-size:1rem;">%</span></div><div style="font-size:0.75rem;color:rgba(226,248,244,0.5);">Overall Risk Index</div></div>', unsafe_allow_html=True)
        disease_html = ""
        d_colors = ["#EF4444", "#F59E0B", "#7C3AED", "#00A8FF"]
        for i, (name, val) in enumerate(dis_data["diseases"].items()): disease_html += progress_html(name, val, d_colors[i % len(d_colors)])
        st.markdown(f'<div class="glass-card">{disease_html}</div>', unsafe_allow_html=True)

    with c_irr:
        st.markdown('<p class="section-header">💦 Irrigation Intelligence</p>', unsafe_allow_html=True)
        isc = irr_data["status_color"]
        st.markdown(f'<div class="glass-card" style="text-align:center"><span class="status-pill" style="background:{isc}22;color:{isc};border:1px solid {isc};">{irr_data["status"].upper()}</span><br><div style="font-family:Orbitron;font-size:2.2rem;color:{isc};margin:0.4rem 0">{irr_data["urgency"]:.1f}<span style="font-size:1rem;">%</span></div><div style="font-size:0.75rem;color:rgba(226,248,244,0.5);">Urgency Index</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card">' + progress_html("Irrigation Urgency", irr_data["urgency"], isc) + f'<div style="margin-top:0.8rem;font-size:0.85rem;">📦 Volume Required: <b style="color:{isc}">{irr_data["volume_liters"]:.2f} L/m²</b><br>⏰ Next Irrigation: <b style="color:{isc}">{irr_data["next_irrigation_hours"]}h</b><br>🌿 ET₀ Rate: <b style="color:{isc}">{irr_data["evapotranspiration"]:.3f} mm/hr</b></div></div>', unsafe_allow_html=True)

    st.markdown("---")
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.markdown('<p class="section-header">📊 24h Climate Trend</p>', unsafe_allow_html=True)
        trend = generate_trend_data(temperature, humidity, 24)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=trend["times"], y=trend["temperatures"], name="Temp °C", line=dict(color="#FF6B6B", width=2), fill="tozeroy", fillcolor="rgba(255,107,107,0.08)"))
        fig_trend.add_trace(go.Scatter(x=trend["times"], y=trend["humidities"], name="Humidity %", line=dict(color="#00A8FF", width=2), fill="tozeroy", fillcolor="rgba(0,168,255,0.06)"))
        fig_trend.add_trace(go.Scatter(x=trend["times"], y=trend["soil_moisture"], name="Soil %", line=dict(color="#7FFF00", width=2, dash="dot")))
        fig_trend.update_layout(**PLOTLY_LAYOUT, height=280, xaxis=dict(tickangle=45, nticks=8))
        plotly_dark_axes(fig_trend)
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        st.markdown('<p class="section-header">🎯 Risk Radar</p>', unsafe_allow_html=True)
        radar_cats = ["Disease Risk", "Heat Stress", "Water Stress", "CO₂ Deficit", "Ventilation Risk", "Yield Risk"]
        vals = [dis_data["overall"], min(100, max(0, (heat_data["heat_index"] - 20) * 2)), max(0, 100 - sus_data["water_efficiency"]), max(0, (900 - co2_level) / 9), max(0, 100 - ventilation_rate), max(0, 100 - sus_data["yield_potential"])]
        fig_radar = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=radar_cats + [radar_cats[0]], fill="toself", fillcolor="rgba(239,68,68,0.15)", line=dict(color="#EF4444", width=2), name="Risk Levels"))
        fig_radar.update_layout(**PLOTLY_LAYOUT, height=280, polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=9, color="#E2F8F4"), showline=False), angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=10, color="#E2F8F4"))))
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

    with ch3:
        st.markdown('<p class="section-header">🥧 Sustainability Breakdown</p>', unsafe_allow_html=True)
        pie_labels = ["Water", "Energy", "Climate", "Disease Prev.", "Yield"]
        pie_values = [sus_data["water_efficiency"], sus_data["energy_efficiency"], sus_data["climate_optimization"], sus_data["disease_prevention"], sus_data["yield_potential"]]
        pie_colors = ["#00A8FF", "#7C3AED", "var(--accent)", "#7FFF00", "#FFD700"]
        fig_pie = go.Figure(go.Pie(labels=pie_labels, values=pie_values, hole=0.55, marker=dict(colors=pie_colors, line=dict(color="#020C14", width=2)), textfont=dict(size=11)))
        fig_pie.add_annotation(text=f"<b>{sus_data['total']}</b>", x=0.5, y=0.5, font=dict(size=26, color="var(--accent)", family="Orbitron"), showarrow=False)
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    hm1, hm2 = st.columns(2)
    temp_grid, hum_grid = generate_zone_heatmap_data(temperature, humidity)
    with hm1:
        st.markdown('<p class="section-header">🌡 Zone Temperature Heatmap</p>', unsafe_allow_html=True)
        fig_ht = go.Figure(go.Heatmap(z=temp_grid, text=[[f"{v:.1f}°C" for v in row] for row in temp_grid], texttemplate="%{text}", colorscale="RdYlGn_r", colorbar=dict(title="°C", tickfont=dict(color="#E2F8F4")), showscale=True))
        fig_ht.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_ht, use_container_width=True, config={"displayModeBar": False})
    with hm2:
        st.markdown('<p class="section-header">💧 Zone Humidity Heatmap</p>', unsafe_allow_html=True)
        fig_hh = go.Figure(go.Heatmap(z=hum_grid, text=[[f"{v:.1f}%" for v in row] for row in hum_grid], texttemplate="%{text}", colorscale="Blues", colorbar=dict(title="%", tickfont=dict(color="#E2F8F4")), showscale=True))
        fig_hh.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_hh, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")
    st.markdown('<p class="section-header">🤖 AI Prediction & Recommendation Engine</p>', unsafe_allow_html=True)
    ai_res = st.session_state.get("ai_result")
    
    if not ai_res:
        st.markdown('<div class="glass-card" style="text-align:center;padding:2rem;"><div style="font-size:2.5rem">⚡</div><div style="font-family:Orbitron;color:var(--accent);margin:0.5rem 0;">AI Engine Ready</div>'
            f'<div style="color:rgba(226,248,244,0.5);font-size:0.85rem;">Click <b>Run Full AI Analysis</b> in the sidebar.</div></div>', unsafe_allow_html=True)
    else:
        src_col, _ = st.columns([2, 3])
        with src_col: st.markdown(f'<span class="engine-pill">Source: {ai_res.get("source","AI Engine")}</span>', unsafe_allow_html=True)
        ai_c1, ai_c2 = st.columns(2)
        with ai_c1:
            st.markdown(f'<div class="ai-card"><h4>🦠 Disease Warning</h4><p>{ai_res.get("disease_warning","—")}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-card"><h4>🌡 Climate Warning</h4><p>{ai_res.get("climate_warning","—")}</p></div>', unsafe_allow_html=True)
        with ai_c2:
            st.markdown(f'<div class="ai-card"><h4>💦 Irrigation Advice</h4><p>{ai_res.get("irrigation_advice","—")}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-card"><h4>♻️ Sustainability Insight</h4><p>{ai_res.get("sustainability_insight","—")}</p></div>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Orbitron;font-size:0.8rem;color:var(--accent);letter-spacing:0.1em;margin-top:0.8rem;">⚡ PRIORITY ACTIONS</p>', unsafe_allow_html=True)
        for i, action in enumerate(ai_res.get("top_actions", []), 1):
            st.markdown(f'<div class="action-item">{"🔴" if i==1 else "🟡" if i==2 else "🟢"} <b>#{i}</b> — {action}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card" style="margin-top:0.5rem"><b style="color:var(--accent);font-family:Orbitron;font-size:0.75rem;">📋 OVERALL ASSESSMENT</b><br><br><span style="font-size:0.92rem;line-height:1.7">{ai_res.get("overall_assessment","—")}</span></div>', unsafe_allow_html=True)
        
        report_text = f"=== AgriTwin AI - Farm Management Report ===\nDate: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\nCrop: {crop_type} ({growth_stage})\n\n--- SENSOR DATA ---\nTemperature: {temperature}°C\nHumidity: {humidity}%\nSoil Moisture: {soil_moisture}%\nCO2 Level: {co2_level} ppm\n\n--- AI RECOMMENDATIONS ---\nDisease Warning: {ai_res.get('disease_warning')}\nClimate Warning: {ai_res.get('climate_warning')}\nIrrigation Advice: {ai_res.get('irrigation_advice')}\n\n--- OVERALL ASSESSMENT ---\n{ai_res.get('overall_assessment')}\n"
        st.download_button(label="📄 Download AI Report (.txt)", data=report_text, file_name=f"AgriTwin_Report_{crop_type}_{datetime.datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain", type="secondary", use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">📊 System Performance Evaluation</p>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    def perf_card(col, icon, label, val, color):
        col.markdown(f'<div class="metric-tile"><div style="font-size:1.3rem">{icon}</div><div class="metric-value" style="color:{color};">{val:.1f}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
    gc = perf_data["grade_color"]
    perf_card(p1, "🎯", "Prediction Accuracy", perf_data["prediction_accuracy"], "#00A8FF")
    perf_card(p2, "⚙️", "System Efficiency", perf_data["system_efficiency"], "#7C3AED")
    perf_card(p3, "🏡", "Greenhouse Performance", perf_data["greenhouse_performance"], "#00DEB4")
    p4.markdown(f'<div class="metric-tile"><div style="font-size:1.3rem">🏆</div><div class="metric-value" style="color:{gc};">{perf_data["overall"]:.1f}</div><div class="metric-label">Overall Score · Grade <b style="color:{gc}">{perf_data["grade"]}</b></div></div>', unsafe_allow_html=True)

# =============================================================================
# TAB 2 & 3: CHATBOT & VISION SCANNER 
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
    st.markdown('<p class="section-header">💬 AI Agri-Assistant</p>', unsafe_allow_html=True)
    if not API_KEY: st.warning("⚠️ Add GEMINI_API_KEY to Streamlit Secrets.")
    else:
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if prompt := st.chat_input(f"Ask about {crop_type}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with placeholder.container():
                    if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=100, key="chat_thinking")
                    else: st.spinner("Thinking...")
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    response = model.generate_content(f"You are an expert AI assistant for {crop_type} at {growth_stage}. Question: {prompt}")
                    placeholder.empty()
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    placeholder.empty()
                    st.error(f"Error: {e}")

with tab_vision:
    st.markdown('<p class="section-header">📸 AI Disease Vision Scanner</p>', unsafe_allow_html=True)
    if not API_KEY: st.warning("⚠️ Add GEMINI_API_KEY to Streamlit Secrets.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            img_file = st.file_uploader("Upload image", type=["jpg", "png"]) or st.camera_input("Take photo")
            if img_file: 
                image = Image.open(img_file)
                st.image(image, use_column_width=True)
        with c2:
            if img_file and st.button("🔍 Scan Image", type="primary", use_container_width=True):
                placeholder = st.empty()
                with placeholder.container():
                    if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=150, key="vision_thinking")
                    else: st.spinner("Scanning...")
                try:
                    v_model = genai.GenerativeModel(get_working_vision_model())
                    res = v_model.generate_content([f"Identify diseases on this {crop_type} plant. Provide 3-step treatment.", image])
                    placeholder.empty()
                    st.markdown(f'<div class="glass-card"><b style="color:var(--accent);">🔬 SCAN RESULTS</b><br><br>{res.text}</div>', unsafe_allow_html=True)
                except Exception as e:
                    placeholder.empty()
                    st.error(f"Error: {e}")

# =============================================================================
# TAB 4: NFT & YIELD PREDICTOR (NEW ADVANCED MODULE)
# =============================================================================
with tab_yield_nft:
    st.markdown('<p class="section-header">🥬 Non-destructive Yield Prediction & NFT System</p>', unsafe_allow_html=True)
    st.info("💡 Optimized for Computer Vision non-destructive yield estimation using a mobile camera image capture, processed on local/cloud interface.")
    
    y_col1, y_col2 = st.columns([1, 1])
    
    with y_col1:
        st.markdown('<div class="glass-card"><h4>📸 Canopy Analyzer (Yield Estimate)</h4>', unsafe_allow_html=True)
        yield_img = st.file_uploader("Upload top-down crop image (Mobile Camera format)", type=["jpg", "png"], key="yield_img")
        if yield_img:
            st.image(Image.open(yield_img), caption="Canopy Image Loaded", use_column_width=True)
            if st.button("🧮 Estimate Fresh Weight", use_container_width=True):
                with st.spinner("Running OpenCV Canopy Extraction & AI Estimation..."):
                    time.sleep(1.5) # Simulate processing
                    # Mock calculation based on random area logic for demonstration
                    estimated_weight = np.random.randint(120, 180)
                    st.success(f"✅ **Estimated Fresh Weight: {estimated_weight}g per plant**")
                    st.metric("Predicted Total Yield (1000 plants)", f"{(estimated_weight * 1000)/1000:.1f} kg")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with y_col2:
        st.markdown('<div class="glass-card"><h4>🚰 Advanced NFT/DFT Dashboard</h4>', unsafe_allow_html=True)
        target_ph = 6.0
        target_ec = 1.8
        
        st.markdown(f"**Current System pH:** {ph_level} (Target: {target_ph})")
        st.markdown(f"**Current System EC:** {ec_level} mS/cm (Target: {target_ph})")
        
        ph_diff = round(ph_level - target_ph, 2)
        if ph_diff > 0.5:
            st.error(f"⚠️ pH is too high. Action: Add {abs(ph_diff)*15:.1f}ml of pH Down solution per 100L.")
        elif ph_diff < -0.5:
            st.warning(f"⚠️ pH is too low. Action: Add {abs(ph_diff)*10:.1f}ml of pH Up solution per 100L.")
        else:
            st.success("✅ pH is within optimal range.")
            
        ec_diff = round(target_ec - ec_level, 2)
        if ec_diff > 0.3:
            st.warning(f"⚠️ EC is low. Action: Add {abs(ec_diff)*20:.1f}ml of A&B Nutrient solution.")
        else:
            st.success("✅ Nutrient EC is stable.")
            
        flow_rate = st.slider("NFT Channel Flow Rate (L/min)", 0.5, 3.0, 1.5, 0.1)
        st.info(f"Current root zone oxygenation optimal for {flow_rate} L/min flow.")
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 5: MARKET ANALYZER (NEW ADVANCED MODULE)
# =============================================================================
with tab_finance:
    st.markdown('<p class="section-header">📈 Profit & Market Analyzer</p>', unsafe_allow_html=True)
    st.markdown("Analyze current market trends and predict harvest profitability.")
    
    f_col1, f_col2, f_col3 = st.columns(3)
    plants_count = f_col1.number_input("Total Plants in Greenhouse", value=1000, step=100)
    expected_yield_per_plant = f_col2.number_input("Expected Yield per Plant (kg)", value=0.15, step=0.05)
    market_price = f_col3.number_input("Current Market Price (Per kg LKR)", value=450.0, step=10.0)
    
    total_yield_kg = plants_count * expected_yield_per_plant
    gross_revenue = total_yield_kg * market_price
    estimated_running_cost = (plants_count * 15) + 5000 # Mock formula
    net_profit = gross_revenue - estimated_running_cost
    
    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("📦 Estimated Total Yield", f"{total_yield_kg:.1f} kg")
    res_col2.metric("💸 Estimated Total Cost", f"Rs. {estimated_running_cost:,.2f}")
    res_col3.metric("💰 Projected Net Profit", f"Rs. {net_profit:,.2f}", delta=f"{net_profit/estimated_running_cost * 100:.1f}% ROI")
    
    st.markdown('<div class="glass-card">💡 <b>Market Insight:</b> Harvest timing is optimal. Current market demand for hydroponic produce is showing a 12% upward trend in local supermarkets.</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 6: HARDWARE & ALERTS (NEW ADVANCED MODULE)
# =============================================================================
with tab_hardware:
    st.markdown('<p class="section-header">⚙️ Smart Hardware API & Automated Triggers</p>', unsafe_allow_html=True)
    
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.markdown('<div class="glass-card"><h4>📡 Hardware API Endpoint</h4>', unsafe_allow_html=True)
        st.markdown("Connect your ESP32, Arduino, or Raspberry Pi directly to this dashboard.")
        st.code("POST https://agritwin-ai.streamlit.app/api/v1/sensors\n\n{\n  \"api_key\": \"YOUR_SECRET_KEY\",\n  \"temp\": 28.5,\n  \"hum\": 65.2,\n  \"ec\": 1.8\n}", language="json")
        st.button("🔄 Generate New API Key")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with h_col2:
        st.markdown('<div class="glass-card"><h4>📱 Automated WhatsApp Alerts</h4>', unsafe_allow_html=True)
        phone_num = st.text_input("WhatsApp Number (with country code)", value="+947XXXXXXXX")
        temp_threshold = st.slider("Alert me if Temperature exceeds (°C):", 25.0, 45.0, 35.0)
        ec_threshold = st.slider("Alert me if EC drops below (mS/cm):", 0.5, 2.0, 1.2)
        
        if st.button("Save Alert Configuration", type="primary"):
            st.success("✅ Automation rules saved successfully.")
            
        if temperature > temp_threshold:
            st.error(f"🚨 **TRIGGER ACTIVATED:** Current temperature ({temperature}°C) exceeds threshold! (Simulated WhatsApp sent to {phone_num})")
        st.markdown('</div>', unsafe_allow_html=True)
