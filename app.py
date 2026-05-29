"""
app.py — AI-Powered Predictive Digital Twin for Smart Protected Agriculture
PREMIUM REDESIGN: Biopunk Dark Theme | Cinematic Animations | High-Tech Agri UI
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
    page_title="AgriTwin AI · Precision Farming",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if API_KEY:
    genai.configure(api_key=API_KEY)

# Session States
if "vision_result" not in st.session_state: st.session_state.vision_result = None
if "vision_messages" not in st.session_state: st.session_state.vision_messages = []
if "market_messages" not in st.session_state: st.session_state.market_messages = []

# ─────────────────────────────────────────────────────────────────────────────
# 🌗 DYNAMIC DAY/NIGHT THEME
# ─────────────────────────────────────────────────────────────────────────────
current_hour = datetime.datetime.now().hour
is_day = 6 <= current_hour < 18

if is_day:
    bg_video_url = "https://cdn.pixabay.com/video/2020/05/25/40141-424888806_large.mp4"
else:
    bg_video_url = "https://cdn.pixabay.com/video/2021/08/11/84687-586749942_large.mp4"

# ─────────────────────────────────────────────────────────────────────────────
# 🎨 PREMIUM BIOPUNK CSS — Full Redesign
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Syncopate:wght@400;700&family=Share+Tech+Mono&display=swap');

/* ══════════════════════════════════════════════
   ROOT VARIABLES — Biopunk Colour System
   ══════════════════════════════════════════════ */
:root {{
  --void:          #030B07;
  --deep:          #050F0A;
  --surface:       #071510;
  --surface2:      #0A1D14;
  --surface3:      #0D2419;

  --lime:          #A8FF3E;
  --lime-dim:      rgba(168,255,62,0.15);
  --lime-glow:     rgba(168,255,62,0.35);
  --teal:          #00FFB2;
  --teal-dim:      rgba(0,255,178,0.12);
  --amber:         #FFB830;
  --amber-dim:     rgba(255,184,48,0.12);
  --crimson:       #FF3E6C;
  --crimson-dim:   rgba(255,62,108,0.12);
  --ice:           #C8FFF4;

  --text-bright:   #E8FFF5;
  --text-mid:      rgba(200,255,244,0.65);
  --text-dim:      rgba(200,255,244,0.35);

  --border:        rgba(168,255,62,0.18);
  --border-bright: rgba(168,255,62,0.55);
  --glass:         rgba(10,29,20,0.75);
  --glass2:        rgba(5,15,10,0.85);
}}

/* ══════════════════════════════════════════════
   VIDEO BACKGROUND
   ══════════════════════════════════════════════ */
#video-background {{
  position: fixed; right: 0; bottom: 0;
  min-width: 100%; min-height: 100%;
  z-index: -2; object-fit: cover;
  filter: blur(12px) brightness(0.18) saturate(0.6) hue-rotate(120deg);
}}

/* Scanline overlay */
body::before {{
  content: '';
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  z-index: -1; pointer-events: none;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 3px,
    rgba(0,0,0,0.08) 3px,
    rgba(0,0,0,0.08) 4px
  );
}}

/* Vignette */
body::after {{
  content: '';
  position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  z-index: -1; pointer-events: none;
  background: radial-gradient(ellipse at center, transparent 55%, rgba(3,11,7,0.85) 100%);
}}

/* ══════════════════════════════════════════════
   BASE LAYOUT
   ══════════════════════════════════════════════ */
.stApp, .main {{ background: transparent !important; }}
#MainMenu, footer {{ visibility: hidden; }}
header {{ background-color: transparent !important; }}
.block-container {{ padding: 1.5rem 2rem 3rem !important; }}

/* ══════════════════════════════════════════════
   SIDEBAR — Biomech Panel
   ══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
  background: var(--glass2) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(24px) saturate(1.4);
}}
section[data-testid="stSidebar"] > div {{
  background: transparent !important;
}}
section[data-testid="stSidebar"]::before {{
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, transparent, var(--lime), var(--teal), transparent);
  animation: scanline-h 4s ease-in-out infinite;
}}

/* ══════════════════════════════════════════════
   TYPOGRAPHY
   ══════════════════════════════════════════════ */
body, .stMarkdown, .stText, label,
.stSelectbox label, .stSlider label,
.stCheckbox label, .stTextInput label, .stNumberInput label {{
  font-family: 'Rajdhani', sans-serif !important;
  color: var(--text-bright) !important;
}}

/* ══════════════════════════════════════════════
   KEYFRAME ANIMATIONS
   ══════════════════════════════════════════════ */
@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(20px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeLeft {{
  from {{ opacity: 0; transform: translateX(-20px); }}
  to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes scanline-h {{
  0%,100% {{ opacity: 0.6; transform: scaleX(0.3); }}
  50%      {{ opacity: 1;   transform: scaleX(1);   }}
}}
@keyframes limePulse {{
  0%,100% {{ text-shadow: 0 0 12px var(--lime), 0 0 30px var(--lime-glow); }}
  50%     {{ text-shadow: 0 0 25px var(--lime), 0 0 60px var(--lime-glow), 0 0 90px var(--lime-dim); }}
}}
@keyframes tealPulse {{
  0%,100% {{ box-shadow: 0 0 8px var(--teal), inset 0 0 8px rgba(0,255,178,0.06); }}
  50%     {{ box-shadow: 0 0 22px var(--teal), inset 0 0 16px rgba(0,255,178,0.12); }}
}}
@keyframes borderSpin {{
  0%   {{ border-color: var(--lime); }}
  33%  {{ border-color: var(--teal); }}
  66%  {{ border-color: var(--amber); }}
  100% {{ border-color: var(--lime); }}
}}
@keyframes blink {{
  0%,100% {{ opacity: 1; }}
  50%     {{ opacity: 0.15; }}
}}
@keyframes progressFill {{
  from {{ width: 0%; }}
}}
@keyframes cardReveal {{
  from {{ opacity: 0; transform: translateY(14px) scale(0.98); }}
  to   {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
@keyframes glitchText {{
  0%,100% {{ clip-path: inset(50% 0 30% 0); transform: translate(-2px,0); }}
  20%     {{ clip-path: inset(20% 0 60% 0); transform: translate(2px,0); }}
  40%     {{ clip-path: inset(80% 0 5% 0);  transform: translate(-1px,0); }}
  60%     {{ clip-path: inset(5% 0 80% 0);  transform: translate(1px,0); }}
  80%     {{ clip-path: inset(40% 0 43% 0); transform: translate(0,0); }}
}}
@keyframes dataFlow {{
  0%   {{ transform: translateY(-100%); opacity: 0; }}
  10%  {{ opacity: 1; }}
  90%  {{ opacity: 1; }}
  100% {{ transform: translateY(100%); opacity: 0; }}
}}
@keyframes hexPulse {{
  0%,100% {{ transform: scale(1); opacity: 0.7; }}
  50%     {{ transform: scale(1.06); opacity: 1; }}
}}
@keyframes cornerBlink {{
  0%,100% {{ opacity: 1; }}
  50%     {{ opacity: 0.3; }}
}}

/* ══════════════════════════════════════════════
   HERO TITLE
   ══════════════════════════════════════════════ */
.hero-wrap {{
  position: relative;
  padding: 0.5rem 0 1.2rem;
}}
.hero-title {{
  font-family: 'Syncopate', sans-serif !important;
  font-size: clamp(1.3rem, 2.8vw, 2.2rem);
  font-weight: 700;
  color: var(--lime) !important;
  letter-spacing: 0.15em;
  animation: fadeUp 0.7s ease both, limePulse 4s ease-in-out 0.7s infinite;
  margin: 0;
  line-height: 1.1;
}}
.hero-title-glitch {{
  position: absolute; top: 0.5rem; left: 0;
  font-family: 'Syncopate', sans-serif !important;
  font-size: clamp(1.3rem, 2.8vw, 2.2rem);
  font-weight: 700;
  color: var(--teal);
  letter-spacing: 0.15em;
  opacity: 0.4;
  animation: glitchText 6s ease-in-out 2s infinite;
  pointer-events: none;
  margin: 0; line-height: 1.1;
}}
.hero-sub {{
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.72rem;
  color: var(--text-mid) !important;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  animation: fadeUp 0.9s 0.2s ease both;
  margin-top: 0.4rem;
}}
.hero-accent-line {{
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, var(--lime), var(--teal) 40%, transparent);
  margin: 0.6rem 0;
  animation: fadeLeft 1s 0.4s ease both;
}}

/* ══════════════════════════════════════════════
   SECTION HEADERS
   ══════════════════════════════════════════════ */
.section-header {{
  font-family: 'Syncopate', sans-serif !important;
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--lime) !important;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.9rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
  position: relative;
}}
.section-header::after {{
  content: '';
  position: absolute; bottom: -1px; left: 0;
  width: 60px; height: 1px;
  background: var(--lime);
  animation: scanline-h 3s ease-in-out infinite;
}}

/* ══════════════════════════════════════════════
   GLASS CARD
   ══════════════════════════════════════════════ */
.glass-card {{
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1.2rem 1.4rem;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  animation: cardReveal 0.5s ease both;
  position: relative;
  overflow: hidden;
  margin-bottom: 1rem;
  transition: border-color 0.3s, box-shadow 0.3s;
}}
.glass-card::before {{
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--lime-glow), transparent);
}}
.glass-card:hover {{
  border-color: var(--border-bright);
  box-shadow: 0 0 30px var(--lime-dim), 0 8px 40px rgba(0,0,0,0.4);
}}

/* Corner decorations */
.glass-card::after {{
  content: '';
  position: absolute; bottom: 0; right: 0;
  width: 16px; height: 16px;
  border-bottom: 2px solid var(--lime);
  border-right: 2px solid var(--lime);
  animation: cornerBlink 2s ease-in-out infinite;
}}

/* ══════════════════════════════════════════════
   METRIC TILES
   ══════════════════════════════════════════════ */
.metric-tile {{
  background: linear-gradient(145deg, var(--surface2), var(--surface3));
  border: 1px solid var(--border);
  border-top: 2px solid var(--lime);
  border-radius: 4px;
  padding: 1.1rem 0.9rem;
  text-align: center;
  backdrop-filter: blur(16px);
  animation: cardReveal 0.5s ease both;
  transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
  position: relative;
  overflow: hidden;
}}
.metric-tile::before {{
  content: '';
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: linear-gradient(135deg, transparent 60%, rgba(168,255,62,0.04) 100%);
}}
.metric-tile:hover {{
  transform: translateY(-5px);
  box-shadow: 0 12px 40px rgba(168,255,62,0.18), 0 4px 20px rgba(0,0,0,0.5);
  border-top-color: var(--teal);
}}
.metric-value {{
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 1.9rem;
  font-weight: 400;
  line-height: 1.15;
}}
.metric-label {{
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-top: 0.3rem;
}}
.metric-emoji {{
  font-size: 1.2rem;
  margin-bottom: 0.3rem;
}}

/* ══════════════════════════════════════════════
   PROGRESS BARS
   ══════════════════════════════════════════════ */
.progress-container {{ margin: 0.55rem 0; }}
.progress-label {{
  display: flex; justify-content: space-between;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.72rem; color: var(--text-mid);
  margin-bottom: 0.25rem;
}}
.progress-track {{
  background: rgba(168,255,62,0.06);
  border: 1px solid rgba(168,255,62,0.12);
  border-radius: 2px; height: 8px; overflow: hidden;
  position: relative;
}}
.progress-fill {{
  height: 100%; border-radius: 2px;
  animation: progressFill 1.4s cubic-bezier(.17,.67,.3,1) both;
  position: relative;
}}
.progress-fill::after {{
  content: '';
  position: absolute; top: 0; right: 0;
  width: 4px; height: 100%;
  background: rgba(255,255,255,0.6);
  filter: blur(1px);
}}

/* ══════════════════════════════════════════════
   AI CARDS
   ══════════════════════════════════════════════ */
.ai-card {{
  background: linear-gradient(135deg, rgba(168,255,62,0.05), rgba(0,255,178,0.04));
  border: 1px solid rgba(168,255,62,0.25);
  border-left: 3px solid var(--lime);
  border-radius: 4px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.8rem;
  animation: cardReveal 0.5s ease both;
  position: relative;
  overflow: hidden;
  transition: border-left-color 0.3s, box-shadow 0.3s;
}}
.ai-card:hover {{
  border-left-color: var(--teal);
  box-shadow: 0 0 20px rgba(0,255,178,0.1);
}}
.ai-card::before {{
  content: '';
  position: absolute; top: 0; right: 0;
  width: 60px; height: 60px;
  background: radial-gradient(circle, rgba(168,255,62,0.06) 0%, transparent 70%);
}}
.ai-card h4 {{
  font-family: 'Syncopate', sans-serif !important;
  font-size: 0.62rem; letter-spacing: 0.2em;
  color: var(--lime) !important;
  text-transform: uppercase; margin: 0 0 0.5rem;
}}
.ai-card p {{
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.92rem; line-height: 1.65;
  color: var(--text-bright); margin: 0;
}}

/* ══════════════════════════════════════════════
   RISK / STATUS BADGES
   ══════════════════════════════════════════════ */
.risk-badge {{
  display: inline-block;
  font-family: 'Syncopate', sans-serif;
  font-size: 0.65rem; font-weight: 700;
  letter-spacing: 0.15em; padding: 0.3rem 0.9rem;
  border-radius: 2px;
  border: 1px solid currentColor;
  text-transform: uppercase;
}}
.status-pill {{
  display: inline-block;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.1em; padding: 0.25rem 0.7rem;
  border-radius: 2px;
  border: 1px solid currentColor;
}}

/* ══════════════════════════════════════════════
   ACTION ITEMS
   ══════════════════════════════════════════════ */
.action-item {{
  background: rgba(168,255,62,0.04);
  border-left: 3px solid var(--lime);
  border-radius: 0 3px 3px 0;
  padding: 0.6rem 1rem;
  margin-bottom: 0.5rem;
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.9rem;
  color: var(--text-bright);
  animation: fadeLeft 0.5s ease both;
  transition: background 0.2s;
}}
.action-item:hover {{
  background: rgba(168,255,62,0.08);
}}

/* ══════════════════════════════════════════════
   DATA TERMINAL BOX
   ══════════════════════════════════════════════ */
.terminal-box {{
  background: rgba(3,11,7,0.9);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1rem 1.2rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.8rem;
  color: var(--lime);
  position: relative;
  overflow: hidden;
}}
.terminal-box::before {{
  content: '// SYSTEM OUTPUT';
  font-size: 0.6rem; letter-spacing: 0.2em;
  color: var(--text-dim);
  display: block; margin-bottom: 0.5rem;
}}

/* ══════════════════════════════════════════════
   LIVE INDICATOR
   ══════════════════════════════════════════════ */
.live-dot {{
  width: 7px; height: 7px;
  background: var(--lime); border-radius: 50%;
  display: inline-block;
  animation: blink 1.2s ease-in-out infinite;
  box-shadow: 0 0 8px var(--lime), 0 0 16px var(--lime-glow);
  vertical-align: middle; margin-right: 5px;
}}

/* ══════════════════════════════════════════════
   GRADE BADGE
   ══════════════════════════════════════════════ */
.grade-badge {{
  display: inline-block;
  font-family: 'Syncopate', sans-serif;
  font-size: 3rem; font-weight: 700;
  padding: 0.3rem 0.8rem;
  border: 2px solid currentColor;
  border-radius: 4px;
  text-shadow: 0 0 20px currentColor;
  animation: tealPulse 2.5s ease-in-out infinite;
}}

/* ══════════════════════════════════════════════
   ENGINE PILL
   ══════════════════════════════════════════════ */
.engine-pill {{
  background: rgba(168,255,62,0.08);
  border: 1px solid rgba(168,255,62,0.3);
  border-radius: 2px;
  padding: 0.2rem 0.8rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.7rem;
  color: var(--lime) !important;
  letter-spacing: 0.1em;
}}

/* ══════════════════════════════════════════════
   HEX DATA DECORATIONS
   ══════════════════════════════════════════════ */
.hex-deco {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.6rem;
  color: var(--text-dim);
  letter-spacing: 0.1em;
}}

/* ══════════════════════════════════════════════
   STREAMLIT OVERRIDES
   ══════════════════════════════════════════════ */
.stSlider > div > div > div {{
  background: var(--lime) !important;
}}
.stSlider > div > div > div > div {{
  background: var(--surface) !important;
  border: 2px solid var(--lime) !important;
}}
.stSelectbox div[data-baseweb],
.stTextInput input,
.stNumberInput input {{
  background: rgba(3,11,7,0.8) !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  color: var(--text-bright) !important;
  font-family: 'Rajdhani', sans-serif !important;
}}
.stButton > button {{
  font-family: 'Syncopate', sans-serif !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  border-radius: 3px !important;
  transition: all 0.2s !important;
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(135deg, var(--lime), var(--teal)) !important;
  color: var(--void) !important;
  border: none !important;
  font-weight: 700 !important;
}}
.stButton > button:not([kind="primary"]) {{
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-bright) !important;
}}
.stButton > button:hover {{
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px var(--lime-dim) !important;
}}
.stTabs [data-baseweb="tab-list"] {{
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0.2rem;
}}
.stTabs [data-baseweb="tab"] {{
  font-family: 'Syncopate', sans-serif !important;
  font-size: 0.62rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.12em !important;
  color: var(--text-dim) !important;
  background: transparent !important;
  border-radius: 3px 3px 0 0 !important;
  padding: 0.7rem 1rem !important;
  transition: all 0.2s !important;
}}
.stTabs [aria-selected="true"] {{
  color: var(--lime) !important;
  background: var(--lime-dim) !important;
  border-bottom: 2px solid var(--lime) !important;
}}
.js-plotly-plot {{ border-radius: 4px !important; }}
.stDownloadButton > button {{
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 0.82rem !important;
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-mid) !important;
  border-radius: 3px !important;
}}
.stMarkdown hr {{ border-color: var(--border) !important; }}

/* ══════════════════════════════════════════════
   SIDEBAR SECTION HEADERS
   ══════════════════════════════════════════════ */
.sidebar-section {{
  font-family: 'Syncopate', sans-serif;
  font-size: 0.6rem; font-weight: 700;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--lime) !important;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.3rem;
  margin: 0.8rem 0 0.6rem;
}}
.sidebar-brand {{
  font-family: 'Syncopate', sans-serif;
  font-size: 1rem; font-weight: 700;
  color: var(--lime) !important;
  letter-spacing: 0.15em;
  animation: limePulse 4s ease-in-out infinite;
}}
.sidebar-ver {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem;
  color: var(--text-dim) !important;
  letter-spacing: 0.15em;
}}

/* Status number big display */
.big-num {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 2.6rem;
  line-height: 1;
  font-weight: 400;
}}

/* Divider with label */
.div-label {{
  display: flex; align-items: center; gap: 0.8rem;
  margin: 1.2rem 0;
}}
.div-label span {{
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem; letter-spacing: 0.25em;
  color: var(--text-dim); white-space: nowrap;
}}
.div-label::before, .div-label::after {{
  content: ''; flex: 1;
  height: 1px; background: var(--border);
}}

</style>
<video autoplay muted loop id="video-background">
  <source src="{bg_video_url}" type="video/mp4">
</video>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

lottie_ai_thinking = load_lottieurl("https://lottie.host/809c91f1-331e-4509-9fc6-9e90098da0da/uJ0q3E1jKz.json")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Rajdhani", color="#C8FFF4", size=12),
    margin=dict(l=16, r=16, t=28, b=16),
    showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11), bordercolor="rgba(168,255,62,0.15)", borderwidth=1)
)

def plotly_dark_axes(fig):
    fig.update_xaxes(gridcolor="rgba(168,255,62,0.07)", zerolinecolor="rgba(168,255,62,0.15)", tickfont=dict(family="Share Tech Mono", size=10))
    fig.update_yaxes(gridcolor="rgba(168,255,62,0.07)", zerolinecolor="rgba(168,255,62,0.15)", tickfont=dict(family="Share Tech Mono", size=10))
    return fig

def progress_html(label: str, value: float, color: str = "#A8FF3E", max_val: float = 100) -> str:
    pct = min(100, max(0, value / max_val * 100))
    return f'''<div class="progress-container">
        <div class="progress-label"><span>{label}</span><span style="color:{color}">{value:.1f}</span></div>
        <div class="progress-track">
            <div class="progress-fill" style="width:{pct}%;background:linear-gradient(90deg,{color}AA,{color});"></div>
        </div>
    </div>'''

def kpi_tile(col, emoji, value, unit, label, color):
    col.markdown(f'''<div class="metric-tile">
        <div class="metric-emoji">{emoji}</div>
        <div class="metric-value" style="color:{color};">{value}{unit}</div>
        <div class="metric-label">{label}</div>
    </div>''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''
        <div class="hex-deco" style="margin-bottom:0.2rem;">0xAGRI_TWIN_v3</div>
        <div class="sidebar-brand">🌿 AgriTwin</div>
        <div class="sidebar-ver">AI · Digital Twin Platform</div>
    ''', unsafe_allow_html=True)
    st.markdown("---")

    crop_type    = st.text_input("🌱 Crop Type", value="Lettuce", placeholder="Lettuce, Tomato…")
    growth_stage = st.selectbox("📈 Growth Stage", GROWTH_STAGES, index=1)

    st.markdown('<div class="sidebar-section">▸ Climate Sensors</div>', unsafe_allow_html=True)
    temperature      = st.slider("Temperature (°C)",       5.0, 50.0,  PARAM_DEFAULTS["temperature"],     0.5)
    humidity         = st.slider("Humidity (%)",           10.0, 99.0, PARAM_DEFAULTS["humidity"],         0.5)
    co2_level        = st.slider("CO₂ (ppm)",             300.0,2000.0,PARAM_DEFAULTS["co2_level"],       10.0)

    st.markdown('<div class="sidebar-section">▸ Soil & Water</div>', unsafe_allow_html=True)
    soil_moisture    = st.slider("Soil Moisture (%)",       5.0, 99.0, PARAM_DEFAULTS["soil_moisture"],    0.5)
    ph_level         = st.slider("pH Level",                4.0,  9.0, PARAM_DEFAULTS["ph_level"],         0.1)
    ec_level         = st.slider("EC Level (mS/cm)",        0.5,  5.0, PARAM_DEFAULTS["ec_level"],         0.1)

    st.markdown('<div class="sidebar-section">▸ Light & Airflow</div>', unsafe_allow_html=True)
    light_intensity  = st.slider("Light Intensity (lux)", 100.0,6000.0,PARAM_DEFAULTS["light_intensity"], 50.0)
    ventilation_rate = st.slider("Ventilation (%)",         0.0, 100.0,PARAM_DEFAULTS["ventilation_rate"],  1.0)
    leaf_area_index  = st.slider("Leaf Area Index",         0.5,  7.0, PARAM_DEFAULTS["leaf_area_index"],   0.1)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">▸ 3D Plant Model</div>', unsafe_allow_html=True)
    components.iframe("https://my.spline.design/miniroom-0b666a0d244958ceef967db0b537c376/", height=240)

    st.markdown("---")
    engine_status = get_ai_engine_status()
    st.markdown(f'<span class="engine-pill">⬡ {engine_status}</span>', unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    run_ai = st.button("⚡ Run Full AI Analysis", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# PARAMS & ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
params = {
    "crop_type": crop_type, "growth_stage": growth_stage, "temperature": temperature,
    "humidity": humidity, "soil_moisture": soil_moisture, "co2_level": co2_level,
    "light_intensity": light_intensity, "ventilation_rate": ventilation_rate,
    "ph_level": ph_level, "ec_level": ec_level, "leaf_area_index": leaf_area_index,
}
sus_data  = compute_sustainability_score(params)
dis_data  = calculate_disease_risk(params)
irr_data  = calculate_irrigation_need(params)
heat_data = calculate_heat_stress(temperature, humidity)
analysis  = {"sustainability": sus_data, "disease": dis_data, "irrigation": irr_data, "heat_stress": heat_data}
perf_data = evaluate_system_performance(sus_data, dis_data, irr_data)
analysis["performance"] = perf_data

if "ai_result" not in st.session_state: st.session_state["ai_result"] = None

if run_ai:
    ph = st.empty()
    with ph.container():
        st.markdown("<h3 style='text-align:center;color:#A8FF3E;font-family:Syncopate,sans-serif;font-size:0.9rem;letter-spacing:0.2em;'>NEURAL NETWORK PROCESSING…</h3>", unsafe_allow_html=True)
        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=180, key="thinking")
        else: st.spinner("Processing…")
    st.session_state["ai_result"] = get_ai_recommendations(params, analysis)
    ph.empty()

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([3, 1])
with hc1:
    now = datetime.datetime.now()
    st.markdown(f'''<div class="hero-wrap">
        <div class="hero-title">{APP_ICON} AgriTwin AI</div>
        <div class="hero-title-glitch">{APP_ICON} AgriTwin AI</div>
        <div class="hero-sub">AI-Powered Predictive Digital Twin · Smart Protected Agriculture</div>
        <div class="hero-accent-line"></div>
    </div>''', unsafe_allow_html=True)
with hc2:
    st.markdown(f'''<div style="text-align:right;padding-top:0.8rem">
        <span class="live-dot"></span><span style="color:#A8FF3E;font-family:Syncopate,sans-serif;font-size:0.65rem;letter-spacing:0.18em;">LIVE</span><br>
        <span style="color:var(--text-mid);font-family:Share Tech Mono,monospace;font-size:0.72rem;">{now.strftime("%d %b %Y  %H:%M:%S")}</span><br>
        <span style="color:#00FFB2;font-family:Share Tech Mono,monospace;font-size:0.7rem;">⬡ {crop_type} · {growth_stage}</span>
    </div>''', unsafe_allow_html=True)

st.markdown('<div class="div-label"><span>SYSTEM STATUS</span></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_dashboard, tab_chat, tab_vision, tab_finance, tab_hardware = st.tabs([
    "◈ SYSTEM DASHBOARD", "◈ AI ASSISTANT", "◈ VISION SCANNER",
    "◈ MARKET ANALYZER", "◈ HARDWARE"
])

# =============================================================================
# TAB 1 — DASHBOARD
# =============================================================================
with tab_dashboard:

    # ── KPI Row ──────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpi_tile(k1,"🌡",f"{temperature:.1f}","°C","Temperature","#FF5C7A" if temperature>30 else "#A8FF3E")
    kpi_tile(k2,"💧",f"{humidity:.1f}","%","Humidity","#FFB830" if humidity>80 else "#00FFB2")
    kpi_tile(k3,"🌱",f"{soil_moisture:.1f}","%","Soil Moisture","#FF5C7A" if soil_moisture<30 else "#A8FF3E")
    kpi_tile(k4,"💨",f"{co2_level:.0f}","","CO₂ ppm","#FFB830" if co2_level<600 else "#00FFB2")
    kpi_tile(k5,"☀️",f"{light_intensity:.0f}","","Lux","#FFB830")
    kpi_tile(k6,"🌬",f"{ventilation_rate:.0f}","%","Ventilation","#FF5C7A" if ventilation_rate<40 else "#A8FF3E")

    st.markdown('<div class="div-label"><span>ANALYTICS</span></div>', unsafe_allow_html=True)

    # ── Sustainability | Disease | Irrigation ─────────────────────────────────
    c_sus, c_dis, c_irr = st.columns(3)

    with c_sus:
        st.markdown('<div class="section-header">♻ Sustainability Score</div>', unsafe_allow_html=True)
        gc = sus_data["grade_color"]
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=sus_data["total"],
            domain={'x':[0,1],'y':[0,1]},
            number={'font':{'color':gc,'size':44,'family':'Share Tech Mono'}},
            gauge={
                'axis':{'range':[None,100],'tickcolor':"rgba(200,255,244,0.4)",'tickfont':{'family':'Share Tech Mono','size':9}},
                'bar':{'color':gc,'thickness':0.22},
                'bgcolor':"rgba(0,0,0,0)",
                'borderwidth':1,'bordercolor':"rgba(168,255,62,0.2)",
                'steps':[
                    {'range':[0,40],'color':"rgba(255,62,108,0.12)"},
                    {'range':[40,75],'color':"rgba(255,184,48,0.12)"},
                    {'range':[75,100],'color':"rgba(168,255,62,0.12)"}
                ],
                'threshold':{'line':{'color':gc,'width':2},'thickness':0.75,'value':sus_data["total"]}
            }
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=200, margin=dict(l=20,r=20,t=20,b=10))
        st.markdown(f'<div class="glass-card" style="text-align:center;padding-bottom:0.5rem">', unsafe_allow_html=True)
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div style="font-family:Syncopate,sans-serif;font-size:0.7rem;color:{gc};letter-spacing:0.2em;">GRADE {sus_data["grade"]}</div></div>', unsafe_allow_html=True)

        html_bars = ""
        for lbl, val, clr in [
            ("Water Efficiency", sus_data["water_efficiency"], "#00FFB2"),
            ("Energy Efficiency", sus_data["energy_efficiency"], "#A8FF3E"),
            ("Climate Control",   sus_data["climate_optimization"], "#FFB830"),
            ("Disease Prevention",sus_data["disease_prevention"], "#00B8FF"),
            ("Yield Potential",   sus_data["yield_potential"], "#FF5C7A")
        ]: html_bars += progress_html(lbl, val, clr)
        st.markdown(f'<div class="glass-card">{html_bars}</div>', unsafe_allow_html=True)

    with c_dis:
        st.markdown('<div class="section-header">🦠 Disease Risk Analysis</div>', unsafe_allow_html=True)
        dlc = dis_data["level_color"]
        st.markdown(f'''<div class="glass-card" style="text-align:center">
            <span class="risk-badge" style="color:{dlc};border-color:{dlc};background:rgba(255,62,108,0.08)">{dis_data["level"].upper()}</span>
            <div class="big-num" style="color:{dlc};margin:0.5rem 0">{dis_data["overall"]:.1f}<span style="font-size:1rem">%</span></div>
            <div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:var(--text-dim);letter-spacing:0.2em">OVERALL RISK INDEX</div>
        </div>''', unsafe_allow_html=True)
        d_clrs = ["#FF5C7A","#FFB830","#A8FF3E","#00FFB2"]
        dh = "".join(progress_html(n,v,d_clrs[i%4]) for i,(n,v) in enumerate(dis_data["diseases"].items()))
        st.markdown(f'<div class="glass-card">{dh}</div>', unsafe_allow_html=True)

    with c_irr:
        st.markdown('<div class="section-header">💦 Irrigation Intelligence</div>', unsafe_allow_html=True)
        isc = irr_data["status_color"]
        st.markdown(f'''<div class="glass-card" style="text-align:center">
            <span class="status-pill" style="color:{isc};border-color:{isc}">{irr_data["status"].upper()}</span>
            <div class="big-num" style="color:{isc};margin:0.5rem 0">{irr_data["urgency"]:.1f}<span style="font-size:1rem">%</span></div>
            <div style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:var(--text-dim);letter-spacing:0.2em">URGENCY INDEX</div>
        </div>''', unsafe_allow_html=True)
        st.markdown(f'''<div class="glass-card">
            {progress_html("Irrigation Urgency", irr_data["urgency"], isc)}
            <div style="margin-top:0.9rem;font-family:Share Tech Mono,monospace;font-size:0.75rem;line-height:2">
                <span style="color:var(--text-dim)">VOLUME :</span> <b style="color:{isc}">{irr_data["volume_liters"]:.2f} L/m²</b><br>
                <span style="color:var(--text-dim)">NEXT   :</span> <b style="color:{isc}">{irr_data["next_irrigation_hours"]}h</b><br>
                <span style="color:var(--text-dim)">ET₀    :</span> <b style="color:{isc}">{irr_data["evapotranspiration"]:.3f} mm/hr</b>
            </div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<div class="div-label"><span>VISUAL ANALYTICS</span></div>', unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.markdown('<div class="section-header">📊 24h Climate Trend</div>', unsafe_allow_html=True)
        trend = generate_trend_data(temperature, humidity, 24)
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=trend["times"], y=trend["temperatures"], name="Temp °C",
            line=dict(color="#FF5C7A",width=2), fill="tozeroy", fillcolor="rgba(255,92,122,0.06)"))
        fig_t.add_trace(go.Scatter(x=trend["times"], y=trend["humidities"], name="Humidity %",
            line=dict(color="#00FFB2",width=2), fill="tozeroy", fillcolor="rgba(0,255,178,0.05)"))
        fig_t.add_trace(go.Scatter(x=trend["times"], y=trend["soil_moisture"], name="Soil %",
            line=dict(color="#A8FF3E",width=2,dash="dot")))
        fig_t.update_layout(**PLOTLY_LAYOUT, height=280, xaxis=dict(tickangle=45,nticks=8))
        plotly_dark_axes(fig_t)
        st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar":False})

    with ch2:
        st.markdown('<div class="section-header">🎯 Risk Radar</div>', unsafe_allow_html=True)
        rc = ["Disease Risk","Heat Stress","Water Stress","CO₂ Deficit","Ventilation","Yield Risk"]
        rv = [dis_data["overall"],
              min(100,max(0,(heat_data["heat_index"]-20)*2)),
              max(0,100-sus_data["water_efficiency"]),
              max(0,(900-co2_level)/9),
              max(0,100-ventilation_rate),
              max(0,100-sus_data["yield_potential"])]
        fig_r = go.Figure(go.Scatterpolar(
            r=rv+[rv[0]], theta=rc+[rc[0]],
            fill="toself", fillcolor="rgba(255,92,122,0.12)",
            line=dict(color="#FF5C7A",width=2), name="Risk"))
        fig_r.add_trace(go.Scatterpolar(
            r=[50]*6+[50], theta=rc+[rc[0]],
            line=dict(color="rgba(168,255,62,0.2)",width=1,dash="dot"),
            name="Benchmark"))
        fig_r.update_layout(**PLOTLY_LAYOUT, height=280,
            polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0,100],gridcolor="rgba(168,255,62,0.08)",
                    tickfont=dict(size=9,color="#A8FF3E",family="Share Tech Mono"),showline=False),
                angularaxis=dict(gridcolor="rgba(168,255,62,0.08)",
                    tickfont=dict(size=9,color="#C8FFF4",family="Rajdhani"))))
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar":False})

    with ch3:
        st.markdown('<div class="section-header">◉ Sustainability Breakdown</div>', unsafe_allow_html=True)
        pl = ["Water","Energy","Climate","Disease","Yield"]
        pv = [sus_data["water_efficiency"],sus_data["energy_efficiency"],
              sus_data["climate_optimization"],sus_data["disease_prevention"],sus_data["yield_potential"]]
        pc = ["#00FFB2","#A8FF3E","#FFB830","#00B8FF","#FF5C7A"]
        fig_p = go.Figure(go.Pie(labels=pl, values=pv, hole=0.58,
            marker=dict(colors=pc, line=dict(color="#030B07",width=2)),
            textfont=dict(size=11, family="Rajdhani")))
        fig_p.add_annotation(text=f"<b>{sus_data['total']}</b>",
            x=0.5, y=0.5, font=dict(size=28, color="#A8FF3E", family="Share Tech Mono"), showarrow=False)
        fig_p.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="div-label"><span>ZONE HEATMAPS</span></div>', unsafe_allow_html=True)

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    hm1, hm2 = st.columns(2)
    temp_grid, hum_grid = generate_zone_heatmap_data(temperature, humidity)
    with hm1:
        st.markdown('<div class="section-header">🌡 Zone Temperature</div>', unsafe_allow_html=True)
        fig_ht = go.Figure(go.Heatmap(z=temp_grid,
            text=[[f"{v:.1f}°C" for v in row] for row in temp_grid],
            texttemplate="%{text}", textfont=dict(family="Share Tech Mono",size=11),
            colorscale=[[0,"#00FFB2"],[0.5,"#FFB830"],[1,"#FF5C7A"]],
            colorbar=dict(title="°C", tickfont=dict(color="#C8FFF4",family="Share Tech Mono",size=10)),
            showscale=True))
        fig_ht.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_ht, use_container_width=True, config={"displayModeBar":False})
    with hm2:
        st.markdown('<div class="section-header">💧 Zone Humidity</div>', unsafe_allow_html=True)
        fig_hh = go.Figure(go.Heatmap(z=hum_grid,
            text=[[f"{v:.1f}%" for v in row] for row in hum_grid],
            texttemplate="%{text}", textfont=dict(family="Share Tech Mono",size=11),
            colorscale=[[0,"#030B07"],[0.5,"#00B8FF"],[1,"#A8FF3E"]],
            colorbar=dict(title="%", tickfont=dict(color="#C8FFF4",family="Share Tech Mono",size=10)),
            showscale=True))
        fig_hh.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_hh, use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="div-label"><span>AI PREDICTION ENGINE</span></div>', unsafe_allow_html=True)

    # ── AI Results ────────────────────────────────────────────────────────────
    ai_res = st.session_state.get("ai_result")
    if not ai_res:
        st.markdown('''<div class="glass-card" style="text-align:center;padding:2.5rem">
            <div style="font-family:Share Tech Mono,monospace;font-size:2rem;color:#A8FF3E">⬡</div>
            <div style="font-family:Syncopate,sans-serif;font-size:0.75rem;color:#A8FF3E;letter-spacing:0.2em;margin:0.6rem 0">AI ENGINE STANDBY</div>
            <div style="font-family:Rajdhani,sans-serif;font-size:0.85rem;color:var(--text-dim)">Press <b>Run Full AI Analysis</b> in the sidebar to activate neural processing.</div>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="engine-pill">⬡ Source: {ai_res.get("source","AI Engine")}</span>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        ai_c1, ai_c2 = st.columns(2)
        with ai_c1:
            st.markdown(f'<div class="ai-card"><h4>🦠 Disease Warning</h4><p>{ai_res.get("disease_warning","—")}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-card"><h4>🌡 Climate Warning</h4><p>{ai_res.get("climate_warning","—")}</p></div>', unsafe_allow_html=True)
        with ai_c2:
            st.markdown(f'<div class="ai-card"><h4>💦 Irrigation Advice</h4><p>{ai_res.get("irrigation_advice","—")}</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-card"><h4>♻ Sustainability Insight</h4><p>{ai_res.get("sustainability_insight","—")}</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="margin-top:0.8rem">⚡ Priority Actions</div>', unsafe_allow_html=True)
        for i, action in enumerate(ai_res.get("top_actions",[]),1):
            icon = "🔴" if i==1 else "🟡" if i==2 else "🟢"
            st.markdown(f'<div class="action-item">{icon} <b>#{i}</b> — {action}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><span style="font-family:Syncopate,sans-serif;font-size:0.62rem;color:#A8FF3E;letter-spacing:0.2em">◈ OVERALL ASSESSMENT</span><br><br><span style="font-family:Rajdhani,sans-serif;font-size:0.95rem;line-height:1.7;color:var(--text-bright)">{ai_res.get("overall_assessment","—")}</span></div>', unsafe_allow_html=True)

        report_text = f"""=== AgriTwin AI · Farm Management Report ===
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Crop: {crop_type} ({growth_stage})

--- SENSOR DATA ---
Temperature   : {temperature}°C
Humidity      : {humidity}%
Soil Moisture : {soil_moisture}%
CO₂ Level     : {co2_level} ppm
Light         : {light_intensity} lux
pH            : {ph_level}
EC            : {ec_level} mS/cm

--- SUSTAINABILITY ---
Total Score   : {sus_data['total']}
Grade         : {sus_data['grade']}

--- AI RECOMMENDATIONS ---
Disease       : {ai_res.get('disease_warning')}
Climate       : {ai_res.get('climate_warning')}
Irrigation    : {ai_res.get('irrigation_advice')}

--- OVERALL ASSESSMENT ---
{ai_res.get('overall_assessment')}
"""
        st.download_button("📄 Download AI Report (.txt)", data=report_text,
            file_name=f"AgriTwin_{crop_type}_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain", use_container_width=True)

    st.markdown('<div class="div-label"><span>SYSTEM PERFORMANCE</span></div>', unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    gc = perf_data["grade_color"]
    def pcard(col, icon, label, val, color):
        col.markdown(f'''<div class="metric-tile">
            <div class="metric-emoji">{icon}</div>
            <div class="metric-value" style="color:{color};">{val:.1f}</div>
            <div class="metric-label">{label}</div>
        </div>''', unsafe_allow_html=True)
    pcard(p1,"🎯","Prediction Accuracy",perf_data["prediction_accuracy"],"#00FFB2")
    pcard(p2,"⚙️","System Efficiency",perf_data["system_efficiency"],"#A8FF3E")
    pcard(p3,"🏡","Greenhouse Performance",perf_data["greenhouse_performance"],"#FFB830")
    p4.markdown(f'''<div class="metric-tile">
        <div class="metric-emoji">🏆</div>
        <div class="metric-value" style="color:{gc};">{perf_data["overall"]:.1f}</div>
        <div class="metric-label">OVERALL · GRADE <b style="color:{gc}">{perf_data["grade"]}</b></div>
    </div>''', unsafe_allow_html=True)

# =============================================================================
# HELPERS — Model Selection
# =============================================================================
def get_working_text_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and 'vision' not in m.name.lower():
                return m.name
    except: pass
    return "models/gemini-1.5-flash"

def get_working_vision_model():
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods and ('vision' in m.name.lower() or 'flash' in m.name.lower()):
                return m.name
    except: pass
    return "models/gemini-1.5-flash"

# =============================================================================
# TAB 2 — AI ASSISTANT CHATBOT
# =============================================================================
with tab_chat:
    st.markdown('<div class="section-header">💬 AI Agri-Assistant</div>', unsafe_allow_html=True)
    if not API_KEY:
        st.markdown('<div class="terminal-box">ERROR :: GEMINI_API_KEY not found in Streamlit secrets.</div>', unsafe_allow_html=True)
    else:
        if "main_messages" not in st.session_state: st.session_state.main_messages = []
        for msg in st.session_state.main_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if prompt := st.chat_input(f"Ask about managing {crop_type}…", key="main_chat"):
            st.session_state.main_messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                ph = st.empty()
                with ph.container():
                    if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=90, key="chat_thinking")
                    else: st.spinner("Processing…")
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    res = model.generate_content(f"You are an expert AI assistant for {crop_type} at {growth_stage} stage. Question: {prompt}")
                    ph.empty(); st.markdown(res.text)
                    st.session_state.main_messages.append({"role":"assistant","content":res.text})
                except Exception as e:
                    ph.empty(); st.error(f"Error: {e}")

# =============================================================================
# TAB 3 — VISION SCANNER
# =============================================================================
with tab_vision:
    st.markdown('<div class="section-header">📸 AI Disease Vision Scanner</div>', unsafe_allow_html=True)
    if not API_KEY:
        st.markdown('<div class="terminal-box">ERROR :: GEMINI_API_KEY not found.</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            img_file = st.file_uploader("Upload plant image", type=["jpg","png"]) or st.camera_input("Take photo")
            if img_file:
                image = Image.open(img_file)
                st.image(image, use_column_width=True)
        with c2:
            if img_file:
                if st.button("🔍 Scan Plant", type="primary", use_container_width=True):
                    ph = st.empty()
                    with ph.container():
                        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=140, key="vision_scan")
                        else: st.spinner("Scanning…")
                    try:
                        v_model = genai.GenerativeModel(get_working_vision_model())
                        res = v_model.generate_content([f"Identify diseases on this {crop_type} plant. Provide structured 3-step treatment plan.", image])
                        ph.empty()
                        st.session_state.vision_result = res.text
                    except Exception as e:
                        ph.empty(); st.error(f"Error: {e}")

        if st.session_state.vision_result:
            st.markdown(f'<div class="glass-card"><span style="font-family:Syncopate,sans-serif;font-size:0.65rem;color:#A8FF3E;letter-spacing:0.2em">◈ SCAN RESULTS</span><br><br><span style="font-family:Rajdhani,sans-serif;font-size:0.92rem;line-height:1.7">{st.session_state.vision_result}</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header" style="margin-top:1rem">💬 Follow-up Analysis</div>', unsafe_allow_html=True)
            for msg in st.session_state.vision_messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            if vp := st.chat_input("Ask about treatment or disease…", key="vision_chat_input"):
                st.session_state.vision_messages.append({"role":"user","content":vp})
                with st.chat_message("user"): st.markdown(vp)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing…"):
                        try:
                            model = genai.GenerativeModel(get_working_text_model())
                            r = model.generate_content(f"You are a plant pathologist. Scan result: '{st.session_state.vision_result}'. Question: {vp}")
                            st.markdown(r.text)
                            st.session_state.vision_messages.append({"role":"assistant","content":r.text})
                        except Exception as e: st.error(f"Error: {e}")

# =============================================================================
# TAB 4 — MARKET ANALYZER
# =============================================================================
with tab_finance:
    st.markdown('<div class="section-header">📈 Profit & Market Analyzer</div>', unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    plants_count = f1.number_input("Total Plants", value=1000, step=100)
    yield_per    = f2.number_input("Yield per Plant (kg)", value=0.15, step=0.05)
    mkt_price    = f3.number_input("Market Price (Rs/kg)", value=450.0, step=10.0)

    tot_yield = plants_count * yield_per
    gross     = tot_yield * mkt_price
    cost      = (plants_count * 15) + 5000
    profit    = gross - cost

    st.markdown('<div class="div-label"><span>PROJECTIONS</span></div>', unsafe_allow_html=True)
    r1,r2,r3 = st.columns(3)
    r1.metric("📦 Total Yield", f"{tot_yield:.1f} kg")
    r2.metric("💸 Running Cost", f"Rs. {cost:,.2f}")
    r3.metric("💰 Net Profit", f"Rs. {profit:,.2f}", delta=f"{profit/cost*100:.1f}% ROI")

    fig_bar = go.Figure()
    cats = ["Gross Revenue","Running Cost","Net Profit"]
    vals = [gross, cost, profit]
    colors = ["#A8FF3E","#FF5C7A","#00FFB2"]
    fig_bar.add_trace(go.Bar(x=cats, y=vals, marker_color=colors,
        marker_line_color="rgba(0,0,0,0.4)", marker_line_width=1,
        text=[f"Rs.{v:,.0f}" for v in vals], textposition="auto",
        textfont=dict(family="Share Tech Mono", size=11, color="#030B07")))
    
    # ── FIXED CODE SECTION ───────────────────────────────────────────────────
    fig_bar.update_layout(**PLOTLY_LAYOUT)
    fig_bar.update_layout(height=280, showlegend=False)
    # ─────────────────────────────────────────────────────────────────────────

    plotly_dark_axes(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})

    st.markdown(f'<div class="glass-card"><span style="color:#A8FF3E;font-family:Syncopate,sans-serif;font-size:0.65rem;letter-spacing:0.18em">◈ MARKET INTEL</span><br><br><span style="font-family:Rajdhani,sans-serif;font-size:0.92rem">Harvest timing is optimal. Hydroponic produce demand is showing a <b style="color:#A8FF3E">+12%</b> upward trend in local supermarkets. Recommended price point: <b style="color:#00FFB2">Rs. {mkt_price*1.12:.0f}/kg</b> within 2 weeks.</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:0.5rem">💬 AI Financial Advisor</div>', unsafe_allow_html=True)
    for msg in st.session_state.market_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if mp := st.chat_input("Ask for market/profitability advice…", key="market_chat_input"):
        st.session_state.market_messages.append({"role":"user","content":mp})
        with st.chat_message("user"): st.markdown(mp)
        with st.chat_message("assistant"):
            with st.spinner("Calculating…"):
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    r = model.generate_content(f"You are an agri-financial advisor. {plants_count} plants of {crop_type}, yield {tot_yield}kg, price Rs.{mkt_price}, cost Rs.{cost}, profit Rs.{profit}. Question: {mp}")
                    st.markdown(r.text)
                    st.session_state.market_messages.append({"role":"assistant","content":r.text})
                except Exception as e: st.error(f"Error: {e}")

# =============================================================================
# TAB 5 — HARDWARE & ALERTS
# =============================================================================
with tab_hardware:
    st.markdown('<div class="section-header">⚙ Smart Hardware & Automation</div>', unsafe_allow_html=True)
    hw1, hw2 = st.columns(2)
    with hw1:
        st.markdown('''<div class="glass-card">
            <div style="font-family:Syncopate,sans-serif;font-size:0.65rem;color:#A8FF3E;letter-spacing:0.18em;margin-bottom:0.8rem">◈ HARDWARE API ENDPOINT</div>
            <div style="font-family:Rajdhani,sans-serif;font-size:0.9rem;color:var(--text-mid);margin-bottom:0.8rem">Connect your ESP32, Arduino or Raspberry Pi directly to this dashboard via REST API.</div>
        </div>''', unsafe_allow_html=True)
        st.code('POST https://agritwin-ai.streamlit.app/api/v1/sensors\n{\n  "api_key": "YOUR_SECRET_KEY",\n  "temp": 28.5,\n  "hum": 65.2,\n  "ec": 1.8\n}', language="json")
        st.button("🔄 Generate New API Key", use_container_width=True)

    with hw2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:Syncopate,sans-serif;font-size:0.65rem;color:#A8FF3E;letter-spacing:0.18em;margin-bottom:0.8rem">◈ WHATSAPP ALERT RULES</div>', unsafe_allow_html=True)
        phone_num       = st.text_input("WhatsApp Number (+country code)", value="+947XXXXXXXX")
        temp_threshold  = st.slider("Alert if Temperature exceeds (°C):", 25.0, 45.0, 35.0)
        ec_threshold    = st.slider("Alert if EC drops below (mS/cm):", 0.5, 2.0, 1.2)
        if st.button("💾 Save Alert Configuration", type="primary", use_container_width=True):
            st.success("✅ Automation rules saved successfully.")
        if temperature > temp_threshold:
            st.markdown(f'<div class="terminal-box" style="border-color:#FF5C7A;color:#FF5C7A">⚠ TRIGGER ACTIVE :: Temp {temperature}°C > threshold {temp_threshold}°C\n→ WhatsApp alert dispatched to {phone_num}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
