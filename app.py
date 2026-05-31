bash

cat > /mnt/user-data/outputs/app.py << 'ENDOFFILE'
"""
app.py — AI-Powered Predictive Digital Twin for Smart Protected Agriculture
CYBER AGRICULTURE DIGITAL TWIN COMMAND CENTER
Premium Glassmorphism | Holographic Panels | Animated Digital Environment
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
    page_title="AgriTwin AI · Digital Twin Command Center",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

API_KEY = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
if API_KEY:
    genai.configure(api_key=API_KEY)

# Session States
if "vision_result"   not in st.session_state: st.session_state.vision_result   = None
if "vision_messages" not in st.session_state: st.session_state.vision_messages = []
if "market_messages" not in st.session_state: st.session_state.market_messages = []
if "ai_result"       not in st.session_state: st.session_state["ai_result"]    = None
if "main_messages"   not in st.session_state: st.session_state.main_messages   = []

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — CYBER AGRICULTURE DIGITAL TWIN
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ══════════════════════════════════════════════
   ROOT — CYBER AGRICULTURE DESIGN TOKENS
   ══════════════════════════════════════════════ */
:root {
  --bg:          #050816;
  --bg2:         #070c1e;
  --bg3:         #0a1228;
  --panel:       rgba(5,8,22,0.75);
  --panel2:      rgba(7,12,30,0.85);
  --panel3:      rgba(10,18,40,0.60);

  --cyan:        #00F5D4;
  --cyan-dim:    rgba(0,245,212,0.12);
  --cyan-glow:   rgba(0,245,212,0.30);
  --cyan-soft:   rgba(0,245,212,0.06);

  --blue:        #00BBF9;
  --blue-dim:    rgba(0,187,249,0.12);
  --blue-glow:   rgba(0,187,249,0.28);

  --purple:      #9B5DE5;
  --purple-dim:  rgba(155,93,229,0.15);
  --purple-glow: rgba(155,93,229,0.30);

  --green:       #A8FF3E;
  --green-dim:   rgba(168,255,62,0.12);
  --green-glow:  rgba(168,255,62,0.28);

  --amber:       #FFD166;
  --amber-dim:   rgba(255,209,102,0.12);
  --red:         #EF476F;
  --red-dim:     rgba(239,71,111,0.12);

  --text:        #EAFBFF;
  --text-mid:    rgba(234,251,255,0.65);
  --text-dim:    rgba(234,251,255,0.35);
  --text-muted:  rgba(234,251,255,0.20);

  --border:      rgba(0,245,212,0.20);
  --border-b:    rgba(0,245,212,0.45);
  --border-s:    rgba(0,187,249,0.20);

  --radius:      8px;
  --radius-sm:   4px;
  --radius-lg:   12px;
}

/* ══════════════════════════════════════════════
   ANIMATED BACKGROUND — PARTICLE GRID
   ══════════════════════════════════════════════ */
.stApp { background: var(--bg) !important; }
.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 80% 50% at 20% 30%, rgba(0,245,212,0.04) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 70%, rgba(155,93,229,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 50% 60% at 50% 10%, rgba(0,187,249,0.04) 0%, transparent 60%);
}
.stApp::after {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,245,212,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,245,212,0.025) 1px, transparent 1px);
  background-size: 48px 48px;
  animation: gridDrift 20s linear infinite;
}
@keyframes gridDrift {
  0%   { background-position: 0 0; }
  100% { background-position: 48px 48px; }
}

/* ══════════════════════════════════════════════
   LAYOUT
   ══════════════════════════════════════════════ */
.main, [data-testid="stAppViewContainer"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
.block-container { padding: 1.2rem 1.8rem 3rem !important; position: relative; z-index: 1; }

/* Top accent bar */
.stApp > div:first-child::before {
  content: '';
  display: block; position: fixed;
  top: 0; left: 0; right: 0; height: 2px; z-index: 9999;
  background: linear-gradient(90deg, var(--purple), var(--cyan), var(--blue), var(--green));
  background-size: 300% 100%;
  animation: accentFlow 4s linear infinite;
}
@keyframes accentFlow {
  0%   { background-position: 0% 0; }
  100% { background-position: 300% 0; }
}

/* ══════════════════════════════════════════════
   SIDEBAR — MISSION CONTROL
   ══════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: var(--panel2) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(24px) saturate(1.6);
  -webkit-backdrop-filter: blur(24px) saturate(1.6);
}
section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"]::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan), var(--blue), transparent);
  animation: scanH 5s ease-in-out infinite;
}
@keyframes scanH {
  0%,100% { opacity: 0.5; transform: scaleX(0.4); }
  50%      { opacity: 1;   transform: scaleX(1); }
}

/* ══════════════════════════════════════════════
   TYPOGRAPHY
   ══════════════════════════════════════════════ */
body, .stMarkdown, .stText, label,
.stSelectbox label, .stSlider label,
.stCheckbox label, .stTextInput label, .stNumberInput label {
  font-family: 'Rajdhani', sans-serif !important;
  color: var(--text) !important;
}

/* ══════════════════════════════════════════════
   KEYFRAME LIBRARY
   ══════════════════════════════════════════════ */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeLeft {
  from { opacity: 0; transform: translateX(-12px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes cardReveal {
  from { opacity: 0; transform: translateY(12px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0)    scale(1); }
}
@keyframes cyanPulse {
  0%,100% { text-shadow: 0 0 16px var(--cyan), 0 0 40px var(--cyan-glow); }
  50%     { text-shadow: 0 0 30px var(--cyan), 0 0 70px var(--cyan-glow), 0 0 100px var(--cyan-dim); }
}
@keyframes bluePulse {
  0%,100% { box-shadow: 0 0 10px var(--blue-glow), inset 0 0 10px rgba(0,187,249,0.05); }
  50%     { box-shadow: 0 0 25px var(--blue-glow), inset 0 0 20px rgba(0,187,249,0.10); }
}
@keyframes ledBlink {
  0%,100% { opacity: 1; }
  50%     { opacity: 0.2; }
}
@keyframes progressFill {
  from { width: 0%; }
}
@keyframes countUp {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes borderGlow {
  0%,100% { border-color: var(--border); box-shadow: 0 0 0 rgba(0,245,212,0); }
  50%     { border-color: var(--border-b); box-shadow: 0 0 20px var(--cyan-dim); }
}
@keyframes sensorPulse {
  0%   { transform: scale(1);   opacity: 0.9; }
  50%  { transform: scale(1.5); opacity: 0.4; }
  100% { transform: scale(1);   opacity: 0.9; }
}
@keyframes dataFlow {
  0%   { stroke-dashoffset: 100; opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 1; }
  100% { stroke-dashoffset: 0;   opacity: 0; }
}
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position:  200% 0; }
}
@keyframes typing {
  from { width: 0; }
  to   { width: 100%; }
}
@keyframes rotate360 {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes orbitPulse {
  0%,100% { transform: scale(1);   box-shadow: 0 0 8px  var(--cyan); }
  50%     { transform: scale(1.08); box-shadow: 0 0 20px var(--cyan-glow); }
}
@keyframes floatY {
  0%,100% { transform: translateY(0); }
  50%     { transform: translateY(-6px); }
}

/* ══════════════════════════════════════════════
   HERO SECTION
   ══════════════════════════════════════════════ */
.hero-outer {
  position: relative;
  padding: 1.2rem 0 1rem;
  overflow: hidden;
}
.hero-outer::before {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), var(--blue) 50%, var(--purple), transparent);
}
.hero-eyebrow {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.62rem;
  color: var(--cyan) !important;
  letter-spacing: 0.5em;
  text-transform: uppercase;
  animation: fadeLeft 0.6s ease both;
  margin-bottom: 0.35rem;
}
.hero-title {
  font-family: 'Orbitron', sans-serif !important;
  font-size: clamp(1.6rem, 3.5vw, 2.8rem);
  font-weight: 900;
  color: var(--text) !important;
  letter-spacing: 0.08em;
  line-height: 1.05;
  animation: fadeUp 0.7s 0.1s ease both;
  margin: 0;
}
.hero-title .accent { color: var(--cyan); animation: cyanPulse 4s ease-in-out infinite; display: inline-block; }
.hero-subtitle {
  font-family: 'Orbitron', sans-serif !important;
  font-size: clamp(0.55rem, 1.2vw, 0.75rem);
  font-weight: 400;
  color: var(--text-mid) !important;
  letter-spacing: 0.35em;
  text-transform: uppercase;
  animation: fadeUp 0.8s 0.2s ease both;
  margin-top: 0.3rem;
}
.hero-desc {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 0.85rem;
  color: var(--text-mid) !important;
  letter-spacing: 0.05em;
  animation: fadeUp 0.9s 0.3s ease both;
  margin-top: 0.5rem;
}
.hero-status {
  text-align: right;
  padding-top: 0.5rem;
  animation: fadeIn 1s 0.4s ease both;
}
.hero-status-live {
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem;
  color: var(--cyan);
  letter-spacing: 0.2em;
}
.live-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--cyan);
  animation: ledBlink 1.4s ease-in-out infinite;
  box-shadow: 0 0 8px var(--cyan), 0 0 16px var(--cyan-glow);
}

/* ══════════════════════════════════════════════
   GLASS CARDS
   ══════════════════════════════════════════════ */
.glass-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.4rem;
  backdrop-filter: blur(20px) saturate(1.4);
  -webkit-backdrop-filter: blur(20px) saturate(1.4);
  animation: cardReveal 0.5s ease both;
  position: relative; overflow: hidden;
  margin-bottom: 0.9rem;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan-glow) 40%, var(--blue-glow) 70%, transparent);
}
.glass-card::after {
  content: '';
  position: absolute; bottom: 0; right: 0;
  width: 60px; height: 60px;
  background: radial-gradient(circle, var(--cyan-soft) 0%, transparent 70%);
  pointer-events: none;
}
.glass-card:hover {
  border-color: var(--border-b);
  box-shadow: 0 0 30px var(--cyan-dim), 0 8px 40px rgba(0,0,0,0.4);
}
.glass-card-purple {
  background: linear-gradient(135deg, rgba(155,93,229,0.08), rgba(5,8,22,0.75));
  border-color: rgba(155,93,229,0.25);
}
.glass-card-purple::before {
  background: linear-gradient(90deg, transparent, var(--purple-glow) 40%, var(--blue-glow) 70%, transparent);
}
.glass-card-blue {
  background: linear-gradient(135deg, rgba(0,187,249,0.07), rgba(5,8,22,0.75));
  border-color: rgba(0,187,249,0.22);
}

/* ══════════════════════════════════════════════
   SECTION HEADERS
   ══════════════════════════════════════════════ */
.section-header {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.6rem;
  font-weight: 700;
  color: var(--cyan) !important;
  letter-spacing: 0.3em;
  text-transform: uppercase;
  display: flex; align-items: center; gap: 0.5rem;
  margin-bottom: 0.8rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
  position: relative;
}
.section-header::after {
  content: '';
  position: absolute; bottom: -1px; left: 0;
  width: 50px; height: 1px;
  background: linear-gradient(90deg, var(--cyan), var(--blue));
  animation: scanH 3s ease-in-out infinite;
}
.section-led {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--cyan);
  animation: ledBlink 2s ease-in-out infinite;
  box-shadow: 0 0 6px var(--cyan);
  flex-shrink: 0;
}
.section-led.purple { background: var(--purple); box-shadow: 0 0 6px var(--purple); animation-duration: 1.6s; }
.section-led.amber  { background: var(--amber);  box-shadow: 0 0 6px var(--amber);  animation-duration: 1.2s; }
.section-led.red    { background: var(--red);    box-shadow: 0 0 6px var(--red);    animation-duration: 0.9s; }
.section-led.green  { background: var(--green);  box-shadow: 0 0 6px var(--green);  animation-duration: 2.5s; }

/* ══════════════════════════════════════════════
   KPI METRIC TILES
   ══════════════════════════════════════════════ */
.kpi-tile {
  background: var(--panel);
  border: 1px solid var(--border);
  border-top: 2px solid var(--cyan);
  border-radius: var(--radius);
  padding: 1rem 0.85rem 0.85rem;
  text-align: center;
  backdrop-filter: blur(16px);
  animation: cardReveal 0.5s ease both;
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
  position: relative; overflow: hidden; cursor: default;
}
.kpi-tile::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, transparent 60%, var(--cyan-soft) 100%);
  pointer-events: none;
}
.kpi-tile:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 36px var(--cyan-dim), 0 4px 16px rgba(0,0,0,0.5);
  border-top-color: var(--blue);
}
.kpi-tile.warn  { border-top-color: var(--amber); }
.kpi-tile.warn:hover { border-top-color: var(--amber); box-shadow: 0 12px 36px var(--amber-dim); }
.kpi-tile.crit  { border-top-color: var(--red);   }
.kpi-tile.crit:hover  { border-top-color: var(--red);   box-shadow: 0 12px 36px var(--red-dim); }
.kpi-tile.good  { border-top-color: var(--green); }
.kpi-tile.info  { border-top-color: var(--blue);  }
.kpi-icon  { font-size: 1.1rem; margin-bottom: 0.2rem; }
.kpi-value {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 1.75rem; font-weight: 400; line-height: 1.1;
  animation: countUp 0.6s ease both;
}
.kpi-unit  { font-size: 0.8rem; opacity: 0.7; margin-left: 1px; }
.kpi-label {
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.62rem; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--text-dim);
  margin-top: 0.25rem;
}
.kpi-trend {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; margin-top: 0.2rem;
  letter-spacing: 0.1em;
}

/* ══════════════════════════════════════════════
   PROGRESS BARS
   ══════════════════════════════════════════════ */
.prog-wrap { margin: 0.5rem 0; }
.prog-header {
  display: flex; justify-content: space-between;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.7rem; color: var(--text-mid);
  margin-bottom: 0.22rem;
}
.prog-track {
  background: rgba(0,245,212,0.05);
  border: 1px solid var(--border);
  border-radius: 2px; height: 7px; overflow: hidden;
  position: relative;
}
.prog-fill {
  height: 100%; border-radius: 2px;
  animation: progressFill 1.2s cubic-bezier(.2,.8,.4,1) both;
  position: relative;
}
.prog-fill::after {
  content: '';
  position: absolute; top: 0; right: 0;
  width: 6px; height: 100%;
  background: rgba(255,255,255,0.5);
  filter: blur(2px);
}
/* Shimmer on hover */
.glass-card:hover .prog-fill {
  background-image: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent) !important;
  background-size: 200% 100%;
  animation: progressFill 1.2s cubic-bezier(.2,.8,.4,1) both, shimmer 2s linear infinite;
}

/* ══════════════════════════════════════════════
   STATUS BADGES
   ══════════════════════════════════════════════ */
.status-badge {
  display: inline-flex; align-items: center; gap: 0.3rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem; font-weight: 600;
  letter-spacing: 0.12em; padding: 0.28rem 0.8rem;
  border-radius: var(--radius-sm); border: 1px solid currentColor;
  text-transform: uppercase;
}
.status-badge .dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: currentColor; animation: ledBlink 1.5s ease-in-out infinite;
}

/* ══════════════════════════════════════════════
   DIGITAL TWIN CENTERPIECE
   ══════════════════════════════════════════════ */
.dt-panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.4rem;
  backdrop-filter: blur(20px) saturate(1.4);
  -webkit-backdrop-filter: blur(20px) saturate(1.4);
  position: relative; overflow: hidden;
  animation: cardReveal 0.6s ease both;
}
.dt-panel::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan-glow), var(--blue-glow), var(--purple-glow), transparent);
}
.dt-map-container {
  position: relative;
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  min-height: 260px;
}
.dt-grid-bg {
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(0,245,212,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,245,212,0.05) 1px, transparent 1px);
  background-size: 32px 32px;
}
.dt-zone {
  position: absolute;
  border: 1px solid;
  border-radius: 4px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--text-mid);
  transition: all 0.3s;
}
.dt-zone:hover { z-index: 10; border-width: 2px; color: var(--text); }
.dt-node {
  position: absolute;
  width: 10px; height: 10px; border-radius: 50%;
  transform: translate(-50%,-50%); cursor: pointer;
}
.dt-node::after {
  content: '';
  position: absolute; inset: -4px;
  border-radius: 50%; border: 1px solid currentColor;
  animation: sensorPulse 2.5s ease-in-out infinite;
}
.dt-node-label {
  position: absolute;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.52rem; letter-spacing: 0.05em;
  white-space: nowrap; top: 14px; left: 50%;
  transform: translateX(-50%);
  color: var(--text-mid);
}
.dt-legend {
  display: flex; gap: 1rem; flex-wrap: wrap;
  margin-top: 0.8rem;
}
.dt-legend-item {
  display: flex; align-items: center; gap: 0.35rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem; color: var(--text-mid);
}
.dt-legend-dot {
  width: 8px; height: 8px; border-radius: 50%;
}

/* ══════════════════════════════════════════════
   AI RESULT CARDS
   ══════════════════════════════════════════════ */
.ai-card {
  background: linear-gradient(135deg, rgba(0,245,212,0.04), rgba(0,187,249,0.03));
  border: 1px solid var(--border);
  border-left: 3px solid var(--cyan);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  margin-bottom: 0.75rem;
  animation: fadeLeft 0.5s ease both;
  transition: border-left-color 0.3s, box-shadow 0.3s;
  position: relative; overflow: hidden;
}
.ai-card::before {
  content: '';
  position: absolute; top: 0; right: 0;
  width: 80px; height: 80px;
  background: radial-gradient(circle, var(--cyan-soft) 0%, transparent 70%);
  pointer-events: none;
}
.ai-card:hover { border-left-color: var(--blue); box-shadow: 0 0 20px var(--blue-dim); }
.ai-card.purple { border-left-color: var(--purple); }
.ai-card.purple::before { background: radial-gradient(circle, var(--purple-dim) 0%, transparent 70%); }
.ai-card.purple:hover { border-left-color: var(--purple); box-shadow: 0 0 20px var(--purple-dim); }
.ai-card.amber  { border-left-color: var(--amber); }
.ai-card.amber:hover  { border-left-color: var(--amber); box-shadow: 0 0 20px var(--amber-dim); }
.ai-card h4 {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.58rem; letter-spacing: 0.22em;
  color: var(--cyan) !important; text-transform: uppercase; margin: 0 0 0.5rem;
}
.ai-card.purple h4 { color: var(--purple) !important; }
.ai-card.amber h4  { color: var(--amber) !important; }
.ai-card p {
  font-family: 'Rajdhani', sans-serif;
  font-size: 0.9rem; line-height: 1.65;
  color: var(--text); margin: 0;
}

/* ══════════════════════════════════════════════
   TERMINAL / DATA STREAM
   ══════════════════════════════════════════════ */
.terminal-box {
  background: rgba(0,0,0,0.6);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.78rem; color: var(--cyan);
  line-height: 1.8; position: relative; overflow: hidden;
}
.terminal-box::before {
  content: '// SYSTEM OUTPUT ──────────────────────';
  font-size: 0.58rem; letter-spacing: 0.15em;
  color: var(--text-muted); display: block; margin-bottom: 0.5rem;
}
.terminal-box .t-key   { color: var(--blue); }
.terminal-box .t-val   { color: var(--text); }
.terminal-box .t-ok    { color: var(--green); }
.terminal-box .t-warn  { color: var(--amber); }
.terminal-box .t-crit  { color: var(--red); }
.terminal-box .t-prompt{ color: var(--purple); margin-right: 0.3rem; }

/* ══════════════════════════════════════════════
   ACTION ITEMS
   ══════════════════════════════════════════════ */
.action-item {
  background: rgba(0,245,212,0.03);
  border-left: 3px solid var(--cyan);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  padding: 0.6rem 1rem;
  margin-bottom: 0.45rem;
  font-family: 'Rajdhani', sans-serif; font-size: 0.9rem;
  color: var(--text);
  animation: fadeLeft 0.5s ease both;
  transition: background 0.2s;
}
.action-item:hover { background: rgba(0,245,212,0.07); }
.action-item.p1 { border-left-color: var(--red); }
.action-item.p2 { border-left-color: var(--amber); }
.action-item.p3 { border-left-color: var(--green); }

/* ══════════════════════════════════════════════
   HEALTH SCORE RING
   ══════════════════════════════════════════════ */
.health-ring-wrap {
  position: relative; text-align: center;
  padding: 0.5rem;
}
.health-score-big {
  font-family: 'Orbitron', monospace !important;
  font-size: 3rem; font-weight: 900; line-height: 1;
  animation: cyanPulse 4s ease-in-out infinite;
}
.health-grade {
  font-family: 'Orbitron', monospace !important;
  font-size: 0.65rem; letter-spacing: 0.3em;
  text-transform: uppercase; margin-top: 0.2rem;
  color: var(--text-mid);
}

/* ══════════════════════════════════════════════
   ENGINE PILL / TAGS
   ══════════════════════════════════════════════ */
.engine-pill {
  background: var(--cyan-dim);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.2rem 0.75rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.68rem; color: var(--cyan) !important;
  letter-spacing: 0.1em;
}
.tag-blue {
  background: var(--blue-dim);
  border: 1px solid rgba(0,187,249,0.25);
  border-radius: var(--radius-sm);
  padding: 0.15rem 0.6rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.62rem; color: var(--blue);
  letter-spacing: 0.08em;
}

/* ══════════════════════════════════════════════
   DIVIDERS
   ══════════════════════════════════════════════ */
.div-label {
  display: flex; align-items: center; gap: 0.8rem;
  margin: 1.1rem 0 0.9rem;
}
.div-label span {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.58rem; letter-spacing: 0.3em;
  color: var(--text-muted); white-space: nowrap;
}
.div-label::before, .div-label::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}

/* ══════════════════════════════════════════════
   SIDEBAR COMPONENTS
   ══════════════════════════════════════════════ */
.sb-brand {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.95rem; font-weight: 800;
  color: var(--cyan) !important; letter-spacing: 0.12em;
  animation: cyanPulse 4s ease-in-out infinite;
}
.sb-sub {
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 0.6rem; color: var(--text-dim) !important;
  letter-spacing: 0.2em; margin-top: 0.1rem;
}
.sb-section {
  font-family: 'Orbitron', sans-serif !important;
  font-size: 0.55rem; font-weight: 700;
  letter-spacing: 0.25em; text-transform: uppercase;
  color: var(--text-muted) !important;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.28rem;
  margin: 0.75rem 0 0.5rem;
}
.sb-stat {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.22rem 0;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.65rem;
}
.sb-stat .k { color: var(--text-dim); }
.sb-stat .v { color: var(--cyan); }
.sb-stat .v.ok   { color: var(--green); }
.sb-stat .v.warn { color: var(--amber); }
.sb-stat .v.crit { color: var(--red); }

/* ══════════════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
   ══════════════════════════════════════════════ */
.stSlider > div > div > div { background: var(--cyan) !important; }
.stSlider > div > div > div > div {
  background: var(--bg3) !important; border: 2px solid var(--cyan) !important;
  border-radius: 3px !important;
}
.stSelectbox div[data-baseweb],
.stTextInput input, .stNumberInput input {
  background: rgba(5,8,22,0.8) !important;
  border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important;
  color: var(--text) !important; font-family: 'Rajdhani', sans-serif !important;
}
.stSelectbox div[data-baseweb]:hover,
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: var(--cyan) !important; box-shadow: 0 0 12px var(--cyan-dim) !important;
}
.stButton > button {
  font-family: 'Orbitron', sans-serif !important; font-size: 0.65rem !important;
  letter-spacing: 0.18em !important; text-transform: uppercase !important;
  border-radius: var(--radius-sm) !important; transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--cyan), var(--blue)) !important;
  color: var(--bg) !important; border: none !important; font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 20px var(--cyan-glow), 0 4px 16px rgba(0,0,0,0.4) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]) {
  background: transparent !important; border: 1px solid var(--border) !important;
  color: var(--text-mid) !important;
}
.stButton > button:not([kind="primary"]):hover {
  border-color: var(--cyan) !important; color: var(--cyan) !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important; border-bottom: 1px solid var(--border) !important;
  gap: 0.1rem;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Orbitron', sans-serif !important; font-size: 0.58rem !important;
  font-weight: 700 !important; letter-spacing: 0.15em !important;
  color: var(--text-dim) !important; background: transparent !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
  padding: 0.65rem 1rem !important; transition: all 0.2s !important;
  border-bottom: 2px solid transparent !important; margin-bottom: -1px;
}
.stTabs [aria-selected="true"] {
  color: var(--cyan) !important; background: var(--cyan-soft) !important;
  border-bottom: 2px solid var(--cyan) !important;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
  color: var(--text-mid) !important; background: var(--panel) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 1rem 0 !important; }
.js-plotly-plot { border-radius: var(--radius) !important; }
.stDownloadButton > button {
  font-family: 'Rajdhani', sans-serif !important; font-size: 0.82rem !important;
  background: transparent !important; border: 1px solid var(--border) !important;
  color: var(--text-mid) !important; border-radius: var(--radius-sm) !important;
  letter-spacing: 0.05em !important;
}
.stDownloadButton > button:hover {
  border-color: var(--cyan) !important; color: var(--cyan) !important;
}
.stMarkdown hr { border-color: var(--border) !important; }
.stExpander {
  background: var(--panel) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
.stExpander header {
  font-family: 'Rajdhani', sans-serif !important; font-size: 0.85rem !important;
  color: var(--text-mid) !important;
}
[data-testid="stChatMessage"] {
  background: var(--panel) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stChatInputContainer"] {
  background: var(--panel2) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stChatInputContainer"] textarea {
  background: transparent !important; font-family: 'Rajdhani', sans-serif !important;
  color: var(--text) !important; font-size: 0.9rem !important;
}
[data-testid="metric-container"] {
  background: var(--panel) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important; padding: 0.6rem 0.8rem !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Share Tech Mono', monospace !important; color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
  font-family: 'Rajdhani', sans-serif !important;
  color: var(--text-dim) !important; font-size: 0.8rem !important;
}
.stAlert { border-radius: var(--radius) !important; border: 1px solid var(--border) !important; }
.stSuccess { background: rgba(168,255,62,0.08) !important; border-color: var(--green) !important; }
</style>
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

# ── PLOTLY LAYOUT — Cyber Style ───────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Rajdhani", color="#EAFBFF", size=12),
    margin=dict(l=12, r=12, t=28, b=12),
    showlegend=True,
    legend=dict(
        bgcolor="rgba(5,8,22,0.6)", font=dict(size=11, family="Share Tech Mono"),
        bordercolor="rgba(0,245,212,0.18)", borderwidth=1
    ),
    hoverlabel=dict(
        bgcolor="#070c1e", bordercolor="rgba(0,245,212,0.4)",
        font=dict(family="Share Tech Mono", size=11, color="#EAFBFF")
    )
)

def plotly_cyber_axes(fig):
    axcfg = dict(
        gridcolor="rgba(0,245,212,0.06)", gridwidth=1,
        zerolinecolor="rgba(0,245,212,0.12)", zerolinewidth=1,
        tickfont=dict(family="Share Tech Mono", size=9, color="rgba(234,251,255,0.40)"),
        linecolor="rgba(0,245,212,0.15)", linewidth=1, showline=True
    )
    fig.update_xaxes(**axcfg)
    fig.update_yaxes(**axcfg)
    return fig

def progress_html(label, value, color="#00F5D4", max_val=100):
    pct = min(100, max(0, value / max_val * 100))
    return f'''<div class="prog-wrap">
        <div class="prog-header"><span>{label}</span>
        <span style="color:{color}">{value:.1f}</span></div>
        <div class="prog-track">
          <div class="prog-fill" style="width:{pct}%;background:linear-gradient(90deg,{color}88,{color});"></div>
        </div></div>'''

def kpi_tile_html(col, icon, value, unit, label, color, cls="", trend=""):
    col.markdown(f'''<div class="kpi-tile {cls}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value" style="color:{color};">{value}<span class="kpi-unit">{unit}</span></div>
        <div class="kpi-label">{label}</div>
        {f'<div class="kpi-trend" style="color:{color}88">{trend}</div>' if trend else ''}
    </div>''', unsafe_allow_html=True)

def section_header(label, led_class="", icon=""):
    return f'''<div class="section-header">
        <span class="section-led {led_class}"></span>{icon} {label}
    </div>'''

def get_kpi_class(value, low_crit=None, low_warn=None, hi_warn=None, hi_crit=None):
    if hi_crit  and value >= hi_crit:  return "crit", "#EF476F"
    if hi_warn  and value >= hi_warn:  return "warn", "#FFD166"
    if low_crit and value <= low_crit: return "crit", "#EF476F"
    if low_warn and value <= low_warn: return "warn", "#FFD166"
    return "good", "#00F5D4"

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

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — MISSION CONTROL
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('''
        <div style="padding:0.4rem 0 0.2rem">
            <div style="font-family:Share Tech Mono,monospace;font-size:0.55rem;color:rgba(0,245,212,0.45);letter-spacing:0.4em;margin-bottom:0.2rem">0xAGRI · TWIN · v3</div>
            <div class="sb-brand">◈ AgriTwin AI</div>
            <div class="sb-sub">Digital Twin Command Center</div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown("---")

    crop_type    = st.text_input("🌱 Crop Type", value="Lettuce", placeholder="Lettuce, Tomato…")
    growth_stage = st.selectbox("📈 Growth Stage", GROWTH_STAGES, index=1)

    st.markdown('<div class="sb-section">// Climate Sensors</div>', unsafe_allow_html=True)
    temperature      = st.slider("Temperature (°C)",       5.0, 50.0,  PARAM_DEFAULTS["temperature"],      0.5)
    humidity         = st.slider("Humidity (%)",          10.0, 99.0,  PARAM_DEFAULTS["humidity"],          0.5)
    co2_level        = st.slider("CO₂ (ppm)",            300.0,2000.0, PARAM_DEFAULTS["co2_level"],        10.0)

    st.markdown('<div class="sb-section">// Soil & Water</div>', unsafe_allow_html=True)
    soil_moisture    = st.slider("Soil Moisture (%)",      5.0, 99.0,  PARAM_DEFAULTS["soil_moisture"],     0.5)
    ph_level         = st.slider("pH Level",               4.0,  9.0,  PARAM_DEFAULTS["ph_level"],          0.1)
    ec_level         = st.slider("EC Level (mS/cm)",       0.5,  5.0,  PARAM_DEFAULTS["ec_level"],          0.1)

    st.markdown('<div class="sb-section">// Light & Airflow</div>', unsafe_allow_html=True)
    light_intensity  = st.slider("Light Intensity (lux)",100.0,6000.0, PARAM_DEFAULTS["light_intensity"],  50.0)
    ventilation_rate = st.slider("Ventilation (%)",        0.0,100.0,  PARAM_DEFAULTS["ventilation_rate"],   1.0)
    leaf_area_index  = st.slider("Leaf Area Index",        0.5,  7.0,  PARAM_DEFAULTS["leaf_area_index"],    0.1)

    st.markdown("---")
    st.markdown('<div class="sb-section">// System Status</div>', unsafe_allow_html=True)
    t_ok   = temperature <= 30
    sm_ok  = soil_moisture >= 30
    vent_ok= ventilation_rate >= 40
    st.markdown(f'''
        <div class="sb-stat"><span class="k">TEMP STATUS</span><span class="v {'ok' if t_ok else 'crit'}">{"NOMINAL" if t_ok else "HIGH"}</span></div>
        <div class="sb-stat"><span class="k">SOIL STATUS</span><span class="v {'ok' if sm_ok else 'warn'}">{"OK" if sm_ok else "LOW"}</span></div>
        <div class="sb-stat"><span class="k">AIRFLOW</span><span class="v {'ok' if vent_ok else 'warn'}">{"OK" if vent_ok else "RESTRICTED"}</span></div>
        <div class="sb-stat"><span class="k">CO₂ CONC.</span><span class="v {'ok' if co2_level>=500 else 'warn'}">{co2_level:.0f} ppm</span></div>
        <div class="sb-stat"><span class="k">pH LEVEL</span><span class="v {'ok' if 5.5<=ph_level<=7.5 else 'warn'}">{ph_level:.1f}</span></div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">// 3D Plant Model</div>', unsafe_allow_html=True)
    components.iframe("https://my.spline.design/miniroom-0b666a0d244958ceef967db0b537c376/", height=230)

    st.markdown("---")
    engine_status = get_ai_engine_status()
    st.markdown(f'<span class="engine-pill">◈ {engine_status}</span>', unsafe_allow_html=True)
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
analysis  = {"sustainability": sus_data, "disease": dis_data,
             "irrigation": irr_data, "heat_stress": heat_data}
perf_data = evaluate_system_performance(sus_data, dis_data, irr_data)
analysis["performance"] = perf_data

if run_ai:
    ph = st.empty()
    with ph.container():
        st.markdown('''<div style="text-align:center;padding:2rem">
            <div style="font-family:Orbitron,sans-serif;font-size:0.75rem;color:#00F5D4;letter-spacing:0.4em;animation:cyanPulse 2s ease-in-out infinite">
            ◈ NEURAL NETWORK INFERENCE ACTIVE ◈</div></div>''', unsafe_allow_html=True)
        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=160, key="thinking")
        else: st.spinner("Processing…")
    st.session_state["ai_result"] = get_ai_recommendations(params, analysis)
    ph.empty()

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
now = datetime.datetime.now()
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown(f'''<div class="hero-outer">
        <div class="hero-eyebrow">◈ &nbsp; Cyber Agriculture Platform &nbsp; · &nbsp; v{VERSION} &nbsp; ◈</div>
        <div class="hero-title">
            AGRI<span class="accent">TWIN</span> AI
        </div>
        <div class="hero-subtitle">Digital Twin Command Center</div>
        <div class="hero-desc">
            AI-Powered Predictive Digital Twin &nbsp;·&nbsp; Smart Protected Agriculture
            &nbsp;·&nbsp; <span style="color:var(--cyan)">{crop_type}</span>
            &nbsp;·&nbsp; <span style="color:var(--blue)">{growth_stage}</span>
        </div>
    </div>''', unsafe_allow_html=True)
with hc2:
    health_color = "#A8FF3E" if sus_data["total"]>=75 else ("#FFD166" if sus_data["total"]>=50 else "#EF476F")
    st.markdown(f'''<div class="hero-status">
        <div class="hero-status-live"><span class="live-dot"></span>LIVE TELEMETRY</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:var(--text-mid);margin-top:0.3rem">
            {now.strftime("%Y-%m-%d  %H:%M:%S")}
        </div>
        <div style="font-family:Share Tech Mono,monospace;font-size:0.68rem;color:var(--blue);margin-top:0.2rem">
            SYS STATUS: <span style="color:{health_color}">{"OPTIMAL" if sus_data["total"]>=75 else "WARNING" if sus_data["total"]>=50 else "CRITICAL"}</span>
        </div>
    </div>''', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_dashboard, tab_twin, tab_chat, tab_vision, tab_finance, tab_hardware = st.tabs([
    "◈ DASHBOARD", "◈ DIGITAL TWIN", "◈ AI ASSISTANT",
    "◈ VISION SCANNER", "◈ MARKET ANALYZER", "◈ HARDWARE"
])

# =============================================================================
# TAB 1 — SYSTEM DASHBOARD
# =============================================================================
with tab_dashboard:

    # ── KPI Row ──────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    t_cls, t_col = get_kpi_class(temperature, hi_warn=28, hi_crit=35)
    h_cls, h_col = get_kpi_class(humidity, hi_warn=80, hi_crit=90)
    sm_cls,sm_col= get_kpi_class(soil_moisture, low_crit=20, low_warn=35)
    c_cls, c_col = get_kpi_class(co2_level, low_warn=500)
    v_cls, v_col = get_kpi_class(ventilation_rate, low_crit=20, low_warn=40)

    kpi_tile_html(k1,"🌡",f"{temperature:.1f}","°C","Temperature",  t_col,  t_cls,  "↑ HIGH" if temperature>30 else "● NOMINAL")
    kpi_tile_html(k2,"💧",f"{humidity:.1f}",   "%", "Humidity",      h_col,  h_cls,  "↑ HIGH" if humidity>80 else "● NOMINAL")
    kpi_tile_html(k3,"🌱",f"{soil_moisture:.1f}","%","Soil Moisture",sm_col, sm_cls, "↓ LOW"  if soil_moisture<30 else "● NOMINAL")
    kpi_tile_html(k4,"💨",f"{co2_level:.0f}",  "ppm","CO₂ Conc.",   c_col,  c_cls,  "↓ LOW"  if co2_level<500 else "● NOMINAL")
    kpi_tile_html(k5,"☀️",f"{light_intensity:.0f}","lux","Irradiance","#00BBF9","info","● ACTIVE")
    kpi_tile_html(k6,"🌬",f"{ventilation_rate:.0f}","%","Ventilation", v_col, v_cls,  "↓ LOW"  if ventilation_rate<40 else "● NOMINAL")

    st.markdown('<div class="div-label"><span>GREENHOUSE HEALTH SCORE</span></div>', unsafe_allow_html=True)

    # ── Health Score + Gauge Row ──────────────────────────────────────────────
    gh1, gh2, gh3 = st.columns([1.2, 2, 1.2])
    gc = sus_data["grade_color"]

    with gh1:
        st.markdown(section_header("OVERALL HEALTH", "green", "♦"), unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=sus_data["total"],
            domain={'x':[0,1],'y':[0,1]},
            number={'font':{'color':gc,'size':42,'family':'Orbitron'},'suffix':''},
            gauge={
                'axis':{'range':[None,100],'tickcolor':"rgba(234,251,255,0.3)",
                        'tickfont':{'family':'Share Tech Mono','size':8}},
                'bar':{'color':gc,'thickness':0.22},
                'bgcolor':"rgba(0,0,0,0)",
                'borderwidth':1,'bordercolor':"rgba(0,245,212,0.15)",
                'steps':[
                    {'range':[0,40],'color':"rgba(239,71,111,0.10)"},
                    {'range':[40,75],'color':"rgba(255,209,102,0.10)"},
                    {'range':[75,100],'color':"rgba(168,255,62,0.10)"}
                ],
                'threshold':{'line':{'color':gc,'width':2},'thickness':0.78,'value':sus_data["total"]}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                height=200, margin=dict(l=18,r=18,t=18,b=8))
        st.markdown('<div class="glass-card" style="text-align:center;padding-bottom:0.5rem">', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'''<div style="font-family:Orbitron,sans-serif;font-size:0.62rem;color:{gc};
            letter-spacing:0.25em;text-align:center;text-transform:uppercase">
            GRADE {sus_data["grade"]} &nbsp;·&nbsp; {sus_data["total"]}/100
        </div></div>''', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(progress_html("Water Efficiency",   sus_data["water_efficiency"],    "#00BBF9"), unsafe_allow_html=True)
        st.markdown(progress_html("Energy Efficiency",  sus_data["energy_efficiency"],   "#A8FF3E"), unsafe_allow_html=True)
        st.markdown(progress_html("Climate Optimization",sus_data["climate_optimization"],"#FFD166"), unsafe_allow_html=True)
        st.markdown(progress_html("Disease Prevention", sus_data["disease_prevention"],  "#9B5DE5"), unsafe_allow_html=True)
        st.markdown(progress_html("Yield Potential",    sus_data["yield_potential"],     "#00F5D4"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with gh2:
        st.markdown(section_header("24H CLIMATE TELEMETRY", "blue", "◉"), unsafe_allow_html=True)
        trend = generate_trend_data(temperature, humidity, 24)
        fig_t = go.Figure()
        t_line = "#EF476F" if temperature > 30 else "#00BBF9"
        fig_t.add_trace(go.Scatter(x=trend["times"], y=trend["temperatures"], name="Temp °C",
            line=dict(color=t_line, width=2),
            fill="tozeroy", fillcolor=f"rgba({'239,71,111' if temperature>30 else '0,187,249'},0.05)"))
        fig_t.add_trace(go.Scatter(x=trend["times"], y=trend["humidities"], name="Humidity %",
            line=dict(color="#00F5D4", width=2),
            fill="tozeroy", fillcolor="rgba(0,245,212,0.04)"))
        fig_t.add_trace(go.Scatter(x=trend["times"], y=trend["soil_moisture"], name="Soil Moist %",
            line=dict(color="#A8FF3E", width=1.5, dash="dot")))
        fig_t.update_layout(**PLOTLY_LAYOUT, height=420, xaxis=dict(tickangle=30, nticks=10))
        plotly_cyber_axes(fig_t)
        st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar":False})

    with gh3:
        st.markdown(section_header("DISEASE RISK", "red", "⬡"), unsafe_allow_html=True)
        dlc = dis_data["level_color"]
        badge_cls = "crit" if dis_data["overall"]>60 else ("warn" if dis_data["overall"]>30 else "")
        st.markdown(f'''<div class="glass-card" style="text-align:center">
            <span class="status-badge {badge_cls}" style="color:{dlc};border-color:{dlc}">
                <span class="dot"></span>{dis_data["level"].upper()}
            </span>
            <div style="font-family:Share Tech Mono,monospace;font-size:2.6rem;color:{dlc};
                margin:0.5rem 0;font-weight:400;letter-spacing:-0.02em">
                {dis_data["overall"]:.1f}<span style="font-size:0.9rem;opacity:0.6">%</span>
            </div>
            <div style="font-family:Share Tech Mono,monospace;font-size:0.55rem;
                color:var(--text-dim);letter-spacing:0.3em">OVERALL RISK INDEX</div>
        </div>''', unsafe_allow_html=True)
        d_clrs = ["#EF476F","#FFD166","#00F5D4","#9B5DE5"]
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        for i,(n,v) in enumerate(dis_data["diseases"].items()):
            st.markdown(progress_html(n, v, d_clrs[i%4]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(section_header("IRRIGATION", "blue", "◎"), unsafe_allow_html=True)
        isc = irr_data["status_color"]
        st.markdown(f'''<div class="glass-card glass-card-blue" style="text-align:center">
            <span class="status-badge" style="color:{isc};border-color:{isc}">
                <span class="dot"></span>{irr_data["status"].upper()}
            </span>
            <div style="font-family:Share Tech Mono,monospace;font-size:2.6rem;color:{isc};
                margin:0.45rem 0;font-weight:400">
                {irr_data["urgency"]:.1f}<span style="font-size:0.9rem;opacity:0.6">%</span>
            </div>
            <div style="font-family:Share Tech Mono,monospace;font-size:0.55rem;
                color:var(--text-dim);letter-spacing:0.3em">URGENCY INDEX</div>
        </div>''', unsafe_allow_html=True)
        st.markdown(f'''<div class="glass-card glass-card-blue">
            {progress_html("Irrigation Urgency", irr_data["urgency"], isc)}
            <div style="margin-top:0.75rem;font-family:Share Tech Mono,monospace;font-size:0.72rem;line-height:2">
                <span style="color:var(--text-dim)">VOLUME &nbsp;</span>
                <b style="color:{isc}">{irr_data["volume_liters"]:.2f} L/m²</b><br>
                <span style="color:var(--text-dim)">NEXT &nbsp;&nbsp;&nbsp;</span>
                <b style="color:{isc}">{irr_data["next_irrigation_hours"]}h</b><br>
                <span style="color:var(--text-dim)">ET₀ &nbsp;&nbsp;&nbsp;&nbsp;</span>
                <b style="color:{isc}">{irr_data["evapotranspiration"]:.3f} mm/hr</b>
            </div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<div class="div-label"><span>RISK INTELLIGENCE ENGINE</span></div>', unsafe_allow_html=True)

    # ── Risk Radar + Sustainability Donut ─────────────────────────────────────
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(section_header("RISK RADAR SCAN", "red", "⬡"), unsafe_allow_html=True)
        rc_labels = ["Disease","Heat Stress","Water Stress","CO₂ Deficit","Ventilation","Yield Risk"]
        rv = [dis_data["overall"],
              min(100,max(0,(heat_data["heat_index"]-20)*2)),
              max(0,100-sus_data["water_efficiency"]),
              max(0,(900-co2_level)/9),
              max(0,100-ventilation_rate),
              max(0,100-sus_data["yield_potential"])]
        fig_r = go.Figure(go.Scatterpolar(
            r=rv+[rv[0]], theta=rc_labels+[rc_labels[0]],
            fill="toself", fillcolor="rgba(239,71,111,0.10)",
            line=dict(color="#EF476F", width=2), name="Risk Profile"))
        fig_r.add_trace(go.Scatterpolar(
            r=[40]*6+[40], theta=rc_labels+[rc_labels[0]],
            line=dict(color="rgba(0,245,212,0.25)", width=1, dash="dot"),
            name="Safe Threshold"))
        fig_r.update_layout(**PLOTLY_LAYOUT, height=310,
            polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0,100], gridcolor="rgba(0,245,212,0.07)",
                    tickfont=dict(size=8,color="#00F5D4",family="Share Tech Mono"), showline=False),
                angularaxis=dict(gridcolor="rgba(0,245,212,0.07)",
                    tickfont=dict(size=9,color="rgba(234,251,255,0.6)",family="Rajdhani"))))
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar":False})

    with rc2:
        st.markdown(section_header("SUSTAINABILITY BREAKDOWN", "blue", "◈"), unsafe_allow_html=True)
        pl = ["Water","Energy","Climate","Disease Prev.","Yield"]
        pv = [sus_data["water_efficiency"],sus_data["energy_efficiency"],
              sus_data["climate_optimization"],sus_data["disease_prevention"],sus_data["yield_potential"]]
        pc = ["#00BBF9","#A8FF3E","#FFD166","#9B5DE5","#00F5D4"]
        fig_p = go.Figure(go.Pie(labels=pl, values=pv, hole=0.60,
            marker=dict(colors=pc, line=dict(color="#050816",width=2)),
            textfont=dict(size=11, family="Rajdhani"),
            hovertemplate="<b>%{label}</b><br>%{value:.1f}<extra></extra>"))
        fig_p.add_annotation(text=f"<b>{sus_data['total']}</b>", x=0.5, y=0.5,
            font=dict(size=28, color="#00F5D4", family="Orbitron"), showarrow=False)
        fig_p.update_layout(**PLOTLY_LAYOUT, height=310)
        st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar":False})

    with rc3:
        st.markdown(section_header("RISK INDICATORS", "amber", "◉"), unsafe_allow_html=True)
        risks = [
            ("Disease Risk",    dis_data["overall"],                            "#EF476F"),
            ("Climate Risk",    min(100,max(0,(heat_data["heat_index"]-20)*2)), "#EF476F" if temperature>30 else "#FFD166"),
            ("Irrigation Risk", irr_data["urgency"],                            irr_data["status_color"]),
            ("Yield Risk",      max(0,100-sus_data["yield_potential"]),         "#FFD166"),
            ("CO₂ Deficit",     max(0,(900-co2_level)/9),                      "#9B5DE5"),
            ("Ventilation",     max(0,100-ventilation_rate),                   "#00BBF9"),
        ]
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        for lbl, val, col in risks:
            st.markdown(progress_html(lbl, val, col), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Heat stress card
        hi = heat_data["heat_index"]
        hi_col = "#EF476F" if hi>35 else ("#FFD166" if hi>28 else "#00F5D4")
        st.markdown(f'''<div class="glass-card glass-card-purple" style="text-align:center;padding:0.9rem">
            <div style="font-family:Orbitron,sans-serif;font-size:0.58rem;color:var(--purple);
                letter-spacing:0.25em;margin-bottom:0.4rem">HEAT STRESS INDEX</div>
            <div style="font-family:Share Tech Mono,monospace;font-size:2.2rem;color:{hi_col};
                font-weight:400">{hi:.1f}<span style="font-size:0.85rem;opacity:0.6">°C</span></div>
            <div style="font-family:Share Tech Mono,monospace;font-size:0.58rem;
                color:{hi_col};margin-top:0.2rem">{"CRITICAL" if hi>35 else "ELEVATED" if hi>28 else "NORMAL"}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown('<div class="div-label"><span>ZONE SPATIAL ANALYSIS</span></div>', unsafe_allow_html=True)

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    hm1, hm2 = st.columns(2)
    temp_grid, hum_grid = generate_zone_heatmap_data(temperature, humidity)
    with hm1:
        st.markdown(section_header("ZONE TEMPERATURE MAP", "red" if temperature>30 else "blue", "⬡"), unsafe_allow_html=True)
        fig_ht = go.Figure(go.Heatmap(z=temp_grid,
            text=[[f"{v:.1f}°C" for v in row] for row in temp_grid],
            texttemplate="%{text}", textfont=dict(family="Share Tech Mono",size=10,color="#EAFBFF"),
            colorscale=[[0,"#00BBF9"],[0.5,"#FFD166"],[1,"#EF476F"]],
            colorbar=dict(title="°C",tickfont=dict(color="rgba(234,251,255,0.5)",family="Share Tech Mono",size=9),
                          titlefont=dict(color="rgba(234,251,255,0.5)",size=9)),
            showscale=True,
            hovertemplate="Zone: %{x},%{y}<br>Temp: %{z:.1f}°C<extra></extra>"))
        fig_ht.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_ht, use_container_width=True, config={"displayModeBar":False})
    with hm2:
        st.markdown(section_header("ZONE HUMIDITY MAP", "blue", "◉"), unsafe_allow_html=True)
        fig_hh = go.Figure(go.Heatmap(z=hum_grid,
            text=[[f"{v:.1f}%" for v in row] for row in hum_grid],
            texttemplate="%{text}", textfont=dict(family="Share Tech Mono",size=10,color="#EAFBFF"),
            colorscale=[[0,"#050816"],[0.5,"#00BBF9"],[1,"#A8FF3E"]],
            colorbar=dict(title="%",tickfont=dict(color="rgba(234,251,255,0.5)",family="Share Tech Mono",size=9),
                          titlefont=dict(color="rgba(234,251,255,0.5)",size=9)),
            showscale=True,
            hovertemplate="Zone: %{x},%{y}<br>Humidity: %{z:.1f}%<extra></extra>"))
        fig_hh.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_hh, use_container_width=True, config={"displayModeBar":False})

    with st.expander("▸ View Raw Telemetry — All Channels"):
        tc1, tc2 = st.columns(2)
        def trow(k,v,unit,ok_range=None):
            if ok_range:
                s = "ok" if ok_range[0]<=v<=ok_range[1] else ("warn" if abs(v-np.mean(ok_range))/max(ok_range)<0.3 else "crit")
            else: s=""
            sc = {"ok":"#A8FF3E","warn":"#FFD166","crit":"#EF476F","":"rgba(234,251,255,0.6)"}.get(s,"rgba(234,251,255,0.6)")
            return f'<div style="display:flex;justify-content:space-between;font-family:Share Tech Mono,monospace;font-size:0.72rem;padding:0.22rem 0;border-bottom:1px solid rgba(0,245,212,0.06)"><span style="color:var(--text-dim)">{k}</span><span style="color:{sc}">{v} {unit}</span></div>'
        with tc1:
            st.markdown('<div class="glass-card">' +
                trow("CH-01 TEMPERATURE",f"{temperature:.1f}","°C",(15,28)) +
                trow("CH-02 HUMIDITY",f"{humidity:.1f}","%",(50,80)) +
                trow("CH-03 SOIL MOISTURE",f"{soil_moisture:.1f}","%",(40,80)) +
                trow("CH-04 CO₂ CONC.",f"{co2_level:.0f}","ppm",(600,1200)) +
                trow("CH-05 IRRADIANCE",f"{light_intensity:.0f}","lux") +
                trow("CH-06 VENTILATION",f"{ventilation_rate:.0f}","%",(40,100)) +
                '</div>', unsafe_allow_html=True)
        with tc2:
            st.markdown('<div class="glass-card">' +
                trow("PH LEVEL",f"{ph_level:.1f}","pH",(5.5,7.5)) +
                trow("EC LEVEL",f"{ec_level:.2f}","mS/cm",(1.0,3.0)) +
                trow("LEAF AREA IDX",f"{leaf_area_index:.2f}","LAI") +
                trow("HEAT INDEX",f"{heat_data['heat_index']:.1f}","°C",(15,30)) +
                trow("DISEASE RISK",f"{dis_data['overall']:.1f}","%") +
                trow("SUST. SCORE",f"{sus_data['total']:.1f}","/100",(75,100)) +
                '</div>', unsafe_allow_html=True)

    st.markdown('<div class="div-label"><span>AI PREDICTION ENGINE</span></div>', unsafe_allow_html=True)

    # ── AI Results ────────────────────────────────────────────────────────────
    ai_res = st.session_state.get("ai_result")
    if not ai_res:
        st.markdown('''<div class="glass-card" style="text-align:center;padding:2.5rem">
            <div style="font-family:Orbitron,sans-serif;font-size:1.6rem;color:rgba(0,245,212,0.25)">◈</div>
            <div style="font-family:Orbitron,sans-serif;font-size:0.65rem;color:var(--cyan);
                letter-spacing:0.3em;margin:0.6rem 0">AI ENGINE STANDBY</div>
            <div style="font-family:Rajdhani,sans-serif;font-size:0.88rem;color:var(--text-mid)">
                Press <b>⚡ Run Full AI Analysis</b> in the sidebar to activate neural inference.
            </div>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="engine-pill">◈ Source: {ai_res.get("source","AI Engine")}</span>&nbsp;<span class="tag-blue">● INFERENCE COMPLETE</span>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        ai_c1, ai_c2 = st.columns(2)
        with ai_c1:
            st.markdown(f'''<div class="ai-card amber">
                <h4>⬡ Disease Warning</h4><p>{ai_res.get("disease_warning","—")}</p></div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="ai-card">
                <h4>◉ Climate Warning</h4><p>{ai_res.get("climate_warning","—")}</p></div>''', unsafe_allow_html=True)
        with ai_c2:
            st.markdown(f'''<div class="ai-card purple">
                <h4>◎ Irrigation Advice</h4><p>{ai_res.get("irrigation_advice","—")}</p></div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="ai-card">
                <h4>♻ Sustainability Insight</h4><p>{ai_res.get("sustainability_insight","—")}</p></div>''', unsafe_allow_html=True)

        st.markdown(section_header("PRIORITY ACTION QUEUE", "red", "⚡"), unsafe_allow_html=True)
        priority_cls = ["p1","p2","p3"]
        for i, action in enumerate(ai_res.get("top_actions",[]),1):
            cls = priority_cls[min(i-1,2)]
            icon = "🔴" if i==1 else ("🟡" if i==2 else "🟢")
            st.markdown(f'<div class="action-item {cls}">{icon} <b>P{i}</b> — {action}</div>', unsafe_allow_html=True)

        st.markdown(f'''<div class="terminal-box" style="margin-top:0.8rem">
            <span style="font-family:Orbitron,sans-serif;font-size:0.55rem;color:var(--cyan);
                letter-spacing:0.3em">◈ OVERALL AI ASSESSMENT</span><br><br>
            <span style="color:var(--text-mid);font-size:0.82rem;line-height:1.8">
                {ai_res.get("overall_assessment","—")}
            </span>
        </div>''', unsafe_allow_html=True)

        report_text = f"""=== AgriTwin AI · Digital Twin Command Center Report ===
Timestamp     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Crop          : {crop_type} ({growth_stage})

--- TELEMETRY CHANNELS ---
CH-01 TEMP    : {temperature}°C
CH-02 HUMID   : {humidity}%
CH-03 SOIL    : {soil_moisture}%
CH-04 CO2     : {co2_level} ppm
CH-05 LIGHT   : {light_intensity} lux
CH-06 VENT    : {ventilation_rate}%
     PH       : {ph_level}
     EC       : {ec_level} mS/cm

--- SUSTAINABILITY ---
Score         : {sus_data['total']} / 100
Grade         : {sus_data['grade']}

--- AI INFERENCE OUTPUT ---
Disease       : {ai_res.get('disease_warning')}
Climate       : {ai_res.get('climate_warning')}
Irrigation    : {ai_res.get('irrigation_advice')}
Sustainability: {ai_res.get('sustainability_insight')}

--- OVERALL ASSESSMENT ---
{ai_res.get('overall_assessment')}
"""
        st.download_button("📄 Download AI Report (.txt)", data=report_text,
            file_name=f"AgriTwin_{crop_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain", use_container_width=True)

    st.markdown('<div class="div-label"><span>SYSTEM PERFORMANCE CENTER</span></div>', unsafe_allow_html=True)

    # ── Performance ───────────────────────────────────────────────────────────
    p1,p2,p3,p4 = st.columns(4)
    gc = perf_data["grade_color"]
    def perf_card(col, icon, ch, label, val, color):
        cls = "good" if val>=75 else ("warn" if val>=50 else "crit")
        col.markdown(f'''<div class="kpi-tile {cls}">
            <div class="kpi-icon">{icon}</div>
            <div style="font-family:Share Tech Mono,monospace;font-size:0.52rem;color:var(--text-muted);
                letter-spacing:0.15em;margin-bottom:0.1rem">{ch}</div>
            <div class="kpi-value" style="color:{color};">{val:.1f}</div>
            <div class="kpi-label">{label}</div>
        </div>''', unsafe_allow_html=True)
    perf_card(p1,"🎯","PA-01","PREDICTION ACCURACY",  perf_data["prediction_accuracy"],   "#00F5D4")
    perf_card(p2,"⚙️","PA-02","SYSTEM EFFICIENCY",     perf_data["system_efficiency"],      "#A8FF3E")
    perf_card(p3,"🏡","PA-03","GREENHOUSE PERFORMANCE",perf_data["greenhouse_performance"], "#FFD166")
    p4.markdown(f'''<div class="kpi-tile" style="border-top-color:{gc}">
        <div class="kpi-icon">🏆</div>
        <div style="font-family:Share Tech Mono,monospace;font-size:0.52rem;color:var(--text-muted);
            letter-spacing:0.15em;margin-bottom:0.1rem">PA-04</div>
        <div class="kpi-value" style="color:{gc};">{perf_data["overall"]:.1f}</div>
        <div class="kpi-label">OVERALL · GRADE <b style="color:{gc}">{perf_data["grade"]}</b></div>
    </div>''', unsafe_allow_html=True)

# =============================================================================
# TAB 2 — DIGITAL TWIN CENTERPIECE
# =============================================================================
with tab_twin:
    st.markdown(section_header("GREENHOUSE DIGITAL TWIN — LIVE SPATIAL MONITOR", "green", "◈"), unsafe_allow_html=True)

    dt1, dt2 = st.columns([2, 1])
    with dt1:
        # Interactive SVG Digital Twin Map
        t_h = min(255, int((temperature / 45) * 255))
        t_g = 255 - t_h
        tc_hex = f"#{t_h:02x}{t_g:02x}80"
        h_b = min(255, int((humidity / 100) * 255))
        hc_hex = f"#00{h_b:02x}ff"

        node_data = [
            (20, 25, "#00F5D4", f"T:{temperature:.0f}°C", "TEMP-N"),
            (50, 25, "#00BBF9", f"H:{humidity:.0f}%",     "HUM-N"),
            (80, 25, "#A8FF3E", f"CO₂:{co2_level:.0f}",   "CO2-N"),
            (20, 65, "#FFD166", f"pH:{ph_level:.1f}",      "PH-N"),
            (50, 65, "#9B5DE5", f"EC:{ec_level:.1f}",      "EC-N"),
            (80, 65, "#EF476F" if soil_moisture<30 else "#00F5D4",
                     f"SM:{soil_moisture:.0f}%", "SOIL-N"),
        ]

        zone_cls = [
            ("Zone A — Seedling", "3%","3%","30%","44%", "#00F5D4"),
            ("Zone B — Vegetative","35%","3%","30%","44%","#00BBF9"),
            ("Zone C — Mature",   "67%","3%","30%","44%","#A8FF3E"),
            ("Zone D — Harvest",  "3%","51%","30%","44%","#FFD166"),
            ("Zone E — Processing","35%","51%","30%","44%","#9B5DE5"),
            ("Zone F — Storage",  "67%","51%","30%","44%","#EF476F"),
        ]

        nodes_svg = ""
        for px,py,nc,label,nid in node_data:
            nodes_svg += f'''
            <circle cx="{px}%" cy="{py}%" r="6" fill="{nc}" opacity="0.9">
              <animate attributeName="r" values="6;10;6" dur="2.5s" repeatCount="indefinite"/>
              <animate attributeName="opacity" values="0.9;0.4;0.9" dur="2.5s" repeatCount="indefinite"/>
            </circle>
            <circle cx="{px}%" cy="{py}%" r="4" fill="{nc}"/>
            <text x="{px}%" y="{py+8}%" text-anchor="middle"
              font-family="Share Tech Mono,monospace" font-size="9" fill="rgba(234,251,255,0.7)">{label}</text>
            '''

        zones_svg = ""
        for zi,(zlabel,zx,zy,zw,zh,zc) in enumerate(zone_cls):
            zones_svg += f'''
            <rect x="{zx}" y="{zy}" width="{zw}" height="{zh}"
              fill="{zc}08" stroke="{zc}" stroke-width="1" stroke-opacity="0.35" rx="4"/>
            <text x="calc({zx} + 2%)" y="calc({zy} + 3%)"
              font-family="Share Tech Mono,monospace" font-size="8" fill="{zc}" opacity="0.7">
              {zlabel.split("—")[0].strip()}
            </text>
            '''

        svg_map = f'''<svg viewBox="0 0 800 320" xmlns="http://www.w3.org/2000/svg"
          style="width:100%;height:100%;min-height:300px;background:rgba(0,0,0,0.5);
          border-radius:8px;border:1px solid rgba(0,245,212,0.18)">
          <!-- Grid -->
          <defs>
            <pattern id="g" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(0,245,212,0.06)" stroke-width="0.8"/>
            </pattern>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
          </defs>
          <rect width="800" height="320" fill="url(#g)"/>

          <!-- Zones -->
          <rect x="8" y="8" width="240" height="140" fill="rgba(0,245,212,0.04)"
            stroke="rgba(0,245,212,0.25)" stroke-width="1" rx="6"/>
          <rect x="280" y="8" width="240" height="140" fill="rgba(0,187,249,0.04)"
            stroke="rgba(0,187,249,0.25)" stroke-width="1" rx="6"/>
          <rect x="552" y="8" width="240" height="140" fill="rgba(168,255,62,0.04)"
            stroke="rgba(168,255,62,0.25)" stroke-width="1" rx="6"/>
          <rect x="8" y="172" width="240" height="140" fill="rgba(255,209,102,0.04)"
            stroke="rgba(255,209,102,0.25)" stroke-width="1" rx="6"/>
          <rect x="280" y="172" width="240" height="140" fill="rgba(155,93,229,0.04)"
            stroke="rgba(155,93,229,0.25)" stroke-width="1" rx="6"/>
          <rect x="552" y="172" width="240" height="140" fill="rgba(239,71,111,0.04)"
            stroke="rgba(239,71,111,0.25)" stroke-width="1" rx="6"/>

          <!-- Zone Labels -->
          <text x="20" y="28" font-family="Share Tech Mono,monospace" font-size="9" fill="rgba(0,245,212,0.6)">ZONE A · SEEDLING</text>
          <text x="292" y="28" font-family="Share Tech Mono,monospace" font-size="9" fill="rgba(0,187,249,0.6)">ZONE B · VEGETATIVE</text>
          <text x="564" y="28" font-family="Share Tech Mono,monospace" font-size="9" fill="rgba(168,255,62,0.6)">ZONE C · MATURE</text>
          <text x="20" y="192" font-family="Share Tech Mono,monospace" font-size="9" fill="rgba(255,209,102,0.6)">ZONE D · HARVEST</text>
          <text x="292" y="192" font-family="Share Tech Mono,monospace" font-size="9" fill="rgba(155,93,229,0.6)">ZONE E · PROCESSING</text>
          <text x="564" y="192" font-family="Share Tech Mono,monospace" font-size="9" fill="rgba(239,71,111,0.6)">ZONE F · STORAGE</text>

          <!-- Connection lines with animation -->
          <line x1="128" y1="78" x2="400" y2="78" stroke="rgba(0,245,212,0.12)" stroke-width="1" stroke-dasharray="4,4">
            <animate attributeName="stroke-dashoffset" from="100" to="0" dur="3s" repeatCount="indefinite"/>
          </line>
          <line x1="400" y1="78" x2="672" y2="78" stroke="rgba(0,187,249,0.12)" stroke-width="1" stroke-dasharray="4,4">
            <animate attributeName="stroke-dashoffset" from="100" to="0" dur="3.5s" repeatCount="indefinite"/>
          </line>
          <line x1="128" y1="78" x2="128" y2="242" stroke="rgba(0,245,212,0.08)" stroke-width="1" stroke-dasharray="3,5">
            <animate attributeName="stroke-dashoffset" from="80" to="0" dur="4s" repeatCount="indefinite"/>
          </line>

          <!-- Sensor nodes: Zone A -->
          <circle cx="128" cy="78" r="7" fill="#00F5D4" filter="url(#glow)">
            <animate attributeName="r" values="7;12;7" dur="2.8s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.9;0.35;0.9" dur="2.8s" repeatCount="indefinite"/>
          </circle>
          <circle cx="128" cy="78" r="4" fill="#00F5D4"/>
          <text x="128" y="98" text-anchor="middle" font-family="Share Tech Mono,monospace" font-size="9" fill="#00F5D4">T:{temperature:.0f}°C</text>

          <!-- Zone B -->
          <circle cx="400" cy="78" r="7" fill="#00BBF9" filter="url(#glow)">
            <animate attributeName="r" values="7;11;7" dur="3.2s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.9;0.35;0.9" dur="3.2s" repeatCount="indefinite"/>
          </circle>
          <circle cx="400" cy="78" r="4" fill="#00BBF9"/>
          <text x="400" y="98" text-anchor="middle" font-family="Share Tech Mono,monospace" font-size="9" fill="#00BBF9">H:{humidity:.0f}%</text>

          <!-- Zone C -->
          <circle cx="672" cy="78" r="7" fill="#A8FF3E" filter="url(#glow)">
            <animate attributeName="r" values="7;11;7" dur="2.5s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.9;0.35;0.9" dur="2.5s" repeatCount="indefinite"/>
          </circle>
          <circle cx="672" cy="78" r="4" fill="#A8FF3E"/>
          <text x="672" y="98" text-anchor="middle" font-family="Share Tech Mono,monospace" font-size="9" fill="#A8FF3E">CO₂:{co2_level:.0f}</text>

          <!-- Zone D -->
          <circle cx="128" cy="242" r="7" fill="#FFD166" filter="url(#glow)">
            <animate attributeName="r" values="7;11;7" dur="3.5s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.9;0.35;0.9" dur="3.5s" repeatCount="indefinite"/>
          </circle>
          <circle cx="128" cy="242" r="4" fill="#FFD166"/>
          <text x="128" y="262" text-anchor="middle" font-family="Share Tech Mono,monospace" font-size="9" fill="#FFD166">pH:{ph_level:.1f}</text>

          <!-- Zone E -->
          <circle cx="400" cy="242" r="7" fill="#9B5DE5" filter="url(#glow)">
            <animate attributeName="r" values="7;11;7" dur="2.9s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.9;0.35;0.9" dur="2.9s" repeatCount="indefinite"/>
          </circle>
          <circle cx="400" cy="242" r="4" fill="#9B5DE5"/>
          <text x="400" y="262" text-anchor="middle" font-family="Share Tech Mono,monospace" font-size="9" fill="#9B5DE5">EC:{ec_level:.1f}</text>

          <!-- Zone F -->
          <circle cx="672" cy="242" r="7" fill="{"#EF476F" if soil_moisture<30 else "#00F5D4"}" filter="url(#glow)">
            <animate attributeName="r" values="7;11;7" dur="{"1.5" if soil_moisture<30 else "3"}s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.9;0.35;0.9" dur="{"1.5" if soil_moisture<30 else "3"}s" repeatCount="indefinite"/>
          </circle>
          <circle cx="672" cy="242" r="4" fill="{"#EF476F" if soil_moisture<30 else "#00F5D4"}"/>
          <text x="672" y="262" text-anchor="middle" font-family="Share Tech Mono,monospace" font-size="9" fill="{"#EF476F" if soil_moisture<30 else "#00F5D4"}">SM:{soil_moisture:.0f}%</text>

          <!-- Central hub -->
          <circle cx="400" cy="160" r="20" fill="rgba(0,245,212,0.08)" stroke="rgba(0,245,212,0.3)" stroke-width="1.5">
            <animate attributeName="r" values="20;26;20" dur="4s" repeatCount="indefinite"/>
          </circle>
          <circle cx="400" cy="160" r="10" fill="rgba(0,245,212,0.15)" stroke="rgba(0,245,212,0.6)" stroke-width="1"/>
          <text x="400" y="164" text-anchor="middle" font-family="Share Tech Mono,monospace" font-size="7" fill="#00F5D4">AI</text>
          <line x1="128" y1="78" x2="400" y2="160" stroke="rgba(0,245,212,0.12)" stroke-width="1" stroke-dasharray="3,6">
            <animate attributeName="stroke-dashoffset" from="60" to="0" dur="2s" repeatCount="indefinite"/>
          </line>
          <line x1="400" y1="78" x2="400" y2="160" stroke="rgba(0,187,249,0.12)" stroke-width="1" stroke-dasharray="3,6">
            <animate attributeName="stroke-dashoffset" from="60" to="0" dur="2.5s" repeatCount="indefinite"/>
          </line>
          <line x1="672" y1="78" x2="400" y2="160" stroke="rgba(168,255,62,0.12)" stroke-width="1" stroke-dasharray="3,6">
            <animate attributeName="stroke-dashoffset" from="60" to="0" dur="3s" repeatCount="indefinite"/>
          </line>
          <line x1="128" y1="242" x2="400" y2="160" stroke="rgba(255,209,102,0.12)" stroke-width="1" stroke-dasharray="3,6">
            <animate attributeName="stroke-dashoffset" from="60" to="0" dur="2.2s" repeatCount="indefinite"/>
          </line>
          <line x1="400" y1="242" x2="400" y2="160" stroke="rgba(155,93,229,0.12)" stroke-width="1" stroke-dasharray="3,6">
            <animate attributeName="stroke-dashoffset" from="60" to="0" dur="2.8s" repeatCount="indefinite"/>
          </line>
          <line x1="672" y1="242" x2="400" y2="160" stroke="rgba(239,71,111,0.12)" stroke-width="1" stroke-dasharray="3,6">
            <animate attributeName="stroke-dashoffset" from="60" to="0" dur="3.2s" repeatCount="indefinite"/>
          </line>
        </svg>'''

        st.markdown('<div class="dt-panel">', unsafe_allow_html=True)
        st.markdown(svg_map, unsafe_allow_html=True)
        st.markdown(f'''<div class="dt-legend">
            <div class="dt-legend-item"><div class="dt-legend-dot" style="background:#00F5D4"></div>Temperature Node</div>
            <div class="dt-legend-item"><div class="dt-legend-dot" style="background:#00BBF9"></div>Humidity Node</div>
            <div class="dt-legend-item"><div class="dt-legend-dot" style="background:#A8FF3E"></div>CO₂ Node</div>
            <div class="dt-legend-item"><div class="dt-legend-dot" style="background:#FFD166"></div>pH Node</div>
            <div class="dt-legend-item"><div class="dt-legend-dot" style="background:#9B5DE5"></div>EC Node</div>
            <div class="dt-legend-item"><div class="dt-legend-dot" style="background:#EF476F"></div>Soil Moisture Node</div>
        </div></div>''', unsafe_allow_html=True)

    with dt2:
        st.markdown(section_header("SENSOR NETWORK STATUS", "green", "◉"), unsafe_allow_html=True)
        sensors = [
            ("TEMP SENSOR",   temperature>35, f"{temperature:.1f}°C"),
            ("HUMIDITY",      humidity>85,    f"{humidity:.1f}%"),
            ("SOIL MOISTURE", soil_moisture<25,f"{soil_moisture:.1f}%"),
            ("CO₂ MONITOR",   co2_level<400,  f"{co2_level:.0f}ppm"),
            ("PH PROBE",      ph_level<5 or ph_level>8, f"{ph_level:.1f}"),
            ("EC SENSOR",     ec_level<0.8,   f"{ec_level:.2f}mS"),
            ("LIGHT SENSOR",  False,          f"{light_intensity:.0f}lx"),
            ("VENTILATION",   ventilation_rate<20, f"{ventilation_rate:.0f}%"),
        ]
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        for sname, has_alert, sval in sensors:
            sc = "#EF476F" if has_alert else "#A8FF3E"
            status = "ALERT" if has_alert else "ONLINE"
            st.markdown(f'''<div style="display:flex;align-items:center;justify-content:space-between;
                padding:0.4rem 0;border-bottom:1px solid rgba(0,245,212,0.06)">
                <div style="display:flex;align-items:center;gap:0.5rem">
                    <div style="width:6px;height:6px;border-radius:50%;background:{sc};
                        box-shadow:0 0 6px {sc};animation:ledBlink {1 if has_alert else 3}s ease-in-out infinite"></div>
                    <span style="font-family:Share Tech Mono,monospace;font-size:0.67rem;
                        color:var(--text-mid)">{sname}</span>
                </div>
                <div style="text-align:right">
                    <div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;color:{sc}">{sval}</div>
                    <div style="font-family:Share Tech Mono,monospace;font-size:0.52rem;color:{sc};
                        opacity:0.7;letter-spacing:0.1em">{status}</div>
                </div>
            </div>''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(section_header("TWIN SYNC STATUS", "blue", "◈"), unsafe_allow_html=True)
        st.markdown(f'''<div class="terminal-box">
            <span class="t-prompt">></span><span class="t-key">SYNC_STATUS</span>&nbsp;&nbsp;= <span class="t-ok">LIVE</span><br>
            <span class="t-prompt">></span><span class="t-key">CROP_TYPE</span>&nbsp;&nbsp;&nbsp;= <span class="t-val">{crop_type}</span><br>
            <span class="t-prompt">></span><span class="t-key">STAGE</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span class="t-val">{growth_stage}</span><br>
            <span class="t-prompt">></span><span class="t-key">HEALTH_SCORE</span>&nbsp;= <span class="{'t-ok' if sus_data['total']>=75 else 't-warn'}">{sus_data['total']}/100</span><br>
            <span class="t-prompt">></span><span class="t-key">RISK_LEVEL</span>&nbsp;&nbsp;= <span class="{'t-crit' if dis_data['overall']>60 else 't-warn' if dis_data['overall']>30 else 't-ok'}">{dis_data['level'].upper()}</span><br>
            <span class="t-prompt">></span><span class="t-key">IRR_STATUS</span>&nbsp;&nbsp;= <span class="t-val">{irr_data['status'].upper()}</span><br>
            <span class="t-prompt">></span><span class="t-key">TIMESTAMP</span>&nbsp;&nbsp;&nbsp;= <span class="t-val">{now.strftime("%H:%M:%S")}</span>
        </div>''', unsafe_allow_html=True)

        st.markdown(section_header("3D PLANT MODEL", "green", "◉"), unsafe_allow_html=True)
        components.iframe("https://my.spline.design/miniroom-0b666a0d244958ceef967db0b537c376/", height=240)

# =============================================================================
# TAB 3 — AI ASSISTANT
# =============================================================================
with tab_chat:
    st.markdown(section_header("AI AGRI-ASSISTANT — CHAT INTERFACE", "blue", "◈"), unsafe_allow_html=True)
    if not API_KEY:
        st.markdown('''<div class="terminal-box">
            <span style="color:var(--red)">ERROR</span> :: GEMINI_API_KEY not found.<br>
            Configure API key in .streamlit/secrets.toml to activate AI features.
        </div>''', unsafe_allow_html=True)
    else:
        for msg in st.session_state.main_messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if prompt := st.chat_input(f"Query: {crop_type} management, diseases, optimization…", key="main_chat"):
            st.session_state.main_messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                ph = st.empty()
                with ph.container():
                    if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=90, key="chat_thinking")
                    else: st.spinner("Processing…")
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    res = model.generate_content(
                        f"You are an expert AI assistant for {crop_type} at {growth_stage} stage. "
                        f"Sensors: Temp={temperature}°C, Humidity={humidity}%, Soil={soil_moisture}%, "
                        f"CO₂={co2_level}ppm, pH={ph_level}, EC={ec_level}mS/cm. Question: {prompt}"
                    )
                    ph.empty(); st.markdown(res.text)
                    st.session_state.main_messages.append({"role":"assistant","content":res.text})
                except Exception as e:
                    ph.empty(); st.error(f"Error: {e}")

# =============================================================================
# TAB 4 — VISION SCANNER
# =============================================================================
with tab_vision:
    st.markdown(section_header("AI DISEASE VISION SCANNER", "amber", "◉"), unsafe_allow_html=True)
    if not API_KEY:
        st.markdown('''<div class="terminal-box"><span style="color:var(--red)">ERROR</span>
            :: GEMINI_API_KEY not found. Vision module offline.</div>''', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            img_file = st.file_uploader("Upload Plant Image", type=["jpg","png"]) or st.camera_input("Capture Frame")
            if img_file:
                image = Image.open(img_file)
                st.image(image, use_column_width=True)
        with c2:
            if img_file:
                st.markdown(f'''<div class="glass-card glass-card-blue" style="padding:0.8rem">
                    <div style="font-family:Share Tech Mono,monospace;font-size:0.62rem;
                        color:var(--blue);letter-spacing:0.2em;margin-bottom:0.5rem">◈ SCAN PARAMETERS</div>
                    <div style="font-family:Share Tech Mono,monospace;font-size:0.7rem;line-height:1.9;color:var(--text-mid)">
                        TARGET &nbsp;&nbsp;= {crop_type}<br>
                        STAGE &nbsp;&nbsp;&nbsp;= {growth_stage}<br>
                        MODE &nbsp;&nbsp;&nbsp;&nbsp;= PATHOLOGY SCAN<br>
                        DEPTH &nbsp;&nbsp;&nbsp;= FULL ANALYSIS<br>
                        OUTPUT &nbsp;&nbsp;= TREATMENT PLAN
                    </div>
                </div>''', unsafe_allow_html=True)
                if st.button("▶ Initiate Vision Scan", type="primary", use_container_width=True):
                    ph = st.empty()
                    with ph.container():
                        if lottie_ai_thinking: st_lottie(lottie_ai_thinking, height=130, key="vision_scan")
                        else: st.spinner("Scanning…")
                    try:
                        v_model = genai.GenerativeModel(get_working_vision_model())
                        res = v_model.generate_content([
                            f"Identify diseases on this {crop_type} plant at {growth_stage} stage. "
                            f"Provide: 1) Disease identification with confidence %, "
                            f"2) Severity assessment, 3) Structured 3-step treatment plan.", image
                        ])
                        ph.empty(); st.session_state.vision_result = res.text
                    except Exception as e:
                        ph.empty(); st.error(f"Error: {e}")

        if st.session_state.vision_result:
            st.markdown(section_header("SCAN RESULTS — PATHOLOGY REPORT", "red", "⬡"), unsafe_allow_html=True)
            st.markdown(f'''<div class="terminal-box">
                <span style="font-family:Orbitron,sans-serif;font-size:0.55rem;color:var(--cyan);letter-spacing:0.3em">
                ◈ AI VISION ANALYSIS · {crop_type.upper()}</span><br><br>
                <span style="color:var(--text-mid);font-family:Rajdhani,sans-serif;font-size:0.9rem;line-height:1.75">
                {st.session_state.vision_result}</span>
            </div>''', unsafe_allow_html=True)

            st.markdown(section_header("FOLLOW-UP ANALYSIS", "blue", "◈"), unsafe_allow_html=True)
            for msg in st.session_state.vision_messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            if vp := st.chat_input("Ask about treatment, pathology, or prevention…", key="vision_chat_input"):
                st.session_state.vision_messages.append({"role":"user","content":vp})
                with st.chat_message("user"): st.markdown(vp)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing…"):
                        try:
                            model = genai.GenerativeModel(get_working_text_model())
                            r = model.generate_content(
                                f"You are a plant pathologist specializing in {crop_type}. "
                                f"Scan result: '{st.session_state.vision_result}'. Question: {vp}"
                            )
                            st.markdown(r.text)
                            st.session_state.vision_messages.append({"role":"assistant","content":r.text})
                        except Exception as e: st.error(f"Error: {e}")

# =============================================================================
# TAB 5 — MARKET ANALYZER
# =============================================================================
with tab_finance:
    st.markdown(section_header("AGRICULTURAL INTELLIGENCE — MARKET ANALYZER", "blue", "◈"), unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    plants_count = f1.number_input("Total Plants",         value=1000,  step=100)
    yield_per    = f2.number_input("Yield per Plant (kg)", value=0.15,  step=0.05)
    mkt_price    = f3.number_input("Market Price (Rs/kg)", value=450.0, step=10.0)

    tot_yield = plants_count * yield_per
    gross     = tot_yield * mkt_price
    cost      = (plants_count * 15) + 5000
    profit    = gross - cost
    roi       = profit / cost * 100

    st.markdown('<div class="div-label"><span>FINANCIAL PROJECTIONS</span></div>', unsafe_allow_html=True)

    km1,km2,km3,km4 = st.columns(4)
    km1.metric("📦 Total Yield",    f"{tot_yield:.1f} kg")
    km2.metric("💸 Running Cost",   f"Rs. {cost:,.2f}")
    km3.metric("💰 Net Profit",     f"Rs. {profit:,.2f}", delta=f"{roi:.1f}% ROI")
    km4.metric("📈 Gross Revenue",  f"Rs. {gross:,.2f}")

    # Revenue chart
    fig_bar = go.Figure()
    cats   = ["Gross Revenue","Running Cost","Net Profit"]
    vals   = [gross, cost, profit]
    colors = ["#00F5D4","#EF476F","#A8FF3E"]
    fig_bar.add_trace(go.Bar(x=cats, y=vals, marker_color=colors,
        marker_line_color="rgba(0,0,0,0.5)", marker_line_width=1,
        text=[f"Rs.{v:,.0f}" for v in vals], textposition="auto",
        textfont=dict(family="Share Tech Mono", size=10, color="#050816")))
    fig_bar.update_layout(**PLOTLY_LAYOUT)
    fig_bar.update_layout(height=280, showlegend=False, bargap=0.35)
    plotly_cyber_axes(fig_bar)
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f'''<div class="terminal-box">
            <span style="font-family:Orbitron,sans-serif;font-size:0.55rem;color:var(--cyan);
                letter-spacing:0.25em">◈ MARKET INTELLIGENCE REPORT</span><br><br>
            <span class="t-prompt">></span><span class="t-key"> HARVEST_TIMING</span>&nbsp;&nbsp;&nbsp;= <span class="t-ok">OPTIMAL</span><br>
            <span class="t-prompt">></span><span class="t-key"> DEMAND_TREND</span>&nbsp;&nbsp;&nbsp;&nbsp;= <span class="t-ok">+12% ↑ (supermarkets)</span><br>
            <span class="t-prompt">></span><span class="t-key"> REC_PRICE</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span class="t-val">Rs. {mkt_price*1.12:.0f}/kg (+12%)</span><br>
            <span class="t-prompt">></span><span class="t-key"> YIELD_TOTAL</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span class="t-val">{tot_yield:.2f} kg</span><br>
            <span class="t-prompt">></span><span class="t-key"> ROI</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span class="{'t-ok' if roi>0 else 't-crit'}">{roi:.1f}%</span><br>
            <span class="t-prompt">></span><span class="t-key"> PRODUCE_TYPE</span>&nbsp;&nbsp;&nbsp;&nbsp;= <span class="t-val">HYDROPONIC · PREMIUM</span>
        </div>''', unsafe_allow_html=True)

    with mc2:
        with st.expander("▸ Detailed Cost Breakdown"):
            st.markdown(f'''<div class="terminal-box">
                <span class="t-key">PLANT STOCK</span>&nbsp;&nbsp;= <span class="t-warn">Rs. {plants_count*15:,.2f}</span><br>
                <span class="t-key">OVERHEAD</span>&nbsp;&nbsp;&nbsp;&nbsp;= <span class="t-warn">Rs. 5,000.00</span><br>
                <span class="t-key">TOTAL COST</span>&nbsp;&nbsp;= <span class="t-crit">Rs. {cost:,.2f}</span><br>
                <span class="t-key">GROSS REV.</span>&nbsp;&nbsp;= <span class="t-ok">Rs. {gross:,.2f}</span><br>
                <span class="t-key">NET PROFIT</span>&nbsp;&nbsp;= <span class="{'t-ok' if profit>0 else 't-crit'}">Rs. {profit:,.2f}</span>
            </div>''', unsafe_allow_html=True)

    st.markdown(section_header("AI FINANCIAL ADVISOR", "blue", "◈"), unsafe_allow_html=True)
    for msg in st.session_state.market_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if mp := st.chat_input("Query: market strategy, pricing, profitability…", key="market_chat_input"):
        st.session_state.market_messages.append({"role":"user","content":mp})
        with st.chat_message("user"): st.markdown(mp)
        with st.chat_message("assistant"):
            with st.spinner("Calculating…"):
                try:
                    model = genai.GenerativeModel(get_working_text_model())
                    r = model.generate_content(
                        f"You are an agri-financial advisor. {plants_count} plants of {crop_type}, "
                        f"yield {tot_yield}kg, price Rs.{mkt_price}/kg, cost Rs.{cost}, "
                        f"profit Rs.{profit} ({roi:.1f}% ROI). Question: {mp}"
                    )
                    st.markdown(r.text)
                    st.session_state.market_messages.append({"role":"assistant","content":r.text})
                except Exception as e: st.error(f"Error: {e}")

# =============================================================================
# TAB 6 — HARDWARE & AUTOMATION
# =============================================================================
with tab_hardware:
    st.markdown(section_header("SMART HARDWARE & AUTOMATION INTERFACE", "amber", "⚙"), unsafe_allow_html=True)
    hw1, hw2 = st.columns(2)
    with hw1:
        st.markdown(section_header("HARDWARE API ENDPOINT", "blue", "◈"), unsafe_allow_html=True)
        st.markdown('''<div class="glass-card glass-card-blue">
            <div style="font-family:Rajdhani,sans-serif;font-size:0.88rem;color:var(--text-mid);
                line-height:1.65;margin-bottom:0.75rem">
                Connect ESP32, Arduino, or Raspberry Pi to this dashboard via REST API.
                Send sensor readings in JSON format for real-time Digital Twin synchronization.
            </div>
        </div>''', unsafe_allow_html=True)
        st.code('POST https://agritwin-ai.streamlit.app/api/v1/sensors\n{\n  "api_key": "YOUR_SECRET_KEY",\n  "temp": 28.5,\n  "hum": 65.2,\n  "ec": 1.8,\n  "ph": 6.2,\n  "co2": 850\n}', language="json")
        st.button("↺ Generate New API Key", use_container_width=True)

        with st.expander("▸ Integration Guide & Supported Hardware"):
            st.markdown(f'''<div class="terminal-box">
                <span class="t-key">ENDPOINT</span>&nbsp;&nbsp;&nbsp;= POST /api/v1/sensors<br>
                <span class="t-key">AUTH</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= Bearer API_KEY<br>
                <span class="t-key">FORMAT</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= application/json<br>
                <span class="t-key">INTERVAL</span>&nbsp;&nbsp;&nbsp;= 30–60s recommended<br>
                <span class="t-key">FIELDS</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= temp,hum,ec,ph,co2,lux<br>
                <br>
                <span class="t-key">ESP32-WROOM</span>&nbsp;&nbsp;= <span class="t-ok">COMPATIBLE</span><br>
                <span class="t-key">ARDUINO UNO</span>&nbsp;&nbsp;= <span class="t-ok">COMPATIBLE</span><br>
                <span class="t-key">RASPBERRY PI</span>&nbsp;= <span class="t-ok">COMPATIBLE</span><br>
                <span class="t-key">DHT22</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span class="t-ok">SUPPORTED</span><br>
                <span class="t-key">EC PROBE</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span class="t-ok">SUPPORTED</span><br>
                <span class="t-key">PH ELECTRODE</span>&nbsp;= <span class="t-warn">BETA</span>
            </div>''', unsafe_allow_html=True)

    with hw2:
        st.markdown(section_header("ALERT RULE CONFIGURATION", "amber" if temperature > 30 else "green", "⚡"), unsafe_allow_html=True)
        phone_num      = st.text_input("Alert Destination (+country code)", value="+947XXXXXXXX")
        temp_threshold = st.slider("Temperature Alert Threshold (°C):", 25.0, 45.0, 35.0)
        ec_threshold   = st.slider("EC Alert Floor (mS/cm):",           0.5,  2.0,  1.2)
        if st.button("▶ Save Alert Configuration", type="primary", use_container_width=True):
            st.success("✅ Automation rules committed. Monitoring active.")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if temperature > temp_threshold:
            st.markdown(f'''<div class="terminal-box" style="border-color:var(--red);color:var(--red)">
                <span style="font-family:Orbitron,sans-serif;font-size:0.58rem;letter-spacing:0.2em">
                ⚡ ALERT TRIGGERED</span><br><br>
                <span class="t-prompt">></span> CONDITION  :: TEMP {temperature}°C &gt; LIMIT {temp_threshold}°C<br>
                <span class="t-prompt">></span> ACTION     :: DISPATCH → {phone_num}<br>
                <span class="t-prompt">></span> PROTOCOL   :: WHATSAPP NOTIFICATION<br>
                <span class="t-prompt">></span> STATUS     :: ALERT SENT
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div class="terminal-box" style="color:var(--text-dim)">
                <span style="font-family:Orbitron,sans-serif;font-size:0.58rem;
                    letter-spacing:0.2em;color:var(--green)">● MONITORING ACTIVE</span><br><br>
                <span class="t-prompt">></span> TEMP     :: {temperature}°C ≤ {temp_threshold}°C &nbsp;<span class="t-ok">✓ OK</span><br>
                <span class="t-prompt">></span> EC       :: {ec_level:.2f} mS/cm ≥ {ec_threshold:.2f} &nbsp;<span class="{'t-ok' if ec_level>=ec_threshold else 't-warn'}">{'✓ OK' if ec_level>=ec_threshold else '⚠ LOW'}</span><br>
                <span class="t-prompt">></span> DEST     :: {phone_num}<br>
                <span class="t-prompt">></span> STATUS   :: STANDBY · NO ALERTS
            </div>''', unsafe_allow_html=True)
ENDOFFILE
echo "Done: $(wc -l < /mnt/user-data/outputs/app.py) lines"
