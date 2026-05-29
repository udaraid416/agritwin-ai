"""
config.py — Central configuration for AI-Powered Predictive Digital Twin
for Smart Protected Agriculture
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# API KEYS  (set as environment variables — never hard-code)
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")   # export GEMINI_API_KEY=...
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")   # optional fallback

# ─────────────────────────────────────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-1.5-flash"                 # free-tier Gemini model
OPENAI_MODEL    = "gpt-3.5-turbo"
MAX_TOKENS      = 512
TEMPERATURE     = 0.35                               # low → deterministic advice

# ─────────────────────────────────────────────────────────────────────────────
# GREENHOUSE PHYSICAL PARAMETERS  (sensible defaults / thresholds)
# ─────────────────────────────────────────────────────────────────────────────
PARAM_DEFAULTS = {
    "temperature":       25.0,   # °C
    "humidity":          65.0,   # %
    "soil_moisture":     55.0,   # %
    "co2_level":        800.0,   # ppm
    "light_intensity":  3500.0,  # lux
    "ventilation_rate":   60.0,  # %
    "ph_level":           6.5,
    "ec_level":           2.2,   # mS/cm
    "leaf_area_index":    3.5,
    "growth_stage":      "Vegetative",
    "crop_type":         "Tomato",
}

CROP_TYPES      = ["Tomato", "Pepper", "Cucumber", "Lettuce", "Strawberry", "Basil"]
GROWTH_STAGES   = ["Seedling", "Vegetative", "Flowering", "Fruiting", "Harvest"]

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD TABLES
# ─────────────────────────────────────────────────────────────────────────────
TEMPERATURE_THRESHOLDS = {"critical_low": 10, "low": 15, "optimal_low": 18,
                           "optimal_high": 28, "high": 32, "critical_high": 38}
HUMIDITY_THRESHOLDS    = {"critical_low": 20, "low": 40, "optimal_low": 55,
                           "optimal_high": 80, "high": 88, "critical_high": 95}
SOIL_THRESHOLDS        = {"critical_low": 15, "low": 30, "optimal_low": 45,
                           "optimal_high": 75, "high": 85, "critical_high": 95}
CO2_THRESHOLDS         = {"low": 400, "optimal_low": 600, "optimal_high": 1200,
                           "high": 1500, "critical_high": 2000}

# ─────────────────────────────────────────────────────────────────────────────
# SUSTAINABILITY WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
SUSTAINABILITY_WEIGHTS = {
    "water_efficiency":       0.25,
    "energy_efficiency":      0.20,
    "climate_optimization":   0.20,
    "disease_prevention":     0.15,
    "crop_yield_potential":   0.20,
}

# ─────────────────────────────────────────────────────────────────────────────
# UI / THEME
# ─────────────────────────────────────────────────────────────────────────────
GRADE_COLORS   = {"A": "#00FF88", "B": "#7FFF00", "C": "#FFD700", "D": "#FF4444"}
RISK_COLORS    = {"Low": "#00C896", "Moderate": "#F59E0B", "High": "#EF4444", "Critical": "#9333EA"}
APP_TITLE      = "AgriTwin AI — Smart Protected Agriculture Digital Twin"
APP_ICON       = "🌿"
VERSION        = "2.1.0"

# ─────────────────────────────────────────────────────────────────────────────
# ZONE GRID (6-zone greenhouse)
# ─────────────────────────────────────────────────────────────────────────────
ZONES = ["Zone A1", "Zone A2", "Zone B1", "Zone B2", "Zone C1", "Zone C2"]
