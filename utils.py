"""
utils.py — Helper functions for the Digital Twin Agriculture system
"""

from __future__ import annotations
import math
import random
import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from config import (
    TEMPERATURE_THRESHOLDS, HUMIDITY_THRESHOLDS,
    SOIL_THRESHOLDS, CO2_THRESHOLDS,
    SUSTAINABILITY_WEIGHTS, GRADE_COLORS, RISK_COLORS, ZONES
)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def score_param(value: float, thresholds: dict) -> float:
    """Return 0-100 score for a parameter given its threshold dict."""
    ol = thresholds.get("optimal_low",  thresholds.get("low", 0))
    oh = thresholds.get("optimal_high", thresholds.get("high", 100))
    cl = thresholds.get("critical_low", ol - 20)
    ch = thresholds.get("critical_high", oh + 20)

    if ol <= value <= oh:
        return 100.0
    if value < cl or value > ch:
        return 0.0
    if value < ol:
        return max(0, (value - cl) / max(ol - cl, 1) * 100)
    return max(0, (ch - value) / max(ch - oh, 1) * 100)


def calculate_water_efficiency(soil_moisture: float, humidity: float,
                                ventilation: float) -> float:
    sm_score  = score_param(soil_moisture, SOIL_THRESHOLDS)
    hum_score = score_param(humidity, HUMIDITY_THRESHOLDS)
    vent_eff  = min(100, ventilation * 1.2) if ventilation < 80 else 100
    return round((sm_score * 0.5 + hum_score * 0.3 + vent_eff * 0.2), 2)


def calculate_energy_efficiency(temperature: float, ventilation: float,
                                  light: float) -> float:
    temp_score  = score_param(temperature, TEMPERATURE_THRESHOLDS)
    vent_score  = 100 - abs(ventilation - 65) * 1.2
    light_score = min(100, light / 50)
    return round((temp_score * 0.4 + max(0, vent_score) * 0.3 + light_score * 0.3), 2)


def calculate_climate_score(temp: float, humidity: float, co2: float) -> float:
    ts = score_param(temp, TEMPERATURE_THRESHOLDS)
    hs = score_param(humidity, HUMIDITY_THRESHOLDS)
    cs = score_param(co2, CO2_THRESHOLDS)
    return round((ts * 0.4 + hs * 0.35 + cs * 0.25), 2)


def calculate_disease_prevention_score(humidity: float, temp: float,
                                        ventilation: float) -> float:
    """Lower humidity + ideal temp + good ventilation → lower disease risk."""
    hum_risk = max(0, humidity - 70) * 2          # penalty above 70 %
    temp_risk = max(0, abs(temp - 23) - 3) * 1.5  # penalty outside 20–26 °C
    vent_bonus = min(20, ventilation / 5)
    raw = 100 - hum_risk - temp_risk + vent_bonus
    return round(max(0, min(100, raw)), 2)


def calculate_yield_potential(temp: float, humidity: float,
                               soil: float, co2: float, light: float) -> float:
    ts = score_param(temp, TEMPERATURE_THRESHOLDS)
    hs = score_param(humidity, HUMIDITY_THRESHOLDS)
    ss = score_param(soil, SOIL_THRESHOLDS)
    cs = score_param(co2, CO2_THRESHOLDS)
    ls = min(100, light / 50)
    return round((ts * 0.25 + hs * 0.20 + ss * 0.25 + cs * 0.15 + ls * 0.15), 2)


def compute_sustainability_score(params: Dict[str, Any]) -> Dict[str, Any]:
    water  = calculate_water_efficiency(params["soil_moisture"],
                                         params["humidity"],
                                         params["ventilation_rate"])
    energy = calculate_energy_efficiency(params["temperature"],
                                          params["ventilation_rate"],
                                          params["light_intensity"])
    climate= calculate_climate_score(params["temperature"],
                                      params["humidity"],
                                      params["co2_level"])
    disease= calculate_disease_prevention_score(params["humidity"],
                                                 params["temperature"],
                                                 params["ventilation_rate"])
    yield_ = calculate_yield_potential(params["temperature"],
                                        params["humidity"],
                                        params["soil_moisture"],
                                        params["co2_level"],
                                        params["light_intensity"])
    w = SUSTAINABILITY_WEIGHTS
    total  = (water  * w["water_efficiency"]
            + energy * w["energy_efficiency"]
            + climate* w["climate_optimization"]
            + disease* w["disease_prevention"]
            + yield_ * w["crop_yield_potential"])
    total  = round(min(100, max(0, total)), 1)
    grade  = ("A" if total >= 80 else "B" if total >= 65
              else "C" if total >= 50 else "D")
    return {
        "total": total, "grade": grade,
        "water_efficiency":     water,
        "energy_efficiency":    energy,
        "climate_optimization": climate,
        "disease_prevention":   disease,
        "yield_potential":      yield_,
        "grade_color":          GRADE_COLORS[grade],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE RISK
# ─────────────────────────────────────────────────────────────────────────────

def calculate_disease_risk(params: Dict[str, Any]) -> Dict[str, Any]:
    temp = params["temperature"]
    hum  = params["humidity"]
    soil = params["soil_moisture"]
    vent = params["ventilation_rate"]

    botrytis = max(0, (hum - 75) * 1.8 + max(0, 22 - temp) * 1.2)
    powdery  = max(0, (hum - 50) * 0.6 + max(0, temp - 26) * 1.5)
    root_rot = max(0, (soil - 70) * 1.5 + max(0, 28 - temp) * 0.8)
    fusarium  = max(0, (temp - 28) * 1.3 + (hum - 60) * 0.7)

    vent_mitigation = (vent / 100) * 15

    diseases = {
        "Botrytis (Gray Mold)":   min(100, botrytis - vent_mitigation),
        "Powdery Mildew":         min(100, powdery  - vent_mitigation * 0.5),
        "Root Rot":               min(100, root_rot),
        "Fusarium Wilt":          min(100, fusarium  - vent_mitigation * 0.7),
    }
    diseases = {k: max(0, round(v, 1)) for k, v in diseases.items()}
    overall  = round(sum(diseases.values()) / len(diseases), 1)
    level    = ("Low" if overall < 20 else "Moderate" if overall < 40
                else "High" if overall < 65 else "Critical")
    return {
        "overall": overall,
        "level":   level,
        "level_color": RISK_COLORS[level],
        "diseases": diseases,
        "top_risk": max(diseases, key=diseases.get),
    }


# ─────────────────────────────────────────────────────────────────────────────
# IRRIGATION INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_irrigation_need(params: Dict[str, Any]) -> Dict[str, Any]:
    soil = params["soil_moisture"]
    temp = params["temperature"]
    hum  = params["humidity"]
    lai  = params["leaf_area_index"]

    et0_base  = 0.0023 * (temp + 17.8) * math.sqrt(abs(temp - 5)) * 4.5
    et0       = et0_base * lai * 0.5
    kc        = {"Seedling": 0.6, "Vegetative": 0.9,
                 "Flowering": 1.1, "Fruiting": 1.2, "Harvest": 0.8}
    stage     = params.get("growth_stage", "Vegetative")
    etc       = et0 * kc.get(stage, 1.0)
    deficit   = max(0, 55 - soil) * 0.3
    urgency   = round(min(100, deficit * 3 + etc * 8), 1)
    volume_L  = round(max(0, (60 - soil) * 0.15 + etc * 0.5), 2)

    status = ("Not Required" if urgency < 15 else "Low Priority" if urgency < 35
              else "Recommended" if urgency < 60 else "Urgent" if urgency < 80
              else "Critical")
    next_h = max(1, round(24 - urgency / 5))

    return {
        "urgency":        urgency,
        "status":         status,
        "volume_liters":  volume_L,
        "evapotranspiration": round(etc, 3),
        "next_irrigation_hours": next_h,
        "status_color":   RISK_COLORS.get(
            "Low" if urgency < 35 else "Moderate" if urgency < 60
            else "High" if urgency < 80 else "Critical", "#00C896"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HEAT STRESS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_heat_stress(temp: float, humidity: float) -> Dict[str, Any]:
    hi = temp + 0.33 * (humidity / 100 * 6.105 *
         math.exp(17.27 * temp / (237.7 + temp))) - 4.0
    level = ("None" if hi < 27 else "Mild" if hi < 32
             else "Moderate" if hi < 38 else "Severe" if hi < 44 else "Extreme")
    return {"heat_index": round(hi, 1), "level": level,
            "percentage": min(100, max(0, round((hi - 20) * 2, 1)))}


# ─────────────────────────────────────────────────────────────────────────────
# ZONE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def generate_zone_heatmap_data(base_temp: float, base_hum: float
                                ) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    temp_grid = base_temp + rng.normal(0, 1.5, (3, 4))
    hum_grid  = base_hum  + rng.normal(0, 3.0, (3, 4))
    return np.clip(temp_grid, 10, 50), np.clip(hum_grid, 20, 100)


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL TREND SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_trend_data(current_temp: float, current_hum: float,
                         hours: int = 24) -> Dict[str, List]:
    now = datetime.datetime.now()
    times, temps, hums, soils = [], [], [], []
    rng = random.Random(42)
    for i in range(hours, 0, -1):
        t = now - datetime.timedelta(hours=i)
        noise_t = rng.gauss(0, 1.5)
        noise_h = rng.gauss(0, 3)
        noise_s = rng.gauss(0, 2)
        times.append(t.strftime("%H:%M"))
        temps.append(round(current_temp + noise_t, 1))
        hums.append(round(min(100, max(0, current_hum + noise_h)), 1))
        soils.append(round(min(100, max(0, 55 + noise_s)), 1))
    return {"times": times, "temperatures": temps,
            "humidities": hums, "soil_moisture": soils}


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_system_performance(sustainability: Dict, disease: Dict,
                                  irrigation: Dict) -> Dict[str, Any]:
    pred_acc  = round(100 - disease["overall"] * 0.3 + sustainability["total"] * 0.1, 1)
    pred_acc  = min(99, max(60, pred_acc))
    sys_eff   = round((sustainability["total"] +
                       (100 - irrigation["urgency"])) / 2, 1)
    gh_perf   = round((sustainability["total"] * 0.6 +
                       (100 - disease["overall"]) * 0.4), 1)
    overall   = round((pred_acc + sys_eff + gh_perf) / 3, 1)
    grade     = ("A" if overall >= 80 else "B" if overall >= 65
                 else "C" if overall >= 50 else "D")
    return {
        "prediction_accuracy":   pred_acc,
        "system_efficiency":     sys_eff,
        "greenhouse_performance": gh_perf,
        "overall": overall, "grade": grade,
        "grade_color": GRADE_COLORS[grade],
    }


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def format_recommendation(text: str) -> str:
    """Clean up whitespace from AI response."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return "\n".join(lines)
