"""
ai_model.py — AI Prediction Engine
Tries Gemini → OpenAI → Rule-based fallback (all feel intelligent)
"""

from __future__ import annotations
import json
import re
from typing import Dict, Any, Optional
from config import (
    GEMINI_API_KEY, OPENAI_API_KEY,
    GEMINI_MODEL, OPENAI_MODEL, MAX_TOKENS, TEMPERATURE
)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(params: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    return f"""You are an expert agricultural AI consultant specialising in smart greenhouse systems.
Analyse the following real-time sensor data and provide concise, actionable recommendations.

## GREENHOUSE STATUS
Crop: {params['crop_type']}  |  Growth Stage: {params['growth_stage']}
Temperature: {params['temperature']}°C  |  Humidity: {params['humidity']}%
Soil Moisture: {params['soil_moisture']}%  |  CO₂: {params['co2_level']} ppm
Light Intensity: {params['light_intensity']} lux  |  pH: {params['ph_level']}
EC Level: {params['ec_level']} mS/cm  |  Ventilation: {params['ventilation_rate']}%
Leaf Area Index: {params['leaf_area_index']}

## PRE-COMPUTED ANALYTICS
Sustainability Score: {analysis['sustainability']['total']}/100 (Grade {analysis['sustainability']['grade']})
Disease Risk: {analysis['disease']['overall']}% ({analysis['disease']['level']})
Top Disease Threat: {analysis['disease']['top_risk']}
Irrigation Urgency: {analysis['irrigation']['urgency']}% ({analysis['irrigation']['status']})
Heat Stress: {analysis['heat_stress']['level']} (Index {analysis['heat_stress']['heat_index']}°C)

## YOUR TASK
Respond ONLY in the following JSON structure (no markdown, no extra text):
{{
  "disease_warning":       "<1-2 sentence disease risk explanation>",
  "irrigation_advice":     "<1-2 sentence irrigation recommendation>",
  "climate_warning":       "<1-2 sentence climate/environment advice>",
  "sustainability_insight":"<1-2 sentence sustainability explanation>",
  "top_actions": [
    "<Priority action 1>",
    "<Priority action 2>",
    "<Priority action 3>"
  ],
  "overall_assessment": "<2-3 sentence overall greenhouse health assessment>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        cfg   = genai.types.GenerationConfig(
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        resp = model.generate_content(prompt, generation_config=cfg)
        return resp.text
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI
# ─────────────────────────────────────────────────────────────────────────────

def _call_openai(prompt: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp   = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[OpenAI] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _rule_based_analysis(params: Dict[str, Any],
                          analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Intelligent rule-based engine that reads like real AI output."""

    crop    = params["crop_type"]
    stage   = params["growth_stage"]
    temp    = params["temperature"]
    hum     = params["humidity"]
    soil    = params["soil_moisture"]
    co2     = params["co2_level"]
    vent    = params["ventilation_rate"]
    sus     = analysis["sustainability"]["total"]
    d_risk  = analysis["disease"]["overall"]
    d_top   = analysis["disease"]["top_risk"]
    irr     = analysis["irrigation"]

    # ── Disease warning ───────────────────────────────────────────────────
    if d_risk >= 65:
        dw = (f"CRITICAL: {d_top} probability is alarmingly high at {d_risk:.0f}%. "
              f"Immediate fungicide application and aggressive ventilation (target ≥75%) "
              f"are required to prevent crop loss in {crop} plants.")
    elif d_risk >= 40:
        dw = (f"Elevated {d_top} risk detected ({d_risk:.0f}%). "
              f"Reduce relative humidity below 70% and increase airflow. "
              f"Schedule a preventive bio-fungicide spray within 24 hours.")
    elif d_risk >= 20:
        dw = (f"Moderate disease pressure ({d_risk:.0f}%) — {d_top} is the primary concern. "
              f"Monitor leaf surfaces daily and maintain current ventilation protocol.")
    else:
        dw = (f"Disease pressure is low ({d_risk:.0f}%); pathogen conditions are unfavourable. "
              f"Continue standard IPM monitoring for {crop} at {stage} stage.")

    # ── Irrigation advice ─────────────────────────────────────────────────
    v_l = irr["volume_liters"]
    nxt = irr["next_irrigation_hours"]
    if irr["urgency"] >= 80:
        ia = (f"URGENT: Soil moisture critically low at {soil:.0f}%. "
              f"Apply {v_l:.1f} L/m² immediately via drip irrigation. "
              f"Check emitter pressure and substrate saturation.")
    elif irr["urgency"] >= 60:
        ia = (f"Irrigation recommended within {nxt} hours — apply {v_l:.1f} L/m². "
              f"ET₀ is {irr['evapotranspiration']:.2f} mm/hr; adjust fertigation EC to "
              f"{params['ec_level'] + 0.2:.1f} mS/cm for current {stage} demand.")
    elif irr["urgency"] >= 35:
        ia = (f"Schedule next irrigation in {nxt} hours ({v_l:.1f} L/m²). "
              f"Substrate moisture is adequate; monitor tensiometer readings closely.")
    else:
        ia = (f"Irrigation not required for approximately {nxt} hours. "
              f"Soil moisture at {soil:.0f}% is within optimal range for {crop}.")

    # ── Climate warning ───────────────────────────────────────────────────
    warnings = []
    if temp > 30:
        warnings.append(f"canopy temperature at {temp}°C exceeds the {crop} "
                         f"photosynthetic optimum — activate shade nets")
    if temp < 16:
        warnings.append(f"sub-optimal temperature ({temp}°C) may delay {stage} development")
    if hum > 85:
        warnings.append(f"very high humidity ({hum}%) promotes fungal sporulation")
    if co2 < 600:
        warnings.append(f"CO₂ at {co2:.0f} ppm is limiting photosynthesis "
                         f"(target 800–1000 ppm for {crop})")
    if vent < 40:
        warnings.append(f"poor ventilation ({vent:.0f}%) trapping heat and moisture")

    if warnings:
        cw = f"Climate alert: {'; '.join(warnings[:2]).capitalize()}."
    else:
        cw = (f"Greenhouse climate is well-balanced. Temperature ({temp}°C) and "
               f"humidity ({hum}%) are within optimal ranges for {crop} at {stage} stage.")

    # ── Sustainability insight ─────────────────────────────────────────────
    grade = analysis["sustainability"]["grade"]
    we    = analysis["sustainability"]["water_efficiency"]
    ee    = analysis["sustainability"]["energy_efficiency"]
    if sus >= 80:
        si = (f"Excellent sustainability performance (Grade {grade}, {sus}/100). "
               f"Resource utilisation is highly efficient — water score {we:.0f}/100, "
               f"energy score {ee:.0f}/100.")
    elif sus >= 65:
        si = (f"Good sustainability score ({sus}/100, Grade {grade}). "
               f"Minor optimisations in water efficiency ({we:.0f}/100) "
               f"and climate control could push you to Grade A.")
    elif sus >= 50:
        si = (f"Sustainability grade {grade} ({sus}/100) signals room for improvement. "
               f"Focus on reducing water waste and stabilising CO₂ levels "
               f"to increase resource efficiency by ~15 points.")
    else:
        si = (f"LOW sustainability score ({sus}/100, Grade {grade}). "
               f"Multiple systems need immediate optimisation — prioritise water "
               f"management and climate control overhaul.")

    # ── Top actions ───────────────────────────────────────────────────────
    actions = []
    if d_risk >= 40:
        actions.append(f"Apply preventive fungicide — {d_top} risk is {d_risk:.0f}%")
    if irr["urgency"] >= 60:
        actions.append(f"Irrigate {v_l:.1f} L/m² within {nxt}h — moisture deficit detected")
    if temp > 30:
        actions.append("Deploy shade netting to reduce canopy heat load")
    if hum > 80:
        actions.append("Increase ventilation rate to ≥70% to reduce humidity")
    if co2 < 600:
        actions.append("Activate CO₂ enrichment — current levels limit photosynthesis")
    if vent < 40:
        actions.append("Open roof vents and side walls to improve air circulation")
    if sus < 60:
        actions.append("Review energy consumption — HVAC schedule may need optimisation")

    if len(actions) < 3:
        actions += [
            f"Maintain current {stage} nutrition programme for {crop}",
            "Log and review sensor calibration data weekly",
            "Schedule next full system inspection in 7 days",
        ]
    actions = actions[:3]

    # ── Overall assessment ────────────────────────────────────────────────
    perf = analysis.get("performance", {})
    g_score = perf.get("overall", sus)
    oa = (f"Your {crop} greenhouse is operating at an overall performance index of "
           f"{g_score:.0f}/100 (Grade {grade}). "
           f"The primary risk vector is {d_top if d_risk > 20 else 'none — conditions are stable'}. "
           f"{'Immediate intervention is recommended.' if g_score < 55 else 'Continue current protocols and monitor sensor trends.'}")

    return {
        "disease_warning":        dw,
        "irrigation_advice":      ia,
        "climate_warning":        cw,
        "sustainability_insight": si,
        "top_actions":            actions,
        "overall_assessment":     oa,
        "source": "Rule-Based AI Engine v2.1"
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARSE LLM JSON SAFELY
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    try:
        # strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        return json.loads(cleaned)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_ai_recommendations(params: Dict[str, Any],
                             analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a structured recommendation dict.
    Tries Gemini → OpenAI → rule-based fallback.
    Always includes a 'source' key describing which engine was used.
    """
    prompt = _build_prompt(params, analysis)

    # 1. Try Gemini
    raw = _call_gemini(prompt)
    if raw:
        parsed = _parse_json(raw)
        if parsed:
            parsed["source"] = f"Google Gemini ({GEMINI_MODEL})"
            return parsed

    # 2. Try OpenAI
    raw = _call_openai(prompt)
    if raw:
        parsed = _parse_json(raw)
        if parsed:
            parsed["source"] = f"OpenAI ({OPENAI_MODEL})"
            return parsed

    # 3. Rule-based fallback
    return _rule_based_analysis(params, analysis)


def get_ai_engine_status() -> str:
    if GEMINI_API_KEY:
        return f"Google Gemini ({GEMINI_MODEL})"
    if OPENAI_API_KEY:
        return f"OpenAI ({OPENAI_MODEL})"
    return "Rule-Based AI Engine v2.1"
