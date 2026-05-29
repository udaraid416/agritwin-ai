# 🌿 AgriTwin AI
## AI-Powered Predictive Digital Twin for Smart Protected Agriculture

> A production-level, cinematic Streamlit dashboard combining real-time sensor analytics,
> AI-driven predictions (Google Gemini / OpenAI / rule-based fallback), sustainability
> scoring, disease risk forecasting, and irrigation intelligence — all in a dark
> glassmorphism UI.

---

## 📸 Features at a Glance

| Module | Description |
|--------|-------------|
| **Live Dashboard** | Temperature, humidity, soil moisture, CO₂, light, ventilation KPIs |
| **Sustainability Score** | Weighted 5-dimension score (0-100) with grade A–D |
| **Disease Risk Engine** | Botrytis, Powdery Mildew, Root Rot, Fusarium probability |
| **Irrigation Intelligence** | ET₀-based urgency index, volume & timing recommendations |
| **Heat Stress Predictor** | Heat index with severity classification |
| **AI Recommendation Panel** | Gemini / OpenAI / rule-based actionable farm advice |
| **Zone Heatmaps** | 3×4 temperature and humidity heatmaps |
| **24h Climate Trend** | Interactive Plotly line charts |
| **Performance Evaluation** | Prediction accuracy, system efficiency, GH performance |

---

## 🚀 Quick Start (Local)

### 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/agritwin-ai.git
cd agritwin-ai
```

### 2 — Create & activate a virtual environment
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Set up API keys (optional — app works without them)
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY or OPENAI_API_KEY
```

Load `.env` before running (or export manually):
```bash
# Option A — use python-dotenv auto-loading
#   (already wired in config.py if you pip install python-dotenv)

# Option B — export manually
export GEMINI_API_KEY="your_key_here"
```

### 5 — Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔑 API Key Integration

### Google Gemini (Preferred — FREE tier)
1. Visit https://aistudio.google.com/app/apikey
2. Create a new API key (free, no credit card required)
3. Add to your `.env`:
   ```
   GEMINI_API_KEY=AIza...
   ```

### OpenAI (Optional fallback)
1. Visit https://platform.openai.com/api-keys
2. Create a key
3. Add to `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

### No API key? No problem.
The **Rule-Based AI Engine** activates automatically — it still produces intelligent,
context-aware recommendations indistinguishable from a lightweight LLM.

---

## ☁️ Deploy to Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit — AgriTwin AI v2.1"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/agritwin-ai.git
git push -u origin main
```

> ⚠️ **Important**: Add `.env` to `.gitignore` before pushing!
> ```bash
> echo ".env" >> .gitignore
> ```

### Step 2 — Connect to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click **"New app"** → connect your GitHub repo
3. Set **Main file path** → `app.py`
4. Click **Advanced settings** → **Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. Click **Deploy**

Your app will be live at:
`https://YOUR_USERNAME-agritwin-ai-app-XXXXX.streamlit.app`

---

## 📁 Project Structure

```
agritwin-ai/
├── app.py                  # 🎨 Streamlit frontend (700+ lines, full UI)
├── ai_model.py             # 🤖 AI engine (Gemini → OpenAI → rule-based)
├── config.py               # ⚙️  Central config, thresholds, constants
├── utils.py                # 🔢 Analytics: scoring, disease, irrigation
├── requirements.txt        # 📦 Python dependencies
├── .env.example            # 🔑 API key template
├── .gitignore              # 🚫 Ignore .env and __pycache__
├── README.md               # 📖 This file
└── .streamlit/
    └── config.toml         # 🎨 Dark theme settings
```

---

## 🧮 Scoring Methodology

### Sustainability Score (0–100)
| Dimension | Weight | Formula |
|-----------|--------|---------|
| Water Efficiency | 25% | f(soil moisture, humidity, ventilation) |
| Energy Efficiency | 20% | f(temperature, ventilation, light) |
| Climate Optimisation | 20% | f(temp, humidity, CO₂) |
| Disease Prevention | 15% | f(humidity, temp, ventilation) |
| Yield Potential | 20% | f(all 5 sensors) |

### Disease Risk
Pathogen-specific models for Botrytis, Powdery Mildew, Root Rot, and Fusarium
based on temperature–humidity interaction coefficients.

### Irrigation Urgency
Uses a simplified Penman-Monteith ET₀ model adjusted by crop stage
coefficient (Kc) and current soil moisture deficit.

---

## 🎨 UI Design Principles

- **Dark futuristic glassmorphism** — deep teal accent on near-black background
- **Orbitron** display font for metrics; **Exo 2** for body
- CSS animation on all cards (fadeSlideDown, pulseGlow)
- Animated progress bars (progressFill keyframe)
- Live blinking dot indicator
- Fully interactive Plotly charts with transparent backgrounds

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit 1.35+ |
| Charts | Plotly 5.x |
| AI (primary) | Google Gemini 1.5 Flash |
| AI (fallback) | OpenAI GPT-3.5 Turbo |
| AI (offline) | Rule-Based Engine v2.1 |
| Numerics | NumPy, Python math |
| Deployment | Streamlit Cloud (free) |

---

## 📜 License

MIT License — free for personal and commercial use.

---

*Built with ❤️ for sustainable agriculture and precision farming.*
