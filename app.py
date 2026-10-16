from pathlib import Path
import os
import re

import joblib
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps
from dotenv import load_dotenv

# =========================================================
# CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parent
CROP_MODEL_PATH = ROOT / "models" / "crop_model.pkl"
YIELD_MODEL_PATH = ROOT / "models" / "yield_model.pkl"

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Mazhai AI Crop",
    page_icon="🌧️",
    layout="wide",
)

# =========================================================
# VISUAL DESIGN: FARM BACKGROUNDS AND CARDS
# =========================================================
st.markdown("""
<style>
:root { --farm-green:#118653; --farm-dark:#103b2b; --farm-light:#eaf6ee; --farm-ink:#18352a; }
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
/* Force readable contrast even when Streamlit's app theme is set to dark. */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: linear-gradient(180deg,#edf7f0 0%,#f8fcf9 48%,#eaf5ee 100%) !important;
  color: var(--farm-ink) !important;
}
[data-testid="stHeader"] { background: rgba(12,58,39,.94) !important; }
[data-testid="stToolbar"] { color: white !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 1500px; }
/* Text contrast fixes for Streamlit's generated styles */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li, .stApp label,
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stHeader"] * { color: var(--farm-ink) !important; }
.stApp h1 { font-size: 2rem !important; }
.stApp h2, .stApp h3 { color: #124b32 !important; }
.stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div {
  background: #fff !important; color: #18352a !important; border-color: #c7dfd0 !important;
}
.stApp input::placeholder, .stApp textarea::placeholder { color:#667b70 !important; }
[data-testid="stTabs"] [role="tablist"] { gap: .45rem; border-bottom: 2px solid #d5e9dc; }
[data-testid="stTabs"] button[role="tab"] { border-radius: 12px 12px 0 0; background: #dcefe3 !important; padding: .65rem 1rem; font-weight: 700; color:#17452f !important; }
[data-testid="stTabs"] button[aria-selected="true"] { background: #138451 !important; color: white !important; }
[data-testid="stTabs"] button[aria-selected="true"] * { color: white !important; }
[data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
.stButton > button, .stFormSubmitButton > button { background: linear-gradient(90deg,#168653,#27a96b) !important; color:white !important; border:0 !important; border-radius:12px; font-weight:700; }
.stButton > button *, .stFormSubmitButton > button * { color:white !important; }
[data-testid="stMetric"] { background: white !important; border: 1px solid #d9eade; border-radius: 14px; padding: .7rem; box-shadow: 0 3px 12px rgba(18,70,43,.06); }
[data-testid="stMetric"] label, [data-testid="stMetric"] [data-testid="stMetricValue"], [data-testid="stMetric"] [data-testid="stMetricDelta"] { color:#173d2c !important; }
[data-testid="stAlert"] p { color:#18352a !important; }
.farm-hero { min-height: 138px; display:flex; flex-direction:column; justify-content:center; padding: 1.4rem 1.6rem; margin: 0 0 1.2rem 0; border-radius: 18px; color:white !important; background-size:cover; background-position:center; box-shadow: inset 0 0 0 2000px rgba(7,38,24,.48); }
.farm-hero h2, .farm-hero p { color:white !important; text-shadow:0 1px 3px rgba(0,0,0,.6); }
.farm-hero h2 { margin:0 0 .35rem 0; font-size:1.8rem; }
.farm-hero p { margin:0; font-size:1rem; }
.brand-hero { min-height:160px; display:flex; flex-direction:column; justify-content:center; padding:1.6rem 1.8rem; margin:.2rem 0 1rem; border-radius:20px; color:#fff !important; background-image:linear-gradient(90deg,rgba(5,42,27,.88),rgba(9,72,43,.45)),url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85'); background-size:cover; background-position:center; box-shadow:0 8px 24px rgba(9,60,36,.15); }
.brand-hero h1, .brand-hero h2, .brand-hero p { color:#fff !important; text-shadow:0 1px 3px rgba(0,0,0,.45); margin:0; }
.brand-hero h1 { font-size:2.25rem !important; }
.brand-hero p { margin-top:.45rem; }
.helper-card { padding:.8rem 1rem; background:#edf8f0; color:#18352a; border-left:4px solid #168653; border-radius:10px; margin:.4rem 0 1rem; }
@media (max-width:700px) { .farm-hero { min-height:110px; padding:1rem; } .farm-hero h2 { font-size:1.3rem; } .brand-hero { min-height:130px; padding:1rem; } .brand-hero h1 { font-size:1.65rem !important; } }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root {
 --mz-forest:#143f2b; --mz-green:#398451; --mz-leaf:#79a96a;
 --mz-sage:#eaf1e7; --mz-cream:#fbfaf6; --mz-ink:#20382b; --mz-muted:#637466;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
 background:#f8faf6 !important; color:var(--mz-ink) !important;
}
[data-testid="stHeader"] { background:rgba(248,250,246,.94) !important; }
.block-container { max-width:1320px !important; padding-top:1.35rem !important; }
.stApp, .stApp button, .stApp input, .stApp textarea, .stApp select {
  font-family:'DM Sans', sans-serif !important;
}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,
.brand-hero h1,.brand-hero h2,.login-title {
  font-family:'Manrope','DM Sans',sans-serif !important;
}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label { color:var(--mz-ink); }
.stApp h2,.stApp h3 { font-weight:750 !important; }
.stApp p { line-height:1.7; }
.news-card {
  height:100%; padding:1rem 1.05rem; background:#fff;
  border:1px solid #e0e8dc; border-radius:15px;
  box-shadow:0 5px 18px rgba(31,66,39,.045);
}
.news-card .news-kicker { color:#4e8057; font-size:.72rem; font-weight:800;
  letter-spacing:.1em; text-transform:uppercase; margin-bottom:.45rem; }
.news-card .news-title { color:#20382b; font-weight:750; font-size:1rem;
  line-height:1.45; margin-bottom:.55rem; }
.news-card .news-meta { color:#718074; font-size:.78rem; }
.news-card a { color:#286b40 !important; font-weight:700; text-decoration:none; }
.news-card a:hover { text-decoration:underline; }
.section-eyebrow { color:#4e8057; text-transform:uppercase;
  font-size:.75rem; font-weight:800; letter-spacing:.13em; }
.stApp h1 { letter-spacing:-.035em; font-weight:760 !important; }
.stApp h2,.stApp h3 { letter-spacing:-.02em; }
[data-testid="stSidebar"] {
 background:#f0f4ed !important; border-right:1px solid #dce6d9;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#345340 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
 border-radius:11px; padding:.38rem .55rem; margin:.12rem 0;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#e1ebdd; }
.stButton > button,.stFormSubmitButton > button {
 background:#3e7d4e !important; color:#fff !important; border:1px solid #3e7d4e !important;
 border-radius:10px !important; font-weight:650 !important; min-height:2.65rem;
 transition:all .15s ease;
}
.stButton > button:hover,.stFormSubmitButton > button:hover {
 background:#285f3b !important; border-color:#285f3b !important;
}
.stApp input,.stApp textarea,[data-baseweb="select"] > div {
 border-radius:10px !important; border-color:#d5e0d2 !important;
}
[data-testid="stMetric"] {
 border:1px solid #e0e8dc !important; border-radius:14px !important;
 background:#fff !important; padding:1rem !important; box-shadow:0 3px 12px #254c3010;
}
[data-testid="stTabs"] [role="tablist"] { gap:.35rem; border-bottom:1px solid #d9e4d5; }
[data-testid="stTabs"] button[role="tab"] { border-radius:10px 10px 0 0; }
.brand-hero {
 min-height:265px !important; padding:2.3rem 2.5rem !important; border-radius:22px !important;
 background-image:linear-gradient(90deg,rgba(12,48,29,.86),rgba(16,65,35,.35)),
 url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85') !important;
 box-shadow:0 14px 36px rgba(26,63,35,.12) !important;
}
.brand-hero h1 { font-size:clamp(2.15rem,4vw,3.35rem) !important; line-height:1.05; }
.brand-hero h2 { font-size:clamp(1.1rem,2vw,1.55rem) !important; margin-top:.8rem !important; }
.brand-hero p { max-width:760px; font-size:1.03rem !important; line-height:1.7; }
.farm-hero { border-radius:17px !important; min-height:150px; box-shadow:none !important; }
.login-shell {
 max-width:560px; margin:2.2rem auto 1rem; padding:2.2rem;
 background:#fff; border:1px solid #e1e8dd; border-radius:22px;
 box-shadow:0 18px 55px rgba(31,66,39,.09);
}
.login-eyebrow { color:#477a4c; text-transform:uppercase; letter-spacing:.14em; font-size:.76rem; font-weight:750; }
.login-title { color:#193f2a; font-size:2.2rem; font-weight:780; letter-spacing:-.04em; margin:.45rem 0; }
.login-copy { color:#637466; line-height:1.65; margin-bottom:1.2rem; }
.feature-tile {
 background:#fff; border:1px solid #e1e8dd; border-radius:15px; padding:1rem 1.05rem;
 min-height:125px; box-shadow:0 5px 16px rgba(35,66,40,.035);
}
.feature-tile .emoji { font-size:1.5rem; margin-bottom:.35rem; }
.feature-tile h4 { color:#234b31; font-size:1rem; margin:.15rem 0 .35rem; }
.feature-tile p { color:#66776a; font-size:.88rem; line-height:1.45; margin:0; }
.home-eyebrow { color:#4d8151; font-size:.8rem; letter-spacing:.12em; text-transform:uppercase; font-weight:750; }
@media(max-width:700px) {
 .block-container { padding:1rem .85rem 2rem !important; }
 .brand-hero { min-height:220px !important; padding:1.25rem !important; }
 .login-shell { margin:1rem auto; padding:1.25rem; }
}
/* Fresh full-page farm backdrop; content panels stay translucent for readability. */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background:
    linear-gradient(135deg, rgba(239,248,237,.82), rgba(247,250,242,.72)),
    url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=2200&q=90')
    center center / cover fixed !important;
  color:var(--mz-ink) !important;
}
[data-testid="stHeader"] { background:rgba(245,250,242,.68) !important; }
[data-testid="stSidebar"] {
  background:rgba(239,246,234,.88) !important;
  backdrop-filter:blur(14px);
  border-right:1px solid rgba(205,224,199,.8) !important;
}
[data-testid="stMain"] .block-container {
  background:rgba(250,252,247,.40);
  border:1px solid rgba(255,255,255,.5);
  border-radius:24px;
  padding:1.35rem 1.4rem 2rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stForm"],
.news-card, [data-testid="stMetric"] {
  background:rgba(255,255,255,.88) !important;
  backdrop-filter:blur(10px);
}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,
.stApp p,.stApp label,.stApp [data-testid="stMarkdownContainer"] p {
  text-shadow:none;
}
/* Keep all banner text crisp and readable over photographs. */
.brand-hero h1,.brand-hero h2,.brand-hero p,
.farm-hero h2,.farm-hero p {
  color:#fff !important;
  text-shadow:0 2px 5px rgba(0,0,0,.72) !important;
}
.brand-hero {
  border:1px solid rgba(255,255,255,.24);
  background-image:
    linear-gradient(90deg,rgba(4,43,29,.90),rgba(8,65,39,.52)),
    url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=90') !important;
}
.farm-hero {
  border:1px solid rgba(255,255,255,.25);
  box-shadow:0 10px 26px rgba(17,62,35,.13) !important;
}
@media(max-width:700px) {
  [data-testid="stMain"] .block-container { padding:.85rem !important; border-radius:16px; }
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background-attachment:scroll !important; }
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TAMIL AND ENGLISH TEXT
# =========================================================
TEXT = {
    "en": {
        "language": "Language / மொழி",
        "title": "🌧️ Mazhai AI Crop",
        "subtitle": "Smart Crop Recommendation, Yield Prediction & Fertilizer Guidance",
        "tagline": "Smarter Farming, Better Harvests.",
        "intro": "An educational AI-powered agricultural assistant for crop recommendations, yield estimates, live weather, and basic fertilizer guidance.",
        "weather": "🌦️ Live Weather",
        "city": "Enter your city",
        "get_weather": "Get Weather",
        "use_weather": "Apply Live Temperature and Humidity",
        "weather_applied": "Live temperature and humidity applied to the crop form. Review rainfall and soil values before predicting.",
        "missing_key": "Weather API key missing. Add OPENWEATHER_API_KEY to your .env file.",
        "empty_city": "Please enter a city name.",
        "invalid_key": "The weather API key is invalid or not active yet.",
        "city_not_found": "City not found. Please check the spelling.",
        "weather_error": "Unable to retrieve weather. Check your connection and API key.",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "conditions": "Conditions",
        "rain_recent": "Rain in the last hour",
        "weather_note": "Current-hour rainfall is not the same as seasonal rainfall. Enter a suitable rainfall value for the crop model.",
        "crop_tab": "🌱 Crop Recommendation",
        "yield_tab": "🌾 Yield Prediction",
        "fertilizer_tab": "🧪 Fertilizer Guidance",
        "about_tab": "ℹ️ About",
        "soil_heading": "Soil and Weather Information",
        "soil_help": "Enter values from a soil test and suitable local climate measurements.",
        "nitrogen": "Nitrogen (N)",
        "phosphorus": "Phosphorus (P)",
        "potassium": "Potassium (K)",
        "ph": "Soil pH",
        "rainfall": "Rainfall (mm)",
        "recommend": "🌾 Recommend Suitable Crop",
        "crop_model_missing": "Crop model not found. Check models/crop_model.pkl.",
        "crop_result": "🌱 Suggested crop: {crop}",
        "alternatives": "Top 5 Alternative Crops",
        "model_score": "Model score (%)",
        "score_note": "Model scores are classifier outputs, not probabilities of farming success.",
        "crop_error": "Crop prediction failed: {error}",
        "yield_heading": "🌾 Crop Yield Prediction",
        "yield_help": "Estimate yield from historical crop, location, year, rainfall, pesticide usage, and temperature data.",
        "yield_model_missing": "Yield model not found. Run python train_yield_model.py first.",
        "yield_units": "Yield is converted from hg/ha to tonnes per hectare.",
        "area": "Country / Area (use dataset spelling)",
        "crop": "Crop",
        "year": "Year",
        "annual_rainfall": "Average annual rainfall (mm)",
        "pesticide": "Pesticide usage (tonnes)",
        "average_temp": "Average temperature (°C)",
        "predict_yield": "Predict Crop Yield",
        "empty_area": "Please enter a country or area.",
        "yield_result": "Estimated Crop Yield",
        "yield_error": "Yield prediction failed: {error}",
        "yield_warning": "This is a historical-data estimate, not a guaranteed future harvest. Predictions may be unreliable for unseen locations or future years. Validate against local agricultural data.",
        "fertilizer_heading": "🧪 Fertilizer Guidance",
        "fertilizer_intro": "Select a crop and enter soil-test N, P, and K values. This provides basic educational guidance, not an exact fertilizer dose.",
        "fert_crop": "Crop for fertilizer guidance",
        "low": "Low",
        "medium": "Moderate",
        "high": "High",
        "nutrient_status": "Nutrient level",
        "nitrogen_guidance": "Nitrogen (N) guidance",
        "phosphorus_guidance": "Phosphorus (P) guidance",
        "potassium_guidance": "Potassium (K) guidance",
        "low_n": "Your entered nitrogen value is relatively low in this simple screening. Ask a local agricultural officer about an appropriate nitrogen source and timing.",
        "medium_n": "Your entered nitrogen value is in the middle screening range. Avoid adding extra nitrogen without checking the soil test and crop growth stage.",
        "high_n": "Your entered nitrogen value is relatively high in this simple screening. Do not add more nitrogen without professional advice; excess can harm crops and increase losses.",
        "low_p": "Your entered phosphorus value is relatively low in this simple screening. Confirm soil-test results before considering a phosphorus fertilizer.",
        "medium_p": "Your entered phosphorus value is in the middle screening range. Base any phosphorus application on soil-test recommendations.",
        "high_p": "Your entered phosphorus value is relatively high in this simple screening. Avoid additional phosphorus unless a soil test or local recommendation supports it.",
        "low_k": "Your entered potassium value is relatively low in this simple screening. Confirm the soil test before considering a potassium source.",
        "medium_k": "Your entered potassium value is in the middle screening range. Use crop- and soil-specific guidance before applying more.",
        "high_k": "Your entered potassium value is relatively high in this simple screening. Avoid extra potassium unless supported by soil testing.",
        "fertilizer_caution": "Important: these low/moderate/high bands are simple app heuristics, not calibrated agronomic thresholds. N, P, and K units and soil-test methods vary. Do not calculate or apply a fertilizer dose from these labels alone. Follow a local soil-test report and official agricultural recommendations.",
        "crop_note": "Different crops have different nutrient needs and growth stages. This tool does not currently calculate crop-specific fertilizer quantities.",
        "about_heading": "About Mazhai AI Crop",
        "about_description": "An educational agricultural AI project combining machine learning with soil and climate information.",
        "features": "Features",
        "feature1": "Crop recommendation using soil and climate inputs",
        "feature2": "Machine-learning crop classification",
        "feature3": "Historical crop yield prediction",
        "feature4": "Live weather lookup",
        "feature5": "Bilingual fertilizer guidance",
        "technology": "Technology Stack",
        "limitation": "Predictions depend on training data and input quality. Consult local agricultural experts before making farming decisions.",
        "footer": "Mazhai AI Crop | AI for Smarter Agriculture",
    },
    "ta": {
        "language": "மொழி / Language",
        "title": "🌧️ மழை AI பயிர்",
        "subtitle": "பயிர் பரிந்துரை, மகசூல் கணிப்பு மற்றும் உர வழிகாட்டி",
        "tagline": "சிறந்த விவசாயம், சிறந்த அறுவடை.",
        "intro": "பயிர் பரிந்துரை, மகசூல் மதிப்பீடு, நேரடி வானிலை மற்றும் அடிப்படை உர வழிகாட்டலுக்கான கல்வி நோக்கிலான AI விவசாய உதவித் தளம்.",
        "weather": "🌦️ நேரடி வானிலை",
        "city": "உங்கள் நகரத்தின் பெயரை உள்ளிடவும்",
        "get_weather": "வானிலையைப் பெறுக",
        "use_weather": "நேரடி வெப்பநிலை மற்றும் ஈரப்பதத்தைப் பயன்படுத்தவும்",
        "weather_applied": "நேரடி வெப்பநிலை மற்றும் ஈரப்பதம் பயிர் படிவத்தில் சேர்க்கப்பட்டுள்ளன. கணிப்பதற்கு முன் மழைப்பொழிவு மற்றும் மண் மதிப்புகளைச் சரிபார்க்கவும்.",
        "missing_key": ".env கோப்பில் OPENWEATHER_API_KEY விசையைச் சேர்க்கவும்.",
        "empty_city": "நகரத்தின் பெயரை உள்ளிடவும்.",
        "invalid_key": "வானிலை API விசை தவறானது அல்லது இன்னும் செயல்படவில்லை.",
        "city_not_found": "நகரம் கிடைக்கவில்லை. பெயரைச் சரிபார்க்கவும்.",
        "weather_error": "வானிலைத் தகவலைப் பெற முடியவில்லை. இணைய இணைப்பு மற்றும் API விசையைச் சரிபார்க்கவும்.",
        "temperature": "வெப்பநிலை",
        "humidity": "ஈரப்பதம்",
        "conditions": "வானிலை நிலை",
        "rain_recent": "கடந்த ஒரு மணி நேர மழை",
        "weather_note": "கடந்த ஒரு மணி நேர மழை என்பது பருவகால மழைப்பொழிவுக்கு சமமல்ல. பயிர் மாதிரிக்கு பொருத்தமான மழைப்பொழிவு மதிப்பை உள்ளிடவும்.",
        "crop_tab": "🌱 பயிர் பரிந்துரை",
        "yield_tab": "🌾 மகசூல் கணிப்பு",
        "fertilizer_tab": "🧪 உர வழிகாட்டி",
        "about_tab": "ℹ️ அறிமுகம்",
        "soil_heading": "மண் மற்றும் வானிலை விவரங்கள்",
        "soil_help": "மண் பரிசோதனை மதிப்புகள் மற்றும் பொருத்தமான உள்ளூர் காலநிலை அளவுகளை உள்ளிடவும்.",
        "nitrogen": "நைட்ரஜன் (N)",
        "phosphorus": "பாஸ்பரஸ் (P)",
        "potassium": "பொட்டாசியம் (K)",
        "ph": "மண்ணின் pH",
        "rainfall": "மழைப்பொழிவு (மிமீ)",
        "recommend": "🌾 பொருத்தமான பயிரைப் பரிந்துரைக்கவும்",
        "crop_model_missing": "பயிர் மாதிரி கிடைக்கவில்லை. models/crop_model.pkl கோப்பைச் சரிபார்க்கவும்.",
        "crop_result": "🌱 பரிந்துரைக்கப்படும் பயிர்: {crop}",
        "alternatives": "சிறந்த 5 மாற்றுப் பயிர்கள்",
        "model_score": "மாதிரி மதிப்பெண் (%)",
        "score_note": "மாதிரி மதிப்பெண்கள் விவசாய வெற்றிக்கான நிகழ்தகவுகள் அல்ல.",
        "crop_error": "பயிர் கணிப்பில் பிழை: {error}",
        "yield_heading": "🌾 பயிர் மகசூல் கணிப்பு",
        "yield_help": "பயிர், பகுதி, ஆண்டு, மழைப்பொழிவு, பூச்சிக்கொல்லி பயன்பாடு மற்றும் வெப்பநிலைத் தரவுகளின் அடிப்படையில் மகசூலை மதிப்பிடவும்.",
        "yield_model_missing": "மகசூல் மாதிரி கிடைக்கவில்லை. முதலில் python train_yield_model.py கட்டளையை இயக்கவும்.",
        "yield_units": "மகசூல் hg/ha அலகிலிருந்து டன்/ஹெக்டேர் அலகிற்கு மாற்றப்படுகிறது.",
        "area": "நாடு / பகுதி (தரவுத்தொகுப்பில் உள்ள பெயரைப் பயன்படுத்தவும்)",
        "crop": "பயிர்",
        "year": "ஆண்டு",
        "annual_rainfall": "சராசரி ஆண்டு மழைப்பொழிவு (மிமீ)",
        "pesticide": "பூச்சிக்கொல்லி பயன்பாடு (டன்)",
        "average_temp": "சராசரி வெப்பநிலை (°C)",
        "predict_yield": "மகசூலைக் கணிக்கவும்",
        "empty_area": "நாடு அல்லது பகுதியின் பெயரை உள்ளிடவும்.",
        "yield_result": "மதிப்பிடப்பட்ட பயிர் மகசூல்",
        "yield_error": "மகசூல் கணிப்பில் பிழை: {error}",
        "yield_warning": "இது வரலாற்றுத் தரவை அடிப்படையாகக் கொண்ட மதிப்பீடு மட்டுமே; உறுதியான எதிர்கால அறுவடை கணிப்பு அல்ல. உள்ளூர் விவசாயத் தரவுகளுடன் சரிபார்க்கவும்.",
        "fertilizer_heading": "🧪 உர வழிகாட்டி",
        "fertilizer_intro": "பயிரைத் தேர்ந்தெடுத்து மண் பரிசோதனையில் கிடைத்த N, P, K மதிப்புகளை உள்ளிடவும். இது அடிப்படை கல்வி வழிகாட்டல் மட்டுமே; துல்லியமான உர அளவு அல்ல.",
        "fert_crop": "உர வழிகாட்டலுக்கான பயிர்",
        "low": "குறைவு",
        "medium": "மிதமானது",
        "high": "அதிகம்",
        "nutrient_status": "ஊட்டச்சத்து நிலை",
        "nitrogen_guidance": "நைட்ரஜன் (N) வழிகாட்டல்",
        "phosphorus_guidance": "பாஸ்பரஸ் (P) வழிகாட்டல்",
        "potassium_guidance": "பொட்டாசியம் (K) வழிகாட்டல்",
        "low_n": "இந்த எளிய மதிப்பீட்டில் நைட்ரஜன் அளவு குறைவாக உள்ளது. பொருத்தமான நைட்ரஜன் உரம் மற்றும் பயன்படுத்தும் நேரம் குறித்து உள்ளூர் வேளாண் அலுவலரிடம் ஆலோசனை பெறவும்.",
        "medium_n": "இந்த எளிய மதிப்பீட்டில் நைட்ரஜன் அளவு மிதமானது. மண் பரிசோதனை மற்றும் பயிரின் வளர்ச்சி நிலையைச் சரிபார்க்காமல் கூடுதல் நைட்ரஜன் இட வேண்டாம்.",
        "high_n": "இந்த எளிய மதிப்பீட்டில் நைட்ரஜன் அளவு அதிகமாக உள்ளது. நிபுணர் ஆலோசனையின்றி கூடுதல் நைட்ரஜன் இட வேண்டாம்; அதிகப்படியான உரம் பயிருக்கும் சுற்றுச்சூழலுக்கும் பாதிப்பை ஏற்படுத்தலாம்.",
        "low_p": "இந்த எளிய மதிப்பீட்டில் பாஸ்பரஸ் அளவு குறைவாக உள்ளது. பாஸ்பரஸ் உரம் இடுவதற்கு முன் மண் பரிசோதனையை உறுதிப்படுத்தவும்.",
        "medium_p": "இந்த எளிய மதிப்பீட்டில் பாஸ்பரஸ் அளவு மிதமானது. மண் பரிசோதனை பரிந்துரைப்படி உரமிடவும்.",
        "high_p": "இந்த எளிய மதிப்பீட்டில் பாஸ்பரஸ் அளவு அதிகமாக உள்ளது. மண் பரிசோதனை அல்லது உள்ளூர் பரிந்துரை ஆதரிக்காவிட்டால் கூடுதல் பாஸ்பரஸ் இட வேண்டாம்.",
        "low_k": "இந்த எளிய மதிப்பீட்டில் பொட்டாசியம் அளவு குறைவாக உள்ளது. பொட்டாசியம் உரம் இடுவதற்கு முன் மண் பரிசோதனையை உறுதிப்படுத்தவும்.",
        "medium_k": "இந்த எளிய மதிப்பீட்டில் பொட்டாசியம் அளவு மிதமானது. கூடுதல் உரமிடுவதற்கு முன் பயிர் மற்றும் மண்ணுக்கான பரிந்துரையைப் பெறவும்.",
        "high_k": "இந்த எளிய மதிப்பீட்டில் பொட்டாசியம் அளவு அதிகமாக உள்ளது. மண் பரிசோதனை ஆதரிக்காவிட்டால் கூடுதல் பொட்டாசியம் இட வேண்டாம்.",
        "fertilizer_caution": "முக்கியம்: குறைவு/மிதமானது/அதிகம் என்ற வகைப்பாடுகள் எளிய செயலி மதிப்பீடுகள் மட்டுமே; வேளாண் அறிவியல் அடிப்படையில் உறுதிப்படுத்தப்பட்ட வரம்புகள் அல்ல. N, P, K அலகுகளும் மண் பரிசோதனை முறைகளும் மாறுபடும். இந்த வகைப்பாட்டை மட்டும் வைத்து உர அளவைத் தீர்மானிக்க வேண்டாம். உள்ளூர் மண் பரிசோதனை அறிக்கை மற்றும் அதிகாரப்பூர்வ வேளாண் பரிந்துரைகளைப் பின்பற்றவும்.",
        "crop_note": "ஒவ்வொரு பயிருக்கும் வளர்ச்சி நிலைக்கும் ஊட்டச்சத்து தேவை மாறுபடும். இந்தக் கருவி தற்போது பயிருக்கு ஏற்ற துல்லியமான உர அளவைக் கணக்கிடாது.",
        "about_heading": "மழை AI பயிர் பற்றி",
        "about_description": "மண் மற்றும் காலநிலைத் தகவல்களுடன் இயந்திரக் கற்றலை இணைக்கும் விவசாய AI திட்டம்.",
        "features": "அம்சங்கள்",
        "feature1": "மண் மற்றும் காலநிலை அடிப்படையிலான பயிர் பரிந்துரை",
        "feature2": "இயந்திரக் கற்றல் பயிர் வகைப்பாடு",
        "feature3": "வரலாற்றுத் தரவு அடிப்படையிலான மகசூல் கணிப்பு",
        "feature4": "நேரடி வானிலைத் தகவல்",
        "feature5": "தமிழ் மற்றும் ஆங்கில உர வழிகாட்டி",
        "technology": "பயன்படுத்தப்படும் தொழில்நுட்பங்கள்",
        "limitation": "கணிப்புகள் பயிற்சித் தரவு மற்றும் உள்ளீடுகளின் தரத்தைப் பொறுத்தது. விவசாய முடிவுகளுக்கு முன் உள்ளூர் வேளாண் நிபுணர்களின் ஆலோசனையைப் பெறவும்.",
        "footer": "மழை AI பயிர் | சிறந்த விவசாயத்திற்கான AI",
    },
}


def tr(key, **kwargs):
    lang = st.session_state.get("language", "en")
    value = TEXT[lang][key]
    return value.format(**kwargs) if kwargs else value


@st.cache_resource
def load_crop_model(path):
    return joblib.load(path)


@st.cache_resource
def load_yield_bundle(path):
    return joblib.load(path)


def nutrient_band(value):
    """Simple illustrative screening bands; not agronomic recommendations."""
    if value < 40:
        return "low"
    if value < 80:
        return "medium"
    return "high"


def hero_banner(title, subtitle, image_url):
    """Show a crop-specific scenic banner above a tab's content."""
    st.markdown(
        f"""<section class="farm-hero" style="background-image: url('{image_url}');">
        <h2>{title}</h2><p>{subtitle}</p></section>""",
        unsafe_allow_html=True,
    )


# =========================================================
# LIVE AGRICULTURE & WEATHER NEWS
# =========================================================
import xml.etree.ElementTree as ET
import html

NEWS_FEEDS = {
    "Agriculture": "agriculture farming India crop farmers",
    "Weather": "India weather monsoon rainfall IMD",
    "Market & Policy": "India agriculture MSP mandi crop prices farmers",
}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_feed(topic: str, limit: int = 6):
    """Fetch recent headlines from Google News RSS; cache the response for 30 minutes."""
    query = NEWS_FEEDS.get(topic, "India agriculture weather")
    response = requests.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
        headers={"User-Agent": "Mozilla/5.0 MazhaiAICrop/1.0"},
        timeout=12,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    items = []
    for item in root.findall("./channel/item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        source = (item.findtext("source") or "Google News").strip()
        if title and link.startswith(("https://", "http://")):
            items.append({"title": title, "link": link, "source": source})
    return items

def render_news_cards(topic: str, limit: int = 6):
    """Render RSS headlines as compact, responsive cards."""
    try:
        items = fetch_news_feed(topic, limit)
    except (requests.RequestException, ET.ParseError, ValueError):
        st.info("Live headlines are temporarily unavailable. Check your internet connection and try again.")
        return
    if not items:
        st.info("No headlines were returned just now. Please try refreshing later.")
        return
    cols = st.columns(3)
    for index, item in enumerate(items):
        with cols[index % 3]:
            title = html.escape(item["title"])
            source = html.escape(item["source"])
            link = html.escape(item["link"], quote=True)
            st.markdown(
                f"""<article class="news-card">
                <div class="news-kicker">{html.escape(topic)} update</div>
                <div class="news-title">{title}</div>
                <div class="news-meta">{source}</div>
                <div style="margin-top:.8rem"><a href="{link}" target="_blank" rel="noopener noreferrer">Read story ↗</a></div>
                </article>""",
                unsafe_allow_html=True,
            )

# =========================================================
# ACCOUNT ACCESS (local SQLite store; passwords are salted + hashed)
# =========================================================
import hashlib
import hmac
import sqlite3
import secrets

USER_DB_PATH = ROOT / "users.db"


def _user_db():
    conn = sqlite3.connect(USER_DB_PATH, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            salt BLOB NOT NULL,
            password_hash BLOB NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)


def create_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,32}", username):
        return False, "Use 3–32 characters: letters, numbers, dot, underscore, or hyphen."
    if len(password) < 10:
        return False, "Choose a password with at least 10 characters."
    salt = secrets.token_bytes(16)
    digest = _password_hash(password, salt)
    try:
        with _user_db() as conn:
            conn.execute(
                "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
                (username, salt, digest),
            )
        return True, "Account created. You can now sign in."
    except sqlite3.IntegrityError:
        return False, "That username already exists. Please sign in or choose another."


def verify_user(username: str, password: str) -> bool:
    username = username.strip().lower()
    with _user_db() as conn:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row:
        # Perform a dummy hash to reduce obvious timing differences for unknown users.
        _password_hash(password, b"mazhai-dummy-salt")
        return False
    return hmac.compare_digest(_password_hash(password, row[0]), row[1])


def render_login_page():
    st.markdown("""
    <div class="login-shell">
      <div class="login-eyebrow">🌿 Smart agriculture, rooted in care</div>
      <div class="login-title">Welcome to Mazhai AI Crop</div>
      <div class="login-copy">A clearer way to explore crop suggestions, local weather, harvest estimates and plant-care guidance.</div>
    </div>
    """, unsafe_allow_html=True)
    left, center, right = st.columns([1, 1.45, 1])
    with center:
        sign_in, sign_up = st.tabs(["Sign in", "Create account"])
        with sign_in:
            with st.form("sign_in_form", clear_on_submit=False):
                username = st.text_input("Username", key="signin_username", autocomplete="username")
                password = st.text_input("Password", type="password", key="signin_password", autocomplete="current-password")
                submitted = st.form_submit_button("Sign in →", type="primary", use_container_width=True)
            if submitted:
                if verify_user(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username.strip().lower()
                    st.rerun()
                else:
                    st.error("We couldn't sign you in. Check your username and password.")
        with sign_up:
            with st.form("sign_up_form", clear_on_submit=True):
                new_username = st.text_input("Choose a username", key="signup_username", help="3–32 characters; letters, numbers, dot, underscore, or hyphen.")
                new_password = st.text_input("Create password", type="password", key="signup_password", help="At least 10 characters.")
                confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
                create = st.form_submit_button("Create my account", type="primary", use_container_width=True)
            if create:
                if new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    ok, message = create_user(new_username, new_password)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
        st.caption("Your password is stored as a salted hash, not as readable text.")
    st.markdown("<div style='text-align:center;color:#738174;font-size:.82rem;margin-top:1.5rem'>MAZHAI AI CROP · Grow with better information</div>", unsafe_allow_html=True)


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    render_login_page()
    st.stop()


# =========================================================
# LANGUAGE SELECTOR AND HEADER
# =========================================================
# Language selector: the user chooses one interface language at a time.
language_choice = st.selectbox(
    "Language / மொழி",
    ["English", "தமிழ்"],
    key="language_choice",
)
st.session_state["language"] = "ta" if language_choice == "தமிழ்" else "en"

# Streamlit's built-in top-left menu opens the sidebar; the sidebar lists every feature.
with st.sidebar:
    st.markdown("## 🌧️ Mazhai AI Crop")
    st.caption("Smart farming companion")
    st.divider()
    nav_labels = {
        "Home": "⌂  Home",
        "Crop Recommendation": "🌱  Crop recommendation",
        "Yield Prediction": "🌾  Yield prediction",
        "Fertilizer Guidance": "🧪  Fertilizer guidance",
        "AI Plant Doctor": "🌿  AI Plant Doctor",
        "Farming Assistant": "🤖  AI Farming Assistant",
        "About": "ℹ️  About Mazhai",
    }
    nav_keys = list(nav_labels)
    selected_page = st.radio(
        "EXPLORE",
        nav_keys,
        format_func=lambda item: nav_labels[item],
        key="mazhai_navigation",
        label_visibility="visible",
    )
    st.divider()
    st.markdown(f"Signed in as **{st.session_state.get('username', 'farmer')}**")
    if st.button("Sign out", use_container_width=True, key="mazhai_logout"):
        st.session_state["authenticated"] = False
        st.session_state.pop("username", None)
        st.rerun()
    st.caption("Made for thoughtful, data-informed farming.")

if selected_page != "Home":
    st.markdown(
        f"""<section class="brand-hero">
        <h1>{tr('title')}</h1>
        <h2>{tr('subtitle')}</h2>
        <p>{tr('tagline')} {tr('intro')}</p>
        </section>""",
        unsafe_allow_html=True,
    )

# =========================================================
# HOME DASHBOARD
# =========================================================
if selected_page == "Home":
    st.markdown("""
    <section class="brand-hero">
      <div class="home-eyebrow" style="color:#d7ebce">YOUR FARMING COMPANION</div>
      <h1>Grow smarter.<br>Harvest with confidence.</h1>
      <h2>Local context. Practical guidance. Better-informed decisions.</h2>
      <p>Explore crop recommendations, live weather, yield estimates and AI-powered plant-care
      guidance in one clear, professional workspace.</p>
    </section>
    """, unsafe_allow_html=True)
    st.markdown("### Your tools, all in one place")
    st.caption("Choose a feature from the left menu to get started.")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown('<div class="feature-tile"><div class="emoji">🌱</div><h4>Crop recommendation</h4><p>Explore crop options from soil and climate inputs.</p></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="feature-tile"><div class="emoji">🌦️</div><h4>Live weather</h4><p>Review current conditions for a selected location.</p></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="feature-tile"><div class="emoji">🌾</div><h4>Yield prediction</h4><p>Estimate harvest potential from historical data.</p></div>', unsafe_allow_html=True)
    f4, f5, f6 = st.columns(3)
    with f4:
        st.markdown('<div class="feature-tile"><div class="emoji">🧪</div><h4>Fertilizer guidance</h4><p>Understand basic nutrient-screening results.</p></div>', unsafe_allow_html=True)
    with f5:
        st.markdown('<div class="feature-tile"><div class="emoji">🌿</div><h4>AI Plant Doctor</h4><p>Screen supported leaf images with clear limitations.</p></div>', unsafe_allow_html=True)
    with f6:
        st.markdown('<div class="feature-tile"><div class="emoji">🤖</div><h4>AI Farming Assistant</h4><p>Ask farming questions in Tamil or English.</p></div>', unsafe_allow_html=True)
    st.markdown("---")

# =========================================================
# LIVE WEATHER (Home only)
# =========================================================
if selected_page == "Home":
    hero_banner(
        "🌦️ " + tr("weather"),
        "Live conditions to support better-informed farm decisions",
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85",
    )
    st.header(tr("weather"))

    with st.container(border=True):
        weather_col, button_col = st.columns([3, 1])

        with weather_col:
            city = st.text_input(
                tr("city"),
                value=st.session_state.get("weather_city", "Cuddalore"),
                key="weather_city_input",
            )

        with button_col:
            st.write("")
            get_weather = st.button(
                tr("get_weather"),
                type="primary",
                use_container_width=True,
            )

        if get_weather:
            if not API_KEY:
                st.error(tr("missing_key"))
            elif not city.strip():
                st.warning(tr("empty_city"))
            else:
                try:
                    response = requests.get(
                        "https://api.openweathermap.org/data/2.5/weather",
                        params={
                            "q": city.strip(),
                            "appid": API_KEY,
                            "units": "metric",
                        },
                        timeout=15,
                    )
                    response.raise_for_status()
                    weather_response = response.json()

                    st.session_state["weather_data"] = weather_response
                    st.session_state["weather_city"] = city.strip()
                    st.session_state["live_temperature"] = float(
                        weather_response["main"]["temp"]
                    )
                    st.session_state["live_humidity"] = float(
                        weather_response["main"]["humidity"]
                    )

                except requests.HTTPError as exc:
                    status = (
                        exc.response.status_code
                        if exc.response is not None else None
                    )
                    if status == 401:
                        st.error(tr("invalid_key"))
                    elif status == 404:
                        st.error(tr("city_not_found"))
                    else:
                        st.error(tr("weather_error"))
                except (requests.RequestException, KeyError, TypeError, ValueError):
                    st.error(tr("weather_error"))

    weather = st.session_state.get("weather_data")

    if weather:
        st.caption(weather.get("name", st.session_state.get("weather_city", "")))
        w1, w2, w3 = st.columns(3)

        w1.metric(
            tr("temperature"),
            f"{weather['main']['temp']:.1f} °C",
        )
        w2.metric(
            tr("humidity"),
            f"{weather['main']['humidity']}%",
        )
        description = weather.get("weather", [{}])[0].get("description", "—")
        w3.metric(tr("conditions"), description.title())

        if weather.get("rain"):
            st.metric(
                tr("rain_recent"),
                f"{weather['rain'].get('1h', 0)} mm",
            )

        # Button runs before crop widgets are created, so setting their state is safe.
        if st.button(tr("use_weather"), key="apply_weather_to_crop"):
            st.session_state["crop_temperature"] = float(weather["main"]["temp"])
            st.session_state["crop_humidity"] = float(weather["main"]["humidity"])
            st.session_state["weather_applied_notice"] = True

        if st.session_state.get("weather_applied_notice"):
            st.success(tr("weather_applied"))

    st.caption(tr("weather_note"))

    st.markdown("---")
    st.markdown('<div class="section-eyebrow">FRESH FROM THE FIELD</div>', unsafe_allow_html=True)
    st.subheader("Live Weather & Agriculture News")
    st.write("Recent headlines on farming, monsoon conditions, crop markets and agricultural policy. Stories open on their publisher's site.")
    news_tabs = st.tabs(["🌾 Agriculture", "🌦️ Weather & Monsoon", "📈 Markets & Policy"])
    with news_tabs[0]:
        render_news_cards("Agriculture", 6)
    with news_tabs[1]:
        render_news_cards("Weather", 6)
    with news_tabs[2]:
        render_news_cards("Market & Policy", 6)
    st.caption("Headlines are retrieved from Google News RSS and cached for 30 minutes. Check the linked publisher and official advisories before making farm decisions.")

# =========================================================
# MAIN TABS
# =========================================================
# Feature content is selected from the collapsible left sidebar.
crop_tab = selected_page == "Crop Recommendation"
yield_tab = selected_page == "Yield Prediction"
fertilizer_tab = selected_page == "Fertilizer Guidance"
plant_doctor_tab = selected_page == "AI Plant Doctor"
assistant_tab = selected_page == "Farming Assistant"
about_tab = selected_page == "About"

# =========================================================
# CROP RECOMMENDATION
# =========================================================
if crop_tab:
    hero_banner(
        "🌱 " + tr("crop_tab"),
        "Choose suitable crops using soil and climate information",
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85",
    )
    st.header(tr("soil_heading"))
    st.write(tr("soil_help"))

    defaults = {
        "crop_nitrogen": 50.0,
        "crop_phosphorus": 40.0,
        "crop_potassium": 40.0,
        "crop_temperature": 28.0,
        "crop_humidity": 70.0,
        "crop_ph": 6.5,
        "crop_rainfall": 150.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    c1, c2, c3 = st.columns(3)

    with c1:
        nitrogen = st.number_input(
            tr("nitrogen"), min_value=0.0, max_value=200.0,
            key="crop_nitrogen",
        )
        phosphorus = st.number_input(
            tr("phosphorus"), min_value=0.0, max_value=200.0,
            key="crop_phosphorus",
        )
        potassium = st.number_input(
            tr("potassium"), min_value=0.0, max_value=200.0,
            key="crop_potassium",
        )

    with c2:
        temperature = st.number_input(
            tr("temperature") + " (°C)",
            min_value=-10.0, max_value=60.0,
            key="crop_temperature",
        )
        humidity = st.number_input(
            tr("humidity") + " (%)",
            min_value=0.0, max_value=100.0,
            key="crop_humidity",
        )

    with c3:
        ph = st.number_input(
            tr("ph"), min_value=0.0, max_value=14.0,
            key="crop_ph",
        )
        rainfall = st.number_input(
            tr("rainfall"), min_value=0.0, max_value=5000.0,
            key="crop_rainfall",
        )

    if st.button(tr("recommend"), type="primary", key="recommend_crop"):
        if not CROP_MODEL_PATH.exists():
            st.error(tr("crop_model_missing"))
        else:
            try:
                crop_model = load_crop_model(str(CROP_MODEL_PATH))
                crop_input = pd.DataFrame([{
                    "N": nitrogen,
                    "P": phosphorus,
                    "K": potassium,
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall,
                }])
                predicted_crop = crop_model.predict(crop_input)[0]
                st.success(tr("crop_result", crop=str(predicted_crop).title()))

                if hasattr(crop_model, "predict_proba"):
                    probabilities = crop_model.predict_proba(crop_input)[0]
                    score_label = tr("model_score")
                    alternatives = pd.DataFrame({
                        "Crop": crop_model.classes_,
                        score_label: probabilities * 100,
                    }).sort_values(score_label, ascending=False).head(5)

                    st.subheader(tr("alternatives"))
                    st.bar_chart(alternatives.set_index("Crop")[score_label])
                    st.caption(tr("score_note"))

            except Exception as exc:
                st.error(tr("crop_error", error=str(exc)))

# =========================================================
# CROP YIELD PREDICTION
# =========================================================
if yield_tab:
    hero_banner(
        "🌾 " + tr("yield_tab"),
        "Estimate harvest potential from historical data",
        "https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?auto=format&fit=crop&w=1800&q=85",
    )
    st.header(tr("yield_heading"))
    st.write(tr("yield_help"))

    if not YIELD_MODEL_PATH.exists():
        st.warning(tr("yield_model_missing"))
    else:
        try:
            yield_bundle = load_yield_bundle(str(YIELD_MODEL_PATH))
            yield_model = yield_bundle["model"]
            yield_features = yield_bundle.get(
                "features",
                [
                    "Area",
                    "Item",
                    "Year",
                    "average_rain_fall_mm_per_year",
                    "pesticides_tonnes",
                    "avg_temp",
                ],
            )

            st.caption(tr("yield_units"))

            crop_options = [
                "Maize",
                "Rice, paddy",
                "Wheat",
                "Potatoes",
                "Sorghum",
                "Soybeans",
                "Cassava",
            ]
            crop_names_ta = {
                "Maize": "மக்காச்சோளம்",
                "Rice, paddy": "நெல்",
                "Wheat": "கோதுமை",
                "Potatoes": "உருளைக்கிழங்கு",
                "Sorghum": "சோளம்",
                "Soybeans": "சோயாபீன்ஸ்",
                "Cassava": "மரவள்ளிக்கிழங்கு",
            }
            display_crops = (
                [crop_names_ta[crop] for crop in crop_options]
                if st.session_state["language"] == "ta"
                else crop_options
            )

            with st.form("yield_prediction_form"):
                area = st.text_input(tr("area"), value="India")
                selected_display = st.selectbox(tr("crop"), display_crops)
                selected_crop = crop_options[display_crops.index(selected_display)]

                y1, y2 = st.columns(2)
                with y1:
                    year = st.number_input(
                        tr("year"), min_value=1961, max_value=2100,
                        value=2020, step=1,
                    )
                    annual_rainfall = st.number_input(
                        tr("annual_rainfall"), min_value=0.0, value=1000.0,
                    )
                with y2:
                    pesticide = st.number_input(
                        tr("pesticide"), min_value=0.0, value=100.0,
                    )
                    average_temp = st.number_input(
                        tr("average_temp"), min_value=-10.0,
                        max_value=60.0, value=25.0,
                    )

                predict_yield = st.form_submit_button(
                    tr("predict_yield"), type="primary"
                )

            if predict_yield:
                if not area.strip():
                    st.error(tr("empty_area"))
                else:
                    yield_input = pd.DataFrame([{
                        "Area": area.strip(),
                        "Item": selected_crop,
                        "Year": year,
                        "average_rain_fall_mm_per_year": annual_rainfall,
                        "pesticides_tonnes": pesticide,
                        "avg_temp": average_temp,
                    }])
                    try:
                        yield_input = yield_input[yield_features]
                        predicted_hg_ha = float(yield_model.predict(yield_input)[0])
                        predicted_tonnes_ha = predicted_hg_ha / 10000
                        st.metric(
                            tr("yield_result"),
                            f"{predicted_tonnes_ha:,.2f} tonnes/ha",
                        )
                        st.warning(tr("yield_warning"))
                    except Exception as exc:
                        st.error(tr("yield_error", error=str(exc)))

        except Exception as exc:
            st.error(f"{tr('yield_model_missing')} Details: {exc}")

# =========================================================
# FERTILIZER GUIDANCE
# =========================================================
if fertilizer_tab:
    hero_banner(
        "🧪 " + tr("fertilizer_tab"),
        "Understand nutrient screening and safe application basics",
        "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?auto=format&fit=crop&w=1800&q=85",
    )
    st.header(tr("fertilizer_heading"))
    st.write(tr("fertilizer_intro"))
    fertilizer_catalogue_note = (
        "இந்தப் பட்டியலில் வயல் பயிர்கள், காய்கறிகள், பழங்கள், மூலிகைகள், பூக்கள், தோட்டச் செடிகள் மற்றும் மரங்கள் உள்ளன. சரியான உர அளவு பயிர் வகை, வயது, மண் பரிசோதனை மற்றும் உள்ளூர் வேளாண் பரிந்துரையைப் பொறுத்தது."
        if st.session_state["language"] == "ta"
        else "The catalogue includes field crops, vegetables, fruits, herbs, flowers, garden plants and trees. Exact fertilizer doses depend on crop variety, age, soil test and local agricultural guidance."
    )
    st.info(fertilizer_catalogue_note)

    fertilizer_crops_en = [
        "Rice", "Maize", "Wheat", "Sorghum", "Millet", "Barley", "Oats",
        "Ragi / Finger millet", "Pearl millet", "Pulses", "Chickpea", "Pigeon pea",
        "Green gram", "Black gram", "Lentil", "Cowpea", "Soybean", "Groundnut",
        "Sesame", "Sunflower", "Mustard", "Cotton", "Sugarcane", "Tobacco",
        "Potato", "Sweet potato", "Cassava", "Tomato", "Brinjal / Eggplant",
        "Chilli", "Onion", "Garlic", "Cabbage", "Cauliflower", "Okra / Lady's finger",
        "Carrot", "Radish", "Cucumber", "Pumpkin", "Bitter gourd", "Bottle gourd",
        "Beans", "Peas", "Banana", "Mango", "Coconut", "Papaya", "Guava", "Lemon",
        "Orange", "Grapes", "Pomegranate", "Jackfruit", "Watermelon", "Pineapple",
        "Turmeric", "Ginger", "Pepper", "Cardamom", "Coriander", "Mint", "Basil",
        "Rose", "Jasmine", "Marigold", "Hibiscus", "Chrysanthemum", "Lawn grass",
        "Neem tree", "Teak tree", "Mango tree", "Coconut tree", "Amla tree",
        "Drumstick tree", "Tamarind tree", "Guava tree", "Eucalyptus", "Bamboo",
        "Areca palm", "Cashew tree", "Coffee", "Tea", "Rubber", "Fodder grass",
        "Ornamental plant", "Medicinal plant", "Other crop / plant / tree",
    ]
    fertilizer_crops_ta = [
        "நெல்", "மக்காச்சோளம்", "கோதுமை", "சோளம்", "சிறுதானியம்", "பார்லி", "ஓட்ஸ்",
        "கேழ்வரகு", "கம்பு", "பயறு வகைகள்", "கொண்டைக்கடலை", "துவரை", "பாசிப்பயறு",
        "உளுந்து", "மசூர் பருப்பு", "தட்டைப்பயறு", "சோயாபீன்ஸ்", "நிலக்கடலை",
        "எள்", "சூரியகாந்தி", "கடுகு", "பருத்தி", "கரும்பு", "புகையிலை", "உருளைக்கிழங்கு",
        "சர்க்கரைவள்ளி", "மரவள்ளிக்கிழங்கு", "தக்காளி", "கத்திரிக்காய்", "மிளகாய்", "வெங்காயம்",
        "பூண்டு", "முட்டைக்கோஸ்", "காலிஃபிளவர்", "வெண்டைக்காய்", "கேரட்", "முள்ளங்கி",
        "வெள்ளரிக்காய்", "பூசணி", "பாகற்காய்", "சுரைக்காய்", "பீன்ஸ்", "பட்டாணி", "வாழை",
        "மாமரம்", "தென்னை", "பப்பாளி", "கொய்யா", "எலுமிச்சை", "ஆரஞ்சு", "திராட்சை",
        "மாதுளை", "பலா", "தர்பூசணி", "அன்னாசி", "மஞ்சள்", "இஞ்சி", "மிளகு", "ஏலக்காய்",
        "கொத்தமல்லி", "புதினா", "துளசி", "ரோஜா", "மல்லிகை", "சாமந்தி", "செம்பருத்தி",
        "சந்திரமல்லி", "புல்வெளிப் புல்", "வேப்பமரம்", "தேக்கு", "மாமரம்", "தென்னைமரம்",
        "நெல்லி", "முருங்கை", "புளியமரம்", "கொய்யாமரம்", "யூகலிப்டஸ்", "மூங்கில்",
        "பாக்கு மரம்", "முந்திரி", "காபி", "தேயிலை", "ரப்பர்", "தீவனப் புல்",
        "அலங்காரச் செடி", "மருத்துவச் செடி", "மற்ற பயிர் / செடி / மரம்",
    ]

    display_fertilizer_crops = (
        fertilizer_crops_ta
        if st.session_state["language"] == "ta"
        else fertilizer_crops_en
    )

    with st.form("fertilizer_form"):
        fertilizer_crop_display = st.selectbox(
            tr("fert_crop"), display_fertilizer_crops
        )
        f1, f2, f3 = st.columns(3)

        with f1:
            fert_n = st.number_input(
                tr("nitrogen"), min_value=0.0, max_value=2000.0,
                value=50.0, key="fert_n",
            )
        with f2:
            fert_p = st.number_input(
                tr("phosphorus"), min_value=0.0, max_value=2000.0,
                value=40.0, key="fert_p",
            )
        with f3:
            fert_k = st.number_input(
                tr("potassium"), min_value=0.0, max_value=2000.0,
                value=40.0, key="fert_k",
            )

        analyze_fertilizer = st.form_submit_button(
            tr("fertilizer_heading"), type="primary"
        )

    if analyze_fertilizer:
        selected_crop_index = display_fertilizer_crops.index(
            fertilizer_crop_display
        )
        selected_fertilizer_crop = fertilizer_crops_en[selected_crop_index]
        st.subheader(f"{tr('crop')}: {selected_fertilizer_crop}")

        nutrient_info = [
            ("N", fert_n, "nitrogen_guidance", "low_n", "medium_n", "high_n"),
            ("P", fert_p, "phosphorus_guidance", "low_p", "medium_p", "high_p"),
            ("K", fert_k, "potassium_guidance", "low_k", "medium_k", "high_k"),
        ]

        status_labels = {
            "low": tr("low"),
            "medium": tr("medium"),
            "high": tr("high"),
        }

        for nutrient, value, heading_key, low_key, medium_key, high_key in nutrient_info:
            band = nutrient_band(value)
            st.markdown(f"**{tr(heading_key)}**")
            st.write(
                f"{nutrient}: {status_labels[band]} "
                f"({value:g}; entered value)"
            )

            advice_key = {
                "low": low_key,
                "medium": medium_key,
                "high": high_key,
            }[band]
            st.info(tr(advice_key))

        st.warning(tr("fertilizer_caution"))
        st.caption(tr("crop_note"))

# =========================================================
# AI PLANT DOCTOR: FREE LOCAL HUGGING FACE MODEL
# =========================================================
PLANT_MODEL_ID = "Khawajaa/plant-disease-detector"
PLANT_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

@st.cache_resource(show_spinner=False)
def load_hf_plant_model():
    """Download the public checkpoint once and load the matching EfficientNet-B4 architecture."""
    import torch
    import torch.nn as nn
    from torchvision import models
    from huggingface_hub import hf_hub_download

    class EfficientNetB4Classifier(nn.Module):
        def __init__(self, num_classes=38, dropout=0.4):
            super().__init__()
            backbone = models.efficientnet_b4(weights=None)
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout), nn.Linear(1792, 512), nn.ReLU(inplace=True),
                nn.Dropout(p=dropout / 2), nn.Linear(512, num_classes),
            )
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    checkpoint_path = hf_hub_download(repo_id=PLANT_MODEL_ID, filename="best_model.pth")
    # This is a public model checkpoint from the selected repository, not a user-supplied file.
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older PyTorch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = EfficientNetB4Classifier(num_classes=len(PLANT_CLASSES), dropout=0.4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def pretty_plant_label(raw_label):
    parts = raw_label.split("___", 1)
    plant = parts[0].replace("_(maize)", " (maize)").replace("_(including_sour)", " (including sour)")
    plant = plant.replace("Pepper,_bell", "Bell pepper").replace("Orange", "Orange / citrus")
    if len(parts) == 1:
        return plant.replace("_", " ")
    condition = parts[1].replace("_", " ").replace("  ", " ").strip()
    condition = condition.replace("Gray leaf spot", "Gray leaf spot")
    return f"{plant.replace('_', ' ')} — {condition}"

if plant_doctor_tab:
    hero_banner(
        "🌿 AI Plant Doctor / AI தாவர மருத்துவர்",
        "Free AI leaf-disease screening powered by Hugging Face / Hugging Face இலவச AI இலை நோய் சோதனை",
        "https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?auto=format&fit=crop&w=1800&q=85",
    )
    is_tamil = st.session_state.get("language", "en") == "ta"
    if is_tamil:
        st.write("இது இணையத்திலிருந்து இலவச Hugging Face மாதிரியை முதல் முறையாகப் பதிவிறக்கி, உங்கள் கணினியிலேயே இயக்கும்.")
        st.info("முக்கிய வரம்பு: இந்த மாதிரி 14 தாவர இனங்களுக்கான 38 வகைகளை மட்டுமே வகைப்படுத்துகிறது. எல்லா பயிர்கள், செடிகள் அல்லது மரங்களையும் இது கண்டறியாது. பட்டியலில் இல்லாத தாவரங்களுக்கு முடிவு நம்பகமற்றதாக இருக்கலாம்.")
        upload_label = "இலைப் புகைப்படத்தைப் பதிவேற்றவும்"
        diagnose_label = "🔍 இலவச AI நோய் சோதனை"
    else:
        st.write("Upload a leaf photo. The model downloads from Hugging Face the first time, then performs inference locally on your computer/server—no API key is needed.")
        st.info("Coverage limitation: this model classifies 38 classes across 14 plant species (including healthy leaves). It does not support all crops, plants, or trees. For an unlisted species, its output may be misleading.")
        upload_label = "Upload a leaf photo"
        diagnose_label = "🔍 Run free AI disease screening"

    st.markdown("**Supported plant groups / ஆதரிக்கப்படும் தாவரங்கள்:** apple, blueberry, cherry, corn, grape, orange/citrus, peach, bell pepper, potato, raspberry, soybean, squash, strawberry, and tomato.")
    plant_image = st.file_uploader(
        upload_label, type=["jpg", "jpeg", "png", "webp"], key="plant_doctor_upload",
        help="Use a clear close-up of one leaf in good light. Avoid screenshots and very busy backgrounds.",
    )
    if plant_image is not None:
        st.image(plant_image, caption=upload_label, use_container_width=True)
        st.caption("Best results: show one leaf clearly, with good lighting and minimal background clutter.")

    if st.button(diagnose_label, key="run_plant_doctor", type="primary", disabled=plant_image is None):
        try:
            with st.spinner("Downloading/loading the model and analysing the image… / AI மாதிரி ஏற்றப்படுகிறது…"):
                import torch
                from torchvision import transforms
                model = load_hf_plant_model()
                image = Image.open(plant_image).convert("RGB")
                # Keep the image aspect ratio and centre-crop to match the model's square training images.
                image = ImageOps.fit(image, (224, 224), method=Image.Resampling.BICUBIC)
                preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                tensor = preprocess(image).unsqueeze(0)
                with torch.inference_mode():
                    probabilities = torch.softmax(model(tensor), dim=1)[0]
                values, indices = torch.topk(probabilities, k=3)

            best_idx = int(indices[0].item())
            best_score = float(values[0].item()) * 100
            raw_label = PLANT_CLASSES[best_idx]
            display_label = pretty_plant_label(raw_label)
            healthy = "healthy" in raw_label.lower()
            st.subheader("AI prediction / AI கணிப்பு")
            st.metric("Top prediction score (not a guarantee)", f"{best_score:.1f}%")
            if best_score < 55:
                st.warning("Low model score. The image may be unclear or outside the model's supported classes. Try a clearer close-up; do not treat this as a diagnosis.")
            st.write(f"**{display_label}**")
            if healthy:
                st.success("The model classified this image as a healthy leaf. This does not rule out other diseases or pests.")
            else:
                st.warning("The model detected a pattern associated with a disease class. This is only a screening result; confirm with a local agricultural expert before treatment.")

            st.markdown("**Other model predictions**")
            for score, idx in zip(values.tolist(), indices.tolist()):
                st.write(f"- {pretty_plant_label(PLANT_CLASSES[int(idx)])}: {float(score)*100:.1f}%")
            st.markdown("**Safe next steps**")
            st.write("1. Photograph another affected leaf and compare the result.\n2. Check both sides of leaves, stems, watering, and recent weather.\n3. Isolate severely affected potted plants where practical and avoid handling healthy plants immediately after affected ones.\n4. Do not apply pesticide based only on this model; seek local agricultural guidance.")
            st.caption("Model source: [Khawajaa/plant-disease-detector](https://huggingface.co/Khawajaa/plant-disease-detector). It is trained on PlantVillage-style leaf images; field performance can be lower, and it always selects among its 38 known classes.")
        except Exception as exc:
            st.error("Could not load or run the free Hugging Face model. Check internet access, installed dependencies, and available memory, then retry.")
            st.exception(exc)

    st.caption("No plant-analysis API key is required. The first run downloads model weights (about 18.5 million parameters); keep the app server online and allow enough memory for PyTorch and torchvision.")

# =========================================================
# ONLINE GROQ AI FARMING ASSISTANT
# =========================================================
def groq_farming_answer(question: str, lang: str, crop: str) -> str:
    """Generate a bilingual farming answer using Groq's hosted chat API."""
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY, timeout=45.0)
    language_name = "Tamil" if lang == "ta" else "English"
    system_prompt = """You are Mazhai AI Crop, a careful agricultural information assistant.
Answer the farmer in the requested language. Be practical, clear, and use short steps.
The farmer may ask about crops, vegetables, fruit plants, ornamental plants, and trees.
Ask for missing details when needed (location, plant age, soil, symptoms, rainfall, irrigation).
Do not claim certainty from incomplete symptoms or a leaf-image model prediction.
Never invent live weather, market prices, or official recommendations. If current/local data is
needed, say it must be checked from a current local source. Do not prescribe exact pesticide
or fertilizer doses without crop, pest/disease confirmation, label directions, and local guidance.
Mention safe, low-risk steps first. Encourage local agricultural extension/KVK advice for serious
or spreading disease. This is educational guidance, not a substitute for an agronomist."""
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Reply in {language_name}. Selected crop/plant/tree: {crop}. Farmer's question: {question}"},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    return completion.choices[0].message.content.strip()


# =========================================================
# TAMIL / ENGLISH FARMING ASSISTANT (GROQ ONLINE AI + SAFE FALLBACK)
# =========================================================
def farming_assistant_answer(question: str, lang: str, crop: str) -> tuple[str, str]:
    """Small offline bilingual farming helper; not a generative LLM."""
    q = question.lower().strip()
    tamil = lang == "ta"

    if not q:
        return (("உங்கள் கேள்வியை உள்ளிடவும்." if tamil else "Please enter your question."), "")

    def contains(*terms):
        return any(term in q for term in terms)

    if contains("water", "irrigat", "watering", "தண்ணீர்", "நீர்ப்பாசனம்", "பாசனம்", "நீர்"):
        answer = ("நீர்ப்பாசனத்திற்கு முன் மண்ணின் ஈரத்தை 5–10 செ.மீ. ஆழத்தில் சரிபார்க்கவும். மண் மற்றும் பயிர் நிலைக்கு ஏற்ப தண்ணீர் அளிக்கவும்; நீர் தேங்குவதைத் தவிர்க்கவும். மழை எதிர்பார்ப்பு மற்றும் வடிகாலையும் கவனிக்கவும்."
                  if tamil else "Before irrigating, check soil moisture about 5–10 cm below the surface. Water according to crop stage and soil type, avoid waterlogging, and consider expected rain and field drainage. This is general guidance, not a sensor-based irrigation prescription.")
        return answer, "water"
    if contains("yellow", "yellowing", "leaf spot", "spots", "disease", "pest", "insect", "நோய்", "மஞ்சள்", "புள்ளி", "பூச்சி", "இலை"):
        answer = ("இலைகளின் இருபுறமும், தண்டு மற்றும் புதிய வளர்ச்சியையும் தெளிவாகப் படம் எடுக்கவும். பாதிக்கப்பட்ட இலைகளை ஆரோக்கியமான செடிகளுடன் தொடாமல் கையாளவும்; வயலில் வடிகால் மற்றும் பூச்சி அறிகுறிகளைச் சரிபார்க்கவும். புகைப்பட AI முடிவு ஒரு ஆரம்பச் சோதனை மட்டுமே—அதன் அடிப்படையில் மட்டும் பூச்சிக்கொல்லி தெளிக்க வேண்டாம்."
                  if tamil else "Photograph both sides of leaves, stems, and new growth in good light. Check drainage and look for insects or spreading symptoms. Handle affected plants separately where practical. Image-model results are only a screening aid; do not spray pesticides based on the prediction alone.")
        return answer, "disease"
    if contains("fertilizer", "fertiliser", "urea", "npk", "manure", "உரம்", "யூரியா", "ஊட்டச்சத்து"):
        answer = ("மண் பரிசோதனை முடிவுகள், பயிர், வளர்ச்சி நிலை மற்றும் பரிந்துரைக்கப்பட்ட அளவைப் பார்த்தே உரத்தைத் தேர்ந்தெடுக்கவும். ஒரே பொதுவான அளவு எல்லாப் பயிர்களுக்கும் பொருந்தாது. உரங்களை கலப்பதற்கு முன் லேபிள் வழிமுறைகளைப் பின்பற்றவும்; சந்தேகம் இருந்தால் வேளாண் அலுவலரை அணுகவும்."
                  if tamil else "Choose fertilizer using a soil-test report, crop, growth stage, and locally recommended rate. One dose does not fit every crop. Follow the product label and consult an agricultural officer if unsure; avoid adding nutrients without evidence of need.")
        return answer, "fertilizer"
    if contains("soil", "ph", "salinity", "மண்", "உப்புத்தன்மை"):
        answer = ("மண் பரிசோதனையில் pH, மின்கடத்துத்திறன் (உப்புத்தன்மை) மற்றும் N-P-K மதிப்புகளைப் பெறுங்கள். மண் வகை, வடிகால் மற்றும் பயிரின் தேவையைப் பொருத்து முடிவு செய்யவும்; pH மட்டும் வைத்து உரம் அல்லது சுண்ணாம்பு அளவைத் தீர்மானிக்க வேண்டாம்."
                  if tamil else "Use a soil test for pH, electrical conductivity (salinity), and N-P-K. Consider soil texture, drainage, and crop needs. Do not decide lime or fertilizer rates from pH alone.")
        return answer, "soil"
    if contains("weather", "rain", "rainfall", "வானிலை", "மழை"):
        answer = ("வானிலை முன்னறிவிப்பை விதைப்பு, உரமிடுதல் மற்றும் தெளிப்பு முடிவுகளுக்கு முன் சரிபார்க்கவும். குறுகிய நேர மழை அளவு பருவ மழைப்பொழிவுக்கு சமமல்ல. உள்ளூர் முன்னறிவிப்பு மற்றும் வயல் நிலையை இணைத்துப் பாருங்கள்."
                  if tamil else "Check the local forecast before sowing, fertilizing, or spraying. Rainfall in the last hour is not the same as seasonal rainfall. Combine the forecast with actual field conditions.")
        return answer, "weather"
    if contains("rice", "paddy", "நெல்", "அரிசி"):
        answer = ("நெற்பயிரில் நீர் மேலாண்மை, களைக் கட்டுப்பாடு மற்றும் உரமிடும் நேரம் நடவு முறை மற்றும் வளர்ச்சி நிலையைப் பொறுத்தது. உள்ளூர் வேளாண் துறை பரிந்துரையைப் பின்பற்றவும்; வயலில் தொடர்ந்து ஆழமான நீர் தேங்க வேண்டியதில்லை."
                  if tamil else "For rice, water management, weed control, and fertilizer timing depend on planting method and growth stage. Follow local agricultural recommendations; continuous deep flooding is not necessary for every system.")
        return answer, "rice"

    fallback = (f"{crop} பயிரைப் பற்றி இன்னும் குறிப்பாக உதவ, அறிகுறி, பயிரின் வயது, மண் வகை, சமீபத்திய மழை/நீர்ப்பாசனம் ஆகியவற்றைச் சொல்லுங்கள். உறுதியான மருந்து அல்லது உர அளவை இங்கு ஊகித்துச் சொல்ல முடியாது."
                if tamil else f"To help with {crop}, share the symptom, crop age, soil type, and recent rain/irrigation. I can't safely guess a specific pesticide or fertilizer dose without diagnosis and local guidance.")
    return fallback, "general"

if assistant_tab:
    hero_banner(
        "🤖 Tamil AI Farming Assistant / AI விவசாய உதவியாளர்",
        "Ask questions about soil, irrigation, pests, weather and crop care",
        "https://images.unsplash.com/photo-1592982537447-7440770cbfc9?auto=format&fit=crop&w=1800&q=85",
    )
    assistant_tamil = st.session_state.get("language", "en") == "ta"
    if assistant_tamil:
        st.write("விவசாயம், மண், நீர்ப்பாசனம், உரம், வானிலை மற்றும் இலை அறிகுறிகள் குறித்து கேளுங்கள்.")
        st.info("இணையத்தில் இயங்கும் Groq AI உதவியாளர். GROQ_API_KEY அமைக்கப்பட வேண்டும். முக்கிய முடிவுகளை உள்ளூர் வேளாண் நிபுணரிடம் உறுதிப்படுத்தவும்.")
        crop_options = ["நெல்", "தக்காளி", "மிளகாய்", "வாழை", "பருத்தி", "கரும்பு", "மக்காச்சோளம்", "தென்னை", "மாமரம்", "மற்ற பயிர் / செடி / மரம்"]
        crop_prompt = "பயிர் / செடி / மரம்"
        question_prompt = "உங்கள் விவசாயக் கேள்வியைத் தமிழில் அல்லது ஆங்கிலத்தில் எழுதவும்"
        ask_label = "💬 வழிகாட்டலைப் பெறுக"
    else:
        st.write("Ask about soil health, irrigation, nutrients, weather risks, pests, leaf symptoms or crop care.")
        crop_options = ["Rice", "Tomato", "Chilli", "Banana", "Cotton", "Sugarcane", "Maize", "Coconut", "Mango", "Other crop / plant / tree"]
        crop_prompt = "Crop / plant / tree"
        question_prompt = "Type your farming question in Tamil or English"
        ask_label = "💬 Get guidance"

    with st.form("farming_assistant_form"):
        assistant_crop = st.selectbox(crop_prompt, crop_options)
        assistant_question = st.text_area(question_prompt, height=110, max_chars=1200)
        ask_assistant = st.form_submit_button(ask_label, type="primary")

    if ask_assistant:
        if not assistant_question.strip():
            st.warning("உங்கள் கேள்வியை உள்ளிடவும்." if assistant_tamil else "Please enter your question.")
        elif not GROQ_API_KEY:
            st.error("GROQ_API_KEY is missing. Add it to your private .env file, then restart Streamlit. Get a key at https://console.groq.com/keys.")
            answer, topic = farming_assistant_answer(assistant_question, "ta" if assistant_tamil else "en", assistant_crop)
            st.markdown("**Basic offline guidance / அடிப்படை வழிகாட்டி:**")
            st.markdown(answer)
        else:
            try:
                with st.spinner("Groq AI is thinking… / Groq AI பதில் தயாரிக்கிறது…"):
                    answer = groq_farming_answer(assistant_question.strip(), "ta" if assistant_tamil else "en", assistant_crop)
                st.subheader("உதவியாளர் பதில்" if assistant_tamil else "Assistant response")
                st.markdown(answer)
                if any(term in assistant_question.lower() for term in ["disease", "spots", "pest", "நோய்", "பூச்சி", "புள்ளி"]):
                    st.warning("படத்தின் அடிப்படையிலான முடிவு உறுதியான நோயறிதல் அல்ல; பூச்சிக்கொல்லி பயன்பாட்டுக்கு முன் உள்ளூர் நிபுணரிடம் உறுதிப்படுத்தவும்." if assistant_tamil else "AI guidance is not a confirmed diagnosis. Confirm locally before pesticide use.")
                st.caption("Mazhai AI Crop • Online Groq AI • Verify local farming decisions")
            except Exception as exc:
                st.error("Groq AI request failed. Check your API key, internet connection, account limits, and selected model.")
                st.caption(str(exc))

# =========================================================
# ABOUT
# =========================================================
if about_tab:
    hero_banner(
        "🌳 " + tr("about_tab"),
        "A practical guide to crop, plant, garden and tree tools",
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1800&q=85",
    )
    st.header(tr("about_heading"))
    st.write(tr("about_description"))
    st.subheader(tr("features"))

    for key in [
        "feature1",
        "feature2",
        "feature3",
        "feature4",
        "feature5",
    ]:
        st.markdown(f"- {tr(key)}")

    st.subheader(tr("technology"))
    st.markdown(
        "- Python\n"
        "- Streamlit\n"
        "- Pandas and NumPy\n"
        "- Scikit-learn\n"
        "- Joblib\n"
        "- OpenWeatherMap API\n"
        "- Hugging Face Plant Disease Detector (local inference; no API key)"
    )
    st.warning(tr("limitation"))

st.divider()
st.caption(tr("footer"))


# Deployment note: SQLite is suitable for local prototypes. On hosts with ephemeral
# filesystems, user accounts may not persist across rebuilds/restarts; use a managed
# database or an identity provider before treating this as production authentication.
