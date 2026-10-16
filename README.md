 <div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Manrope&weight=800&size=36&pause=1000&color=238636&center=true&vCenter=true&width=850&lines=%F0%9F%8C%A7%EF%B8%8F+Mazhai+AI+Crop;Smarter+Farming.+Better+Harvests.;AI+for+the+Future+of+Agriculture." alt="Mazhai AI Crop animated heading" />

### 🌱 Smarter Farming. Better Harvests.

**An AI-powered agriculture assistant for data-informed farming.**

[![GitHub](https://img.shields.io/badge/GitHub-Mazhai%20AI%20Crop-238636?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Nagulanvelu/mazhai-ai-crop)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![AI/ML](https://img.shields.io/badge/AI%20%7C%20ML-Agriculture-2E7D32?style=for-the-badge)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B3D2E,50:238636,100:9ACD32&height=140&section=header&text=Growing%20a%20Smarter%20Future&fontSize=25&fontColor=ffffff&animation=fadeIn" width="100%" alt="Green agriculture banner"/>

</div>

## 🌾 About Mazhai AI Crop

**Mazhai AI Crop** is a Streamlit-based agricultural AI application that combines machine learning, live weather information, and AI-powered farming guidance in one place.

The goal is to make agricultural information easier to explore through a simple, modern interface available in English and Tamil.

## ✨ Key Features

| Feature | Description |
|---|---|
| 🌱 Crop Recommendation | Suggests crop options using soil and climate inputs. |
| 🌾 Yield Prediction | Estimates crop yield from historical agricultural data. |
| 🌦️ Live Weather | Retrieves current weather conditions for a selected city. |
| 🧪 Fertilizer Guidance | Provides basic N-P-K screening information and safety reminders. |
| 🌿 AI Plant Doctor | Analyzes supported plant-leaf images using the configured model. |
| 🤖 AI Farming Assistant | Answers farming questions using the configured Groq AI integration. |
| 📰 Live Agriculture News | Displays recent agriculture, weather, monsoon, and market headlines. |
| 🌐 Bilingual Interface | Lets users choose English or Tamil. |

## 🛠️ Technology Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

- **Frontend:** Streamlit and custom CSS
- **Machine learning:** scikit-learn and Joblib
- **Data processing:** Pandas
- **Plant image analysis:** PyTorch and Hugging Face Hub, as configured
- **Weather data:** OpenWeatherMap API
- **AI assistant:** Groq API
- **Local account prototype:** SQLite with salted password hashing

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Nagulanvelu/mazhai-ai-crop.git
cd mazhai-ai-crop
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure API keys

Create a local `.env` file in the project root:

```dotenv
OPENWEATHER_API_KEY=your_openweathermap_api_key
GROQ_API_KEY=your_groq_api_key
```

Replace the example values with your own API keys. Never commit your `.env` file or publish real credentials.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL printed in your terminal to test the application during development.

## ☁️ Deploy to Streamlit Community Cloud

1. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in using GitHub.
3. Create an app from `Nagulanvelu/mazhai-ai-crop`.
4. Select branch `main` and main file path `app.py`.
5. Add `OPENWEATHER_API_KEY` and `GROQ_API_KEY` through the deployment's Secrets settings.
6. Deploy and review the logs for any dependency or model-loading errors.

The hosting service will provide the public web address after a successful deployment.

## 🔒 Security

- Keep `.env` and all API credentials private.
- Exclude `.venv/`, `__pycache__/`, `.streamlit/secrets.toml`, and private user data from Git.
- Configure secrets through the hosting provider.
- Check model file sizes against GitHub and hosting limits.
- The local SQLite login is a prototype; production authentication and persistent storage require appropriate hosting support.

## ⚠️ Responsible Use

Mazhai AI Crop is an educational decision-support project. Crop and yield predictions depend on model training data and input quality. Plant disease predictions can be incorrect, and nutrient screening is not a precise fertilizer prescription.

Always validate important crop, irrigation, pesticide, and fertilizer decisions using soil tests, official advisories, and qualified local agricultural experts.

## 🗺️ Roadmap

- [x] Crop recommendation interface
- [x] Historical yield prediction interface
- [x] Live weather lookup
- [x] Basic fertilizer guidance
- [x] AI plant-health analysis
- [x] AI farming assistant
- [x] Agriculture and weather headlines
- [x] English and Tamil selection
- [ ] Public deployment and cloud testing
- [ ] Improve mobile responsiveness and performance
- [ ] Expand model validation with regional agricultural data

## 👨‍💻 Maintainer

**Nagulan V**

GitHub: [@Nagulanvelu](https://github.com/Nagulanvelu)

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B3D2E,50:238636,100:9ACD32&height=110&section=footer" width="100%" alt="Green wave footer"/>

### 🌱 Grow smarter. Harvest with confidence.

*Built with curiosity and AI for smarter agriculture.*

</div>
