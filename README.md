 <div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:052E16,35:166534,70:22C55E,100:BBF7D0&height=230&section=header&text=MAZHAI%20AI%20CROP&fontSize=46&fontColor=FFFFFF&fontAlignY=38&desc=SMARTER%20FARMING%20%E2%80%A2%20BETTER%20HARVESTS&descAlignY=59&descSize=17&animation=fadeIn" width="100%" alt="Mazhai AI Crop animated green hero banner"/>

<br/>

<a href="https://mazhai-ai-crop.streamlit.app/">
  <img src="https://img.shields.io/badge/%E2%96%B6%20LAUNCH%20LIVE%20APP-16A34A?style=for-the-badge&logo=streamlit&logoColor=white" alt="Launch live app"/>
</a>
<a href="https://github.com/Nagulanvelu/mazhai-ai-crop">
  <img src="https://img.shields.io/badge/VIEW%20SOURCE%20CODE-111827?style=for-the-badge&logo=github&logoColor=white" alt="View source code"/>
</a>

<br/><br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=2800&pause=900&color=22C55E&center=true&vCenter=true&width=700&lines=AI-powered+agriculture+%F0%9F%8C%B1;Predict.+Analyze.+Grow.+%F0%9F%8C%BE;Building+a+smarter+farming+future+%F0%9F%8C%8D" alt="Animated project tagline"/>

<br/>

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
<img src="https://img.shields.io/badge/Machine%20Learning-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" alt="Machine Learning"/>
<img src="https://img.shields.io/badge/AI%20Assistant-7C3AED?style=flat-square" alt="AI Assistant"/>
<img src="https://img.shields.io/badge/Language-English%20%7C%20Tamil-0D9488?style=flat-square" alt="English and Tamil"/>

</div>

---

## 🌿 The vision

**What if technology could help farmers make more informed decisions?**

Mazhai AI Crop is an AI-powered agricultural web application that brings machine learning predictions, weather information, plant image analysis, and AI-driven farming assistance into one accessible platform.

Built with Python and Streamlit, the project explores how AI can support smarter agricultural decisions through an easy-to-use interface in **English and Tamil**.

<div align="center">

| 🌱 Smart Recommendations | 🌦️ Weather Insights | 🤖 AI Assistance |
|:---:|:---:|:---:|
| Explore crop suitability | Check current conditions | Get farming guidance |

</div>

## ✨ Explore the features

<table>
<tr>
<td width="50%" valign="top">

### 🌱 Crop Recommendation

Use soil and climate inputs to receive crop suggestions from a trained machine learning model.

### 🌾 Yield Prediction

Estimate agricultural yield using the project's trained prediction model.

### 🌦️ Live Weather

Retrieve current weather information for a selected city through the configured weather API.

### 🧪 Fertilizer Guidance

Explore basic nutrient information and fertilizer-related guidance.

</td>
<td width="50%" valign="top">

### 🌿 AI Plant Doctor

Analyze supported plant-leaf images using the configured plant-health model.

### 🤖 AI Farming Assistant

Ask agricultural questions using the configured Groq AI integration.

### 📰 Agriculture News

Explore recent agriculture, weather, monsoon, and market headlines from configured feeds.

### 🌐 English + தமிழ்

Switch between English and Tamil for a more accessible experience.

</td>
</tr>
</table>

## 🧠 How the system works

<div align="center">

<img src="https://mermaid.ink/img/pako:eNpVj8FqwzAMhl9F6NRC2k2hQ9hC2E5a2m6H3aVJtqR2M6lK2W0T0v3-JE0n2m8_8v9I2QqV0xq0k3K2KQ0q2pQ3v3f7w2rQ0pWQ1m7k0b2g3o7W2Q5o4c0g3b4p2o7q2q0v1v3o7b0g3r0q0p0x7m4q0o7b0m3r2g0v2w0" width="0" alt=""/>

</div>

```text
             👨‍🌾 USER
                |
                v
       🌿 STREAMLIT INTERFACE
                |
       +--------+---------+
       |        |         |
       v        v         v
   🌱 ML MODELS 🌦️ APIs  🤖 AI SERVICES
       |        |         |
       +--------+---------+
                |
                v
       📊 RESULTS & GUIDANCE
```

**Workflow:** User input → application interface → relevant model or external service → results presented to the user.

## 🛠️ Technology ecosystem

<div align="center">

<img src="https://skillicons.dev/icons?i=python,git,github" alt="Python and development tools"/>

<br/><br/>

<img src="https://img.shields.io/badge/Streamlit-Web%20Framework-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
<img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"/>
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>

</div>

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web application |
| Pandas | Data processing |
| scikit-learn | Machine learning models |
| Joblib | Model serialization and loading |
| PyTorch / Hugging Face Hub | Configured plant image analysis |
| OpenWeatherMap | Weather data |
| Groq API | AI farming assistant |
| SQLite | Local account prototype |
| Git, GitHub and Git LFS | Version control and model storage |
| Streamlit Community Cloud | Web deployment |

## 🚀 Run it on your machine

### Prerequisites

- Python installed on your system
- Git installed
- Git LFS installed for model files
- API credentials for the external services you want to use

### 1. Clone the project

```bash
git clone https://github.com/Nagulanvelu/mazhai-ai-crop.git
cd mazhai-ai-crop
```

### 2. Download model files

```bash
git lfs install
git lfs pull
git lfs ls-files
```

Confirm that the required model files are available under the `models/` directory.

### 3. Create a virtual environment

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

### 4. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure your credentials

Add your own API credentials through a local `.env` file if supported by your application:

```dotenv
OPENWEATHER_API_KEY=your_weather_api_key
GROQ_API_KEY=your_groq_api_key
```

Keep real credentials private. Do not commit `.env` files to GitHub.

### 6. Launch the application

Use the actual Streamlit entry-point filename in your repository:

```bash
streamlit run app.py
```

If your entry point has a different name, replace `app.py` with that filename.

## ☁️ Live deployment

<div align="center">

<a href="https://mazhai-ai-crop.streamlit.app/">
<img src="https://img.shields.io/badge/OPEN%20MAZHAI%20AI%20CROP-Launch%20Application-16A34A?style=for-the-badge&logo=streamlit&logoColor=white" alt="Launch deployed application"/>
</a>

<br/><br/>

**Live app:** [mazhai-ai-crop.streamlit.app](https://mazhai-ai-crop.streamlit.app/)

**Source code:** [github.com/Nagulanvelu/mazhai-ai-crop](https://github.com/Nagulanvelu/mazhai-ai-crop)

</div>

The application is hosted on Streamlit Community Cloud. A working deployment requires compatible dependencies, accessible model files, and correctly configured secrets.

For your own deployment:

1. Connect the repository in [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select the `main` branch and the correct entry-point file.
3. Add the required API keys under the app's Secrets settings.
4. Verify that Git LFS model objects are available.
5. Check the deployment logs if any service or model fails to load.

## 🔐 Security and responsible AI

- Never commit API keys, `.env`, private databases, or user credentials.
- Keep virtual environments and cache files out of version control.
- Use hosting-provider secrets for deployed applications.
- Treat local SQLite authentication as a prototype rather than production-grade account infrastructure.
- Crop and yield predictions depend on the model's training data and the quality of user inputs.
- Plant-health predictions can be inaccurate; nutrient guidance is not a laboratory soil analysis or precise fertilizer prescription.

**Always validate important agricultural decisions** using soil tests, official agricultural advisories, and qualified local experts.

## 🗺️ Project roadmap

- [x] Crop recommendation
- [x] Yield prediction
- [x] Live weather integration
- [x] Basic fertilizer guidance
- [x] AI plant-health analysis
- [x] AI farming assistant
- [x] Agriculture news
- [x] English and Tamil interface
- [x] Public cloud deployment
- [x] Model files managed using Git LFS
- [ ] Add real application screenshots and a video walkthrough
- [ ] Improve mobile responsiveness
- [ ] Expand model evaluation using regional agricultural datasets
- [ ] Add more regional language support

## 👨‍💻 About the developer

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=19&pause=1000&color=22C55E&center=true&vCenter=true&width=550&lines=Nagulan+V;AI+Student+%7C+Python+%7C+Machine+Learning;Building+projects+that+solve+real+problems" alt="Developer introduction"/>

<br/><br/>

<a href="https://github.com/Nagulanvelu">
<img src="https://img.shields.io/badge/GitHub-Follow%20my%20work-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub profile"/>
</a>

<a href="https://mazhai-ai-crop.streamlit.app/">
<img src="https://img.shields.io/badge/Project-Try%20the%20Live%20Demo-16A34A?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live demo"/>
</a>

<br/><br/>

*Learning, building, and exploring the possibilities of AI.*

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:BBF7D0,35:22C55E,70:166534,100:052E16&height=150&section=footer&animation=fadeIn" width="100%" alt="Animated green footer"/>

### 🌱 Grow smarter. Harvest with confidence.

**Mazhai AI Crop — AI for a smarter agricultural future.**

</div>
