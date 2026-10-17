# 🤖 Fake Influencer & Bot Detection System

<p align="center">
  <strong>AI-Powered Social Media Analysis</strong>
  <br>
  Machine Learning · Graph Neural Networks · Streamlit
</p>

<p align="center">
  <a href="https://fake-influencer-bot-detection.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">
  </a>
  <a href="https://github.com/amitkumar97097/fake-influencer-bot-detection">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
  <img src="https://img.shields.io/badge/Python-ML-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

---

## 🌐 Live Demo

**Try the application:** [Launch Fake Influencer Detector](https://fake-influencer-bot-detection.streamlit.app/)

## 📌 About the Project

A machine learning-based application designed to analyze social media content and account features to identify potentially suspicious or automated activity.

The project combines traditional machine learning techniques with graph-based analysis to support social media account investigation through an interactive web interface.

## ✨ Key Features

- 🔐 User authentication
- 📝 Tweet text prediction
- 📂 Batch predictions from CSV files
- 👤 Social media profile analysis
- 🕸️ Graph-based account analysis
- 🧠 Machine learning predictions
- 🔗 GCN / GraphSAGE functionality where required artifacts are available
- 📊 Model evaluation and visualizations
- 📥 CSV and PDF result exports
- 🗂️ Prediction history, where enabled

## 🛠️ Tech Stack

`Python` · `Streamlit` · `Pandas` · `Scikit-learn` · `PyTorch` · `PyTorch Geometric` · `SQLite`

## 🚀 Run Locally

```bash
git clone https://github.com/amitkumar97097/fake-influencer-bot-detection.git
cd fake-influencer-bot-detection
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## 📁 Project Structure

```text
├── app.py
├── auth.py
├── evaluation.py
├── gcn_predict.py
├── graph_analysis.py
├── modeling.py
├── requirements.txt
├── ml_outputs/
├── processed_data/
├── graph_outputs/
└── Twitter_DataSET/
```

## ⚠️ Limitations

Predictions are model-based estimates and should not be treated as definitive proof that an account is fake or automated. Results depend on data quality, model performance, and the availability of required model and dataset files.

## 👨‍💻 Author

**Amit Kumar**

- GitHub: [@amitkumar97097](https://github.com/amitkumar97097)
- Repository: [Fake Influencer & Bot Detection](https://github.com/amitkumar97097/fake-influencer-bot-detection)
- Live Demo: [Open Application](https://fake-influencer-bot-detection.streamlit.app/)

---

<p align="center">
  Made with Python, Machine Learning & Streamlit
</p>
