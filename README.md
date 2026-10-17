# 🤖 Fake Influencer & Bot Detection System

<p align="center">
  <strong>Machine Learning | Graph Neural Networks | Streamlit</strong>
</p>

<p align="center">
  A machine learning-based application for detecting suspicious social media accounts using tweet analysis, profile features, and graph-based predictions.
</p>

<p align="center">
  [![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-FF4B4B?logo=streamlit&logoColor=white)](https://fake-influencer-bot-detection.streamlit.app/)
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/amitkumar97097/fake-influencer-bot-detection)
</p>

---

## 🌐 Live Demo

**Try the application here:**

👉 https://fake-influencer-bot-detection.streamlit.app/

> Note: The demo requires a working deployment and access to the required model and dataset files. If the app is temporarily unavailable, check the deployment logs.

## 📌 About the Project

The **Fake Influencer & Bot Detection System** is designed to help analyze social media accounts and identify potentially suspicious or automated activity.

It combines machine learning, tweet-level analysis, account profile features, and graph-based approaches to produce predictions that can support further investigation.

The application provides a user-friendly Streamlit interface for running predictions, exploring analysis results, and reviewing available evaluation metrics.

**Important:** Predictions are model-based estimates, not definitive proof that an account is fake or operated by a bot.

## ✨ Features

- **🔐 User Authentication** — Login functionality provided by the application.
- **📝 Tweet Analysis** — Analyze tweet text using the available prediction pipeline.
- **📂 Batch CSV Prediction** — Upload a supported CSV file to analyze multiple records.
- **👤 Profile Analysis** — Examine available account profile features.
- **🕸️ Graph Analysis** — Explore relationships between accounts using graph-based analysis.
- **🧠 Machine Learning** — Use the project's trained models for supported predictions.
- **🔗 Graph Neural Networks** — Includes GCN/GraphSAGE-related functionality where the required artifacts are available.
- **📊 Evaluation Dashboard** — Review supported model evaluation metrics and visualizations.
- **📈 Data Visualization** — Explore charts and analysis outputs.
- **📥 Export Results** — Download supported prediction results in CSV or PDF format.
- **🗂️ Prediction History** — Review stored prediction history where enabled.
- **⚙️ Settings and Help** — Access available application settings and guidance.

*The availability of individual features depends on the files, model artifacts, and configuration included in the deployed version.*

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and evaluation utilities |
| PyTorch | Deep learning |
| PyTorch Geometric | Graph neural network functionality |
| Matplotlib / Plotly | Data visualization, where used |
| SQLite | Local prediction history, where configured |
| Git & GitHub | Version control and project hosting |

## 🏗️ Project Structure

```text
fake-influencer-bot-detection/
│
├── app.py                  # Main Streamlit application
├── auth.py                 # Authentication functionality
├── evaluation.py           # Model evaluation
├── gcn_predict.py          # Graph neural network prediction
├── graph_analysis.py       # Graph analysis
├── modeling.py              # Modeling utilities
├── requirements.txt         # Python dependencies
├── .gitignore              # Files excluded from Git
│
├── ml_outputs/              # Model artifacts and outputs
├── processed_data/          # Processed data and supported artifacts
├── graph_outputs/           # Graph analysis outputs
├── Twitter_DataSET/         # Dataset files used by the project
│
└── README.md                # Project documentation
```

*This is a representative structure. Exact files and folders may differ in your repository.*

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/amitkumar97097/fake-influencer-bot-detection.git
cd fake-influencer-bot-detection
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some PyTorch or PyTorch Geometric packages may require installation instructions specific to your operating system and Python environment. Follow the official installation guides if those dependencies fail.

### 4. Start the application

```bash
streamlit run app.py
```

The local application will usually open at:

```text
http://localhost:8501
```

## 📁 Dataset and Model Requirements

The application may require dataset files, trained model artifacts, processed tensors, and graph-related files to operate correctly.

Before running or deploying the application:

1. Confirm that required files exist in the expected directories.
2. Check that dataset paths used in the code match the actual filenames.
3. Ensure the deployed environment has all required model artifacts.
4. Keep large files within the hosting platform's supported limits.
5. Avoid uploading passwords, API keys, private datasets, or personal information to a public repository.

If a required file is missing, the relevant feature may fail even when the main Streamlit application starts successfully.

## ☁️ Deployment

This project is intended to run as a Streamlit web application.

General deployment steps:

1. Push the application code to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect the GitHub repository.
4. Select the appropriate branch and `app.py`.
5. Configure any required secrets using the platform's secrets settings.
6. Confirm that all required dependencies, datasets, and model files are available.
7. Review the deployment logs and test the application's major features.

**Live application:**  
https://fake-influencer-bot-detection.streamlit.app/

## 📊 Evaluation

The project includes evaluation-related functionality for examining model performance.

Depending on the configured evaluation pipeline and available labeled data, useful metrics may include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Actual results should be generated from the project's evaluation code and test data. No performance scores are claimed here because they should be verified against reproducible evaluation results.

## 🔒 Privacy and Limitations

- Predictions should be treated as indicators for further investigation, not conclusive evidence.
- Model performance depends on the quality, coverage, and representativeness of the training data.
- Publicly accessible deployment does not automatically make authentication or stored history secure.
- Do not upload sensitive account information or confidential datasets.
- Review authentication, database handling, and secrets management before using the application with real users.

## 🔮 Future Improvements

- Add more comprehensive model evaluation and explainability.
- Improve handling of missing or incomplete profile data.
- Add stronger validation for uploaded CSV files.
- Improve deployment reliability and model artifact management.
- Expand testing for authentication, predictions, and downloads.
- Add reproducible evaluation reports and documented benchmark results.

## 👨‍💻 Author

**Amit Kumar**

- **GitHub:** [@amitkumar97097](https://github.com/amitkumar97097)
- **Project Repository:** [Fake Influencer & Bot Detection](https://github.com/amitkumar97097/fake-influencer-bot-detection)
- **Live Demo:** [Open Application](https://fake-influencer-bot-detection.streamlit.app/)

## 📄 License

No license is specified in this README. Add a `LICENSE` file if you intend to grant others explicit permission to use, modify, or distribute the project.

---

<p align="center">
  <strong>Built with Python, Machine Learning, Graph Analysis, and Streamlit.</strong>
</p>
