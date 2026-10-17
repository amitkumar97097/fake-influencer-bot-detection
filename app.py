# app.py 
import os
import sys
from pathlib import Path
import time
import io
import sqlite3
import re
from contextlib import closing
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import joblib
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from auth import (
    DEFAULT_SETTINGS,
    authenticate_user,
    create_user,
    initialize_auth_db,
    load_user_settings,
    save_user_settings,
)
from evaluation import evaluate_all_models
from gcn_predict import load_graph_data, predict_graph_user
from graph_analysis import analyze_labeled_graph, visual_subgraph
from modeling import (
    load_model_resources,
    positive_class_probability,
    profile_hybrid_metadata,
    transform_hybrid,
)

# ---------------------------

# ---------------------------
st.set_page_config(
    page_title="Fake Influencer Detection",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# Base project directory (adjust if needed). Using Path for Windows-safe paths.
ML_DIR = BASE_DIR / "ml_outputs"
GRAPH_DIR = BASE_DIR / "graph_outputs"
DATA_DIR = BASE_DIR / "Twitter_DataSET"

# Local prediction history helpers (SQLite)
DB_PATH = Path(os.environ.get("FAKEINFLUENCER_DB_PATH", BASE_DIR / "prediction_history.db")).expanduser()
initialize_auth_db(DB_PATH)


def save_prediction(source, prediction, confidence=None, details=""):
    """Save a prediction locally without interrupting the prediction workflow."""
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            with conn:
                conn.execute(
                    "INSERT INTO predictions (created_at, username, source, prediction, confidence, details) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        st.session_state.get("auth_username"),
                        str(source),
                        str(prediction),
                        float(confidence) if confidence is not None else None,
                        str(details)[:4000],
                    ),
                )
    except Exception as exc:
        st.warning(f"Prediction was made, but history could not be saved: {exc}")

def make_history_pdf(history):
    """Create a simple PDF report; return None if ReportLab is unavailable."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
        styles = getSampleStyleSheet()
        elements = [Paragraph("Fake Influencer / Bot Detection — Prediction History", styles["Title"]),
                    Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Records: {len(history)}", styles["Normal"]), Spacer(1, 12)]
        cols = ["created_at", "source", "prediction", "confidence", "details"]
        data = [["Date / Time", "Source", "Prediction", "Confidence", "Details"]]
        for _, row in history[cols].fillna("").iterrows():
            conf = row["confidence"]
            conf_text = f"{float(conf):.1%}" if conf != "" and pd.notna(conf) else "N/A"
            data.append([str(row["created_at"])[:19], str(row["source"])[:24], str(row["prediction"])[:24], conf_text, str(row["details"])[:100]])
        table = Table(data, repeatRows=1, colWidths=[115, 90, 95, 70, 300])
        table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#263858")),
                                   ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                                   ("VALIGN", (0,0), (-1,-1), "TOP"), ("FONTSIZE", (0,0), (-1,-1), 8),
                                   ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#eef2f7")]),
                                   ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5)]))
        elements.append(table)
        doc.build(elements)
        return buffer.getvalue()
    except ImportError:
        return None

# ---------------------------
# PROFILE PDF REPORT
# ---------------------------
def make_profile_pdf(report):
    """Generate a PDF for one completed profile analysis."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from xml.sax.saxutils import escape
    except ImportError:
        return None

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    rows = [
        ["Metric", "Value"],
        ["Heuristic risk score", f"{report['risk']}/100"],
        ["Risk level", report["risk_level"]],
        ["Followers", str(report["followers"])],
        ["Following", str(report["following"])],
        ["Total posts / tweets", str(report["posts"])],
        ["Account age (days)", str(report["account_age_days"])],
        ["Average posts per day", f"{report['avg_posts_day']:.2f}"],
        ["Engagement rate (%)", f"{report['engagement_rate']:.2f}"],
        ["Verified", "Yes" if report["verified"] else "No"],
        ["Following / follower ratio", f"{report['following_ratio']:.2f}"],
        ["Posts per account-age day", f"{report['posts_per_age_day']:.2f}"],
    ]
    if report.get("trained_model_name"):
        rows.append(["Trained model", str(report["trained_model_name"])])
    if report.get("bot_probability") is not None:
        rows.append(["Trained model bot probability", f"{report['bot_probability']:.1%}"])
    table = Table(rows, colWidths=[220, 260])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263858")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")]),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story = [
        Paragraph("Fake Influencer Detection - Profile Risk Report", styles["Title"]),
        Paragraph("This rule-based score is a heuristic, not proof that an account is a bot.", styles["BodyText"]),
        Spacer(1, 14),
        table,
        Spacer(1, 14),
        Paragraph("Observed risk signals", styles["Heading2"]),
    ]
    if report["signals"]:
        story.extend(Paragraph(escape(str(signal)), styles["BodyText"]) for signal in report["signals"])
    else:
        story.append(Paragraph("No configured heuristic risk indicators were triggered.", styles["BodyText"]))
    if report.get("profile_text"):
        story.extend([Spacer(1, 10), Paragraph("Supplied bio / post text", styles["Heading2"]), Paragraph(escape(report["profile_text"]), styles["BodyText"])])
    document.build(story)
    return buffer.getvalue()

# ---------------------------
# THEME / STYLES (Light & Dark)
# ---------------------------
authenticated_username = st.session_state.get("auth_username")
if authenticated_username and st.session_state.get("_settings_loaded_for") != authenticated_username:
    saved_settings = load_user_settings(DB_PATH, authenticated_username)
    st.session_state["setting_theme"] = saved_settings["theme"]
    st.session_state["setting_default_model"] = saved_settings["default_model"]
    st.session_state["setting_show_probs"] = saved_settings["show_probs"]
    st.session_state["setting_show_sample_tweets"] = saved_settings["show_sample_tweets"]
    st.session_state["_settings_loaded_for"] = authenticated_username
else:
    st.session_state.setdefault("setting_theme", DEFAULT_SETTINGS["theme"])
    st.session_state.setdefault("setting_default_model", DEFAULT_SETTINGS["default_model"])
    st.session_state.setdefault("setting_show_probs", DEFAULT_SETTINGS["show_probs"])
    st.session_state.setdefault("setting_show_sample_tweets", DEFAULT_SETTINGS["show_sample_tweets"])

theme = st.sidebar.radio("🌓 Theme Mode", ["Light", "Dark"], key="setting_theme")

# Better CSS so text is readable in both modes and elements adapt
if theme == "Dark":
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background-color: #0E1117; color: #E6EDF3; }
        .css-1d391kg { color: #E6EDF3; } /* generic */
        .stTextInput>div>input, .stTextArea>div>textarea { background-color: #111318; color: #E6EDF3; }
        .stButton>button { color: #E6EDF3; }
        .stSelectbox>div>div { color: #E6EDF3; }
        .stDataFrame div { color: #E6EDF3; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp { background-color: #F7F7F7; color: #0B0B0B; }
        .stTextInput>div>input, .stTextArea>div>textarea { background-color: #FFFFFF; color: #0B0B0B; }
        .stButton>button { color: #0B0B0B; }
        .stSelectbox>div>div { color: #0B0B0B; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# MODERN DASHBOARD STYLING
# ---------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(145deg, #0b1020 0%, #101827 55%, #111326 100%); }
    [data-testid="stSidebar"] { background: #111827; border-right: 1px solid #263249; }
    [data-testid="stSidebar"] * { color: #e8edf7; }
    h1, h2, h3 { letter-spacing: -0.025em; }
    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 700 !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30,41,67,.96), rgba(18,27,45,.96));
        border: 1px solid #2b3b59; padding: 18px 20px; border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,.16);
    }
    [data-testid="stMetricLabel"] { color: #aab8d0 !important; }
    [data-testid="stMetricValue"] { color: #f3f6ff !important; font-weight: 800; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #2b3b59 !important; border-radius: 16px !important;
    }
    .stButton > button, .stDownloadButton > button {
        border: 0; border-radius: 10px; font-weight: 700;
        background: linear-gradient(90deg, #6d5dfc, #3b82f6); color: white;
        padding: .55rem 1rem; transition: transform .15s ease, filter .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        color: white; filter: brightness(1.12); transform: translateY(-1px);
        border: 0;
    }
    .stTextArea textarea, .stTextInput input {
        border-radius: 10px !important; border-color: #344563 !important;
        background: #0d1524 !important; color: #eef3ff !important;
    }
    [data-baseweb="select"] > div { border-radius: 10px; }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #2b3b59; border-radius: 12px; overflow: hidden;
    }
    hr { border-color: #2b3b59 !important; }
    .hero {
        padding: 26px 28px; border: 1px solid #33436a; border-radius: 20px;
        background: radial-gradient(circle at 85% 10%, rgba(109,93,252,.25), transparent 32%),
                    linear-gradient(120deg, #17233d, #121a2d);
        margin-bottom: 20px;
    }
    .hero-kicker { color: #9daeff; font-size: .82rem; text-transform: uppercase;
                   font-weight: 800; letter-spacing: .12em; }
    .hero-title { color: #f4f6ff; font-size: clamp(1.7rem, 3vw, 2.55rem);
                  line-height: 1.15; font-weight: 800; margin: 8px 0; }
    .hero-copy { color: #bdc9df; font-size: 1rem; margin-bottom: 0; }
    .section-note { color: #9eacc4; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# LOCAL ACCOUNT GATE
# ---------------------------
if not st.session_state.get("auth_username"):
    st.title("Sign in to Fake Influencer Detection")
    st.caption("Create a local account on first use. Passwords are stored as salted hashes in the project SQLite database.")
    st.warning("Local SQLite authentication is intended for a single-machine deployment, not a public multi-user service.")
    auth_mode = st.radio("Account", ["Sign in", "Create account"], horizontal=True, key="auth_mode")
    with st.form("local_auth_form"):
        account_username = st.text_input("Username", key="account_username")
        account_password = st.text_input("Password", type="password", key="account_password")
        confirm_password = None
        if auth_mode == "Create account":
            confirm_password = st.text_input("Confirm password", type="password", key="account_confirm_password")
        submitted = st.form_submit_button(auth_mode)
    if submitted and auth_mode == "Sign in":
        username = authenticate_user(DB_PATH, account_username, account_password)
        if username:
            st.session_state["auth_username"] = username
            st.rerun()
        st.error("Username or password was not recognized.")
    elif submitted and auth_mode == "Create account":
        if account_password != confirm_password:
            st.error("The passwords do not match.")
        else:
            try:
                st.session_state["auth_username"] = create_user(DB_PATH, account_username, account_password)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.stop()

# ---------------------------
# Helper: find model file by candidates
# ---------------------------
def find_file(dir_path: Path, candidates):
    """
    Return first candidate file path that exists in dir_path.
    candidates: list of filenames (str) or patterns
    """
    for c in candidates:
        p = dir_path / c
        if p.exists():
            return p
    # try case-insensitive match
    if dir_path.exists():
        files = {f.name.lower(): f for f in dir_path.iterdir() if f.is_file()}
        for c in candidates:
            low = c.lower()
            if low in files:
                return files[low]
    return None

# ---------------------------
# Load resources (with graceful fallback)
# ---------------------------
@st.cache_resource
def load_models_and_vectorizer():
    results = load_model_resources(ML_DIR)
    results["summary_df"] = results.get("summary_csv")
    return results

resources = load_models_and_vectorizer()


def evaluation_signature():
    candidates = [DATA_DIR / "bot_detection_data.csv"]
    candidates.extend(ML_DIR.glob("*.pkl"))
    candidates.extend(ML_DIR.glob("*.pt"))
    candidates.extend(ML_DIR.glob("*.csv"))
    candidates.extend(ML_DIR.glob("*.json"))
    candidates.extend((BASE_DIR / "processed_data").glob("*.pt"))
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(candidates)
        if path.is_file()
    )


@st.cache_data(show_spinner="Evaluating saved models on labeled holdout data...", max_entries=2)
def cached_model_evaluation(project_dir: str, signature: tuple):
    return evaluate_all_models(project_dir)


def get_model_evaluation():
    return cached_model_evaluation(str(BASE_DIR), evaluation_signature())


def render_evaluation_results(evaluation, key_prefix: str, show_curves: bool = False):
    metrics = evaluation["metrics"]
    if metrics.empty:
        st.info("No compatible model checkpoints and labeled holdout data were available for evaluation.")
    else:
        st.dataframe(metrics, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Download measured evaluation metrics (CSV)",
            metrics.to_csv(index=False).encode("utf-8"),
            file_name="measured_model_metrics.csv",
            mime="text/csv",
            key=f"{key_prefix}_metrics_csv",
        )
        predictions = evaluation["predictions"]
        if not predictions.empty:
            st.download_button(
                "⬇️ Download labeled test predictions (CSV)",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="model_test_predictions.csv",
                mime="text/csv",
                key=f"{key_prefix}_predictions_csv",
            )

        confusion_matrices = evaluation["confusion_matrices"]
        if confusion_matrices:
            model_name = st.selectbox(
                "Confusion matrix model",
                list(confusion_matrices),
                key=f"{key_prefix}_confusion_model",
            )
            matrix = confusion_matrices[model_name]
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Human", "Bot"],
                yticklabels=["Human", "Bot"],
                ax=ax,
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
            ax.set_title(f"{model_name}: confusion matrix")
            st.pyplot(fig)
            plt.close(fig)

        classification_reports = evaluation.get("classification_reports", {})
        if classification_reports:
            report_model = st.selectbox(
                "Class-wise classification report",
                list(classification_reports),
                key=f"{key_prefix}_class_report_model",
            )
            report_table = pd.DataFrame(classification_reports[report_model]).T
            st.dataframe(report_table, width="stretch")

        if show_curves and evaluation["curves"]:
            curve_names = list(evaluation["curves"])
            selected_curves = st.multiselect(
                "Models to plot",
                curve_names,
                default=curve_names,
                key=f"{key_prefix}_curve_models",
            )
            roc_fig, roc_ax = plt.subplots(figsize=(7, 4.5))
            pr_fig, pr_ax = plt.subplots(figsize=(7, 4.5))
            for name in selected_curves:
                row = metrics.loc[metrics["Model"] == name]
                auc_text = f" (AUC={row['ROC-AUC'].iloc[0]:.3f})" if not row.empty and pd.notna(row["ROC-AUC"].iloc[0]) else ""
                fpr, tpr, _ = evaluation["curves"][name]["roc"]
                precision, recall, _ = evaluation["curves"][name]["pr"]
                roc_ax.plot(fpr, tpr, label=f"{name}{auc_text}")
                pr_ax.plot(recall, precision, label=name)
            roc_ax.plot([0, 1], [0, 1], "k--", label="Chance")
            roc_ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
            pr_ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves")
            for axis in (roc_ax, pr_ax):
                axis.legend(loc="best")
                axis.grid(alpha=0.2)
            st.pyplot(roc_fig)
            st.pyplot(pr_fig)
            plt.close(roc_fig)
            plt.close(pr_fig)

        cross_seed_metrics = evaluation.get("cross_seed_metrics")
        if cross_seed_metrics is not None and not cross_seed_metrics.empty:
            st.subheader("GraphSAGE repeated-seed results")
            st.dataframe(cross_seed_metrics, width="stretch", hide_index=True)
            score_columns = [
                column for column in ("accuracy", "precision_bot", "recall_bot", "f1_bot", "f1_macro", "roc_auc", "pr_auc")
                if column in cross_seed_metrics.columns
            ]
            if score_columns:
                st.caption("Across-seed mean and standard deviation (the test set is scored once per seed).")
                st.dataframe(cross_seed_metrics[score_columns].agg(["mean", "std"]).T, width="stretch")

    for error in evaluation["errors"]:
        st.warning(error)

# ---------------------------
# TITLE & AUTHORS
# ---------------------------
st.markdown("<h1 style='text-align:center; color:#6C63FF;'>🤖 Fake Influencer / Bot Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Built by <b>Udaykant Choubey & Amit Kumar</b></p>", unsafe_allow_html=True)
st.divider()

# ---------------------------
# SIDEBAR
# ---------------------------
menu = [
    "Home",
    "Predict Bot",
    "Batch Predict",
    "Graphs",
    "Network Visualization",
    "ROC Analysis",
    "Model Metrics",
    "Graph GCN Detection",
    "Influencer Profile Analyzer",
    "Prediction History",
    "Research Evaluation & Export",
    "Settings",
    "About"
]
choice = st.sidebar.selectbox("📌 Menu", menu, key="app_navigation")

st.sidebar.caption(f"Signed in as {authenticated_username}")
if st.sidebar.button("Sign out", key="sign_out"):
    st.session_state.pop("auth_username", None)
    st.session_state.pop("profile_report", None)
    st.session_state.pop("_settings_loaded_for", None)
    st.rerun()

st.sidebar.markdown("### 🧭 Quick Controls")
model_choice_default = st.sidebar.radio(
    "Default Model for Quick Predict",
    ["Auto (best available)", "Logistic Regression", "Random Forest"],
    key="setting_default_model",
)
show_probs = st.sidebar.checkbox("Show prediction probabilities", key="setting_show_probs")
sample_tweet = st.sidebar.checkbox("Show sample tweets", key="setting_show_sample_tweets")
save_user_settings(DB_PATH, authenticated_username, {
    "theme": theme,
    "default_model": model_choice_default,
    "show_probs": show_probs,
    "show_sample_tweets": sample_tweet,
})

st.sidebar.markdown("---")
st.sidebar.write("Model files found:")
st.sidebar.write(f"- Logistic: {resources['lr_path'].name if resources['lr_path'] else 'Not found'}")
st.sidebar.write(f"- RandomForest: {resources['rf_path'].name if resources['rf_path'] else 'Not found'}")
st.sidebar.write(f"- TF-IDF: {resources['tfidf_path'].name if resources['tfidf_path'] else 'Not found'}")
for artifact_name, load_error in resources.get("load_errors", {}).items():
    st.sidebar.warning(f"{artifact_name}: {load_error}")

# ---------------------------
# HOME
# ---------------------------
if choice == "Home":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">AI · Machine Learning · Graph Intelligence</div>
          <div class="hero-title">Fake Influencer <span style="color:#8f83ff">/ Bot Detection</span></div>
          <p class="hero-copy">Analyze suspicious account behavior with text-based machine learning and social-network graph models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        graph_snapshot = load_graph_data(BASE_DIR)
        graph_labels = graph_snapshot["labels"].cpu().numpy()
        graph_nodes = graph_snapshot["num_nodes"]
        graph_edges = int(graph_snapshot["edge_index"].shape[1])
        labeled_users = graph_snapshot["num_labeled"]
        bot_count = int((graph_labels == 1).sum())
        human_count = int((graph_labels == 0).sum())
        graph_snapshot_error = None
    except Exception as exc:
        graph_labels = None
        graph_nodes = graph_edges = labeled_users = bot_count = human_count = None
        graph_snapshot_error = str(exc)

    available_models = sum([
        resources.get("lr_model") is not None,
        resources.get("rf_model") is not None,
        resources.get("hybrid_model") is not None,
        (ML_DIR / "gcn_model.pt").is_file(),
        (ML_DIR / "graphsage_model.pt").is_file(),
    ])
    st.markdown("### Project at a glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Graph nodes", f"{graph_nodes:,}" if graph_nodes is not None else "Unavailable")
    m2.metric("Graph edges", f"{graph_edges:,}" if graph_edges is not None else "Unavailable")
    m3.metric("Labeled users", f"{labeled_users:,}" if labeled_users is not None else "Unavailable")
    m4.metric("Available model artifacts", available_models)

    st.markdown("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("📌 Project overview")
        st.write(
            "This system explores whether a Twitter account is more likely to be a bot "
            "or a human by combining traditional machine learning with graph-based learning."
        )
        st.markdown(
            """
            **Model approaches**
            - Logistic Regression — feature-based baseline
            - Random Forest — tree-based baseline
            - GCN — Graph Convolutional Network
            - GraphSAGE — neighborhood aggregation
            - Hybrid — features combined with graph embeddings
            """
        )
        st.info("Research goal: evaluate whether social-network structure adds useful signal beyond feature-based models.")
    with right:
        st.subheader("🗂️ Dataset snapshot")
        dataset_info = pd.DataFrame({
            "Metric": ["Dataset", "Total nodes", "Total edges", "Labeled users", "Task", "Graph type"],
            "Value": [
                "TwiBot-20",
                f"{graph_nodes:,}" if graph_nodes is not None else "Unavailable",
                f"{graph_edges:,}" if graph_edges is not None else "Unavailable",
                f"{labeled_users:,}" if labeled_users is not None else "Unavailable",
                "Binary node classification",
                "Follower network",
            ],
        })
        st.dataframe(dataset_info, width="stretch", hide_index=True)
        if graph_snapshot_error:
            st.caption(f"Graph snapshot unavailable: {graph_snapshot_error}")

    st.divider()
    st.subheader("🏁 Measured model evaluation")
    st.info("Accuracy, ROC-AUC, PR-AUC, and confusion matrices are computed from labeled holdout predictions on the Research Evaluation & Export page. No illustrative scores are shown as measured results here.")

    st.divider()
    st.subheader("👥 Labeled account distribution")
    if graph_labels is None:
        st.info("Labeled graph data is unavailable; no distribution is displayed.")
    else:
        stats = {"Bots": bot_count, "Humans": human_count}
        c1, c2 = st.columns([0.8, 1.2], gap="large")
        with c1:
            total_labels = max(sum(stats.values()), 1)
            st.metric("Bot accounts", f"{stats['Bots']:,}", f"{stats['Bots'] / total_labels:.1%} of labeled users")
            st.metric("Human accounts", f"{stats['Humans']:,}", f"{stats['Humans'] / total_labels:.1%} of labeled users")
        with c2:
            fig3, ax3 = plt.subplots(figsize=(6, 3.6))
            ax3.pie(list(stats.values()), labels=list(stats.keys()), autopct="%1.1f%%",
                    startangle=90, colors=["#fb7185", "#38bdf8"],
                    textprops={"color": "#edf2ff"}, wedgeprops={"width": 0.42, "edgecolor": "#111827"})
            ax3.set_title("Bot vs human distribution", color="#f3f6ff")
            fig3.patch.set_alpha(0)
            st.pyplot(fig3, width="stretch")
            plt.close(fig3)

    st.divider()
    st.subheader("🧠 Model architecture")
    st.code(
        "Dataset (TwiBot-20)\\n"
        "       ↓\\n"
        "Feature-based models + social graph\\n"
        "       ↓\\n"
        "Logistic Regression / Random Forest / GCN / GraphSAGE / Hybrid\\n"
        "       ↓\\n"
        "Evaluation: Accuracy · ROC-AUC · Confusion Matrix\\n"
        "       ↓\\n"
        "Bot vs Human prediction",
        language="text",
    )
    st.caption("Note: the figures above are the project values currently hard-coded in app.py. Replace them with measured evaluation outputs when available.")
    st.success("Dataset: TwiBot-20  •  Task: fake influencer / bot detection")

# ---------------------------
# PREDICT SINGLE TWEET
# ---------------------------
elif choice == "Predict Bot":
    st.header("🔍 Detect Bot from a Tweet")

    if sample_tweet:
        st.info("Example tweet: \"Just boarded my private jet! 😎✈️ #luxurylifestyle #travel\"")

    user_input = st.text_area("✏️ Enter Tweet Here", height=120)

    model_choice = st.selectbox(
        "Choose Model",
        ["Auto (choose best available)"] + [m for m in ["Logistic Regression", "Random Forest"]]
    )

    # decide model to use
    def get_model_by_name(name):
        if name == "Logistic Regression":
            return resources.get("lr_model")
        if name == "Random Forest":
            return resources.get("rf_model")
        # Auto uses the stable Logistic Regression path first.
        return resources.get("lr_model") or resources.get("rf_model") or None

    chosen_model = get_model_by_name(model_choice if model_choice != "Auto (choose best available)" else model_choice_default)

    # resolve model from choice or default string
    if isinstance(chosen_model, str):  # if sidebar returned string label
        chosen_model = get_model_by_name(chosen_model)

    predict_button = st.button("🚀 Predict")

    if predict_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a tweet to predict.")
        elif resources.get("tfidf") is None:
            st.error("TF-IDF vectorizer not loaded. Prediction disabled.")
        elif chosen_model is None:
            st.error("No model loaded to perform prediction.")
        else:
            with st.spinner("⏳ Analyzing tweet with selected model..."):
                time.sleep(0.6)
                vector = resources["tfidf"].transform([user_input])
                label_idx = int(chosen_model.predict(vector)[0])
                label = "🤖 Bot Account" if label_idx == 1 else "🧑 Human Account"
                st.success(f"### ✅ Prediction: **{label}**")

                # Show probability and confidence only when the loaded model supports it.
                confidence = None
                bot_prob = None
                try:
                    probs = chosen_model.predict_proba(vector)[0]
                    classes = list(getattr(chosen_model, "classes_", range(len(probs))))
                    bot_position = classes.index(1) if 1 in classes else None
                    human_position = classes.index(0) if 0 in classes else None

                    if bot_position is not None:
                        bot_prob = float(probs[bot_position])
                    else:
                        bot_prob = None
                    if human_position is not None:
                        human_prob = float(probs[human_position])
                    else:
                        human_prob = None

                    if bot_prob is not None and human_prob is not None:
                        confidence = max(bot_prob, human_prob)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Bot probability", f"{bot_prob:.1%}")
                        col2.metric("Human probability", f"{human_prob:.1%}")
                        col3.metric("Model confidence", f"{confidence:.1%}")

                        if show_probs:
                            st.markdown("#### 📊 Prediction probability")
                            probability_df = pd.DataFrame(
                                {"Probability": [human_prob, bot_prob]},
                                index=["Human", "Bot"],
                            )
                            st.bar_chart(probability_df, width="stretch")
                except (AttributeError, ValueError, TypeError, IndexError):
                    st.info("This model does not expose class probabilities, so a confidence score cannot be shown.")

                # Honest explanation: these are observable text characteristics, not model feature attribution.
                words = user_input.split()
                hashtags = [word for word in words if word.startswith("#")]
                urls = [word for word in words if "http://" in word.lower() or "https://" in word.lower()]
                mentions = [word for word in words if word.startswith("@")]
                st.markdown("#### 🧾 Tweet analysis")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Word count", len(words))
                e2.metric("Hashtags", len(hashtags))
                e3.metric("Mentions", len(mentions))
                e4.metric("URLs", len(urls))
                st.caption(
                    "These counts describe the submitted text; they are not proof that an account is a bot. "
                    "The current text model does not provide a per-prediction feature explanation."
                )
                save_prediction(
                    source=f"Tweet / {model_choice}",
                    prediction=label,
                    confidence=confidence,
                    details=f"text={user_input[:500]} | words={len(words)} | hashtags={len(hashtags)} | mentions={len(mentions)} | urls={len(urls)}"
                )
                st.caption("✓ This prediction was saved to local history.")

# ---------------------------
# BATCH PREDICTIONS (CSV)
# ---------------------------
elif choice == "Batch Predict":
    st.header("📥 Batch Prediction (CSV)")

    st.markdown(
        """
        Upload a CSV containing at least one column with raw tweet text (column named `tweet_text` or `text`).
        The app will add `pred_label` and `pred_prob_bot` (if available) columns and allow you to download results.
        """
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    csv_has_col = None
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("Preview:")
            st.dataframe(df_upload.head())
            # pick text column
            text_cols = [c for c in df_upload.columns if c.lower() in ("tweet_text", "text", "tweet")]
            if not text_cols:
                st.warning("CSV does not contain 'tweet_text' or 'text' column. Please rename the column or upload a new file.")
            else:
                text_col = text_cols[0]
                st.success(f"Using column: `{text_col}` for prediction.")
                if st.button("Run Batch Prediction"):
                    if resources.get("tfidf") is None or (resources.get("lr_model") is None and resources.get("rf_model") is None):
                        st.error("Required model or TF-IDF missing.")
                    else:
                        with st.spinner("Predicting in batch..."):
                            vectors = resources["tfidf"].transform(df_upload[text_col].astype(str).tolist())
                            # Use the stable linear baseline by default; Random Forest remains available explicitly.
                            model_to_use = resources.get("lr_model") or resources.get("rf_model")
                            preds = model_to_use.predict(vectors)
                            df_upload["pred_label"] = ["bot" if p == 1 else "human" for p in preds]
                            bot_probabilities = positive_class_probability(model_to_use, vectors)
                            df_upload["pred_prob_bot"] = (
                                bot_probabilities if bot_probabilities is not None else np.nan
                            )

                            st.success("Batch prediction complete. Preview results below:")
                            st.dataframe(df_upload.head(), width="stretch")

                            # allow download
                            todownload = df_upload.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download Results CSV", data=todownload, file_name="batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {type(e).__name__}: {e}")

# ---------------------------
# GRAPHS
# ---------------------------
elif choice == "Graphs":
    st.header("📊 Social Graph Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Enhanced Graph Preview")
        enhanced_path = GRAPH_DIR / "enhanced_graph.png"
        if enhanced_path.exists():
            st.image(Image.open(enhanced_path), width="stretch")
            st.caption("Static preview asset; model inference uses the processed graph tensors on the Network Visualization page.")
        else:
            st.info("enhanced_graph.png not found in graph_outputs.")

    with col2:
        st.subheader("Mention Graph")
        mention_path = GRAPH_DIR / "mention_graph.png"
        if mention_path.exists():
            st.image(Image.open(mention_path), width="stretch")
        else:
            st.info("mention_graph.png not found in graph_outputs.")

# ---------------------------
# MODEL METRICS
# ---------------------------
elif choice == "Model Metrics":
    st.header("📈 Model Accuracy & Metrics")
    st.caption("All displayed values are recomputed from actual saved-model predictions and labeled holdout rows.")
    render_evaluation_results(get_model_evaluation(), "model_metrics")

# ---------------------------
# NETWORK VISUALIZATION
# ---------------------------
elif choice == "Network Visualization":
    st.header("🌐 Interactive Social Network Visualization")
    st.markdown("Red nodes are labeled bots; blue nodes are labeled humans. Clusters and edges below come from the processed graph tensors.")
    try:
        graph_analysis = analyze_labeled_graph(BASE_DIR)
        community_table = graph_analysis["communities"]
        st.caption(
            f"Labeled graph: {graph_analysis['num_nodes']:,} nodes, "
            f"{graph_analysis['num_edges']:,} edges, {len(community_table):,} communities."
        )
        suspicious_communities = community_table[community_table["suspicious"]]
        if suspicious_communities.empty:
            st.info("No communities met the configured suspicious-cluster threshold (at least 5 nodes and 60% bots).")
        else:
            st.subheader("Bot-heavy connected communities")
            st.dataframe(suspicious_communities.head(20), width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Export community analysis (CSV)",
            community_table.to_csv(index=False).encode("utf-8"),
            file_name="graph_community_analysis.csv",
            mime="text/csv",
            key="graph_community_export",
        )

        node_limit = st.slider(
            "Labeled nodes to visualize",
            min_value=100,
            max_value=min(1000, graph_analysis["num_nodes"]),
            value=min(300, graph_analysis["num_nodes"]),
            step=100,
        )
        labels, visual_edges = visual_subgraph(graph_analysis, node_limit=node_limit, edge_limit=3000)
        edge_rows = [
            {
                "source": int(source),
                "source_label": "Bot" if labels[source] == 1 else "Human",
                "target": int(target),
                "target_label": "Bot" if labels[target] == 1 else "Human",
            }
            for source, target in visual_edges
        ]
        st.download_button(
            "⬇️ Export displayed graph edges (CSV)",
            pd.DataFrame(edge_rows, columns=["source", "source_label", "target", "target_label"]).to_csv(index=False).encode("utf-8"),
            file_name="displayed_graph_edges.csv",
            mime="text/csv",
            key="graph_edges_export",
        )
        try:
            from pyvis.network import Network

            nodes = graph_analysis["nodes"].set_index("node_id")
            net = Network(height="620px", width="100%", bgcolor="#111111", font_color="white", directed=False)
            for node_id in range(len(labels)):
                row = nodes.loc[node_id]
                label = "Bot" if labels[node_id] == 1 else "Human"
                net.add_node(
                    node_id,
                    label=f"{label} {node_id}",
                    title=f"{label} account | degree {int(row['degree'])} | community {int(row['community_id'])}",
                    color="#ef476f" if labels[node_id] == 1 else "#38a3ff",
                )
            for source, target in visual_edges:
                net.add_edge(int(source), int(target))
            net.repulsion(node_distance=110, central_gravity=0.25)
            html_file = BASE_DIR / "network_graph.html"
            net.save_graph(str(html_file))
            st.iframe(html_file, height=660, alt="Interactive labeled bot and human social graph")
        except ImportError:
            st.error("Interactive graph visualization requires pyvis. Install it with: python -m pip install pyvis")
    except Exception as exc:
        st.error(f"Graph analysis is unavailable: {type(exc).__name__}: {exc}")
# ---------------------------
# ROC CURVE ANALYSIS
# ---------------------------
elif choice == "ROC Analysis":
    st.header("📈 ROC Curve Comparison")
    st.header("📈 ROC and Precision-Recall Curves")
    st.caption("Curves are calculated from actual positive-class probabilities on each model's labeled holdout set.")
    render_evaluation_results(get_model_evaluation(), "roc_analysis", show_curves=True)
# ---------------------------
# PHASE 1: INFLUENCER PROFILE ANALYZER
# ---------------------------
elif choice == "Influencer Profile Analyzer":
    st.header("🕵️ Influencer Profile Analyzer")
    st.caption("Profile indicators are heuristic signals, not proof that an account is a bot.")
    with st.form("profile_analyzer_form"):
        c1, c2 = st.columns(2)
        with c1:
            followers = st.number_input("Followers", min_value=0, value=500, step=10)
            following = st.number_input("Following", min_value=0, value=500, step=10)
            posts = st.number_input("Total posts / tweets", min_value=0, value=100, step=10)
        with c2:
            account_age_days = st.number_input("Account age (days)", min_value=0, value=365, step=30)
            avg_posts_day = st.number_input("Average posts per day", min_value=0.0, value=1.0, step=0.5)
            engagement_rate = st.number_input("Engagement rate (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
        verified = st.checkbox("Verified / officially authenticated (if known)")
        profile_text = st.text_area("Bio or recent post text (optional)", placeholder="Paste profile bio or a recent post...")
        run_profile = st.form_submit_button("🔎 Analyze Profile", width="stretch")
    if run_profile:
        following_ratio = following / max(followers, 1)
        posts_per_age_day = posts / max(account_age_days, 1)
        signals = []
        risk = 0
        if followers < 10 and following > 500:
            risk += 20; signals.append("Very low followers relative to following count")
        if following_ratio > 10:
            risk += 20; signals.append("Following-to-follower ratio is unusually high")
        if avg_posts_day > 50:
            risk += 25; signals.append("Very high stated posting frequency")
        if account_age_days < 30 and followers > 1000:
            risk += 15; signals.append("High follower count for a very new account")
        if engagement_rate < 0.1 and followers >= 1000:
            risk += 15; signals.append("Very low engagement rate relative to follower count")
        if not profile_text.strip():
            if resources.get("hybrid_model") is not None:
                signals.append("No bio/post text supplied; hybrid prediction uses the available account metrics")
            else:
                signals.append("No bio/post text supplied; text-only model was not run")
        if verified:
            risk = max(0, risk - 10); signals.append("Verification selected (small heuristic adjustment only)")
        risk = min(100, max(0, risk))
        risk_level = "High" if risk >= 60 else "Moderate" if risk >= 30 else "Low"
        a, b, c = st.columns(3)
        a.metric("Heuristic risk score", f"{risk}/100")
        b.metric("Following / follower ratio", f"{following_ratio:.2f}")
        c.metric("Posts per account-age day", f"{posts_per_age_day:.2f}")
        st.progress(risk / 100)
        if risk >= 60: st.warning("Several risk indicators detected. Review the account manually.")
        elif risk >= 30: st.info("Some risk indicators detected. More context is recommended.")
        else: st.success("Few heuristic risk indicators detected; this does not prove the account is human.")
        st.markdown("#### Signals observed")
        if signals:
            for signal in signals: st.write(f"• {signal}")
        else:
            st.write("No configured heuristic risk indicators were triggered.")
        profile_model_name = None
        profile_prediction = None
        bot_probability = None
        if all(resources.get(key) is not None for key in ("hybrid_model", "hybrid_tfidf", "hybrid_scaler")):
            try:
                metadata = profile_hybrid_metadata(
                    followers=followers,
                    following=following,
                    posts=posts,
                    account_age_days=account_age_days,
                    engagement_rate=engagement_rate,
                    verified=verified,
                    profile_text=profile_text,
                )
                hybrid_features = transform_hybrid(
                    [profile_text], metadata, resources["hybrid_tfidf"], resources["hybrid_scaler"]
                )
                hybrid_model = resources["hybrid_model"]
                profile_prediction = int(hybrid_model.predict(hybrid_features)[0])
                probabilities = positive_class_probability(hybrid_model, hybrid_features)
                bot_probability = float(probabilities[0]) if probabilities is not None else None
                profile_model_name = "Hybrid (tweet text + account metadata)"
            except Exception as exc:
                st.warning(f"Hybrid profile prediction unavailable: {type(exc).__name__}: {exc}")
        elif profile_text.strip() and resources.get("tfidf") is not None:
            text_model = resources.get("lr_model") or resources.get("rf_model")
            if text_model is not None:
                try:
                    text_features = resources["tfidf"].transform([profile_text])
                    profile_prediction = int(text_model.predict(text_features)[0])
                    probabilities = positive_class_probability(text_model, text_features)
                    bot_probability = float(probabilities[0]) if probabilities is not None else None
                    profile_model_name = "Text-only fallback"
                except Exception as exc:
                    st.warning(f"Text model could not analyze this text: {type(exc).__name__}: {exc}")
        else:
            st.info("A trained profile/text model is unavailable; only the transparent heuristic score is shown.")

        if profile_prediction is not None:
            st.markdown("#### Trained model result (separate from heuristic score)")
            st.write("🤖 Bot-like prediction" if profile_prediction == 1 else "🧑 Human-like prediction")
            if bot_probability is not None:
                st.metric("Trained model bot probability", f"{bot_probability:.1%}")
            if profile_model_name == "Hybrid (tweet text + account metadata)":
                st.caption("The Hybrid model uses text and supported account features. Retweet count is estimated from the entered engagement rate; following count and post frequency contribute to the separate heuristic score.")
        st.caption("The risk score is a transparent rule-based heuristic, not a trained account classifier. Use verified data and human review for decisions.")
        st.session_state["profile_report"] = {
            "risk": risk,
            "risk_level": risk_level,
            "followers": int(followers),
            "following": int(following),
            "posts": int(posts),
            "account_age_days": int(account_age_days),
            "avg_posts_day": float(avg_posts_day),
            "engagement_rate": float(engagement_rate),
            "verified": bool(verified),
            "following_ratio": following_ratio,
            "posts_per_age_day": posts_per_age_day,
            "signals": signals,
            "profile_text": profile_text.strip()[:1500],
            "trained_model_name": profile_model_name,
            "trained_prediction": profile_prediction,
            "bot_probability": bot_probability,
        }
        confidence = max(bot_probability, 1 - bot_probability) if bot_probability is not None else None
        save_prediction(
            source="Influencer Profile Analyzer",
            prediction=(
                f"Heuristic risk {risk}/100; trained bot prediction {profile_prediction}"
                if profile_prediction is not None else f"Heuristic risk {risk}/100"
            ),
            confidence=confidence,
            details=(f"risk_level={risk_level} | followers={int(followers)} | following={int(following)} | "
                     f"posts={int(posts)} | account_age_days={int(account_age_days)} | "
                     f"following_ratio={following_ratio:.2f} | signals={'; '.join(signals)}"),
        )

    profile_report = st.session_state.get("profile_report")
    if profile_report is not None:
        try:
            pdf_report = make_profile_pdf(profile_report)
        except Exception as exc:
            st.error(f"Could not generate the profile PDF: {exc}")
        else:
            if pdf_report is None:
                st.info("PDF export requires ReportLab. Install it with: python -m pip install reportlab")
            else:
                st.download_button(
                    "📄 Download Profile PDF Report",
                    data=pdf_report,
                    file_name="influencer_profile_risk_report.pdf",
                    mime="application/pdf",
                    key="download_profile_pdf",
                )

# ---------------------------
# PHASE 2: GRAPH ANALYSIS SUMMARY
# ---------------------------
elif choice == "Graph GCN Detection":
    st.header("🔗 Graph-Based Fake Influencer Detection")
    st.markdown("Predictions use the saved graph checkpoint and processed node features. The displayed true label is available only for labeled nodes.")
    try:
        graph_data = load_graph_data(BASE_DIR)
        graph_model_name = st.selectbox("Graph model", ["GCN", "GraphSAGE"], key="graph_inference_model")
        user_index = st.number_input(
            f"Labeled user index (0–{graph_data['num_labeled'] - 1})",
            min_value=0,
            max_value=graph_data["num_labeled"] - 1,
            step=1,
        )
        if st.button(f"🚀 Predict Using {graph_model_name}"):
            start_time = time.perf_counter()
            result = predict_graph_user(int(user_index), graph_model_name, BASE_DIR)
            elapsed = time.perf_counter() - start_time
            label_text = "🤖 Bot Account" if result["prediction"] == 1 else "🧑 Human Account"
            true_text = "🤖 Bot Account" if result["true_label"] == 1 else "🧑 Human Account"
            st.success(f"Prediction: {label_text}")
            st.metric("Bot probability", f"{result['bot_probability']:.1%}")
            st.metric("Model confidence", f"{result['confidence']:.1%}")
            st.metric("Inference time", f"{elapsed * 1000:.1f} ms")
            st.info(f"True label: {true_text}")
            save_prediction(
                source=f"{graph_model_name} / graph user index",
                prediction=label_text,
                confidence=result["confidence"],
                details=f"user_index={int(user_index)} | true_label={true_text} | inference_ms={elapsed * 1000:.1f}",
            )
            st.caption("✓ Prediction saved to local history.")
            st.markdown("### 📊 User Profile Snapshot")
            user_info = result["user_info"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Followers", user_info["followers"])
                st.metric("Following", user_info["following"])
            with col2:
                st.metric("Tweets", user_info["tweets"])
                st.metric("Account Age", user_info["account_age"])
            with col3:
                st.metric("Graph Degree", user_info["graph_degree"])
                st.metric("Engagement Score", user_info["engagement_score"])
    except Exception as exc:
        st.error(f"Graph prediction is unavailable: {type(exc).__name__}: {exc}")

# ---------------------------
# PHASE 4: LOCAL PREDICTION HISTORY
# ---------------------------
elif choice == "Prediction History":
    st.header("🗃️ Prediction History")
    st.caption("History is stored locally in prediction_history.db inside the project folder.")
    db_path = DB_PATH
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            history = pd.read_sql_query(
                "SELECT created_at, source, prediction, confidence, details "
                "FROM predictions WHERE username = ? ORDER BY id DESC",
                conn,
                params=(authenticated_username,),
            )
        if history.empty:
            st.info("No saved predictions yet. Run a prediction from the Predict Bot page to begin building your local history.")
        else:
            st.dataframe(history, width="stretch", hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Export history as CSV", history.to_csv(index=False).encode("utf-8"), "prediction_history.csv", "text/csv")
            with c2:
                pdf_bytes = make_history_pdf(history)
                if pdf_bytes is not None:
                    st.download_button("📄 Download history PDF", pdf_bytes, "prediction_history_report.pdf", "application/pdf")
                else:
                    st.warning("PDF export needs ReportLab. Install it with: python -m pip install reportlab")
    except Exception as exc:
        st.error(f"Could not open local history database: {exc}")

# ---------------------------
# SETTINGS
# ---------------------------
elif choice == "Settings":
    st.header("⚙️ Application Settings")
    current_settings = load_user_settings(DB_PATH, authenticated_username)
    st.write(f"Signed in as **{authenticated_username}**.")
    st.write(f"Theme: **{current_settings['theme']}**")
    st.write(f"Default quick-predict model: **{current_settings['default_model']}**")
    st.write(f"Show probabilities: **{'Yes' if current_settings['show_probs'] else 'No'}**")
    st.write(f"Show sample tweets: **{'Yes' if current_settings['show_sample_tweets'] else 'No'}**")
    st.caption("Change these settings using the sidebar controls; preferences are saved to the local SQLite database.")
    st.info("This app uses local SQLite authentication and storage. Use an identity provider and persistent managed database before public multi-user deployment.")

# ---------------------------
# PHASE 3: RESEARCH EVALUATION AND EXPORT
# ---------------------------
elif choice == "Research Evaluation & Export":
    st.header("📚 Research-Grade Evaluation & Export")
    st.write("Metrics are recomputed from labeled holdout rows and actual model outputs. No historical summary CSV is treated as ground truth.")
    evaluation = get_model_evaluation()
    render_evaluation_results(evaluation, "research_export", show_curves=True)
    st.markdown("#### Evaluation checklist")
    st.write("- Confusion matrix, precision, recall and F1-score: require held-out labels and predictions.")
    st.write("- ROC-AUC / PR-AUC: require ground-truth labels and probability scores.")
    st.write("- Inference time: measured on an individual prediction when that model runs.")
    st.write("- Text and Hybrid models share the tweet holdout. Graph models use labeled nodes from the graph trainer split.")
    st.warning("The GCN checkpoint has no saved split provenance; its computed row is diagnostic only. GraphSAGE uses a saved stratified 60/20/20 research split and repeated-seed results when the research checkpoint is available. Legacy GraphSAGE weights without saved splits are labeled diagnostic.")

# ---------------------------
# ABOUT
# ---------------------------
elif choice == "About":
    st.header("ℹ️ About / Help")
    st.markdown(
        """
        **How to use**
        - Sign in or create a local account on first use.
        - Run `python train_ml.py` to create matching text and Hybrid checkpoints from the labeled tweet dataset.
        - For batch predictions, upload a CSV with a `tweet_text`, `text`, or `tweet` column.
        - GCN and GraphSAGE require their checkpoint files plus the tensors in `processed_data`.
        """
    )
    st.markdown("**Troubleshooting**")
    st.write("- The app prefers versioned `*_retrained.pkl` artifacts and validates their feature counts against the matching vectorizers.")
    st.write("- Graph checkpoints must match the feature width in `processed_data`; missing or mismatched artifacts are reported without generating predictions.")
    st.write("- Local account/password data and prediction history are stored in `prediction_history.db`.")
    st.warning("The SQLite login is for local/single-machine use. Public deployment should use an identity provider and persistent managed database.")
    st.markdown("---")
    st.subheader("Project File Paths (detected)")
    st.write(f"Base dir: `{BASE_DIR}`")
    st.write(f"ml_outputs: `{ML_DIR}`")
    st.write(f"Detected Logistic: `{resources['lr_path']}`")
    st.write(f"Detected RandomForest: `{resources['rf_path']}`")
    st.write(f"Detected TF-IDF: `{resources['tfidf_path']}`")
    st.write(f"Detected Hybrid: `{resources['hybrid_model_path']}`")
    st.write(f"Detected Model Summary CSV: `{resources['summary_csv_path']}`")