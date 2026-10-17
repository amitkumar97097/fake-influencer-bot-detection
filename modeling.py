from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack


HYBRID_METADATA_COLUMNS = (
    "Retweet Count",
    "Mention Count",
    "Log_Followers",
    "Verified",
    "Account_Age_Days",
    "Hashtag_Count",
    "Engagement_Rate",
    "tweet_length",
    "num_urls",
)

MODEL_CANDIDATES = {
    "lr_model": ["logistic_model_retrained.pkl", "logistic_model.pkl", "logistic_model.sav", "log_reg.pkl", "log_reg.sav", "logistic.pkl"],
    "rf_model": ["random_forest_model_retrained.pkl", "random_forest_model_repaired.pkl", "random_forest_model.pkl", "random_forest.pkl", "rf.pkl", "rf_model.pkl"],
    "tfidf": ["tfidf_retrained.pkl", "tfidf.pkl", "tfidf_vectorizer.pkl", "vectorizer.pkl"],
    "hybrid_model": ["hybrid_model_retrained.pkl", "hybrid_model_v2.pkl"],
    "hybrid_tfidf": ["hybrid_tfidf_retrained.pkl", "hybrid_tfidf.pkl"],
    "hybrid_scaler": ["hybrid_scaler_retrained.pkl", "hybrid_scaler.pkl"],
    "hybrid_reference": ["hybrid_reference_date.txt"],
    "summary_csv": ["model_summary.csv"],
}


def find_file(directory: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = directory / candidate
        if path.is_file():
            return path
    if directory.is_dir():
        files = {path.name.casefold(): path for path in directory.iterdir() if path.is_file()}
        for candidate in candidates:
            if candidate.casefold() in files:
                return files[candidate.casefold()]
    return None


def load_model_resources(model_dir: Path) -> dict[str, Any]:
    resources: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, candidates in MODEL_CANDIDATES.items():
        path = find_file(model_dir, candidates)
        resources[f"{name}_path"] = path
        resources[name] = None
        if path is None:
            continue
        try:
            if name == "hybrid_reference":
                resources[name] = path.read_text(encoding="utf-8").strip()
            elif name == "summary_csv":
                resources[name] = pd.read_csv(path)
            else:
                resources[name] = joblib.load(path)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    resources["lr_path"] = resources.get("lr_model_path")
    resources["rf_path"] = resources.get("rf_model_path")
    resources["load_errors"] = errors
    _validate_feature_pairs(resources)
    return resources


def _validate_feature_pairs(resources: dict[str, Any]) -> None:
    vectorizer = resources.get("tfidf")
    if vectorizer is not None:
        feature_count = len(getattr(vectorizer, "vocabulary_", {}))
        for model_name in ("lr_model", "rf_model"):
            model = resources.get(model_name)
            if model is not None and getattr(model, "n_features_in_", feature_count) != feature_count:
                resources["load_errors"][model_name] = (
                    f"Feature mismatch: model expects {model.n_features_in_}, "
                    f"but vectorizer creates {feature_count}."
                )
                resources[model_name] = None

    hybrid_model = resources.get("hybrid_model")
    hybrid_vectorizer = resources.get("hybrid_tfidf")
    hybrid_scaler = resources.get("hybrid_scaler")
    if all(value is not None for value in (hybrid_model, hybrid_vectorizer, hybrid_scaler)):
        expected = len(getattr(hybrid_vectorizer, "vocabulary_", {})) + int(hybrid_scaler.n_features_in_)
        if getattr(hybrid_model, "n_features_in_", expected) != expected:
            resources["load_errors"]["hybrid_model"] = (
                f"Feature mismatch: model expects {hybrid_model.n_features_in_}, "
                f"but vectorizer and scaler create {expected}."
            )
            resources["hybrid_model"] = None


def extract_hybrid_metadata(dataset: pd.DataFrame, reference_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    tweets = dataset["Tweet"].fillna("").astype(str)
    retweets = pd.to_numeric(dataset["Retweet Count"], errors="coerce").fillna(0).clip(lower=0)
    followers = pd.to_numeric(dataset["Follower Count"], errors="coerce").fillna(0).clip(lower=0)
    verified_raw = dataset["Verified"]
    verified = verified_raw.astype(str).str.casefold().isin({"true", "1", "yes", "verified"}).astype(int)

    created_at = pd.to_datetime(dataset["Created At"], errors="coerce", utc=True).dt.tz_convert(None)
    as_of = pd.Timestamp(reference_date) if reference_date else pd.Timestamp.now().normalize()
    account_age = (as_of - created_at).dt.days.fillna(0).clip(lower=0)
    mentions = tweets.str.count(r"@[A-Za-z0-9_]+")
    hashtags = tweets.str.count(r"#\w+")
    urls = tweets.str.count(r"http|www|\.com")

    return pd.DataFrame({
        "Retweet Count": retweets.astype(float),
        "Mention Count": mentions.astype(float),
        "Log_Followers": np.log1p(followers.astype(float)),
        "Verified": verified.astype(float),
        "Account_Age_Days": account_age.astype(float),
        "Hashtag_Count": hashtags.astype(float),
        "Engagement_Rate": (retweets / (followers + 1)).astype(float),
        "tweet_length": tweets.str.len().astype(float),
        "num_urls": urls.astype(float),
    }, columns=HYBRID_METADATA_COLUMNS)


def profile_hybrid_metadata(
    *,
    followers: int,
    following: int,
    posts: int,
    account_age_days: int,
    engagement_rate: float,
    verified: bool,
    profile_text: str,
) -> pd.DataFrame:
    text = str(profile_text or "")
    follower_count = max(0, int(followers))
    estimated_retweets = max(0.0, float(engagement_rate)) / 100.0 * (follower_count + 1)
    retweet_count = round(estimated_retweets)
    return pd.DataFrame([{
        "Retweet Count": float(retweet_count),
        "Mention Count": float(len(re.findall(r"@[A-Za-z0-9_]+", text))),
        "Log_Followers": float(np.log1p(follower_count)),
        "Verified": float(bool(verified)),
        "Account_Age_Days": float(max(0, int(account_age_days))),
        "Hashtag_Count": float(len(re.findall(r"#\w+", text))),
        "Engagement_Rate": float(retweet_count / (follower_count + 1)),
        "tweet_length": float(len(text)),
        "num_urls": float(len(re.findall(r"http|www|\.com", text, flags=re.IGNORECASE))),
    }], columns=HYBRID_METADATA_COLUMNS)


def transform_hybrid(texts: list[str], metadata: pd.DataFrame, vectorizer, scaler):
    text_vectors = vectorizer.transform([str(text or "") for text in texts])
    metadata_values = scaler.transform(metadata.loc[:, HYBRID_METADATA_COLUMNS].to_numpy(dtype=float))
    return hstack([text_vectors, csr_matrix(metadata_values)], format="csr")


def positive_class_probability(model, features) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = np.asarray(model.predict_proba(features))
    classes = list(getattr(model, "classes_", range(probabilities.shape[1])))
    if 1 not in classes:
        return None
    return probabilities[:, classes.index(1)]