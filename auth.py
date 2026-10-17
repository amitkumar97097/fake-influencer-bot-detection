from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


PASSWORD_ITERATIONS = 310_000
DEFAULT_SETTINGS = {
    "theme": "Dark",
    "default_model": "Auto (best available)",
    "show_probs": True,
    "show_sample_tweets": True,
}


def initialize_auth_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS users ("
                "username TEXT PRIMARY KEY COLLATE NOCASE, salt BLOB NOT NULL, "
                "password_hash BLOB NOT NULL, created_at TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS user_settings ("
                "username TEXT PRIMARY KEY COLLATE NOCASE, settings_json TEXT NOT NULL, "
                "FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS predictions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, username TEXT, "
                "source TEXT, prediction TEXT, confidence REAL, details TEXT)"
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
            if "username" not in columns:
                connection.execute("ALTER TABLE predictions ADD COLUMN username TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS predictions_user_created "
                "ON predictions(username, id DESC)"
            )


def normalize_username(username: str) -> str:
    normalized = username.strip().casefold()
    if not re.fullmatch(r"[a-z0-9_.-]{3,32}", normalized):
        raise ValueError("Username must be 3-32 characters: letters, digits, dot, underscore or hyphen.")
    return normalized


def create_user(db_path: Path, username: str, password: str) -> str:
    normalized = normalize_username(username)
    if len(password) < 10:
        raise ValueError("Use a password with at least 10 characters.")

    salt = secrets.token_bytes(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    with closing(sqlite3.connect(db_path)) as connection:
        try:
            with connection:
                connection.execute(
                    "INSERT INTO users(username, salt, password_hash, created_at) "
                    "VALUES (?, ?, ?, datetime('now'))",
                    (normalized, salt, password_hash),
                )
                connection.execute(
                    "INSERT INTO user_settings(username, settings_json) VALUES (?, ?)",
                    (normalized, json.dumps(DEFAULT_SETTINGS)),
                )
                connection.execute(
                    "UPDATE predictions SET username = ? WHERE username IS NULL",
                    (normalized,),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("That username is already registered.") from exc
    return normalized


def authenticate_user(db_path: Path, username: str, password: str) -> str | None:
    try:
        normalized = normalize_username(username)
    except ValueError:
        return None
    with closing(sqlite3.connect(db_path)) as connection:
        row = connection.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?",
            (normalized,),
        ).fetchone()
    if row is None:
        return None
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), row[0], PASSWORD_ITERATIONS
    )
    return normalized if hmac.compare_digest(candidate, row[1]) else None


def load_user_settings(db_path: Path, username: str) -> dict[str, Any]:
    with closing(sqlite3.connect(db_path)) as connection:
        row = connection.execute(
            "SELECT settings_json FROM user_settings WHERE username = ?",
            (username,),
        ).fetchone()
    settings = dict(DEFAULT_SETTINGS)
    if row:
        try:
            settings.update(json.loads(row[0]))
        except (TypeError, json.JSONDecodeError):
            pass
    return settings


def save_user_settings(db_path: Path, username: str, settings: dict[str, Any]) -> None:
    safe_settings = {key: settings[key] for key in DEFAULT_SETTINGS if key in settings}
    with closing(sqlite3.connect(db_path)) as connection:
        with connection:
            connection.execute(
                "INSERT INTO user_settings(username, settings_json) VALUES (?, ?) "
                "ON CONFLICT(username) DO UPDATE SET settings_json = excluded.settings_json",
                (username, json.dumps(safe_settings)),
            )