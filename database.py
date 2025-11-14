# database.py
import sqlite3
import datetime
import json

DB_NAME = "qa_logs.db"


def init_db():
    """データベースとテーブルを初期化する"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # ユーザーテーブル (認証用 - シンプルな実装)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """
    )

    # 質問ログテーブル
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY,
            timestamp DATETIME,
            user_id TEXT,
            question TEXT,
            answer TEXT,
            sources TEXT
        )
    """
    )

    # ダミーユーザーの追加 (passwordはハッシュ化せずにプレーンテキストとして保持 *デモ用*)
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            ("testuser", "password123"),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # すでにユーザーが存在する場合

    conn.close()
    print("Database initialized.")


def get_user(username, password):
    """ユーザー認証を行う"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username FROM users WHERE username=? AND password=?",
        (username, password),
    )
    user = cursor.fetchone()
    conn.close()
    return {"id": user[0], "username": user[1]} if user else None


def log_interaction(user_id, question, answer, sources):
    """質問と回答のログを記録する"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO logs (timestamp, user_id, question, answer, sources) 
        VALUES (?, ?, ?, ?, ?)
    """,
        (datetime.datetime.now(), user_id, question, answer, json.dumps(sources)),
    )
    conn.commit()
    conn.close()


# DB初期化
init_db()
