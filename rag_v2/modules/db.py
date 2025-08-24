import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "db.sqlite3"

def init_db():
    # Create database and chat_history table if not exists
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        conn.commit()

def save_chat_history(question: str, answer: str):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_history (question, answer) VALUES (?, ?);",
            (question, answer)
        )
        conn.commit()

def get_chat_history(limit=20):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT question, answer FROM chat_history ORDER BY timestamp DESC LIMIT ?;", (limit,)
        )
        rows = cur.fetchall()
        # Return a list of dicts for convenience
        return [{"question": row[0], "answer": row} for row in rows]
