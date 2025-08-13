import sqlite3
import json
import time
from typing import Dict, List, Optional


def init_db(db_path: str, synchronous: str = "FULL"):
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()
    # durability and concurrency settings
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute(f"PRAGMA synchronous = {synchronous};")
    # tables
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS mappings (
            student_id INTEGER PRIMARY KEY,
            teacher_ids TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def load_all_mappings(db_path: str) -> Dict[int, List[int]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT student_id, teacher_ids FROM mappings;")
    result = {row[0]: json.loads(row[1]) for row in cur.fetchall()}
    conn.close()
    return result


def get_last_processed(db_path: str) -> Optional[int]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT value FROM meta WHERE key = 'last_processed';")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else None


def save_chunk(
    db_path: str, mapping_chunk: Dict[int, List[int]], last_processed_index: int
):
    """
    Upsert mapping_chunk (student_id -> list of teacher_ids).
    Also stores last_processed index in meta table.
    Use a single transaction to make it atomic.
    """
    if not mapping_chunk:
        # still update last_processed
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("BEGIN;")
        cur.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_processed', ?);",
            (str(last_processed_index),),
        )
        conn.commit()
        conn.close()
        return

    conn = sqlite3.connect(db_path, timeout=60)
    cur = conn.cursor()
    cur.execute("BEGIN;")
    ts = int(time.time() * 1000)
    params = [(sid, json.dumps(tlist), ts) for sid, tlist in mapping_chunk.items()]
    cur.executemany(
        "INSERT OR REPLACE INTO mappings (student_id, teacher_ids, updated_at) VALUES (?, ?, ?);",
        params,
    )
    cur.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_processed', ?);",
        (str(last_processed_index),),
    )
    conn.commit()
    conn.close()
