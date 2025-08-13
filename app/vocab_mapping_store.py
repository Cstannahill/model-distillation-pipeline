# vocab_mapping_store.py
import os
import json
import math
import time
from typing import Dict, List, Optional

from .distilldb import init_db, save_chunk  # uses your existing DB helper


# keep same get_mapping_path signature as before so other code doesn't change
def get_mapping_path(
    student_tokenizer,
    teacher_tokenizer,
    dir_path="/home/christian/dev/ai/distill-pipeline/app",
):
    student_name = getattr(
        student_tokenizer, "name_or_path", str(type(student_tokenizer))
    )
    teacher_name = getattr(
        teacher_tokenizer, "name_or_path", str(type(teacher_tokenizer))
    )
    key = f"{student_name}|{teacher_name}"
    # short stable hash so filename isn't huge
    import hashlib

    key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    filename = f"vocab_mapping_{key_hash}.json"
    return os.path.join(dir_path, filename)


# atomic JSON writer (fsync + atomic replace)
def atomic_json_write(path: str, data) -> None:
    dirpath = os.path.dirname(path) or "."
    tmp = os.path.join(dirpath, os.path.basename(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_vocab_mapping(
    mapping: Dict[int, List[int]],
    student_tokenizer,
    teacher_tokenizer,
    chunk_size: int = 1000,
):
    """
    Lightweight JSON export of the full mapping (for human-readable snapshot).
    Prefer the DB as the source of truth; this is optional / used as a final snapshot.
    """
    path = get_mapping_path(student_tokenizer, teacher_tokenizer)
    # Convert keys to str for JSON
    out = {str(k): v for k, v in mapping.items()}
    atomic_json_write(path, out)
    return path


def load_vocab_mapping(
    student_tokenizer, teacher_tokenizer
) -> Optional[Dict[int, List[int]]]:
    """
    Read the JSON snapshot if it exists. Return mapping with integer keys.
    Note: DB is still preferred for resume.
    """
    path = get_mapping_path(student_tokenizer, teacher_tokenizer)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return {int(k): v for k, v in m.items()}
    except Exception as e:
        # corrupted file — warn and return None
        print(f"[WARN] Failed to load JSON mapping {path}: {e}")
        return None


# -----------------------
# Migration helpers
# -----------------------


def import_json_to_db(json_path: str, db_path: str, batch_size: int = 1000):
    """
    Import a big JSON mapping into the sqlite DB in batches.
    This will call init_db(db_path) and then use distilldb.save_chunk to upsert in transactions.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    mapping = {int(k): v for k, v in raw.items()}

    init_db(db_path)

    # do batched upserts using save_chunk (atomic per batch)
    student_ids = sorted(mapping.keys())
    total = len(student_ids)
    for i in range(0, total, batch_size):
        batch_ids = student_ids[i : i + batch_size]
        chunk = {sid: mapping[sid] for sid in batch_ids}
        last_processed_sid = batch_ids[-1]
        save_chunk(db_path, chunk, last_processed_sid)
        print(
            f"[LOG] Imported batch {i//batch_size + 1} / {math.ceil(total/batch_size)} up to {last_processed_sid}"
        )

    print(f"[LOG] Imported {total} mappings from {json_path} -> {db_path}")


def find_latest_backup_json(backups_dir: str, base_filename: str) -> Optional[str]:
    """
    If you still kept the old backup folder, use this to pick the latest .bak JSON
    (you used to name files like 'vocab_mapping_<hash>.json.<count>.<ts>.bak').
    """
    if not os.path.isdir(backups_dir):
        return None
    candidates = []
    for fn in os.listdir(backups_dir):
        if fn.startswith(base_filename) and fn.endswith(".bak"):
            candidates.append(fn)
    if not candidates:
        return None
    # sort by mtime of the backup file
    candidates.sort(
        key=lambda x: os.path.getmtime(os.path.join(backups_dir, x)), reverse=True
    )
    return os.path.join(backups_dir, candidates[0])
