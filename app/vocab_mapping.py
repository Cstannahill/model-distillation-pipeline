import os
import time
import torch
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional
from tqdm import tqdm

# keep your existing JSON backup helpers (optional)
from .vocab_mapping_store import (
    save_vocab_mapping,
    load_vocab_mapping,
    get_mapping_path,
)

# new sqlite-backed durability
from .distilldb import (
    init_db,
    save_chunk,
    load_all_mappings,
    get_last_processed,
)


def build_vocab_mapping(student_tokenizer, teacher_tokenizer, partial_mapping=None):
    """
    Build student->teacher token mapping with robust SQLite-backed checkpoints.

    Resuming behavior:
      - The sqlite DB stores mappings (student_id -> teacher_ids) and a meta key
        'last_processed' which is the last student_id we processed.
      - On startup we initialize the DB, load DB mappings, and resume from
        (last_processed + 1).
      - We still optionally create JSON backups via save_vocab_mapping() as you had.
    """
    # DB path derived from your existing mapping path (replace .json -> .db)
    json_path = get_mapping_path(student_tokenizer, teacher_tokenizer)
    db_path = os.path.splitext(json_path)[0] + ".db"

    # ensure DB & tables exist
    init_db(db_path)

    # load what's already in DB (fast)
    db_mapping = load_all_mappings(db_path) or {}
    # load any existing JSON checkpoint (optional fallback)
    disk_mapping = load_vocab_mapping(student_tokenizer, teacher_tokenizer) or {}

    # start from the most complete mapping we know of
    if partial_mapping is None:
        # prefer DB mapping (authoritative), then disk mapping
        mapping = db_mapping if db_mapping else disk_mapping.copy()
    else:
        mapping = dict(partial_mapping)
        # prefer larger known mappings
        if db_mapping and len(db_mapping) > len(mapping):
            mapping = dict(db_mapping)
        elif disk_mapping and len(disk_mapping) > len(mapping):
            mapping = dict(disk_mapping)

    print(f"[DEBUG] Initial mapping size: {len(mapping)}")

    vocab_size = student_tokenizer.vocab_size
    # Read last processed student_id from DB (None -> start at 0)
    last_processed = get_last_processed(db_path)
    if last_processed is None:
        last_processed = -1
    print(f"[DEBUG] Resuming from last_processed student_id: {last_processed}")

    chunk_size = 1000  # save every N student ids
    chunk_dict: Dict[int, List[int]] = {}
    processed_since_save = 0
    last_saved_student_id = last_processed

    # Iterate sequentially so resume index is stable
    for student_id in tqdm(range(0, vocab_size), desc="Student ids (0..vocab_size-1)"):
        # skip already-processed IDs (based on DB's last_processed).
        if student_id <= last_processed:
            # ensure mapping has any DB values for this id
            if student_id in db_mapping and student_id not in mapping:
                mapping[student_id] = db_mapping[student_id]
            continue

        # skip if mapping already exists (maybe from disk or partial)
        if student_id in mapping:
            # still count this as processed for resume purposes
            processed_since_save += 1
            last_saved_student_id = student_id
        else:
            # decode the student token
            try:
                token_str = student_tokenizer.decode(
                    [student_id], skip_special_tokens=True
                )
            except Exception:
                # treat as processed without mapping
                token_str = ""
            teacher_ids: List[int] = []

            if token_str != "":
                # brute force search teacher vocab (can optimize later)
                for teacher_id in range(teacher_tokenizer.vocab_size):
                    try:
                        teacher_str = teacher_tokenizer.decode(
                            [teacher_id], skip_special_tokens=True
                        )
                    except Exception:
                        continue
                    if teacher_str == token_str and token_str != "":
                        teacher_ids.append(teacher_id)

            if teacher_ids:
                mapping[student_id] = teacher_ids
                chunk_dict[student_id] = teacher_ids
                print(
                    f"[DEBUG] Added mapping for student_id {student_id} (now {len(mapping)})"
                )

            # count as processed (whether mapped or not)
            processed_since_save += 1
            last_saved_student_id = student_id

        # commit every chunk_size processed student IDs
        if processed_since_save >= chunk_size:
            try:
                # Save the chunk to sqlite (atomic)
                save_chunk(db_path, chunk_dict, last_saved_student_id)
                print(
                    f"[LOG] Saved chunk upto student_id {last_saved_student_id} ({len(chunk_dict)} new mappings)."
                )
                # Optional: keep disk JSON backup as well (cheap human-readable backup)
                try:
                    save_vocab_mapping(
                        mapping,
                        student_tokenizer,
                        teacher_tokenizer,
                        chunk_size=chunk_size,
                    )
                except Exception as e:
                    print(f"[WARN] JSON backup failed after chunk save: {e}")
            except Exception as e:
                print(
                    f"[ERROR] Failed to save chunk to DB at student_id {last_saved_student_id}: {e}"
                )
                # on failure, continue — DB should be stable next run
            # reset chunk counters
            chunk_dict = {}
            processed_since_save = 0
            last_processed = last_saved_student_id

    # final commit for any remaining chunk
    if processed_since_save > 0 or last_saved_student_id > last_processed:
        try:
            save_chunk(db_path, chunk_dict, last_saved_student_id)
            print(
                f"[LOG] Final chunk saved upto student_id {last_saved_student_id} ({len(chunk_dict)} new mappings)."
            )
            try:
                save_vocab_mapping(
                    mapping,
                    student_tokenizer,
                    teacher_tokenizer,
                    chunk_size=chunk_size,
                )
            except Exception as e:
                print(f"[WARN] JSON backup failed on final save: {e}")
        except Exception as e:
            print(f"[ERROR] Final DB save failed: {e}")

    # In case DB was written by others during run, reload authoritative DB mapping
    final_db_mapping = load_all_mappings(db_path) or {}
    # Merge disk mapping and final_db_mapping and in-memory mapping — prefer largest entries
    final_mapping = dict(final_db_mapping)
    for k, v in mapping.items():
        if k not in final_mapping:
            final_mapping[k] = v

    print(
        f"Mapped {len(final_mapping)} out of {student_tokenizer.vocab_size} student tokens."
    )
    return final_mapping


def map_teacher_probs_to_student(
    teacher_probs: torch.Tensor,  # [batch, seq, teacher_vocab]
    vocab_mapping: Dict[int, List[int]],
    student_vocab_size: int,
) -> torch.Tensor:
    """
    Map teacher probabilities to student vocab size using the mapping.
    Returns tensor of shape [batch, seq, student_vocab]
    """
    batch, seq, _ = teacher_probs.shape
    student_probs = torch.zeros(
        (batch, seq, student_vocab_size), device=teacher_probs.device
    )
    mapped_ids = set(vocab_mapping.keys())
    unmapped_ids = set(range(student_vocab_size)) - mapped_ids
    for student_id, teacher_ids in vocab_mapping.items():
        # Sum teacher probabilities for all teacher tokens that map to this student token
        student_probs[..., student_id] = teacher_probs[..., teacher_ids].sum(dim=-1)
    # Assign small uniform probability to unmapped tokens
    if unmapped_ids:
        student_probs[..., list(unmapped_ids)] = 1e-8
    # Normalize so probabilities sum to 1 for each position
    sum_probs = student_probs.sum(dim=-1, keepdim=True)
    student_probs = student_probs / (sum_probs + 1e-8)
    return student_probs
