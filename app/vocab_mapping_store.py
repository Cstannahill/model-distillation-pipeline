import os
import json
import hashlib


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
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    filename = f"vocab_mapping_{key_hash}.json"
    return os.path.join(dir_path, filename)


def save_vocab_mapping(mapping, student_tokenizer, teacher_tokenizer, chunk_size=1000):
    import shutil
    import time

    path = get_mapping_path(student_tokenizer, teacher_tokenizer)
    keys = list(mapping.keys())
    total = len(keys)
    temp_path = path + ".tmp"
    backup_dir = os.path.join(os.path.dirname(path), "vocab_mapping_backups")
    os.makedirs(backup_dir, exist_ok=True)
    # Backup original file if it exists and not already backed up
    if os.path.exists(path) and not os.path.exists(
        os.path.join(backup_dir, os.path.basename(path) + ".orig.bak")
    ):
        shutil.copy2(
            path, os.path.join(backup_dir, os.path.basename(path) + ".orig.bak")
        )
        print(f"[LOG] Original backup created: {os.path.basename(path)}.orig.bak")
    written = 0
    try:
        with open(temp_path, "w") as f:
            f.write("{")
            for i, k in enumerate(keys):
                entry = f'"{k}": {json.dumps(mapping[k])}'
                if i > 0:
                    f.write(",")
                f.write(entry)
                written += 1
                # Flush every chunk_size entries
                if written % chunk_size == 0:
                    f.flush()
                # Versioned backup every 1000 tokens
                if written % 1000 == 0:
                    ts = int(time.time())
                    backup_path = os.path.join(
                        backup_dir, f"{os.path.basename(path)}.{written}.{ts}.bak"
                    )
                    try:
                        shutil.copy2(temp_path, backup_path)
                        print(f"[LOG] Backup saved to {backup_path}")
                        # Keep only 3 most recent backups (plus original)
                        backups = sorted(
                            [
                                f
                                for f in os.listdir(backup_dir)
                                if f.startswith(os.path.basename(path) + ".")
                                and f.endswith(".bak")
                                and not f.endswith(".orig.bak")
                            ],
                            reverse=True,
                        )
                        if len(backups) > 3:
                            for old_bak in backups[3:]:
                                try:
                                    os.remove(os.path.join(backup_dir, old_bak))
                                    print(f"[LOG] Deleted old backup: {old_bak}")
                                except Exception as e:
                                    print(
                                        f"[ERROR] Failed to delete backup {old_bak}: {e}"
                                    )
                    except Exception as e:
                        print(f"[ERROR] Backup error: {e}")
            f.write("}")
        os.replace(temp_path, path)
        print(
            f"[LOG] Saved vocab mapping to {path} in {((total-1)//chunk_size)+1} chunks."
        )
        # Integrity check: reload and compare key sets
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
            loaded_keys = set(loaded.keys())
            mem_keys = set(str(k) for k in mapping.keys())
            if loaded_keys == mem_keys:
                print(f"[LOG] Integrity check passed: {len(loaded_keys)} keys.")
            else:
                print(
                    f"[WARN] Integrity check failed: {len(loaded_keys)} loaded vs {len(mem_keys)} in memory."
                )
        except Exception as e:
            print(f"[ERROR] Integrity check failed: {e}")
    except Exception as e:
        print(f"[ERROR] Error saving vocab mapping: {e}")


def load_vocab_mapping(student_tokenizer, teacher_tokenizer):
    import glob

    path = get_mapping_path(student_tokenizer, teacher_tokenizer)
    backup_dir = os.path.join(os.path.dirname(path), "vocab_mapping_backups")

    def safe_merge(map1, map2):
        # Prefer the mapping with more keys
        if map1 is None:
            return map2
        if map2 is None:
            return map1
        return map1 if len(map1) >= len(map2) else map2

    mapping = None
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                mapping = json.load(f)
            print(f"[LOG] Loaded vocab mapping from {path}")
            mapping = {int(k): v for k, v in mapping.items()}
        except Exception as e:
            print(f"[ERROR] Error loading vocab mapping: {e}")
    # Try to restore from original backup if main file is corrupted or incomplete
    orig_backup = os.path.join(backup_dir, os.path.basename(path) + ".orig.bak")
    if os.path.exists(orig_backup):
        try:
            with open(orig_backup, "r") as f:
                backup_mapping = json.load(f)
            print(f"[LOG] Restored vocab mapping from backup {orig_backup}")
            backup_mapping = {int(k): v for k, v in backup_mapping.items()}
            mapping = safe_merge(mapping, backup_mapping)
        except Exception as e2:
            print(f"[ERROR] Backup restore failed: {e2}")
    # Try to restore from most recent versioned backup if still incomplete
    versioned_backups = sorted(
        glob.glob(os.path.join(backup_dir, f"{os.path.basename(path)}.*.bak")),
        reverse=True,
    )
    for backup_path in versioned_backups:
        try:
            with open(backup_path, "r") as f:
                backup_mapping = json.load(f)
            print(f"[LOG] Restored vocab mapping from backup {backup_path}")
            backup_mapping = {int(k): v for k, v in backup_mapping.items()}
            mapping = safe_merge(mapping, backup_mapping)
            break
        except Exception as e3:
            print(f"[ERROR] Versioned backup restore failed: {e3}")
    if mapping is not None:
        print(f"[LOG] Final vocab mapping loaded with {len(mapping)} keys.")
        return mapping
    print(f"[ERROR] No valid vocab mapping found.")
    return None
