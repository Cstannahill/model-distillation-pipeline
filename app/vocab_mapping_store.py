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
    path = get_mapping_path(student_tokenizer, teacher_tokenizer)
    # Write in chunks to avoid losing all progress if interrupted
    keys = list(mapping.keys())
    total = len(keys)
    temp_path = path + ".tmp"
    written = 0
    with open(temp_path, "w") as f:
        f.write("{")
        for i, k in enumerate(keys):
            # Write key-value pair
            entry = f'"{k}": {json.dumps(mapping[k])}'
            if i > 0:
                f.write(",")
            f.write(entry)
            written += 1
            # Flush every chunk_size entries
            if written % chunk_size == 0:
                f.flush()
        f.write("}")
    os.replace(temp_path, path)
    print(f"Saved vocab mapping to {path} in {((total-1)//chunk_size)+1} chunks.")


def load_vocab_mapping(student_tokenizer, teacher_tokenizer):
    path = get_mapping_path(student_tokenizer, teacher_tokenizer)
    if os.path.exists(path):
        with open(path, "r") as f:
            mapping = json.load(f)
        print(f"Loaded vocab mapping from {path}")
        return {int(k): v for k, v in mapping.items()}
    return None
