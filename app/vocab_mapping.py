import torch
from transformers import PreTrainedTokenizer
from typing import Dict, List
from tqdm import tqdm
from .vocab_mapping_store import save_vocab_mapping, load_vocab_mapping


def build_vocab_mapping(student_tokenizer, teacher_tokenizer, partial_mapping=None):
    if partial_mapping is None:
        mapping = {}
    else:
        mapping = dict(partial_mapping)
    missing_ids = [
        sid for sid in range(student_tokenizer.vocab_size) if sid not in mapping
    ]
    chunk_size = 1000
    for i, student_id in enumerate(
        tqdm(missing_ids, desc="Student tokens (missing only)")
    ):
        try:
            token_str = student_tokenizer.decode([student_id], skip_special_tokens=True)
        except Exception:
            continue
        teacher_ids = []
        for teacher_id in tqdm(
            range(teacher_tokenizer.vocab_size),
            desc=f"Teacher tokens for student {student_id}",
            leave=False,
        ):
            try:
                teacher_str = teacher_tokenizer.decode(
                    [teacher_id], skip_special_tokens=True
                )
            except Exception:
                continue
            if token_str == teacher_str and token_str != "":
                teacher_ids.append(teacher_id)
        if teacher_ids:
            mapping[student_id] = teacher_ids
        # Save progress every chunk_size tokens
        if (i + 1) % chunk_size == 0:
            save_vocab_mapping(
                mapping, student_tokenizer, teacher_tokenizer, chunk_size=chunk_size
            )
            # Reload and merge mapping from disk to ensure progress is not lost
            disk_mapping = load_vocab_mapping(student_tokenizer, teacher_tokenizer)
            if disk_mapping:
                mapping.update(disk_mapping)
    # Final save after all tokens
    save_vocab_mapping(
        mapping, student_tokenizer, teacher_tokenizer, chunk_size=chunk_size
    )
    disk_mapping = load_vocab_mapping(student_tokenizer, teacher_tokenizer)
    if disk_mapping:
        mapping.update(disk_mapping)
    print(
        f"Mapped {len(mapping)} out of {student_tokenizer.vocab_size} student tokens."
    )
    return mapping


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
