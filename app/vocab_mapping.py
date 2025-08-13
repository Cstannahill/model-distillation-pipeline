import torch
from transformers import PreTrainedTokenizer
from typing import Dict, List
from tqdm import tqdm


def build_vocab_mapping(student_tokenizer, teacher_tokenizer, partial_mapping=None):
    if partial_mapping is None:
        mapping = {}
    else:
        mapping = dict(partial_mapping)
    missing_ids = [
        sid for sid in range(student_tokenizer.vocab_size) if sid not in mapping
    ]
    for student_id in tqdm(missing_ids, desc="Student tokens (missing only)"):
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
    for student_id, teacher_ids in vocab_mapping.items():
        # Sum teacher probabilities for all teacher tokens that map to this student token
        student_probs[..., student_id] = teacher_probs[..., teacher_ids].sum(dim=-1)
    return student_probs
