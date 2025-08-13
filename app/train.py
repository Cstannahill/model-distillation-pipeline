import torch
import torch.nn.functional as F
from typing import cast
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset, get_dataset_split_names
from transformers import DataCollatorWithPadding
from .models import teacher_tokenizer, teacher, student_tokenizer, student, device
from .vocab_mapping import build_vocab_mapping, map_teacher_probs_to_student
from .vocab_mapping_store import load_vocab_mapping, save_vocab_mapping
from tqdm import tqdm

datasplit = get_dataset_split_names("bigcode/the-stack", data_dir="data/python")
print(datasplit)
print("Loading dataset...")
dataset = load_dataset(
    "bigcode/the-stack",
    data_dir="data/python",
    data_files="train-00000-of-00206.parquet",
    split="train[:1%]",
)  # small sample
print("Dataset loaded.")


def tokenize_fn(example):
    return teacher_tokenizer(example["content"], truncation=True, max_length=256)


print("Tokenizing dataset...")
remove_columns = dataset.column_names
if isinstance(remove_columns, dict):
    # Use the first split's columns (usually 'train')
    remove_columns = next(iter(remove_columns.values()))
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)
print("Tokenization complete.")
data_collator = DataCollatorWithPadding(tokenizer=teacher_tokenizer)
# Hint to Pylance that HuggingFace Dataset conforms to the torch Dataset protocol
dataloader = DataLoader(
    cast(TorchDataset, dataset), batch_size=2, shuffle=True, collate_fn=data_collator
)
print("Dataloader created.")

print("Initializing optimizer...")
optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
print("Optimizer initialized.")

T = 2.0  # temperature
alpha = 0.9  # weight for soft loss
beta = 0.1  # weight for hard loss


print("Checking for cached vocab mapping...")
vocab_mapping = load_vocab_mapping(student_tokenizer, teacher_tokenizer)
if vocab_mapping is None:
    print("No cached mapping found. Building vocab mapping...")
    vocab_mapping = build_vocab_mapping(student_tokenizer, teacher_tokenizer)
    save_vocab_mapping(vocab_mapping, student_tokenizer, teacher_tokenizer)
else:
    print("Using cached vocab mapping. Checking for missing tokens...")
    updated_mapping = build_vocab_mapping(
        student_tokenizer, teacher_tokenizer, partial_mapping=vocab_mapping
    )
    if len(updated_mapping) > len(vocab_mapping):
        print(
            f"Added {len(updated_mapping) - len(vocab_mapping)} new mappings. Saving updated mapping."
        )
        save_vocab_mapping(updated_mapping, student_tokenizer, teacher_tokenizer)
    vocab_mapping = updated_mapping
student_vocab_size = student_tokenizer.vocab_size
print("Vocab mapping ready.")
print(f"Vocab mapping size: {len(vocab_mapping)}")

print("Starting training loop...")
for epoch in range(1):
    print(f"Epoch {epoch+1}...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training batches")):
        input_ids = batch["input_ids"].to(device)

        with torch.no_grad():
            teacher_logits = teacher(input_ids).logits  # [B, L, teacher_vocab]
        teacher_probs = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
        mapped_teacher_probs = map_teacher_probs_to_student(
            teacher_probs, vocab_mapping, student_vocab_size
        )
        # Diagnostic prints
        print(
            f"Batch {batch_idx}: mapped_teacher_probs shape: {mapped_teacher_probs.shape}"
        )
        print(
            f"Batch {batch_idx}: mapped_teacher_probs min: {mapped_teacher_probs.min().item()}, max: {mapped_teacher_probs.max().item()}"
        )
        print(
            f"Batch {batch_idx}: mapped_teacher_probs contains NaN: {torch.isnan(mapped_teacher_probs).any().item()}"
        )
        print(
            f"Batch {batch_idx}: mapped_teacher_probs contains zeros: {(mapped_teacher_probs == 0).sum().item()} out of {mapped_teacher_probs.numel()}"
        )
        teacher_log_probs = torch.log(mapped_teacher_probs + 1e-8)  # avoid log(0)

        student_logits = student(input_ids).logits  # [B, L, student_vocab]
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        loss_soft = F.kl_div(
            student_log_probs, teacher_log_probs, reduction="batchmean"
        ) * (T * T)

        target_ids = input_ids[:, 1:].contiguous()
        student_preds = student_logits[:, :-1, :].contiguous()
        loss_hard = F.cross_entropy(
            student_preds.view(-1, student_preds.size(-1)),
            target_ids.view(-1),
            ignore_index=teacher_tokenizer.pad_token_id,
        )

        loss = alpha * loss_soft + beta * loss_hard

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
