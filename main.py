def main():
    from app.models import teacher_tokenizer, student_tokenizer, teacher, student
    from app.vocab_mapping import build_vocab_mapping, map_teacher_probs_to_student
    import torch

    # Build vocab mapping from student to teacher
    print("Building vocab mapping...")
    vocab_mapping = build_vocab_mapping(student_tokenizer, teacher_tokenizer)
    print(f"Mapped {len(vocab_mapping)} student tokens to teacher tokens.")

    # Example: get teacher logits and map to student vocab
    # Dummy input for demonstration
    input_text = "def hello_world():\n    print('Hello, world!')"
    student_inputs = student_tokenizer(input_text, return_tensors="pt")
    teacher_inputs = teacher_tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        teacher_logits = teacher(**teacher_inputs).logits  # [1, seq, teacher_vocab]
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)

    # Map teacher probabilities to student vocab
    student_vocab_size = student_tokenizer.vocab_size
    mapped_probs = map_teacher_probs_to_student(
        teacher_probs, vocab_mapping, student_vocab_size
    )
    print(f"Mapped teacher probs shape: {mapped_probs.shape}")


if __name__ == "__main__":
    main()
