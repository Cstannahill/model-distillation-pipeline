from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
import torch
from typing import Any, cast

# Use a string for libraries that expect a device name and a torch.device for torch APIs
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

# Teacher (AWQ quantized)
teacher_model_path = "/home/christian/dev/llm_models/Deepseek-Coder-6.7B-AWQ"
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
# Ensure teacher tokenizer has a pad token (fallback to EOS if missing)
if getattr(teacher_tokenizer, "pad_token", None) is None and getattr(teacher_tokenizer, "eos_token", None) is not None:
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
teacher = AutoAWQForCausalLM.from_quantized(
    teacher_model_path,
    device=device_str,  # AWQ expects a string device like 'cpu' or 'cuda'
    fuse_layers=True
).eval()

# Student (GPT-2-medium)
student_model_name = "/home/christian/dev/llm_models/GPT2-md"
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
# GPT-2 doesn't define a pad token by default; set it to EOS to support padding/collation
if getattr(student_tokenizer, "pad_token", None) is None and getattr(student_tokenizer, "eos_token", None) is not None:
    student_tokenizer.pad_token = student_tokenizer.eos_token
student: Any = AutoModelForCausalLM.from_pretrained(student_model_name)
student = student.to(device)
# Propagate pad_token_id to model config for loss/attention masking compatibility
if hasattr(student, "config") and getattr(student_tokenizer, "pad_token_id", None) is not None:
    # type: ignore[attr-defined]
    setattr(student.config, "pad_token_id", int(student_tokenizer.pad_token_id))
student.train()
