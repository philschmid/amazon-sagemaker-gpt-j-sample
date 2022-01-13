import os
import torch
from transformers import AutoTokenizer, pipeline

GPT_WEIGHTS_NAME = "gptj.pt"


def model_fn(model_dir):
    model = torch.load(os.path.join(model_dir, GPT_WEIGHTS_NAME))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if torch.cuda.is_available():
        device = 0
    else:
        device = -1

    generation = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device=device
    )

    return generation
