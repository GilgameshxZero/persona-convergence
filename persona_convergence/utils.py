import textwrap

import torch
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedTokenizer,
)


def print_with_wrap(text, width=80):
    print(textwrap.fill(text, width=width))


def gen_text(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    prompt_tokenized = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)
    prompt_len = prompt_tokenized.input_ids.shape[-1]

    with torch.no_grad():
        gen_out = model.generate(
            prompt_tokenized.input_ids,
            attention_mask=prompt_tokenized.attention_mask,
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            ),
        )[0]

    return tokenizer.decode(gen_out[prompt_len:])
