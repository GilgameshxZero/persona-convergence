import datasets
from transformers import PreTrainedTokenizer


def load_ds(
    split: str,
    tokenizer: PreTrainedTokenizer,
    seq_len: int,
    batch_size: int = 1000,
    num_proc: int = 16,
) -> datasets.Dataset:
    """
    Examples of valid splits:
        "train", "validation", "train[:100]", "train[10%:20%]"
    """
    ds = datasets.load_dataset(
        "roneneldan/TinyStories",
        split=split,
    )

    ds = (
        ds.map(
            lambda batch_texts: tokenizer(
                batch_texts,
                max_length=seq_len,
                truncation=True,
            ),
            input_columns="text",
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
        )
        .remove_columns(["attention_mask"])
        .rename_column("input_ids", "orig_input_ids")
    )

    ds = ds.filter(
        lambda batch_input_ids: [len(x) == seq_len for x in batch_input_ids],
        input_columns="orig_input_ids",
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    ds = (
        ds.map(
            lambda batch_texts: tokenizer(
                [" " + x for x in batch_texts],
                max_length=1,
                truncation=True,
            ),
            input_columns="text",
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
        )
        .remove_columns(["attention_mask"])
        .rename_column("input_ids", "first_token_w_space")
    )

    return ds
