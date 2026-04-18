from datasets import DatasetDict, load_dataset

def load_dataset_numina(dataset_name_or_path: str) -> DatasetDict:
    """
    Load the NuminaMath dataset and add a 'messages' field alongside the original
    'problem'/'solution' fields.

    - use_chat_template=True  → default_sft_map_fn uses 'messages' via apply_chat_template
    - use_chat_template=False → default_sft_map_fn uses 'problem'/'solution' directly (raw text)
    """
    dataset = load_dataset(dataset_name_or_path)

    def map_fn(example):
        return {
            "messages": [
                {"role": "user", "content": example["problem"].strip()},
                {"role": "assistant", "content": example["solution"].strip()},
            ],
            "problem": example["problem"],
            "solution": example["solution"],
        }

    # Keep only the fields we need; drop any other original columns
    cols_to_remove = [
        c for c in dataset["train"].column_names
        if c not in ("problem", "solution", "messages")
    ]
    return dataset.map(map_fn, remove_columns=cols_to_remove)
