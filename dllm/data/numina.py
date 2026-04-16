from datasets import DatasetDict, load_dataset

def load_dataset_numina(dataset_name_or_path: str) -> DatasetDict:
    """
    Load the NuminaMath dataset and map 'problem'/'solution' to 'messages'.
    This matches the format used by Tulu and other SFT datasets in this repo.
    """
    dataset = load_dataset(dataset_name_or_path)
    
    def map_fn(example):
        return {
            "messages": [
                {"role": "user", "content": example["problem"].strip()},
                {"role": "assistant", "content": example["solution"].strip()},
            ]
        }
    
    # Remove original columns to avoid confusion
    return dataset.map(map_fn, remove_columns=dataset["train"].column_names)
