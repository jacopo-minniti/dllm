from datasets import load_dataset, DatasetDict

def load_dataset_gsm8k(dataset_name_or_path: str) -> DatasetDict:
    """
    Load GSM8K (main subset) and expose raw prompt/response fields.
    """
    dataset = load_dataset(dataset_name_or_path, "main")
    
    def map_fn(example):
        return {
            "prompt": example["question"].strip(),
            "response": f"Answer: {example['answer'].strip()}"
        }
        
    # We remove the original question/answer columns to keep it clean
    return dataset.map(map_fn, remove_columns=dataset["train"].column_names)
