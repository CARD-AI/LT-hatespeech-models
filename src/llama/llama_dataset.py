import json

from datasets import load_dataset

dataset = load_dataset(
    "CARD-AI/Lithuanian-hatespeech",
    data_files={
        "train": "train.csv",
        "test": "mini/sampled_test_data_50_per_label.csv",
        "validation": "validation.csv",
    },
)


def create_instruction_dataset(output_path, split="train"):
    # Read the CSV file
    # df = pd.read_csv(csv_path)
    df = dataset[split].to_pandas()

    # Define the system context that explains the task
    system_context = """Tu esi neapykantos kalbos aptikimo ekspertas. Tavo užduotis - įvertinti tekstą dėl neapykantos ar įžeidžios kalbos.

Galimi atsakymai:
- neapykanta: tekstas, kuris skatina neapykantą, smurtą ar diskriminaciją
- įžeidus: tekstas, kuris yra įžeidžiantis, bet neskatina neapykantos
- neutralus: tekstas be neapykantos ar įžeidžių elementų"""

    # Create instruction-response pairs
    instruction_data = []

    for _, row in df.iterrows():
        # Map binary hate/non-hate to three-class classification
        label_mapping = {
            "hate": "neapykanta",
            "nothate": "neutralus",
            "non-hate": "neutralus",
            "offensive": "įžeidus",
        }

        example = {
            "instruction": f"Įvertink šį tekstą dėl neapykantos kalbos: {row['data']}",
            "input": system_context,
            "output": label_mapping.get(row["labels"], row["labels"]),
        }
        instruction_data.append(example)

    # Save as JSONL (one JSON object per line)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in instruction_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# Sample of how to load and use the dataset for training
def load_dataset_for_training(jsonl_path):
    from datasets import load_dataset

    # Load the JSONL file as a Hugging Face dataset
    dataset = load_dataset("json", data_files=jsonl_path)

    # Prepare the prompts in the format Llama 2 expects
    def format_prompt(example):
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

    # Format all examples
    formatted_dataset = dataset.map(
        lambda x: {"text": format_prompt(x)},
        remove_columns=dataset["train"].column_names,
    )

    return formatted_dataset


# Example usage
if __name__ == "__main__":
    create_instruction_dataset("./train.jsonl", split="train")
    create_instruction_dataset("./validation.jsonl", split="validation")
