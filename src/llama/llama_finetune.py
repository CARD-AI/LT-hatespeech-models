from typing import Dict

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    default_data_collator,
    BitsAndBytesConfig,
)


def verify_dataset_format(dataset_dict):
    """Verify that the dataset has the expected format and features"""
    required_features = ["instruction", "input", "output"]

    if not all(
            feature in dataset_dict["train"].features for feature in required_features
    ):
        missing = [
            f for f in required_features if f not in dataset_dict["train"].features
        ]
        raise ValueError(f"Dataset missing required features: {missing}")

    print("\nRaw sample data point:")
    sample = dataset_dict["train"][0]
    for feature in required_features:
        print(f"{feature}: {sample[feature][:100]}...")

    print("\nFormatted sample (Gemma template):")
    formatted_sample = format_prompt(sample)
    print(formatted_sample[:500])


def prepare_model_and_tokenizer(
        model_name: str = "google/gemma-2b-it",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
):
    """
    Prepare the Gemma model and tokenizer with LoRA configuration
    """
    try:
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model with Gemma-specific configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=False,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            add_eos_token=True,
            use_fast=True,
        )

        # Configure LoRA for Gemma architecture
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )

        # Get PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    except Exception as e:
        print(f"Error in model preparation: {str(e)}")
        raise


def format_prompt(example: Dict) -> str:
    """Format a single example using Gemma's specific template with multi-turn conversation"""
    comment = example["instruction"].strip()  # This contains the text to evaluate
    system_prompt = example["input"].strip()  # This contains the expert role definition
    response = example["output"].strip()

    return f"""<bos>
<start_of_turn>user
{system_prompt}
<end_of_turn>
<start_of_turn>user
Įvertink šį tekstą dėl neapykantos kalbos: {comment}
<end_of_turn>
<start_of_turn>model
{response}
<end_of_turn>"""


def prepare_dataset(
        tokenizer,
        dataset_path: str,
        max_length: int = 512,
        batch_size: int = 8,
):
    """
    Prepare dataset with Gemma-specific tokenization
    """
    try:
        dataset = load_dataset("json", data_files=dataset_path)
        print(f"Dataset loaded: {dataset}")
        verify_dataset_format(dataset)

        def tokenize_function(example: Dict) -> Dict:
            try:
                # Format using Gemma template
                prompt = format_prompt(example)

                # Tokenize with padding and truncation
                tokenized = tokenizer(
                    prompt,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                    return_token_type_ids=False,
                )

                tokenized["input_ids"] = tokenized["input_ids"].squeeze()
                tokenized["attention_mask"] = tokenized["attention_mask"].squeeze()

                # Create labels
                tokenized["labels"] = tokenized["input_ids"].clone()
                tokenized["labels"][
                    tokenized["input_ids"] == tokenizer.pad_token_id
                    ] = -100

                return tokenized

            except Exception as e:
                print(f"Error tokenizing example: {str(e)}")
                return None

        print("Starting tokenization...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset",
            load_from_cache_file=False,
            num_proc=1,
        )

        tokenized_dataset = tokenized_dataset.filter(lambda x: x is not None)
        return tokenized_dataset

    except Exception as e:
        print(f"Error in dataset preparation: {str(e)}")
        raise


def train(
        model: str = "google/gemma-2b-it",
        train_dataset_path: str = "train.jsonl",
        val_dataset_path: str = "validation.jsonl",
        output_dir: str = "./gemma-2b-finetuned",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        max_grad_norm: float = 0.3,
        warmup_ratio: float = 0.03,
):
    """
    Main training function adapted for Gemma
    """
    try:
        model_name = model.split("/")[-1] if "/" in model else model

        wandb.init(project=f"{model_name}-finetuning", name=f"{model_name}_training_run")

        model, tokenizer = prepare_model_and_tokenizer(model_name=model)

        train_dataset = prepare_dataset(tokenizer, train_dataset_path)
        val_dataset = prepare_dataset(tokenizer, val_dataset_path)

        num_update_steps_per_epoch = len(train_dataset["train"]) // (
                per_device_train_batch_size * gradient_accumulation_steps
        )
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        warmup_steps = int(max_train_steps * warmup_ratio)

        total_batch_size = per_device_train_batch_size * gradient_accumulation_steps

        print(f"\nTraining configuration:")
        print(f"Total batch size: {total_batch_size}")
        print(f"Updates per epoch: {num_update_steps_per_epoch}")
        print(f"Total training steps: {max_train_steps}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=True,
            save_total_limit=3,
            logging_steps=10,
            save_strategy="steps",
            save_steps=200,
            warmup_steps=warmup_steps,
            optim="paged_adamw_8bit",
            logging_dir="./logs",
            remove_unused_columns=False,
            report_to="wandb",
            max_grad_norm=max_grad_norm,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            eval_steps=200,
            eval_strategy="steps"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset["train"],
            eval_dataset=val_dataset["train"],
            data_collator=default_data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=5, early_stopping_threshold=0.01
                ),
            ],
        )

        try:
            trainer.train()
        except Exception as e:
            print(f"Training error: {str(e)}")
            trainer.save_model(f"{output_dir}/checkpoint-error")
            raise

        trainer.save_model(f"{output_dir}/final")
        wandb.finish()

    except Exception as e:
        print(f"Critical error in training process: {str(e)}")
        wandb.finish()
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Gemma model")
    parser.add_argument("--model", type=str, default="google/gemma-2b-it", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./gemma-2b-finetuned", help="Output directory for the model")
    args = parser.parse_args()

    train(model=args.model, output_dir=args.output_dir)

    # Example usage:
    # python llama_finetune.py --model google/gemma-2b-it --output_dir ./gemma-2b-finetuned