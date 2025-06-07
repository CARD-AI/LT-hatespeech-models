import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import wandb

wandb.init(project="hatespeech_rwkv")
model_name = "RWKV/rwkv-5-world-1b5"


# Prepare dataset

df = pd.read_csv(
    "/content/drive/MyDrive/pre-processed-dataset.csv",
    encoding="utf-8",
    names=["data", "labels"],
)
df.dropna(inplace=True)
df = df[~(df.labels == "labels")]


def generate_prompt(data):
    return f"""
    Instruction: Analyze the sentiment of the internet comment enclosed in square brackets, determine if it is non-hate, offensive, or hate, and return the answer as the corresponding sentiment label "non-hate" or "offensive" or "hate"

    Input: [{data['data']}]

    Response: {data['labels']}
    """.strip()


def generate_test_prompt(data):
    return f"""
    Instruction: Analyze the sentiment of the internet comment enclosed in square brackets, determine if it is non-hate, offensive, or hate, and return the answer as the corresponding sentiment label "non-hate" or "offensive" or "hate"

    Input: [{data['data']}]
    """.strip()


X_train = []
X_test = []

for sentiment in ["non-hate", "offensive", "hate"]:
    train, test = train_test_split(
        df[df.labels == sentiment], train_size=1041, test_size=1041, random_state=42
    )
    X_train.append(train)
    X_test.append(test)

X_train = pd.concat(X_train).sample(frac=1.0, random_state=21)
X_test = pd.concat(X_test)

eval_idx = [idx for idx in df.index if idx not in list(train.index) + list(test.index)]
X_eval = df[df.index.isin(eval_idx)]
X_eval = X_eval.groupby("labels", group_keys=False).apply(
    lambda x: x.sample(n=50, random_state=10, replace=True)
)
X_train = X_train.reset_index(drop=True)

X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), columns=["data"])
X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), columns=["data"])

y_true = X_test.labels
X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["data"])

# Generating Torch Dataset

train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)

# Loading Model

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="left",
    add_bos_token=True,
    add_eos_token=True,
)

tokenizer.pad_token = tokenizer.eos_token

# Prediction and Evaluation


def predict(X_test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["data"]
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=15,
            temperature=1.0,
        )
        result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)
        answer = result[0]["generated_text"].split("Response:")[-1].lower()
        if "non-hate" in answer:
            y_pred.append("non-hate")
        elif "hate" in answer:
            y_pred.append("hate")
        elif "offensive" in answer:
            y_pred.append("offensive")
        else:
            y_pred.append("none")
    return y_pred


def evaluate(y_true, y_pred):
    labels = ["non-hate", "hate", "offensive"]
    mapping = {"non-hate": 2, "hate": 1, "none": 2, "offensive": 0}

    def map_func(x):
        return mapping.get(x, 1)

    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f"Accuracy: {accuracy:.3f}")

    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f"Accuracy for label {label}: {accuracy:.3f}")

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print("\nClassification Report:")
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print("\nConfusion Matrix:")
    print(conf_matrix)


# Trainer

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["feed_forward.value"],
)

training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # 4
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="wandb",
    evaluation_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="data",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    max_seq_length=1024,
)

# Start Training

trainer.train()

# Evaluate Model
y_pred = predict(X_test, model, tokenizer)
evaluate(y_true, y_pred)

# Save Model
# trainer.model.save_pretrained("")
