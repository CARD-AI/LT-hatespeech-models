from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from dataset_class import Dataset
import numpy as np
import pandas as pd
from datasets import load_dataset
from random import randrange
import wandb

wandb.init(project="Hatespeech_publication_2024-11")
SEED = randrange(10000)
labels_map = {
    "hate": 0,
    "non-hate": 1,
    "offensive": 2
}


# Define pretrained tokenizer and model
model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)


data_files = {"train": "balanced-more/balanced_augmented_train.csv"}
dataset = load_dataset("CARD-AI/Lithuanian-hatespeech-augmented", data_files=data_files)
train_data = dataset['train'].to_pandas()

data_files = {"test": "validation.csv"}
dataset = load_dataset("CARD-AI/Lithuanian-hatespeech", data_files=data_files)
val_data = dataset['test'].to_pandas()

train_data['labels'] = train_data['labels'].map(labels_map)
train_data = train_data.dropna(subset=['data', 'labels'])
val_data['labels'] = val_data['labels'].map(labels_map)
val_data = val_data.dropna(subset=['data', 'labels'])

train_comments = list(train_data.data)
val_comments = list(val_data.data)

train_labels = list(train_data.labels)
train_labels = [int(x) for x in train_labels]

val_labels = list(val_data.labels)
val_labels = [int(x) for x in val_labels]

X_train_tokenized = tokenizer(train_comments, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(val_comments, padding=True, truncation=True, max_length=512)


# Create torch dataset
train_dataset = Dataset(X_train_tokenized, train_labels)
eval_dataset = Dataset(X_val_tokenized, val_labels)

# Define metrics for evaluation
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, zero_division=0, average='weighted')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# Define trainer and its arguments
training_args = TrainingArguments(
    output_dir=f"models/multilingual_finetuned/multilingual_aug_more_{SEED}",
    eval_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_train_epochs=10,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)


# Train and evaluate model
trainer.train()
trainer.evaluate()


# Save model
#model.save_pretrained('./multilingual_finetuned/')

