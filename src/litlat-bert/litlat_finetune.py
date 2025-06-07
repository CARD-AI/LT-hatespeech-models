from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from dataset_class import Dataset
from datasets import load_dataset
import numpy as np
from random import randrange
import wandb

wandb.init(project="Hatespeech_publication")
SEED = randrange(10000)
labels_map = {
    "hate": 0,
    "non-hate": 1,
    "offensive": 2
}

# Define pretrained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/litlat-bert")
model = AutoModelForSequenceClassification.from_pretrained("EMBEDDIA/litlat-bert", num_labels=3)

# Create train and evaluate datasets
data_files = {"train": "balanced-less/balanced_less_original_train.csv"}
dataset = load_dataset("CARD-AI/Lithuanian-hatespeech-augmented", data_files=data_files)
train_data = dataset['train'].to_pandas()

data_files = {"test": "test.csv"}
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
    output_dir=f"models/litlat_finetuned/litlat_aug_less_mini_{SEED}",
    eval_strategy="steps",
    eval_steps=500,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=7,
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
