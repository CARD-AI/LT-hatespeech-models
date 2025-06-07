from transformers import ElectraTokenizer, ElectraForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
from transformers import Trainer
from dataset_class import Dataset
from datasets import load_dataset
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd


# Define pretrained tokenizer and model

MODEL_PATH = "models/electra_finetuned/electra_aug_less_7066/checkpoint-1000"
tokenizer = ElectraTokenizer.from_pretrained("google/electra-large-discriminator")
model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)

# Create train and evaluate datasets
data_files = {"test": "test.csv"}
dataset = load_dataset("CARD-AI/Lithuanian-hatespeech", data_files=data_files)
data = dataset['test'].to_pandas()

labels_map = {
    "hate": 0,
    "non-hate": 1,
    "offensive": 2
}


data['labels'] = data['labels'].map(labels_map)
data = data.dropna(subset=['data'])

comments = list(data.data)
labels = list(data.labels)

X_test_tokenized = tokenizer(comments, padding=True, truncation=True, max_length=512)
test_dataset = Dataset(X_test_tokenized)


# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)


print(confusion_matrix(labels, y_pred))
print(classification_report(labels, y_pred, zero_division=0))

data['predicted'] = y_pred
#data.to_csv('../../dataset/electra_predicted.csv', index=False, encoding='utf8')

print(f"f1 score: {f1_score(labels, y_pred, average='weighted')}")
