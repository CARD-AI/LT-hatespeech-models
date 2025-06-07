from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
from transformers import Trainer
from dataset_class import Dataset
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score
import pandas as pd


# Define pretrained tokenizer and model

model_name = "models/multilingual_finetuned/multilingual_aug_more_6424/checkpoint-2000"
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)


# Loading dataset

# Create train and evaluate datasets
data_files = {"test": "mini/sampled_test_data_50_per_label.csv"}
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
#data.to_csv('../../dataset/multilingual_predicted.csv', index=False, encoding='utf8')

print(f"f1 score: {f1_score(labels, y_pred, average='weighted')}")
