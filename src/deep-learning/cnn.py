import fasttext
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from keras.src.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Loading data...")
dataset = load_dataset("CARD-AI/Lithuanian-hatespeech")
train_df = dataset['train'].to_pandas()
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
test_df = dataset['test'].to_pandas()

print("Preprocessing data...")
# Fill missing values with empty strings
train_df['data'] = train_df['data'].fillna('').astype(str)
val_df['data'] = val_df['data'].fillna('').astype(str)
test_df['data'] = test_df['data'].fillna('').astype(str)

print("Loading FastText model...")
# Load your FastText model
model_path = hf_hub_download(repo_id="facebook/fasttext-lt-vectors", filename="model.bin")
fasttext_model = fasttext.load_model(model_path)

print("Tokenizing data...")
# Tokenize the text for both training and testing data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['data'])

print("Preparing data...")
train_sequences = tokenizer.texts_to_sequences(train_df['data'])
val_sequences = tokenizer.texts_to_sequences(val_df['data'])
test_sequences = tokenizer.texts_to_sequences(test_df['data'])

# Find max sequence length across both train and test to ensure consistent input size
max_sequence_len = max(max(len(x) for x in train_sequences), max(len(x) for x in test_sequences))

# Pad sequences to have the same length
X_train = pad_sequences(train_sequences, maxlen=max_sequence_len)
X_val = pad_sequences(val_sequences, maxlen=max_sequence_len)
X_test = pad_sequences(test_sequences, maxlen=max_sequence_len)

print("Building model...")
# Encode labels for the training data
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['labels'])
num_classes = np.max(y_train) + 1

# Encode labels for the validation data
y_val = label_encoder.transform(val_df['labels'])

# Encode labels for the testing data using the same encoder to ensure consistency
y_test = label_encoder.transform(test_df['labels'])

print("Training model...")
# Prepare the embedding matrix
word_index = tokenizer.word_index
embedding_dim = fasttext_model.get_dimension()
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = fasttext_model.get_word_vector(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define the CNN model
print("Building model...")
model = Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_len,
                    trainable=False))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(f'Model summary: {model.summary()}')

# stop training when validation accuracy stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[early_stopping])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, callbacks=[early_stopping])

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# calculate F1 weighted score
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
precision = precision_score(y_test, y_pred_classes, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_classes, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print(f"Precision: {precision}, Recall: {recall}")
print(f"f1 score: {f1}")