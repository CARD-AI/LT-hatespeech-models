import pandas as pd
import numpy as np
import fasttext
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping

# Preprocessing Data
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

# Downloading FastText model
print("Loading FastText model...")
model_path = hf_hub_download(repo_id="facebook/fasttext-lt-vectors", filename="model.bin", )
fasttext_model = fasttext.load_model(model_path)

# Tokenizing data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.concat([train_df['data'], val_df['data'], test_df['data']]))  # Combining for vocab

# Preparing data
max_sequence_len = max(max(len(x.split()) for x in train_df['data']),
                       max(len(x.split()) for x in val_df['data']),
                       max(len(x.split()) for x in test_df['data']))

X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['data']), maxlen=max_sequence_len)
X_val = pad_sequences(tokenizer.texts_to_sequences(val_df['data']), maxlen=max_sequence_len)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_df['data']), maxlen=max_sequence_len)

# Encoding labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['labels'])
y_val = label_encoder.transform(val_df['labels'])
y_test = label_encoder.transform(test_df['labels'])

# Prepare the embedding matrix
word_index = tokenizer.word_index
embedding_dim = fasttext_model.get_dimension()
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = fasttext_model.get_word_vector(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# Model building function with L2 regularization
def create_model(input_length, num_classes, embedding_matrix, word_index, embedding_dim):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=input_length,
                        trainable=False))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch > 0 and epoch % 5 == 0:
        return lr * 0.5
    else:
        return lr


# Cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    print(f"Training on fold {fold + 1}/{n_splits}...")
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model = create_model(max_sequence_len, len(np.unique(y_train)), embedding_matrix, word_index, embedding_dim)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
              epochs=10, batch_size=32, callbacks=[lr_scheduler, early_stopping])

# Final evaluation on the test set
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