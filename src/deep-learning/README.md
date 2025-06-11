# Lithuanian Hate Speech Classification Models

This repository contains multiple deep learning models for classifying Lithuanian text data into categories such as hate speech. All models use the [CARD-AI/Lithuanian-hatespeech](https://huggingface.co/datasets/CARD-AI/Lithuanian-hatespeech) dataset and FastText embeddings from `facebook/fasttext-lt-vectors`.

## Models

### 1. CNN (Convolutional Neural Network)
- Uses 1D convolution and global max pooling layers.
- Embedding layer is initialized with FastText Lithuanian word vectors.
- Includes dropout regularization.
- Good for capturing local n-gram patterns in text.

### 2. LSTM (Long Short-Term Memory)
- Sequential model with stacked LSTM layers.
- Captures long-range dependencies in text.
- Uses FastText embeddings and dropout.

### 3. BiLSTM (Bidirectional LSTM)
- Similar to LSTM but processes input in both forward and backward directions.
- Useful for understanding context from both directions.

### 4. CNN Advanced
- CNN-based architecture with enhancements:
  - L2 regularization
  - Learning rate scheduler
  - 5-fold stratified cross-validation
- More robust for smaller or imbalanced datasets.

## Running the Models

Each model is defined in a separate Python script:

- `cnn.py` - Basic CNN model
- `lstm.py` - LSTM-based model
- `bilstm.py` - Bidirectional LSTM model
- `cnn_adv.py` - Advanced CNN with regularization and cross-validation

### Example Usage
To run a model, navigate to the `src/deep-learning` directory and execute the desired script. For example, to run the CNN model:

### Dependencies
Make sure to install the required libraries before running the scripts. You can use the following command:

```bash
pip install -r requirements.txt
```

Then, navigate to the `src/deep-learning` directory and run the desired model script:

```bash
python cnn.py # or lstm.py, bilstm.py, cnn_adv.py
```


