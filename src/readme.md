# Hatespeech detection models

We trained 7 groups of models for hatespeech detection: Electra transformer, Multilingual-BERT, LitLat-BERT, RWKV, fine-tuned Lithuanian Llama 2, ChatGPT and several other deep learning models. Each of the models was either trained or finetuned to classify lithuanian media comments into three classes: hate, non-hate and offensive. 

## Electra transformer

The [Electra model](https://huggingface.co/docs/transformers/model_doc/electra) was chosen since in theory it should work better will smaller amounts of data.


- The `electra/electra_finetune.py` script was used to train the embeddings model from scratch (can also be used to additionaly train already pretrained models for english language). After that, the discriminator part of the transformer is saved into `electra/outputs_base/discriminator_model/` folder and can be later fine-tuned to do the maistream tasks.
- The `electra/electra_finetune.py` script was used to fine-tune the model for a classification task. The resulting fine-tuned model is stored in the `electra/electra_finetuned/` folder.
- The `electra/electra_predict.py` scripts was used to test the fine-tuned model. This script calculates cofusion matrix and the precision, accuracy, recall and F-score of the test-set.

In order to train the model from scratch, use:

```python
model = LanguageModelingModel(
    "electra",
    None,
    args=train_args,
    train_files=train_file,
)
```

In order to additionaly train already pretrained embeddings model, use (for ELECTRA english model):

```python
model = LanguageModelingModel(
    "electra",
    "electra",
    generator_name="google/electra-small-generator",
    discriminator_name="google/electra-large-discriminator",
    args=train_args,
    train_files=train_file,
)
```

### Running the scripts

In order to run these scripts, go to the root directory of this project and simply run the selected script using python, for example:
```
python src/electra/electra_finetune.py
```


## Multilingual-BERT

The multilingual-BERT model was chosen since it can already be fine-tuned without any additional training. In this case we used the [multilingual-cased model](https://huggingface.co/bert-base-multilingual-cased).

- The `multilingual-bert/multilingual_finetune.py` script was used to additionaly train the bert-base-multilingual-cased embeddings model. After that, the embeddings model is saved into `multilingual-bert/outputs_bert/` folder and can be later fine-tuned to do the maistream tasks.
- The `multilingual-bert/multilingual_finetune.py` script was used to fine-tune the model for a classification task. The resulting fine-tuned model is stored in the `multilingual-bert/multilingual_finetuned/` folder.
- The `multilingual-bert/multilingual_predict.py` scripts was used to test the fine-tuned model. This script calculates cofusion matrix and the precision, accuracy, recall and F-score of the test-set.

### Running the scripts

In order to run these scripts, go to the root directory of this project and simply run the selected script using python, for example:
```
python src/multilingual-bert/multilingual_finetune.py
```


## LitLat-BERT

Similarly, the [LitLat-BERT](https://huggingface.co/EMBEDDIA/litlat-bert) model was chosen since in can already be fine-tuned withoud any additional training. This model was trained using English, Latvian and Lithuanian languages, therefore in most of the tasks it outperformes the multilingual BERT for these languages.

- The `litlat-bert/litlat_finetune.py` script was used to additionaly train the bert-base-litlat-cased embeddings model. After that, the embeddings model is saved into `litlat-bert/outputs_litlatbert/` folder and can be later fine-tuned to do the maistream tasks.
- The `litlat-bert/litlat_finetune.py` script was used to fine-tune the model for a classification task. The resulting fine-tuned model is stored in the `litlat-bert/litlat_finetuned/` folder.
- The `litlat-bert/litlat_predict.py` scripts was used to test the fine-tuned model. This script calculates cofusion matrix and the precision, accuracy, recall and F-score of the test-set.

### Running the scripts

In order to run these scripts, go to the root directory of this project and simply run the selected script using python, for example:
```
python src/lilat-bert/litlat_finetune.py
```


## RWKV

RWKV model was chosen because it combines the strengths of recurrent neural networks (RNNs) and transformers. In particular the RWKV 4 was used in this research. The model has various different multilingual architecture variations (`169M`, `430M`, `1B5`, `3B`, `7B`, `14B` parameters) that can be fine-tuned for sequence classification.

### Running the scripts

1. Rename `RWKV/models_` to `RWKV/models`

2. Download the desired model from [HuggingFace](https://huggingface.co/BlinkDL/rwkv-4-world/tree/main) and run the converter script

```
python train/convert.py
```

3. Either manually prepare the dataset or use predefined script for dataset preparation using command below

```
python train/data_process.py --corpus_file <path_to_csv_file> --tokenizer_file <path_to_vocab_file> --output_dir <path_to_data_save_dir> --train_ratio <float_number of train_size>
```

4. Run training script to fine-tune RWKV model for Sequence Classification

```
python train/train_from_lightning_classifier.py --train_file <path_to_train_csv> --test_file <path_to_test_csv> --model_path <path_to_converted_model> --tokenizer_file <path_to_vocab_file> --batch_size <int_batch_size> --num_classes <int_output_classes> --max_epochs <int_epochs> --output_dir <train_model_output_path>
```

For more detailed explanation check out `RWKV` folder.

## Large Language Models

We used large language models (LLMs) to classify Lithuanian media comments into three categories: `hate`, `offensive`, and `neutral`. These models include fine-tuned versions of `Gemma-2B` and `Lt-Llama-2-7B`.

### Inference

1. Install dependencies:
2. Navigate to the `src/llama` directory
3. Prepare a dataset CSV file with the following structure:

   ```csv
   data,labels
   "Example text","hate"
   ```

4. Download a GGUF model file, e.g.:
   [Lt-Llama-13b-q8.gguf](https://www.jottacloud.com/s/1580a97507a3efe4c14b0d3cede3caf3584/list/)

5. Run the inference script:
To classify new examples using a local LLM, run:

```bash
python llama_test.py --model_path ./Lt-Llama-13b-q8.gguf --csv_path ./data.csv --output_path ./results.txt
```

### Fine-Tuning the Model

#### Dataset Preparation for Fine-Tuning

Use `llama_dataset.py` to convert data into instruction-style prompt format:

```bash
python llama_dataset.py
```

This will generate:

* `train.jsonl` – training data
* `validation.jsonl` – validation data

#### Running Fine-Tuning

Models to choose from for fine-tuning:

* `google/gemma-2b-it` (requires ~17GB VRAM)
* `neurotechnology/Lt-Llama-2-7b-instruct-hf` (requires ~19GB VRAM)

Run the fine-tuning script:

```bash
python llama_finetune.py --model google/gemma-2b-it --output_dir ./gemma-2b-finetuned
python llama_finetune.py --model neurotechnology/Lt-Llama-2-7b-instruct-hf --output_dir ./llama-2-7b-finetuned
```

## ChatGPT

We used OpenAI's ChatGPT models to classify Lithuanian text into three categories: `neapykanta` (hate), `įžeidus` (offensive), and `neutralus` (neutral). The classification is guided using a carefully crafted system prompt that defines each category with examples in Lithuanian.

### Features

* Supports models like `gpt-4o`, `gpt-4`, and `gpt-3.5-turbo`.
* Uses OpenAI's API via the `openai` Python library.

### System Prompt (Lithuanian)

The system prompt instructs the model to return only one category label

### Example Usage

Before running, export your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Then run the classification script:

```bash
python chatgpt_test.py --csv_path ./data.csv --model gpt-4o
```

* `--csv_path`: path to the CSV file with `data` and `labels` columns.
* `--model`: OpenAI model name (e.g., `gpt-4o`, `gpt-4`, `gpt-3.5-turbo`).

### Output

The script generates a results file (`results-{model}.csv`)

## Other deep learning models

In addition to transformer-based and LLM models, we developed and tested several classic deep learning architectures for Lithuanian hate speech classification. These models use FastText word embeddings and are trained on the [CARD-AI/Lithuanian-hatespeech](https://huggingface.co/datasets/CARD-AI/Lithuanian-hatespeech) dataset. The goal is to classify text into `hate`, `offensive`, or `neutral`.

### Models

#### 1. CNN (Convolutional Neural Network)

* Incorporates 1D convolutional layers followed by global max pooling.
* Embedding layer is initialized with FastText Lithuanian vectors (`facebook/fasttext-lt-vectors`).
* Includes dropout for regularization.
* Effective in capturing local n-gram features.

#### 2. LSTM (Long Short-Term Memory)

* Sequential model using stacked LSTM layers to capture long-range dependencies in text.
* Pre-trained FastText embeddings are used as input.
* Dropout is applied for regularization.

#### 3. BiLSTM (Bidirectional LSTM)

* Extension of the LSTM model that processes sequences in both forward and backward directions.
* Provides better context awareness from both ends of the text.

#### 4. CNN Advanced

* Enhanced CNN model with:

  * L2 regularization
  * Learning rate scheduler
  * 5-fold stratified cross-validation
* Designed for robustness with smaller or imbalanced datasets.

### Running the Models

All model scripts are located in the `src/deep-learning/` directory. To run a model:

```bash
cd src/deep-learning/
python cnn.py        # For basic CNN
python lstm.py       # For LSTM
python bilstm.py     # For BiLSTM
python cnn_adv.py    # For advanced CNN
```