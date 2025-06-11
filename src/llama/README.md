# LLM Hate Speech Detection Model

Uses the LLM for detecting hate and offensive speech in Lithuanian texts. The model classifies texts into three categories: `hate`, `offensive`, `neutral`.

## Requirements

- Python 3.x
- `llama_cpp` library
- `pandas` library

## Installation

1. Install the required dependencies:
    ```bash
    pip install llama-cpp-python pandas
    ```

2. Use a data CSV file with the following structure:
    ```csv
    data,labels
    "Example text", "label"
    ```

3. Download the llama gguf model
    https://www.jottacloud.com/s/1580a97507a3efe4c14b0d3cede3caf3584/list/

## Usage

Run the main script with these arguments:

```bash
python main.py --model_path ./Lt-Llama-13b-q8.gguf --csv_path ./data.csv --output_path ./results.txt
```

## Fine-tuning

To fine-tune the model, first you need to prepare your dataset.
Dataset must contain a full prompt and comment.

To generate the dataset, you can use the `llama_dataset.py` script:

```bash
python llama_dataset.py
```

It will create a train and validation dataset files:

* `train.jsonl` - contains training data
* `validation.jsonl` - contains validation data

## Model Training

We used two models:

* `google/gemma-2b-it` - that requires 17GB of VRAM
* `neurotechnology/Lt-Llama-2-7b-instruct-hf` - that requires 19GB of VRAM

To fine-tune the model, you can use the `llama_finetune.py` script:

```bash
python llama_finetune.py --model google/gemma-2b-it --output_dir ./gemma-2b-finetuned
python llama_finetune.py --model neurotechnology/Lt-Llama-2-7b-instruct-hf --output_dir ./llama-2-7b-finetuned
```

