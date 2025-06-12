# RWKV for Sequence Classification

## Model Description

RWKV is a type of NN architecture that combines the strengths of recurrent neural networks (RNNs) and transformers. Designed to be as efficient as RNNs while maintaining the performance and scalability of transformers, RWKV operates in a linear time and memory complexity per token, making it highly suitable for long-context language modeling. Unlike traditional transformers that rely on self-attention mechanisms, RWKV uses time-mixing and channel-mixing layers to capture sequential dependencies without incurring the quadratic cost.

## Requirements

    transformers == 4.30.2
    torch >= 1.13.1+cu117
    pandas == 2.0.2
    peft == 0.3.0


## Code Structure

```
├── models/                     # RWKV model + tokenizer
│   ├── __init__.py
│   ├── customized_tokenizers   # RWKV tokenizer script
│   └── model.py                # RWKV model implementation for sequence classification
├── train/                      # Scripts used for model training and data preparation
│   ├── convert.py                            # Convert model ckpt to training/inference ready model
│   ├── data_process.py                       # Dataset preparation script
│   └── train_from_lightning_classifier.py    # Model training script
└── README.md                   # Project documentation
```

## Usage intructions

1. Download the desired model from [HuggingFace](https://huggingface.co/BlinkDL/rwkv-4-world/tree/main)
2. Modify paths in `train/convert.py` file

| Line Number | Place to Edit | Explanation |
|-------------|---------------|-------------|
| 4 | ckpt_file | Locate the ckpt you have downloaded from HuggingFace in step 1|
| 153 | output_dir | Select the desirable location for model |
| 154 | size | Adjust accordingly to your model ckpt. Select one from `169M`, `430M`, `1B5`, `3B`, `7B`, `14B`. |

3. Convert the model using `convert.py` in `train` dir

```console
python train/convert.py
```

4. Open up config file (`config.json`) in exported model and change `vocab_size` to 65536 and `context_length` to 4096

5. Prepare your dataset

    5.1 If you are using `World` or `Raven` model types. You will need to download and use vocab instead of tokenizer. [Vocab file](https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_vocab_v20230424.txt). Save this file in your working dir.

    5.2 ***Optional***. Process your dataset `csv` file using `train/data_process.py`

    ```console
    python train/data_process.py --corpus_file <path_to_csv_file> --tokenizer_file <path_to_vocab_file> --output_dir <path_to_data_save_dir> --train_ratio <float_number of train_size>
    ```

    5.3 ***If 5.2 skipped*** For more regulated data splitting, if you have any specific needs in preparing the dataset manually. Structure of final csv file should contain columns `label`, `sentiment`, `length`.

6. Train RWKV for squence classification model.

```console
python train/train_from_lightning_classifier.py --train_file <path_to_train_csv> --test_file <path_to_test_csv> --model_path <path_to_converted_model> --tokenizer_file <path_to_vocab_file> --batch_size <int_batch_size> --num_classes <int_output_classes> --max_epochs <int_epochs> --output_dir <train_model_output_path>
```

***Full list of arguments in training script.***

| Argument Name | Argument Type | Argument Explanation | Default Value |
|---------------|---------------|----------------------|---------------|
| train_file | str | Train dataset path | datasets/train_2.csv |
| test_file | str | Test dataset path| datasets/test_2.csv |
| model_path | str | Converted model path| model/raven-0.4b-world|
| tokenizer_file | str | Vocab file path | rwkv_vocab_v20230424.txt |
| device | str | Device to train on `gpu`, `cpu`| gpu|
| batch_size | int | Samples per step | 1|
| num_devices | int | Number of devices used to train | 1|
| num_classes | int | Class count in dataset| 3|
| max_length | int | max length of sequence | 2048|
| max_epoches | int | Training epochs | 9|
| accumulate_grad_batches | int | Accumulates gradients over k batches before stepping the optimizer | 4|
| ckpt_path | str | Path to save ckpts | None|
| strategy | str | Different training strategies | deepspeed_stage_2_offload|
| is_adalora | bool | Use AdaLoRA training strategy| True|
| is_qlora | bool| Use QLoRA training strategy | False|
| output_dir | str | Trained model output path| output_dir/|


## Acknowledgements

Sequence classification code from [RWKV-Classification by @yynil](https://github.com/yynil/RWKV-Classification), which implements text classification using the [RWKV language model](https://github.com/BlinkDL/RWKV-LM).