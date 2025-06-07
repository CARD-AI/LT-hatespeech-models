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

## Lithuanian Llama 2

## ChatGPT

## Other deep learning models
