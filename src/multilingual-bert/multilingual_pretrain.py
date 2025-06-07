from simpletransformers.language_modeling import LanguageModelingModel
import logging
import torch


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    "num_train_epochs": 6,
    "save_eval_checkpoints": True,
    "save_model_every_epoch": False,
    "learning_rate": 5e-4,
    "warmup_steps": 10000,
    "train_batch_size": 48,
    "eval_batch_size": 128,
    "gradient_accumulation_steps": 1,
    "block_size": 128,
    "max_seq_length": 128,
    "dataset_type": "simple",
    "wandb_project": "Lithuanian - BERT",
    "wandb_kwargs": {"name": "BERT_pretrain"},
    "logging_steps": 100,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 50000,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    "sliding_window": True,
    "vocab_size": 119547,
    'encoding': 'utf-8',
    'output_dir': 'outputs_bert',
}

train_file = "../data/for_pretrain/train.txt"
test_file = "../data/for_pretrain/test.txt"

model = LanguageModelingModel(
    "bert",
    "bert-base-multilingual-cased",
    args=train_args,
    train_files=train_file,
)


model.train_model(
    train_file, eval_file=test_file,
)

model.eval_model(test_file)
