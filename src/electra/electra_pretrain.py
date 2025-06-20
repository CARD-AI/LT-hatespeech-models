from simpletransformers.language_modeling import LanguageModelingModel
import logging
import torch
from random import randrange


SEED = randrange(10000)
#SEED = 3575
torch.manual_seed(SEED)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    "num_train_epochs": 20,
    "save_eval_checkpoints": True,
    "save_model_every_epoch": True,
    "learning_rate": 5e-4,
    "warmup_steps": 10000,
    "train_batch_size": 128,
    #"train_batch_size": 48,
    "eval_batch_size": 128,
    #"eval_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "block_size": 128,
    "max_seq_length": 128,
    "dataset_type": "simple",
    "wandb_project": "LT_Electra2024",
    "wandb_kwargs": {"name": f"ElectraLT2024_{SEED}"},
    "logging_steps": 100,
    "evaluate_during_training": True,
    #"evaluate_during_training_steps": 50000,
    "evaluate_during_training_steps": False,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    "sliding_window": True,
    "vocab_size": 52000,
    'encoding': 'utf-8',
    'output_dir': f'models/electra_base2024_{SEED}',
    "generator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
        "num_hidden_layers": 3,
    },
    "discriminator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
    },
}

train_file = "data/final_dataset/train.txt"
test_file = "data/final_dataset/test.txt"

model = LanguageModelingModel(
    # Use these all the time
    model_type="electra",
    model_name=None,
    #model_name="outputs/best_model",
    # Use these when training from checkpoint
    #generator_name="aitrash/lt_models//electra_base_7050/generator_moodel",
    #discriminator_name="aitrash/lt_models/electra_base_7050/discriminator_model",
    ########
    # Other lines
    #generator_name="google/electra-large-generator",
    #discriminator_name="google/electra-large-discriminator",
    args=train_args,
    train_files=train_file
)

print("Training model...")

model.train_model(
    train_file, eval_file=test_file,
)

model.eval_model(test_file)
