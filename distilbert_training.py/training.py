"""
@File    :   training.py
@Time    :   12/2024
@Author  :   nikifori
@Version :   -
"""

import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from hyperpyyaml import load_hyperpyyaml
import argparse
from pathlib import Path
import os
import neptune
from neptune.utils import stringify_unsupported
import logging
import pwd
from pytorch_lightning.loggers import NeptuneLogger
import sys
from datetime import datetime
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import ModelSummary

from genius_dataset import Genius_dataset
from model_lightning import GenreClassifier_lightning


def print_info(msg):
    print(f"INFO: {msg}")


# neptune bug
# https://github.com/neptune-ai/neptune-client/issues/1702#issuecomment-2376615676
class _FilterCallback(logging.Filterer):
    def filter(self, record: logging.LogRecord):
        return not (
            record.name == "neptune"
            and record.getMessage().startswith(
                "Error occurred during asynchronous operation processing: X-coordinates (step) must be strictly increasing for series attribute"
            )
        )


neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
    _FilterCallback()
)


def setup_neptune_logger(cfg, use_neptune, exp_name):
    """
    Sets up the neptune logger if use_neptune is True.
    Args:
        cfg (dict): The configuration dictionary.
        use_neptune (bool): Whether to use neptune or not.
    Returns:
        NeptuneLogger: The neptune logger object.
        neptune.run: The neptune run object.
    """
    import hashlib

    if not use_neptune:
        print_info("Not Using Neptune Logger.")
        return None, None

    # Create a unique id for the experiment
    neptune_id = pwd.getpwuid(os.getuid())[0]
    hash_key = hashlib.new("sha256")  # sha256 can be replaced with diffrent algorithms
    hash_key.update(
        exp_name.encode()
    )  # give a encoded string. Makes the String to the Hash
    neptune_id = neptune_id + str(hash_key.hexdigest())
    print(f"exp id {neptune_id}")
    neptune_id = neptune_id[:31]

    neptune_run = neptune.init_run(
        capture_hardware_metrics=cfg["capture_hardware_metrics"],
        capture_stdout=cfg["capture_stdout"],
        capture_stderr=cfg["capture_stderr"],
        capture_traceback=cfg["capture_traceback"],
        project=cfg["project"],
        api_token=os.environ["NEPTUNE_API_TOKEN"],
        tags=cfg["tags"],
        source_files=["./*.py"],
        custom_run_id=neptune_id,
    )
    neptune_run["train_config"] = stringify_unsupported(cfg)
    neptune_run["cmd_arguments"] = " ".join(sys.argv[0:])
    neptune_run["exp_id"] = neptune_id
    neptune_run["sys/name"] = ""
    neptune_logger = NeptuneLogger(run=neptune_run, log_model_checkpoints=False)
    neptune_logger.experiment["model/hyper-parameters"] = stringify_unsupported(cfg)
    return neptune_logger, neptune_run


def safe_load_yaml(yaml_file):
    """
    Safely loads a yaml file, using hyperpyyaml.
    Args:
        yaml_file (str): The path to the yaml file.
    Returns:
        dict: The dictionary containing the yaml file data.
    """
    with open(yaml_file, "r") as f:
        cfg_dict = load_hyperpyyaml(f)
    return cfg_dict


def safe_load_yaml(yaml_file):
    """
    Safely loads a yaml file, using hyperpyyaml.
    Args:
        yaml_file (str): The path to the yaml file.
    Returns:
        dict: The dictionary containing the yaml file data.
    """
    with open(yaml_file, "r") as f:
        cfg_dict = load_hyperpyyaml(f)
    return cfg_dict


def train(cfg):
    dataset_config = cfg["dataset_config"]
    model_config = cfg["model_config"]
    training_config = cfg["training_config"]
    neptune_config = cfg["neptune_config"]

    exp_name = (
        f"{training_config['exp_name']}_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
    )

    exp_folder = Path(training_config.get("exp_folder", "./")) / exp_name
    exp_folder.mkdir(parents=True, exist_ok=True)

    if "seed" in training_config:
        pl.seed_everything(dataset_config.get("seed", 42))
    else:
        print("SEED EVERYTHING TO 0")
        pl.seed_everything(0)

    # SetUp neptune logger
    neptune_logger, neptune_run = setup_neptune_logger(
        neptune_config,
        neptune_config.get("use_neptune", False),
        exp_name=exp_name,
    )

    if neptune_run != None:
        neptune_run["conf_file"].upload(str(cfg["config_filename"]))
        logger = neptune_logger
    else:
        logger = None

    # Datasets and Dataloaders
    train_dataset = Genius_dataset(
        csv_path=dataset_config.get("csv_path"),
        random_seed=dataset_config.get("random_seed"),
        transformer_model=model_config.get("transformer_model"),
        split_name="train",
    )
    val_dataset = Genius_dataset(
        csv_path=dataset_config.get("csv_path"),
        random_seed=dataset_config.get("random_seed"),
        transformer_model=model_config.get("transformer_model"),
        split_name="validation",
    )
    test_dataset = Genius_dataset(
        csv_path=dataset_config.get("csv_path"),
        random_seed=dataset_config.get("random_seed"),
        transformer_model=model_config.get("transformer_model"),
        split_name="test",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=dataset_config.get("num_workers", 8),
        batch_size=training_config.get("train_batch_size", 32),
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=dataset_config.get("num_workers", 8),
        batch_size=training_config.get("train_batch_size", 32),
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=dataset_config.get("num_workers", 8),
        batch_size=training_config.get("train_batch_size", 32),
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = GenreClassifier_lightning(
        transformer_model=model_config.get("transformer_model"),
        exp_folder=exp_folder,
        use_activation_func_before_class_layer=model_config.get("use_activation_func_before_class_layer", True)
    )

    # print("[INFO] Model Summary:")
    # print(ModelSummary(model, max_depth=3))
    # return 0

    # lr monitoring and model checkpointing
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_folder,
        monitor="val_loss",
        mode="min",
        filename="checkpoint-ts-{val_loss:.4f}-epoch={epoch:.0f}",
        save_top_k=training_config.get("save_top_k_checkpoints", 5),
        auto_insert_metric_name=False,
        save_last=False,
    )


    # Trainer
    trainer = pl.Trainer(
        default_root_dir=exp_folder,
        accelerator=training_config.get("accelerator", "gpu"),
        strategy=training_config.get("strategy", "ddp"),
        enable_checkpointing=training_config.get("enable_checkpointing", True),
        num_nodes=training_config.get("num_nodes", 1),
        enable_model_summary=training_config.get("enable_model_summary", True),
        enable_progress_bar=training_config.get("enable_progress_bar", True),
        max_epochs=training_config.get("max_epochs", 1000),
        check_val_every_n_epoch=training_config.get("check_val_every_n_epoch", 1),
        log_every_n_steps=training_config.get("log_every_n_steps", 10),
        num_sanity_val_steps=training_config.get("num_sanity_val_steps", 1),
        gradient_clip_val=training_config.get("gradient_clip_val", 10),
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 2),
        # Experimental, use mixed precision training for more efficient training (In previous projects there were some stability issues so keep in mind during debugging)
        precision=training_config.get("precision", "32"),
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor_callback],
    )

    if training_config.get("pre_validate", True):
        print("[INFO] Running pre-validation...")
        trainer.validate(
            model,
            val_loader
        )
    
    trainer.fit(model, train_loader, val_loader)

    # Testing the model
    print("[INFO] Testing the best model based on val_loss...")
    # trainer.test(model=None, dataloaders=test_loader, ckpt_path="best")
    trainer.test(model=model, ckpt_path="/home/nikifori/Desktop/Master/NLP/final_project/classification_song_lyrics/exps/removing_special_characters_04-01-2025_16-50-33/checkpoint-ts-0.83-epoch=1.ckpt", dataloaders=test_loader,)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="YAML file containing the training configuration.",
    )
    parser.add_argument("-g", "--gpu", type=str, default="0", help="Using gpu #")
    args = parser.parse_args()

    hparams_file = Path(args.config).resolve()
    print(f"config file {hparams_file}")

    cfg = safe_load_yaml(args.config)
    cfg["config_filename"] = hparams_file

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train(cfg)
