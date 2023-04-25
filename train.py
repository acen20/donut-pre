import argparse
import datetime
import json
import os
import random
from io import BytesIO
from os.path import basename
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from sconf import Config

from donut import get_data
from lightning_module import DonutDataPLModule, DonutModelPLModule


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {"model." + key: value for key, value in state_dict.items()}
        return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")


def train(config):
    #pl.utilities.seed(config.get("seed", 42), workers=True)

    model_module = DonutModelPLModule(config)
    data_module = DonutDataPLModule(config)

    # add datasets to data_module
    datasets = {"train": [], "test": []}
    for dataset_name_or_path in config.dataset_name_or_paths:
        task_name = "custom"
        #task_name = os.path.basename(dataset_name_or_path)  # e.g., cord-v2, docvqa, rvlcdip, ...
        
        # add categorical special tokens (optional)
        if task_name == "rvlcdip":
            model_module.model.decoder.add_special_tokens([
                "<advertisement/>", "<budget/>", "<email/>", "<file_folder/>", 
                "<form/>", "<handwritten/>", "<invoice/>", "<letter/>", 
                "<memo/>", "<news_article/>", "<presentation/>", "<questionnaire/>", 
                "<resume/>", "<scientific_publication/>", "<scientific_report/>", "<specification/>"
            ])
        if task_name == "docvqa":
            model_module.model.decoder.add_special_tokens(["<yes/>", "<no/>"])
            
        for split in ["train", "test"]:
            datasets[split].append(
                get_data(filepath = f"{dataset_name_or_path}/{split}",
                         split=split,
                         task_name=task_name,
                         donut_model=model_module.model)
            )
            # prompt_end_token is used for ignoring a given prompt in a loss function
            # for docvqa task, i.e., {"question": {used as a prompt}, "answer": {prediction target}},
            # set prompt_end_token to "<s_answer>"
    data_module.train_datasets = datasets["train"]
    data_module.val_datasets = datasets["test"]

    logger = TensorBoardLogger(
        save_dir=config.save_path,
        default_hp_metric=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(config.save_path),
        filename="artifacts",
        save_top_k=1,
        save_last=False,
        mode="min",
    )

    custom_ckpt = CustomCheckpointIO()
    trainer = pl.Trainer(
        resume_from_checkpoint=config.get("resume_from_checkpoint_path", None),
        num_nodes=config.get("num_nodes", 1),
        gpus=torch.cuda.device_count(),
        strategy="ddp",
        accelerator="cpu",
        plugins=custom_ckpt,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision=32,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=[lr_callback, checkpoint_callback],
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    config.dataset_name_or_paths = ["/home/ahsenmailbox/Desktop/22-04-2023/donut/data"]
    config.save_path = "/home/ahsenmailbox/Desktop/22-04-2023/donut/model"

    save_config_file(config, config.save_path)
    train(config)
