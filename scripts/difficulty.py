import hydra
from omegaconf import DictConfig
import hook
import os, sys, random
# sys.path.extend(['..', '../data_collection'])
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


from .utils import encoding_shape, load_latest_checkpoint, checkpointing_paths
from ..data_collection.dataset import DifficultyCPDataloader
from .modules_ import PredictionHead



@hydra.main(config_path="../conf/", config_name="difficulty")
def main(cfg: DictConfig):

    random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.backends.cudnn.deterministic = True

    # Set the environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
    os.environ['MASTER_PORT'] = str(cfg.distributed.masterport)  # an open port
    os.environ['WORLD_SIZE'] = '1'           # total number of processes
    os.environ['RANK'] = '0'                 # rank of this process
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    dist.init_process_group(backend="nccl", init_method="env://")

    experiment_name, checkpoint_dir = checkpointing_paths(cfg)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',    # The metric to monitor
        min_delta=0.001,        # The minimum change in the monitored metric to qualify as an improvement
        patience=200,            # Number of epochs with no improvement after which training will be stopped
        verbose=False,         # Whether to log more info
        mode='max',            # In 'min' mode, training will stop when the quantity monitored has stopped decreasing
    )

    # Wandb logger
    wandb_logger = WandbLogger(name=experiment_name, project="piano_judge", entity="huanz")

    # Initialize the trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=cfg.gpu,
    )

    print("init model...")
    if 'model' in cfg.model: # some hydra config version diff...
        cfg.model = cfg.model.model

    # Initialize your Lightning module
    model = PredictionHead(cfg, 
                            embedding_dim=encoding_shape(cfg.encoder)[1], 
                            embedding_len=encoding_shape(cfg.encoder)[0])
    train_loader = DataLoader(
        DifficultyCPDataloader(mode='train', num_classes=cfg.dataset.num_classes), 
        **cfg.dataset.train
    )
    valid_loader = DataLoader(
        DifficultyCPDataloader(mode='test', num_classes=cfg.dataset.num_classes), 
        **cfg.dataset.eval, 
    )

    # Train the model

    if cfg.mode == 'train':
        trainer.fit(model, train_loader, valid_loader)
    elif cfg.mode == 'test':
        model = PredictionHead.load_from_checkpoint(checkpoint_path=load_latest_checkpoint(checkpoint_dir), 
                                    cfg=cfg,
                                    embedding_dim=encoding_shape(cfg.encoder)[1], 
                                    embedding_len=encoding_shape(cfg.encoder)[0]
                                    )
        trainer.test(model, dataloaders=valid_loader)

    


if __name__ == "__main__":
    main()