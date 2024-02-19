import hydra
from omegaconf import DictConfig
from .. import hook
import os, sys, random
# sys.path.extend(['..', '../data_collection'])
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from .utils import init_encoder, load_or_compute_embedding, encoding_shape
from ..data_collection.dataset import DifficultyDataloader
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

    print('here')
    experiment_name = f"{cfg.task}_{cfg.encoder}"

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{experiment_name}",
        filename="{epoch}-{step}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )

    # Wandb logger
    wandb_logger = WandbLogger(name=experiment_name, project="piano_judge", entity="huanz")

    # Initialize the trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=cfg.max_epochs,
        accelerator="gpu",
        devices=cfg.gpu,
    )

    print("init model...")
    device = torch.device(f'cuda:5')

    # Initialize your Lightning module
    model = PredictionHead(cfg, 
                            ae_device=device,
                            embedding_dim=encoding_shape(cfg.encoder)[1], 
                            embedding_len=encoding_shape(cfg.encoder)[0])
    train_loader = DataLoader(
        DifficultyDataloader(mode='train'), 
        **cfg.dataset.train
    )
    valid_loader = DataLoader(
        DifficultyDataloader(mode='test'), 
        **cfg.dataset.eval, 
    )

    # Train the model

    if cfg.mode == 'train':
        trainer.fit(model, train_loader, valid_loader)
    elif cfg.mode == 'test':
        model = PredictionHead.load_from_checkpoint(checkpoint_path="checkpoints/enc_dac/epoch=44-step=3915-val_loss=0.47.ckpt", 
                                    cfg=cfg,
                                    ae_device=device,
                                    embedding_dim=encoding_shape(cfg.encoder)[1], 
                                    embedding_len=encoding_shape(cfg.encoder)[0]
                                    )
        trainer.test(model, dataloaders=valid_loader)

    


if __name__ == "__main__":
    main()