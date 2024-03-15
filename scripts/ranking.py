import hydra
from omegaconf import DictConfig
import hook
import os, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from .utils import encoding_shape, load_latest_checkpoint, checkpointing_paths
from ..data_collection.dataset import ExpertiseDataloader, ICPCDataloader
from .modules_ import PredictionHead



@hydra.main(config_path="../conf/", config_name="ranking")
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
        save_last=False
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
    if 'model' in cfg.model: # some hydra config version diff...
        cfg.model = cfg.model.model

    # Initialize your Lightning module
    model = PredictionHead(cfg, 
                            embedding_dim=encoding_shape(cfg.encoder)[1], 
                            embedding_len=encoding_shape(cfg.encoder)[0])
    train_loader = DataLoader(
        ExpertiseDataloader(mode='train', pair_mode=cfg.dataset.pair_mode, num_classes=cfg.dataset.num_classes), 
        **cfg.dataset.train
    )
    valid_loader = DataLoader(
        ExpertiseDataloader(mode='test', pair_mode=cfg.dataset.pair_mode, num_classes=cfg.dataset.num_classes), 
        **cfg.dataset.eval, 
    )
    test_loader = DataLoader(
        ICPCDataloader(pair_mode='all', num_classes=2, mode="test"),  # use all pairs in the testing set
        # ExpertiseDataloader(mode='test', pair_mode=cfg.dataset.pair_mode, num_classes=cfg.dataset.num_classes), 
        **cfg.dataset.test, 
    )
    hook()

    # Train the model

    if cfg.mode == 'train':
        trainer.fit(model, train_loader, valid_loader)
    else: # testing
        model = PredictionHead.load_from_checkpoint(
                            checkpoint_path=load_latest_checkpoint(checkpoint_dir), 
                            # checkpoint_path='/homes/hz009/Research/PianoJudge/checkpoints/rank_jukebox_c_2/epoch=86-step=9831-val_loss=0.35.ckpt',
                            cfg=cfg,
                            embedding_dim=encoding_shape(cfg.encoder)[1], 
                            embedding_len=encoding_shape(cfg.encoder)[0],
                            strict=False
                            )
        if cfg.mode == 'fit_test':
            trainer = pl.Trainer(
                logger=wandb_logger,
                callbacks=[checkpoint_callback],
                accelerator="gpu",
                devices=cfg.gpu,
                max_epochs=30,
                limit_val_batches=0,
                num_sanity_val_steps=0
            )
            fit_test_loader = DataLoader(
                ICPCDataloader(pair_mode='all', num_classes=2, mode="train"),  # use all pairs in the testing set
                # ExpertiseDataloader(mode='test', pair_mode=cfg.dataset.pair_mode, num_classes=cfg.dataset.num_classes), 
                **cfg.dataset.test, 
            )
            trainer.fit(model, fit_test_loader, test_loader)

        trainer.test(model, dataloaders=test_loader)

    


if __name__ == "__main__":
    main()