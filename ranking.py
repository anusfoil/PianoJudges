import hydra
from omegaconf import DictConfig
import hook
import os, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torchmetrics
import torch.distributed as dist
from sklearn.metrics import classification_report
from tqdm import tqdm
from einops import reduce, rearrange

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils import init_encoder, load_or_compute_embedding, encoder_input_dim
from data_collection.dataset import PerformanceDataloader, ICPCDataloader



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Skip Connection - Ensuring the channel dimensions match
        if identity.shape != out.shape:
            identity = nn.Conv2d(identity.shape[1], out.shape[1], kernel_size=1)(identity)

        out += identity
        return out


class AudioCNNTransformer(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, cnn_channels, kernel_size, dropout=0.1):
        super(AudioCNNTransformer, self).__init__()

        # CNN blocks with skip connections
        self.conv_block1 = ConvBlock(1, cnn_channels, kernel_size, dropout)
        self.conv_block2 = ConvBlock(cnn_channels, 2 * cnn_channels, kernel_size, dropout)

        # Calculate the new sequence length after CNN
        self.new_seq_len = (input_dim // 4)  # Adjust based on your CNN architecture

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.new_seq_len,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Regression Layer
        self.regression = nn.Linear(self.new_seq_len * 2 * cnn_channels, 1)

    def forward(self, x):
        # x shape: (batch, n_segs, seg_timesteps * 2, embedding dimension)
        batch_size, n_segs, seq_len, emb_dim = x.shape

        # Apply CNN blocks
        x = x.view(batch_size * n_segs, 1, seq_len, emb_dim)  # Reshape for CNN
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        # Flatten and transpose for Transformer
        x = x.view(batch_size, n_segs, -1).transpose(0, 1)  # (n_segs, batch, flattened emb_dim)

        # Apply Transformer Encoder
        transformer_output = self.transformer_encoder(x)

        # Aggregate the output and reshape
        agg_output = transformer_output.transpose(0, 1).reshape(batch_size, -1)

        # Regression to get final output
        output = self.regression(agg_output).squeeze(-1)

        return output

# # Example Usage
# model = AudioCNNTransformer(
#     input_dim=1024,  # Embedding dimension
#     nhead=4,
#     num_encoder_layers=2,
#     dim_feedforward=512,
#     cnn_channels=16,  # Number of CNN channels
#     kernel_size=(3, 3),  # Kernel size for CNN
#     dropout=0.1
# ).to('cuda:1')

# # Example input tensor
# input_tensor = torch.rand(2, 70, 870*2, 1024).to('cuda:1')  # (batch, n_segs, seg_timesteps * 2, embedding dimension)
# output = model(input_tensor)
# print(output.shape)  # Should be (batch_size,)


class RankerLightning(pl.LightningModule):
    def __init__(self, cfg, device_, input_dim, learning_rate=0.001, encoder='jukebox'):
        super(RankerLightning, self).__init__()
        # Define your model architecture
        self.model = AudioCNNTransformer(
            input_dim=input_dim,  # Embedding dimension
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=512,
            cnn_channels=8,  # Number of CNN channels
            kernel_size=(3, 3),  # Kernel size for CNN
            dropout=0.1
        )

        self.learning_rate = learning_rate
        self.cfg = cfg
        self.device_ = device_
        self.encoder = encoder
        self.audio_inits = init_encoder(encoder=encoder, device=device_)

        self.criterion = nn.MSELoss()

        self.precision_metric = torchmetrics.Precision(num_classes=3, average='macro', task='multiclass')
        self.recall = torchmetrics.Recall(num_classes=3, average='macro', task='multiclass')
        self.f1 = torchmetrics.F1Score(num_classes=3, average='macro', task='multiclass')
        self.accuracy = torchmetrics.Accuracy(num_classes=3, task='multiclass')


    def forward(self, x):
        x = self.layers(x)
        x = rearrange(x, "b t 1 -> b t")
        return torch.mean(x, dim=-1)


    def training_step(self, batch, batch_idx):

        outputs, loss = self.train_valid_pass(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        outputs, loss = self.train_valid_pass(batch)
        self.log("val_loss", loss)

        # Convert labels to tensor (note that I added 1 to both label and pred for the metrics)
        labels = torch.tensor(batch["label"], dtype=torch.float32, device=self.device) + 1

        # Convert regression output to classification labels
        predicted_labels = torch.tensor([torch.argmin(torch.abs(torch.tensor([x + 1, x, x - 1]))).item() for x in outputs], device=self.device)

        # Update metrics
        precision = self.precision_metric(predicted_labels, labels)
        recall = self.recall(predicted_labels, labels)
        f1_score = self.f1(predicted_labels, labels)
        accuracy = self.accuracy(predicted_labels, labels)

        self.log('val_precision', precision, on_epoch=True, prog_bar=True)
        self.log('val_recall', recall, on_epoch=True, prog_bar=True)
        self.log('val_f1_score', f1_score, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

        return loss

    
    def test_step(self, batch, batch_idx):

        combined_emb = self.batch_to_embedding(batch)
        outputs = self(combined_emb)  # Regression output

        # Convert labels to tensor
        labels = torch.tensor(batch["label"], dtype=torch.float32, device=self.device)

        # Convert regression output to classification labels based on nearest point
        predicted_labels = torch.tensor([np.argmin([abs(x + 1), abs(x), abs(x - 1)]) - 1 for x in outputs], device=outputs.device)

        # Store labels and predictions for later use in test_epoch_end
        return {'true_labels': labels, 'predicted_labels': predicted_labels, 'outputs:': outputs}


    def test_epoch_end(self, outputs):
        # Concatenate all labels and predictions
        all_true_labels = torch.cat([x['true_labels'] for x in outputs], dim=0).cpu().numpy()
        all_predicted_labels = torch.cat([x['predicted_labels'] for x in outputs], dim=0).cpu().numpy()
        all_predicted_outputs = torch.cat([x['predicted_outputs'] for x in outputs], dim=0).cpu().numpy()

        # Print classification summary
        print(classification_report(all_true_labels, all_predicted_labels, labels=[-1, 0, 1]))
        hook()


    def train_valid_pass(self, batch):

        combined_emb = self.batch_to_embedding(batch).to(self.device)

        # Convert labels to tensor
        labels = torch.tensor(batch["label"], dtype=torch.float32, device=self.device)

        # Forward pass
        outputs = self(combined_emb)
        loss = self.criterion(outputs, labels)

        return outputs, loss

    def batch_to_embedding(self, batch):
        emb1 = load_or_compute_embedding(batch["audio_path_1"], self.encoder, self.device_, self.audio_inits, recompute=self.cfg.recompute)
        emb2 = load_or_compute_embedding(batch["audio_path_2"], self.encoder, self.device_, self.audio_inits, recompute=self.cfg.recompute)

        # emb: ()
        emb1 = emb1.to(self.device_)
        emb2 = emb2.to(self.device_)

        # Concatenate or otherwise combine the embeddings
        combined_emb = torch.cat((emb1, emb2), dim=-2)  # (b, t1+t2, e)

        return combined_emb

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)



@hydra.main(config_path=".", config_name="ranking")
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


    experiment_name = f"enc_{cfg.encoder}"

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
    wandb_logger = WandbLogger(name=experiment_name, project="expertise_ranking", entity="huanz")

    # Initialize the trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=50,
        accelerator="gpu",
        devices=[cfg.gpu],
    )

    print("init model...")
    device = torch.device(f'cuda:{cfg.gpu}')

    # Initialize your Lightning module
    model = RankerLightning(cfg, device, 
                            input_dim=encoder_input_dim(cfg.encoder), 
                            encoder=cfg.encoder,
                            learning_rate=cfg.learning_rate)
    train_loader = DataLoader(
        PerformanceDataloader(mode='train', pair_mode=cfg.dataset.pair_mode), 
        **cfg.dataset.train
    )
    valid_loader = DataLoader(
        PerformanceDataloader(mode='test', pair_mode=cfg.dataset.pair_mode), 
        **cfg.dataset.eval, 
    )
    test_loader = DataLoader(
        ICPCDataloader, 
        **cfg.dataset.test, 
    )

    # Train the model

    if cfg.mode == 'train':
        trainer.fit(model, train_loader, valid_loader)
    elif cfg.mode == 'test':
        model.load_from_checkpoint("checkpoints/last.ckpt", device=device)
        trainer.test(model, dataloaders=test_loader)

    


if __name__ == "__main__":
    main()