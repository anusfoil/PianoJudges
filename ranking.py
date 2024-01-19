import hydra
from omegaconf import DictConfig
import hook
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torch.distributed as dist
from sklearn.metrics import classification_report
from tqdm import tqdm
from einops import reduce, rearrange
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils import setup_jbx, load_audio_for_jbx, encode
from data_collection.dataset import PerformanceDataloader



def compute_mert_embeddings(audio_paths, model, processor, resample_rate):
    import torchaudio
    import torchaudio.transforms as T

    batch_embeddings = []

    for path in audio_paths:
        # Load and process each audio file
        audio, sampling_rate = torchaudio.load(path)
        if resample_rate != sampling_rate:
            resampler = T.Resample(sampling_rate, resample_rate)
            audio = resampler(audio)

        inputs = processor(audio.squeeze(0), sampling_rate=resample_rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
        # Flatten the tensor to concatenate all layers
        flattened_embedding = all_layer_hidden_states.view(-1)
        batch_embeddings.append(flattened_embedding)

    # Stack all embeddings into a batch
    return torch.stack(batch_embeddings)


def compute_jukebox_embeddings(audio_path, vqvae, device_):
    audio = load_audio(audio_path).to(device_)
    embeddings = encode(vqvae, audio.contiguous())

    return rearrange(embeddings, "1 e t -> t e")

def load_audio(audio_path):
    try: # in case there is problem with audio loading.
        audio = load_audio_for_jbx(audio_path, dur=100)
    except:
        print(f"audio failed to read for {audio_path}")
        audio = torch.zeros(4410000)

    # You may need to pad the audio to have the same length
    # batch_audio = torch.nn.utils.rnn.pad_sequence(batch_audio, batch_first=True)

    return rearrange(audio, "t -> 1 t 1")


def load_or_compute_embedding(audio_paths, method, device_, vqvae=None, mert_model=None, mert_processor=None):

    embeddings = []
    for audio_path in audio_paths:
        save_path = audio_path[:-4] + f"_{method}.npy"
        save_path = save_path.replace('ATEPP-audio', 'ATEPP-audio-embeddings')
        os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)

        if os.path.exists(save_path):
            embedding = torch.from_numpy(np.load(save_path)).to(device_)
        else:
            if method == 'jukebox':
                embedding = compute_jukebox_embeddings(audio_path, vqvae, device_)
            elif method == 'mert':
                embedding = compute_mert_embeddings(audio_path, mert_model, mert_processor, device_, mert_processor.sampling_rate)
            else:
                raise ValueError("Invalid method specified")

            np.save(save_path, embedding.cpu().numpy())

        embeddings.append(embedding)

    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)  # (b t e)

    return embeddings



def init_encoder(mode, device):

    if mode == 'jukebox':
        audio_encoder = setup_jbx('5b', device)
        return {"audio_encoder": audio_encoder}
    
    if mode == 'mert':
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoModel

        audio_encoder = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        # loading the corresponding preprocessor config
        processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
        return {"audio_encoder": audio_encoder, 
                "processor": processor}

    



class SimpleNNLightning(pl.LightningModule):
    def __init__(self, device, learning_rate=0.001, encoder='jukebox'):
        super(SimpleNNLightning, self).__init__()
        # Define your model architecture
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.learning_rate = learning_rate

        self.device_ = device
        self.encoder = encoder
        self.inits = init_encoder(mode=encoder, device=device)

        self.criterion = nn.MSELoss()
        print(self.device)

    def forward(self, x):
        x = rearrange(self.layers(x), "b t 1 -> b t")
        return torch.mean(x, dim=-1)

    def training_step(self, batch, batch_idx):

        loss = self.train_valid_pass(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):

        loss = self.train_valid_pass(batch)
        self.log("val_loss", loss)

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

        combined_emb = self.batch_to_embedding(batch)

        # Convert labels to tensor
        labels = torch.tensor(batch["label"], dtype=torch.float32, device=self.device)

        # Forward pass
        outputs = self(combined_emb)
        loss = self.criterion(outputs, labels)

        return loss

    def batch_to_embedding(self, batch):
        if self.encoder == 'jukebox':
            # Compute embeddings for both sets of audio paths
            emb1 = load_or_compute_embedding(batch["audio_path_1"], 'jukebox', self.device_, vqvae=self.inits['audio_encoder'])
            emb2 = load_or_compute_embedding(batch["audio_path_2"], 'jukebox', self.device_, vqvae=self.inits['audio_encoder'])
        elif self.encoder == 'mert':
            emb1 = load_or_compute_embedding(batch["audio_path_1"], 'mert', self.device_,
                                            mert_model=self.inits['audio_encoder'], mert_processor=self.inits['processor'])
            emb2 = load_or_compute_embedding(batch["audio_path_2"], 'mert', self.device_,
                                            mert_model=self.inits['audio_encoder'], mert_processor=self.inits['processor'])

        # Concatenate or otherwise combine the embeddings
        combined_emb = torch.cat((emb1, emb2), dim=-2)  # (b, t1+t2, e)

        return combined_emb

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)




@hydra.main(config_path=".", config_name="ranking")
def main(cfg: DictConfig):

    # Set the environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
    os.environ['MASTER_PORT'] = str(cfg.distributed.masterport)  # an open port
    os.environ['WORLD_SIZE'] = '1'           # total number of processes
    os.environ['RANK'] = '0'                 # rank of this process
    os.environ['HYDRA_FULL_ERROR'] = '1'

    dist.init_process_group(backend="nccl", init_method="env://")

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{step}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )

    # Wandb logger
    wandb_logger = WandbLogger(project="expertise_ranking", entity="huanz")

    # Initialize the trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=50,
        accelerator="gpu",
        devices=[1],
    )

    print("init model...")
    device = torch.device(cfg.device)

    # Initialize your Lightning module
    model = SimpleNNLightning(device)

    train_loader = DataLoader(
        PerformanceDataloader(mode='train'), 
        batch_size=cfg.train.batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        PerformanceDataloader(mode='test'), 
        batch_size=cfg.eval.batch_size, 
        shuffle=False
    )

    # Train the model

    if cfg.mode == 'train':
        trainer.fit(model, train_loader, test_loader)
    elif cfg.mode == 'test':
        model.load_from_checkpoint("checkpoints/epoch=1-step=64184-val_loss=1.23.ckpt", device=device)
        trainer.test(model, dataloaders=test_loader)

    


if __name__ == "__main__":
    main()