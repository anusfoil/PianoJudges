import hydra
from omegaconf import DictConfig
import hook
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torchmetrics
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



def compute_mert_embeddings(audio_path, model, processor, resample_rate, segment_duration=10):
    import torchaudio
    import torchaudio.transforms as T

    try:
        audio, sampling_rate = torchaudio.load(audio_path)
    except:
        print(f"audio failed to read for {audio_path}")
        return []

    if resample_rate != sampling_rate: # 24000
        resampler = T.Resample(sampling_rate, resample_rate)
        audio = resampler(audio)
    audio = reduce(audio, 'c t -> t', 'mean')

    embeddings = []
    num_segments = int(len(audio) / resample_rate // segment_duration)
    for i in range(num_segments):

        offset = i * segment_duration
        audio_seg = audio[offset * resample_rate: (offset + segment_duration) * resample_rate]
        input_features = processor(audio_seg.squeeze(0), sampling_rate=resample_rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**input_features, output_hidden_states=True)

        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze() 
        embeddings.append(all_layer_hidden_states[-1, :, :])

    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    assert(embeddings.shape[1:] == torch.Size([749, 1024]))
    return embeddings # (n_segs, 749, 1024)  For some reason the output missed timestep. Should be 75 as frame rate.


def compute_jukebox_embeddings(audio_path, vqvae, device_, segment_duration=10):
    JUKEBOX_SAMPLE_RATE = 44100

    try: # in case there is problem with audio loading.
        audio = load_audio_for_jbx(audio_path)
    except:
        print(f"audio failed to read for {audio_path}")
        return torch.zeros(1, 3445, 64)

    embeddings = []
    num_segments = int(len(audio) / JUKEBOX_SAMPLE_RATE // segment_duration)
    for i in range(num_segments):
        offset = i * segment_duration
        audio_seg = audio[offset * JUKEBOX_SAMPLE_RATE: (offset + segment_duration) * JUKEBOX_SAMPLE_RATE]
        audio_seg =  rearrange(audio_seg, "t -> 1 t 1").to(device_)

        embeddings.append(encode(vqvae, audio_seg.contiguous()))

    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    embeddings = rearrange(embeddings, "b 1 e t -> b t e")
    assert(embeddings.shape[1:] == torch.Size([3445, 64]))
    return embeddings # (n_segs, 3445, 64)


def compute_dac_embeddings(audio_path, dac_model,  segment_duration=1, concat_duration=10):
    from audiotools import AudioSignal

    # Load the full audio signal
    signal = AudioSignal(audio_path)
    signal.resample(44100)
    signal.to_mono()

    # Calculate the number of samples per segment and per concatenation
    samples_per_segment = segment_duration * signal.sample_rate

    # Process in segments
    total_samples = signal.audio_data.shape[-1]
    num_segments = (total_samples + samples_per_segment - 1) // samples_per_segment  # Ceiling division
    embeddings = []

    signal.audio_data = signal.audio_data.to(dac_model.device)
    for i in range(num_segments):
        start = i * samples_per_segment
        end = min(start + samples_per_segment, total_samples)
        segment = signal.audio_data[:, :, start:end]

        # Process each segment
        x = dac_model.preprocess(segment, signal.sample_rate)
        z, _, _, _, _ = dac_model.encode(x)
        embeddings.append(rearrange(z, '1 e t -> t e'))

    embeddings =  torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    end = embeddings.shape[0] - embeddings.shape[0] % concat_duration
    final_embeddings = rearrange(embeddings[:end], '(s x) t e -> s (x t) e', x=5)

    assert(final_embeddings.shape[1:] == torch.Size([870, 1024]))
    return final_embeddings  # (n_segs, 870, 1024)



def compute_audiomae_embeddings(audio_path, amae, fp, device_, segment_duration=10):
    import torchaudio

    try:
        audio, sr = torchaudio.load(audio_path)
        num_segments = int(len(audio[0]) / sr // segment_duration)
    except:
        print(f"audio failed to read for {audio_path}")
        return torch.zeros(1, 512, 768)
        
    embeddings = []
    for i in range(num_segments):
        wav, fbank, _ = fp(audio_path, start_sec=i*segment_duration, dur_sec=segment_duration)

        fbank = rearrange(fbank, 'c t f -> 1 c t f').to(device_).to(torch.float16)
        output = amae.get_audio_embedding(fbank)
        embeddings.append(output["patch_embedding"])

    embeddings =  torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    assert(embeddings.shape[1:] == torch.Size([512, 768]))
    return rearrange(embeddings, "s 1 t e -> s t e") # (n_segs, 512, 768)


def load_or_compute_embedding(audio_paths, method, device_, audio_inits, recompute=False):

    embeddings = []
    for audio_path in audio_paths:
        save_path = audio_path[:-4] + f"_{method}.npy"
        save_path = save_path.replace('ATEPP-audio', 'ATEPP-audio-embeddings')
        os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)

        if os.path.exists(save_path) and (not recompute):
            embedding = torch.from_numpy(np.load(save_path)).to(device_)

            if len(embedding.shape) >= 3:
                embedding = rearrange(embedding, "1 t e -> t e")
        else:
            if method == 'jukebox':
                vqvae = audio_inits['audio_encoder']
                embedding = compute_jukebox_embeddings(audio_path, vqvae, device_)
            elif method == 'mert':
                mert_model = audio_inits['audio_encoder']
                mert_processor = audio_inits['processor']
                embedding = compute_mert_embeddings(audio_path, mert_model, mert_processor, mert_processor.sampling_rate)
            elif method == 'dac':
                dac_model = audio_inits['audio_encoder']
                embedding = compute_dac_embeddings(audio_path, dac_model)
            elif method == 'audiomae':
                amae = audio_inits['audio_encoder']
                fp = audio_inits['fp']
                embedding = compute_audiomae_embeddings(audio_path, amae, fp, device_)
            else:
                raise ValueError("Invalid method specified")

            np.save(save_path, embedding.cpu().detach().numpy())

        embeddings.append(embedding)

    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)  # (b t e)
    hook()
    embeddings.to(device_)

    return embeddings



def init_encoder(encoder, device):

    if encoder == 'jukebox':
        audio_encoder = setup_jbx('5b', device)
        return {"audio_encoder": audio_encoder}
    
    if encoder == 'mert':
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoModel

        audio_encoder = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        # loading the corresponding preprocessor config
        processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
        return {"audio_encoder": audio_encoder, 
                "processor": processor}

    if encoder == 'dac':
        import dac
        model_path = dac.utils.download(model_type="44khz")
        model = dac.DAC.load(model_path)

        model.to(device)
        return {'audio_encoder': model}
    

    if encoder == 'audiomae':
        from audio_processor import _fbankProcessor
        from audiomae_wrapper import AudioMAE

        amae = AudioMAE.create_audiomae(ckpt_path='AudioMAE/finetuned.pth', device=device)

        return {'audio_encoder': amae,
                'fp': _fbankProcessor.build_processor()}
    



class SimpleNNLightning(pl.LightningModule):
    def __init__(self, cfg, device_, input_dim, learning_rate=0.001, encoder='jukebox'):
        super(SimpleNNLightning, self).__init__()
        # Define your model architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
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

        emb1 = emb1.to(self.device_)
        emb2 = emb2.to(self.device_)

        # Concatenate or otherwise combine the embeddings
        combined_emb = torch.cat((emb1, emb2), dim=-2)  # (b, t1+t2, e)

        return combined_emb

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def encoder_input_dim(encoder):
    if encoder == 'jukebox':
        return 64
    elif encoder == 'mert':
        return 1024
    elif encoder == 'dac':
        return 1024
    elif encoder == 'audiomae':
        return 768


@hydra.main(config_path=".", config_name="ranking")
def main(cfg: DictConfig):

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
    model = SimpleNNLightning(cfg, device, input_dim=encoder_input_dim(cfg.encoder), encoder=cfg.encoder)

    train_loader = DataLoader(
        PerformanceDataloader(mode='train'), 
        batch_size=cfg.train.batch_size, 
        num_workers=8,
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
        model.load_from_checkpoint("checkpoints/last.ckpt", device=device)
        trainer.test(model, dataloaders=test_loader)

    


if __name__ == "__main__":
    main()