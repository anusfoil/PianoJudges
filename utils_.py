import librosa
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Sampler
import pandas as pd

# from transformers import PerceiverModel
# from transformers import MT5ForConditionalGeneration, T5Tokenizer

from jukebox.make_models import make_vqvae, MODELS
from jukebox.hparams import setup_hparams, Hyperparams, DEFAULTS
# from perceiver_pytorch import PerceiverIO

import glob, random, json
from tqdm import tqdm
import  hook

JUKEBOX_SAMPLE_RATE = 44100  # Hz
SEG_DUR = 2205000
BATCH_SIZE = 2
MINIBATCH = 8
N_COMPETITIOR = 120

# utility functions from https://github.com/ethman/tagbox

def setup_jbx(model, device, levels=3, sample_length=1048576):
    """Sets up the Jukebox VQ-VAE."""
    vqvae = MODELS[model][0]
    hparams = setup_hparams(vqvae, dict(sample_length=sample_length,
                                        levels=levels))

    for default in ["vqvae", "vqvae_conv_block", "train_test_eval"]:
        for k, v in DEFAULTS[default].items():
            hparams.setdefault(k, v)

    hps = Hyperparams(**hparams)
    return make_vqvae(hps, device)


def audio_for_jbx(audio, trunc_sec=None, device='cuda'):
    """Readies an audio array for Jukebox."""
    if audio.ndim == 1:
        audio = audio[None]
        audio = audio.mean(axis=0)

    # normalize audio
    norm_factor = np.abs(audio).max()
    if norm_factor > 0:
        audio /= norm_factor

    audio = audio.flatten()
    if trunc_sec is not None:
        audio = audio[: int(JUKEBOX_SAMPLE_RATE * trunc_sec)]

    return torch.tensor(audio, device=device)[None, :, None]


def load_audio_for_jbx(path, offset=0.0, dur=None, trunc_sec=None, device='cpu'):
    """Loads a path for use with Jukebox."""
    audio, sr = librosa.load(path, sr=None, offset=offset, duration=dur)

    if sr != JUKEBOX_SAMPLE_RATE:
        audio = librosa.resample(audio, sr, JUKEBOX_SAMPLE_RATE)

    return audio_for_jbx(audio, trunc_sec, device=device)

def encode(vqvae, x):
    """Encode audio, `x`, to an unquantized embedding using `vqvae`."""
    x_in = vqvae.preprocess(x)
    xs = []
    for level in range(vqvae.levels):
        encoder = vqvae.encoders[level]
        x_out = encoder(x_in)
        xs.append(x_out[-1])

    return xs



device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
vqvae = setup_jbx('5b', device)

# read the results table
results_path = "../Datasets/ICPC2015-dataset/data/results.tsv"
results = pd.read_table(results_path, index_col=None)


class PerfProgressRank():
    def __init__(self):
        # compile the dataset
        perf_paths = glob.glob("../Datasets/ICPC2015-dataset/data/raw/00_preliminary/**/*.wav", recursive=True)

        with open('train_indices.txt', 'r') as json_file:
            self.train_indices = json.load(json_file)    

        self.audio_segs, self.gt_progress = {}, {}
        for perf_path in tqdm(perf_paths[:N_COMPETITIOR]):
            competitor = perf_path.split(" – ")[0].split("/")[-1].split('_')[-1]
            competitor_pass = results[results["#name"].str.contains(competitor)]['#preliminary'].values
            if len(competitor_pass) == 0:
                print(competitor)
                continue # can't find the competitor in results table

            perf_audio = load_audio_for_jbx(perf_path).to(torch.device('cpu'))
            perf_audio_segs = torch.stack([perf_audio[:, i: i+SEG_DUR, :] for i in (range(0, perf_audio.shape[1]-SEG_DUR, SEG_DUR))])
            
            self.audio_segs[competitor] = perf_audio_segs # (num_segs, 1, seg_dur, 1)
            self.gt_progress[competitor] = competitor_pass

            # self.train_indices[competitor] = random.sample(range(len(perf_audio_segs)), int(len(perf_audio_segs) * 0.8))
            # self.test_indices[competitor] = [i for i in range(len(perf_audio_segs)) if i not in self.train_indices[competitor]]     

    def __getitem__(self, idx):
        """sample one audio segment from each performer. returns:
            audio_segs: (, num_competitors, seg_dur)
            gt_progress: (, num_competitors)

        TODO: try different sampling strategy.
        """
        audio_segs, gt_progress = [], []
        for competitor, perf_audio_segs in self.audio_segs.items():
            
            if idx % 2 == 1: # sampler in train mode
                perf_audio_segs = perf_audio_segs[self.train_indices[competitor]]
            else:
                mask = [x for x in range(perf_audio_segs.shape[0]) if x not in self.train_indices[competitor]]
                perf_audio_segs = perf_audio_segs[mask]

            audio_segs.append(random.choice(perf_audio_segs))
            gt_progress.append(self.gt_progress[competitor])
        audio_segs = torch.stack(audio_segs).squeeze(1).to(device)
        gt_progress = torch.tensor(np.array(gt_progress)).squeeze(1).to(device)
        return audio_segs, gt_progress
    
    def __len__(self):
        return 10 * BATCH_SIZE


class DummySampler(Sampler):
    """dummy sampler, for giving even/odd indices into the train& test set"""
    def __init__(self, data, train=True):
        self.train = train
        self.num_samples = len(data)

    def __iter__(self):
        return iter(range(int(self.train), self.num_samples, 2))

    def __len__(self):
        return self.num_samples

perf_progress_data = PerfProgressRank()
train_sampler, test_sampler = DummySampler(perf_progress_data), DummySampler(perf_progress_data, train=False)
train_loader = DataLoader(perf_progress_data, sampler=train_sampler, batch_size=BATCH_SIZE)
test_loader = DataLoader(perf_progress_data, sampler=test_sampler, batch_size=BATCH_SIZE)

"""models to try: 
    AST: will need to start from audio samples, huggingface implementation
    WavLM: speech, also need to restart from audio samples

    AudioLM: it's mainly generation, no sequence transfer (and no pretrained) 
    Perceiver AR: no usable pretrained, can start from scratch (lucidrains implementation only in tokenized format)
    """

model = PerceiverIO(
    dim = 64,                    # dimension of sequence to be encoded
    queries_dim = 2,             # dimension of decoder queries
    logits_dim = 2,              # dimension of final logits
    depth = 6,                   # depth of net
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 256,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,         # number of dimensions per cross attention head
    latent_dim_head = 64,        # number of dimensions per latent self attention head
    weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
    seq_dropout_prob = 0.2       # fraction of the tokens from the input sequence to dropout (structured dropout, for saving compute and regularizing effects)
).to(device)


queries = torch.randn(N_COMPETITIOR, 2).to(device)

optim = Adam(model.parameters(), lr=1e-5)

for epoch in tqdm(range(30)):

    for audio_segs, gt_progress in train_loader:
        model.train()
        # merge the batch dimension and sequence dimension 
        audio_segs_ = audio_segs.reshape(-1, SEG_DUR, 1)

        """process the jukebox embeddings in smaller batchs, avoiding OOM"""
        print("generating jukebox embeddings...")
        embedding_top = []
        for i in tqdm(range(0, audio_segs_.shape[0], MINIBATCH)):
            audio_segs_minibatch = audio_segs_[i:i+MINIBATCH, :, :]
            embedding_top.append(encode(vqvae, audio_segs_minibatch)[2].detach())
        embedding_top = torch.vstack(embedding_top) # (batch * num_competitor, 64, 17226)

        assert(embedding_top.shape[0] == audio_segs_.shape[0])
        # embedding_top = encode(vqvae, audio_segs_)[2].detach()

        """TODO: ways to reduce """
        # embeddings = embedding_top.reshape(num_segs, -1) 
        embeddings = torch.mean(embedding_top, dim=2) # (batch * num_competitor, 64)
        embeddings = embeddings.reshape(BATCH_SIZE, -1, embeddings.shape[-1]) # (batch, num_competitors, 64)

        logits = model(embeddings, queries=queries)
        progress = logits.argmax(dim=2)

        loss = F.cross_entropy(logits.permute(0, 2, 1), gt_progress.to(device))
        print(loss)
        loss.backward()

        optim.step()
        optim.zero_grad()


    if epoch % 5 == 0:
        for audio_segs, gt_progress in test_loader:

            model.eval()
            # merge the batch dimension and sequence dimension 
            audio_segs_ = audio_segs.reshape(-1, SEG_DUR, 1)

            """process the jukebox embeddings in smaller batchs, avoiding OOM"""
            embedding_top = []
            for i in range(0, audio_segs_.shape[0], MINIBATCH):
                audio_segs_minibatch = audio_segs_[i:i+MINIBATCH, :, :]
                embedding_top.append(encode(vqvae, audio_segs_minibatch)[2].detach())
            embedding_top = torch.vstack(embedding_top) # (batch * num_competitor, 64, 17226)
            
            # embeddings = embedding_top.reshape(num_segs, -1) 
            embeddings = torch.mean(embedding_top, dim=2) # (batch * num_competitor, 64)
            embeddings = embeddings.reshape(BATCH_SIZE, N_COMPETITIOR, -1) # (batch, num_competitors, 64)

            logits = model(embeddings, queries=queries) # (batch, num_competitors, 2)

            loss = F.cross_entropy(logits.permute(0, 2, 1), gt_progress)
            progress = logits.argmax(dim=2) # (batch, num_competitors)
            acc = torch.round((progress == gt_progress).float().mean(), decimals=4) # (batch, num_competitors)
            print(f"loss: {loss}; acc: {acc}")

            torch.save(model.state_dict(), f"perceiver_ep{epoch}_acc{acc}.pt")



hook()
# loss = outputs.loss