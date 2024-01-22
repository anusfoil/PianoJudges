import librosa
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
# import torchaudio
# import torchaudio.transforms as T
from torch.utils.data import DataLoader, Sampler

import soundfile as sf
from scipy.signal import resample


from tqdm import tqdm
import  hook

################# Utility function for Jukebox ###################


JUKEBOX_SAMPLE_RATE = 44100  # Hz
SEG_DUR = 2205000
BATCH_SIZE = 2
MINIBATCH = 8
N_COMPETITIOR = 120

# utility functions from https://github.com/ethman/tagbox

def setup_jbx(model, device, levels=3, sample_length=1048576):
    """Sets up the Jukebox VQ-VAE."""

    from jukebox.make_models import make_vqvae, MODELS
    from jukebox.hparams import setup_hparams, Hyperparams, DEFAULTS

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

    return torch.tensor(audio, device=device)


def load_audio_for_jbx(path, offset=0.0, dur=None, trunc_sec=None, device='cpu'):

    if 'mp3' in path:
        """Loads a path for use with Jukebox."""
        audio, sr = librosa.load(path, sr=None, offset=offset, duration=dur)

        if sr != JUKEBOX_SAMPLE_RATE:
            audio = librosa.resample(audio, sr, JUKEBOX_SAMPLE_RATE)

        return audio_for_jbx(audio, trunc_sec, device=device)

    # Load audio file. 'sf.read' returns both audio data and the sample rate
    audio, sr = sf.read(path, dtype='float32')

    # Handle offset and duration
    if offset or dur:
        start_sample = int(offset * sr)
        end_sample = start_sample + int(dur * sr) if dur else None
        audio = audio[start_sample:end_sample]

    # Resample if necessary
    if sr != JUKEBOX_SAMPLE_RATE:
        num_samples = round(len(audio) * float(JUKEBOX_SAMPLE_RATE) / sr)
        audio = resample(audio, num_samples)


    return  audio_for_jbx(audio, trunc_sec, device=device)


def encode(vqvae, x):
    """Encode audio, `x`, to an unquantized embedding using `vqvae`."""
    x_in = vqvae.preprocess(x)
    # for level in range(vqvae.levels):
    #     print('here')
    #     encoder = vqvae.encoders[level]
    #     x_out = encoder(x_in)
    #     xs.append(x_out[-1])

    # only using the most coerse encodings
    encoder = vqvae.encoders[2]
    x_out = encoder(x_in)


    return x_out[-1]




################# Utility function for MERT ###################



