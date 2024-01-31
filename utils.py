import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
# import torchaudio
# import torchaudio.transforms as T
from torch.utils.data import DataLoader, Sampler

from scipy.signal import resample
from einops import rearrange, reduce, repeat


from tqdm import tqdm
import  hook

################# Utility function for encoders ###################


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
    final_embeddings = rearrange(embeddings[:end], '(s x) t e -> s (x t) e', x=concat_duration)

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

    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)  # (b s t e)

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
    


def encoder_input_dim(encoder):
    if encoder == 'jukebox':
        return 64
    elif encoder == 'mert':
        return 1024
    elif encoder == 'dac':
        return 1024
    elif encoder == 'audiomae':
        return 768



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

    import librosa
    import soundfile as sf
    if 'mp3' in path:
        """Loads a path for use with Jukebox."""
        audio, sr = librosa.load(path, sr=None, offset=offset, duration=dur)

        if sr != JUKEBOX_SAMPLE_RATE:
            audio = librosa.resample(audio, sr, JUKEBOX_SAMPLE_RATE)

        return audio_for_jbx(audio, trunc_sec, device=device)

    # Load audio file. 'sf.read' returns both audio data and the sample rate
    audio, sr = sf.read(path, dtype='float32')
    audio = reduce(audio, 't c -> t', 'mean')

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








