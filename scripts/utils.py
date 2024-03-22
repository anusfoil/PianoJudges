import os, glob
import numpy as np
import torch
import pandas as pd
from torch.optim import Adam
# import torchaudio
# import torchaudio.transforms as T
from torch.utils.data import DataLoader, Sampler
import h5py
import torch.distributed as dist

from scipy.signal import resample
from einops import rearrange, reduce, repeat

import hook
from tqdm import tqdm
from ..data_collection.dataset import *

################# Utility function for encoders ###################


def compute_mert_embeddings(audio_path, model, processor, resample_rate, segment_duration=10, max_segs=None, compute_all=False):
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
    if max_segs and (not compute_all):
        num_segments = min(num_segments, max_segs)
    for i in range(num_segments):

        offset = i * segment_duration
        audio_seg = audio[offset * resample_rate: (offset + segment_duration) * resample_rate]
        input_features = processor(audio_seg.squeeze(0), sampling_rate=resample_rate, return_tensors="pt", padding=True).to(model.device)

        try:
            with torch.no_grad():
                outputs = model(**input_features, output_hidden_states=True)
        except:
            print(f"embedding failed to compute for {audio_path}, seg {i}")

        all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze().to('cpu') 
        embeddings.append(all_layer_hidden_states[-1, :, :])

    if num_segments == 0:
        embeddings = [torch.zeros(749, 1024)]

    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    if compute_all:
        end = embeddings.shape[0] - embeddings.shape[0] % 30
        embeddings = rearrange(embeddings[:end], '(p s) t e -> p s t e', s = 30)
    
    assert(embeddings.shape[-2:] == torch.Size([749, 1024]))
    return embeddings # (n_segs, 749, 1024)  For some reason the output missed timestep. Should be 75 as frame rate.


def compute_jukebox_embeddings(audio_path, vqvae, device_, segment_duration=10, max_segs=None, compute_all=False):
    JUKEBOX_SAMPLE_RATE = 44100

    try: # in case there is problem with audio loading.
        audio = load_audio_for_jbx(audio_path)
    except Exception as e:
        print(e)
        print(f"audio failed to read for {audio_path}")
        return torch.zeros(1, 3445, 64) # 10s of empty embedding

    if compute_all: # slice the audio into 5mins segments
        audios = [audio[i * JUKEBOX_SAMPLE_RATE: (i + 300) * JUKEBOX_SAMPLE_RATE] for i in range(0, int(len(audio) / JUKEBOX_SAMPLE_RATE) - 300, 300)]
        all_embeddings = []
    else:
        audios = [audio]

    for audio in audios:
        embeddings = []
        num_segments = int(len(audio) / JUKEBOX_SAMPLE_RATE // segment_duration)
        if max_segs:
            num_segments = min(num_segments, max_segs)
        for i in range(num_segments):
            offset = i * segment_duration
            audio_seg = audio[offset * JUKEBOX_SAMPLE_RATE: (offset + segment_duration) * JUKEBOX_SAMPLE_RATE]
            audio_seg =  rearrange(audio_seg, "t -> 1 t 1").to(device_)

            embeddings.append(encode(vqvae, audio_seg.contiguous()))

        if num_segments == 0:
            embeddings = [torch.zeros(1, 64, 3445)]

        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        embeddings = rearrange(embeddings, "b 1 e t -> b t e").to('cpu')

        if compute_all: all_embeddings.append(embeddings)

    if compute_all: embeddings = torch.stack(all_embeddings)

    assert(embeddings.shape[-2:] == torch.Size([3445, 64]))
    return embeddings # ((n_parts), n_segs, 3445, 64)


def compute_dac_embeddings(audio_path, dac_model, process_duration=1, segment_duration=10, max_segs=None, compute_all=False):
    from audiotools import AudioSignal

    if compute_all: max_segs = None

    try:
        # Load the full audio signal
        signal = AudioSignal(audio_path)
        signal.resample(44100)
        signal.to_mono()

        # Calculate the number of samples per segment and per concatenation
        samples_per_segment = process_duration * signal.sample_rate

        # Process in segments
        total_samples = signal.audio_data.shape[-1]
        num_segments = (total_samples + samples_per_segment - 1) // samples_per_segment  # Ceiling division
        if max_segs:
            # since dac process in 1s and have to concatenate 10s
            num_segments = min(num_segments, max_segs  * int(segment_duration / process_duration))
        embeddings = []

        signal.audio_data = signal.audio_data
        for i in range(num_segments):

            start = i * samples_per_segment
            end = min(start + samples_per_segment, total_samples)
            segment = signal.audio_data[:, :, start:end].to(dac_model.device)

            # Process each segment
            x = dac_model.preprocess(segment, signal.sample_rate)
            with torch.no_grad():
                z, codes, latents, commitment_loss, codebook_loss = dac_model.encode(x)
            embeddings.append(rearrange(z, '1 e t -> t e')) 
    except Exception as e:
        print(e)
        print(f"audio failed to read for {audio_path}")
        embeddings = [torch.zeros(1, 870, 1024)]

    embeddings =  torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    end = embeddings.shape[0] - embeddings.shape[0] % segment_duration
    final_embeddings = rearrange(embeddings[:end], '(s x) t e -> s (x t) e', x=segment_duration)
    if compute_all: 
        end = final_embeddings.shape[0] - final_embeddings.shape[0] % 30
        final_embeddings = rearrange(final_embeddings[:end], "(p s) t e -> p s t e", s = 30)

    assert(final_embeddings.shape[-2:] == torch.Size([870, 1024]))
    return final_embeddings  # ((n_parts), n_segs, 870, 1024)



def compute_audiomae_embeddings(audio_path, amae, fp, device_, segment_duration=10, max_segs=None, compute_all=False):
    import torchaudio
    
    if compute_all: max_segs = None

    try:
        audio, sr = torchaudio.load(audio_path)
        num_segments = int(len(audio[0]) / sr // segment_duration)
        if max_segs:
            num_segments = min(num_segments, max_segs)
    except:
        print(f"audio failed to read for {audio_path}")
        return torch.zeros(1, 512, 768)
        
    embeddings = []
    for i in range(num_segments):
        wav, fbank, _ = fp(audio_path, start_sec=i*segment_duration, dur_sec=segment_duration)

        fbank = rearrange(fbank, 'c t f -> 1 c t f').to(device_).to(torch.float16)
        with torch.no_grad():
            output = amae.get_audio_embedding(fbank)
        embeddings.append(output["patch_embedding"])

    if num_segments == 0:
        embeddings = [torch.zeros(1, 512, 768)]

    embeddings =  torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
    embeddings = rearrange(embeddings, "s 1 t e -> s t e")
    if compute_all:
        end = embeddings.shape[0] - embeddings.shape[0] % 30
        embeddings = rearrange(embeddings[:end], '(p s) t e -> p s t e', s = 30)

    assert(embeddings.shape[-2:] == torch.Size([512, 768]))
    return embeddings # ((n_parts), n_segs, 512, 768)



def compute_audio_embeddings(audio_path, audio_inits, method, device_, segment_duration=10, max_segs=None, compute_all=False):
    """_summary_

    Args:
        audio_path (str): path of audio file to compute
        audio_inits (dict): audio encoders 
        method (str): _description_
        segment_duration (int, optional): duration of 1 segment. Defaults to 10.
        max_segs (int, optional): 30 segments, with is 5mins. Defaults to None.
        compute_all (bool, optional): Compute all 5 mins segements instead of only the first one (for ICPC dataset). The output have one more dimension.

    """
    if method == 'jukebox':
        vqvae = audio_inits['audio_encoder']
        return compute_jukebox_embeddings(audio_path, vqvae, device_, segment_duration, max_segs, compute_all=compute_all)
    
    elif method == 'mert':
        mert_model = audio_inits['audio_encoder']
        mert_processor = audio_inits['processor']
        return compute_mert_embeddings(audio_path, mert_model, mert_processor, mert_processor.sampling_rate, segment_duration, max_segs, compute_all=compute_all)
    
    elif method == 'dac':
        dac_model = audio_inits['audio_encoder']
        return compute_dac_embeddings(audio_path, dac_model, segment_duration=segment_duration, max_segs=max_segs, compute_all=compute_all)
    
    elif method == 'audiomae':
        amae = audio_inits['audio_encoder']
        fp = audio_inits['fp']
        return compute_audiomae_embeddings(audio_path, amae, fp, device_, segment_duration, max_segs, compute_all=compute_all)
    
    else:
        raise ValueError("Invalid method specified")


def pad_and_clip_sequences(sequences, max_length):
    # Clip and pad the sequences
    clipped = [seq if len(seq) <= max_length else seq[:max_length] for seq in sequences]
    padded_sequences = torch.nn.utils.rnn.pad_sequence(clipped, batch_first=True)

    if padded_sequences.shape[-3] != max_length:
        padding_shape = list(padded_sequences.shape)
        padding_shape[-3] =  max_length - padded_sequences.shape[-3]
        # padding_shape = (padded_sequences.shape[0], max_length - padded_sequences.shape[1], padded_sequences.shape[2],  padded_sequences.shape[3])
        padded_sequences = torch.cat((padded_sequences, torch.zeros(padding_shape).to(padded_sequences.device)), dim=1)

    assert(padded_sequences.shape[-3] == max_length)
    return padded_sequences


def load_embedding_from_hdf5(audio_paths, encoder, device, max_segs=30, use_trained=False):

    embeddings = []

    for audio_path in audio_paths:
        parent_folder = "/".join(audio_path.split("/")[:-1])

        trained = "_trained" if use_trained else ""
        hdf5_path = os.path.join(parent_folder, f"{encoder}{trained}_embeddings.hdf5")
        if "ATEPP" in audio_path:
            hdf5_path = f"/import/c4dm-scratch-02/ATEPP-audio-embeddings/{encoder}{trained}_embeddings.hdf5"        

        with h5py.File(hdf5_path, 'r') as hdf5_file:
            audio_id = os.path.basename(audio_path).split('.')[0]
            # hook()
            if "ATEPP" in audio_path:
                audio_id = audio_path.replace("/import/c4dm-datasets/ATEPP-audio/", "")[:-4]
                dataset = traverse_long_id(hdf5_file, audio_id)
                if dataset is not None:
                    embedding = dataset[:]  # Use [:] to read the data if it's a dataset
                else:
                    print(f"Embedding for {audio_id} not found.")
                    embedding = torch.zeros(1, 870, 1024)
            else:
                if audio_id in hdf5_file:
                    embedding_np = hdf5_file[audio_id][:]
                    embedding = torch.from_numpy(embedding_np).to(device)
                else:
                    print(f"Embedding for {audio_id} not found.")
                    hook()
        embeddings.append(torch.tensor(embedding))
    
    embeddings = pad_and_clip_sequences(embeddings, max_segs)  # maximum 30 segs (5mins)
    return embeddings


def traverse_long_id(hdf5_file, long_id):
    """
    Fetches an embedding from an HDF5 file given a long hierarchical ID.
    
    Parameters:
    - hdf5_file: An open HDF5 file object or the root group of the file.
    - long_id: A string representing the hierarchical path to the dataset.
    
    Returns:
    - The dataset corresponding to the long_id, or None if not found.
    """
    # Split the long_id by slashes to get individual group names
    groups = long_id.split('/')
    
    # Navigate through the groups based on the split ID
    current_group = hdf5_file
    for group_name in groups:
        # Check if the current part of the path exists as a group or dataset
        if group_name in current_group:
            current_group = current_group[group_name]
        else:
            # If any part of the path doesn't exist, return None
            print(f"Path '{long_id}' not found in the HDF5 file.")
            return None
    
    # Return the final dataset or group reached
    return current_group



def init_encoder(encoder, device, use_trained=False):

    if encoder == 'jukebox':
        audio_encoder = setup_jbx('5b', device)
        return {"audio_encoder": audio_encoder}
    
    if encoder == 'mert':
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import AutoModel

        audio_encoder = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device)
        # loading the corresponding preprocessor config
        processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)
        return {"audio_encoder": audio_encoder, 
                "processor": processor}

    if encoder == 'dac':
        import dac
        if use_trained:
            model_path = '/homes/hz009/Research/descript-audio-codec/runs/baseline/best/dac/weights_best.pth'
        else:
            model_path = dac.utils.download(model_type="44khz")

        model = dac.DAC.load(model_path)

        model.to(device)
        return {'audio_encoder': model}
    

    if encoder == 'audiomae':
        from .audio_processor import _fbankProcessor
        from .audiomae_wrapper import AudioMAE

        if use_trained:
            ckpt_path = '/homes/hz009/audiomae_checkpoint/experiments/checkpoint-40.pth'
        else:
            ckpt_path = '/homes/hz009/Research/PianoJudge/AudioMAE/finetuned.pth'
        amae = AudioMAE.create_audiomae(ckpt_path=ckpt_path, device=device)

        return {'audio_encoder': amae,
                'fp': _fbankProcessor.build_processor()}
    


def encoding_shape(encoder):
    if encoder == 'jukebox':
        return (3445, 64)
    elif encoder == 'mert':
        return (749, 1024)
    elif encoder == 'dac':
        return (870, 1024)
    elif encoder == 'audiomae':
        return (512, 768)



def compute_all_embeddings(encoder, folder_with_wavs, metadata, save_path, device, audio_inits, use_trained=False, max_segs=30):
    hdf5_path = os.path.join(save_path, f"{encoder}_embeddings.hdf5")
    if use_trained:
        hdf5_path = os.path.join(save_path, f"{encoder}_trained_embeddings.hdf5")

    metadata = pd.read_csv(metadata)
    if (('ICPC' not in folder_with_wavs) 
        and ('technique' not in folder_with_wavs)
        and ('cipi' not in folder_with_wavs)): 
        if 'duration' in metadata.columns:
            metadata = metadata[metadata['duration'].astype(int) < 400]
        else:
            metadata = metadata[metadata['audio_duration'].astype(int) < 300]

    # Use 'a' mode for read/write access and creating the file if it doesn't exist
    with h5py.File(hdf5_path, 'a') as hdf5_file:
        if 'id' in metadata.columns:
            audio_ids = metadata['id'].tolist()
            audio_paths = [os.path.join(folder_with_wavs, aid + ".wav") for aid in audio_ids]
        else:
            audio_paths = metadata['audio_path'].tolist()
            audio_paths = [os.path.join(folder_with_wavs, aid) for aid in audio_paths]
        
        for audio_path in tqdm(audio_paths):
            audio_name = audio_path.replace(folder_with_wavs, '')[:-4]

            # Check if the embedding already exists
            if audio_name in hdf5_file:
                print(f"Embedding for {audio_name} already exists. Skipping computation.")
                continue

            # Compute the embeddings
            embedding = compute_audio_embeddings(audio_path, audio_inits, encoder, device, 
                                                 max_segs=max_segs, compute_all=('ICPC' in folder_with_wavs))
            embedding_np = embedding.cpu().detach().numpy()

            # Save the embedding
            try:
                hdf5_file.create_dataset(audio_name, data=embedding_np, compression="gzip")
                print(f"Saved embedding for {audio_name}.")
            except RuntimeError as e:
                print(f"Error saving embedding for {audio_name}: {e}")
 

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
    if len(audio.shape) > 1:
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



################### General Utilities ###########################


def load_latest_checkpoint(checkpoint_dir):
    """
    Load the latest checkpoint from the given directory.
    
    Parameters:
    - checkpoint_dir: Path to the directory containing the checkpoint files.
    
    Returns:
    - The loaded checkpoint, or None if the directory is empty or does not exist.
    """
    # Check if the checkpoint directory exists
    if not os.path.isdir(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None
    
    # List all files in the checkpoint directory
    all_checkpoints = os.listdir(checkpoint_dir)

    # Filter out files that are not checkpoints (do not end with '.ckpt')
    checkpoint_files = [f for f in all_checkpoints if f.endswith('.ckpt')]

    # Sort the checkpoints by modification time (newest first)
    checkpoint_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)

    if checkpoint_files:
        latest_checkpoint = checkpoint_files[0]
        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Loading latest checkpoint: {latest_checkpoint_path}")

        return latest_checkpoint_path
    else:
        print("No checkpoints found.")
        return None

def checkpointing_paths(cfg):
    
    trained = 't' if cfg.use_trained else 'nt'

    # experiment_name = f"{cfg.task}_{cfg.encoder}"
    experiment_name = f"{cfg.task}_{cfg.encoder}_{cfg.objective[0]}_{cfg.dataset.num_classes}_{trained}"
    checkpoint_dir = f"/homes/hz009/Research/PianoJudge/checkpoints/{experiment_name}"

    return experiment_name, checkpoint_dir


import torch
from torchmetrics import Metric

class PlusMinusOneAccuracy(Metric):
    def __init__(self, num_classes, average, dist=1):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.dist = dist
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Calculate if predictions are within the acceptable range
        correct = torch.abs(preds - target) <= self.dist
        # Update metric states
        self.correct += correct.sum()
        self.total += target.numel()

    def compute(self):
        # Compute the final accuracy
        return self.correct.float() / self.total

# Assuming the apply_label_smoothing function as defined previously
def apply_label_smoothing(targets, alpha=0.1):
    smoothed_targets = targets * (1 - alpha) + (1 - targets) * alpha
    return smoothed_targets


import hydra
from omegaconf import DictConfig

@hydra.main(config_path="/homes/hz009/Research/PianoJudge/conf/utils/", config_name='compute_embeddings')
def main(cfg: DictConfig):
    # Set the environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
    os.environ['MASTER_PORT'] = str(cfg.port)  # an open port
    os.environ['WORLD_SIZE'] = '1'           # total number of processes
    os.environ['RANK'] = '0'                 # rank of this process
    os.environ['HYDRA_FULL_ERROR'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f'cuda:{cfg.gpu}')

    for category in cfg.category:
        for encoder in cfg.encoder:
            audio_inits = init_encoder(encoder, device, use_trained=cfg.use_trained)
            compute_all_embeddings(encoder, cfg.category[category].folder_path, 
                                            cfg.category[category].metadata, 
                                            cfg.category[category].save_path, 
                                            device, audio_inits,
                                            use_trained=cfg.use_trained,
                                            max_segs=cfg.max_segs)
            print(f"Embeddings for {encoder} in {category} computed and saved.")


if __name__ == '__main__':

    main()



