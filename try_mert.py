# from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
from torchaudio import transforms
from datasets import load_dataset
from data_collection.dataset import PerformanceDataloader
import hook

# loading our model weights
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)


def resample_for_mert(audio_array):
    # load demo audio and set processor
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    resample_rate = processor.sampling_rate
    # make sure the sample_rate aligned
    if resample_rate != sampling_rate:
        print(f'setting rate from {sampling_rate} to {resample_rate}')
        resampler = transforms.Resample(sampling_rate, resample_rate)
    else:
        resampler = None

    # audio file is decoded on the fly
    if resampler is None:
        input_audio = audio_array
    else:
        input_audio = resampler(torch.from_numpy(audio_array).float())
    

    return input_audio


hook()
pdl = PerformanceDataloader(mode='train')


inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# take a look at the output shape, there are 25 layers of representation
# each layer performs differently in different downstream tasks, you should choose empirically
all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
print(all_layer_hidden_states.shape) # [25 layer, Time steps, 1024 feature_dim]

# for utterance level classification tasks, you can simply reduce the representation in time
time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
print(time_reduced_hidden_states.shape) # [25, 1024]

