# from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
# import torchaudio.transforms as T
from datasets import load_dataset
import hook

from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    # target_modules=
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)


# loading our model weights
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)


model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
hook()

# load demo audio and set processor
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

resample_rate = processor.sampling_rate
# make sure the sample_rate aligned
if resample_rate != sampling_rate:
    print(f'setting rate from {sampling_rate} to {resample_rate}')
    resampler = T.Resample(sampling_rate, resample_rate)
else:
    resampler = None

# audio file is decoded on the fly
if resampler is None:
    input_audio = dataset[0]["audio"]["array"]
else:
  input_audio = resampler(torch.from_numpy(dataset[0]["audio"]["array"]).float())
  
inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hook()
# take a look at the output shape, there are 25 layers of representation
# each layer performs differently in different downstream tasks, you should choose empirically
all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
print(all_layer_hidden_states.shape) # [25 layer, Time steps, 1024 feature_dim]

# for utterance level classification tasks, you can simply reduce the representation in time
time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
print(time_reduced_hidden_states.shape) # [25, 1024]

# you can even use a learnable weighted average representation
aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1)
weighted_avg_hidden_states = aggregator(time_reduced_hidden_states.unsqueeze(0)).squeeze()
print(weighted_avg_hidden_states.shape) # [1024]


