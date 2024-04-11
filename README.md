# PianoJudge

Code repository following paper *From Audio Encoders to Piano Judges: Benchmarking Performance Understanding for Solo Piano*. 

- [PianoJudge](#pianojudge)
  - [Environment](#environment)
  - [Data](#data)
      - [Fetching](#fetching)
      - [Embedding computation](#embedding-computation)
  - [Training for the three main tasks](#training-for-the-three-main-tasks)
  - [ICPC-2015 Prediction](#icpc-2015-prediction)


## Environment
```
pip install requirement.txt
```
For the Jukebox model, you would also need to install the [jukebox package](https://github.com/openai/jukebox) according to their doc. 

## Data 

The Pianism-labeling dataset (PLD) is a ~138 hours dataset featuring clips that's labeled with expertise level, difficulty (curated originally from [CIPI](https://zenodo.org/records/8037327) dataset), and solo piano technique. For a demonstration of data please refer to [project page](https://bit.ly/3SYzozY).  We provide metadata of youtube link correspondance.

#### Fetching
```
pyathon -m PianoJudge.data_collection.fetch
```
List of channels for novice, advanced, and virtuoso levels are found in ```data_collection/*_channels.txt```, please modify the paths in ```fetch.py```. Downloaded audio files can be also requested from the author.


#### Embedding computation
```
python -m PianoJudge.scripts.utils
```
Config can be found in ```conf/utils/compute_embeddings.yaml```. This saves the computed embeddings into ```hdf5``` files.
- -encoder: 'Jukebox' or 'MERT' or 'DAC' or 'AudioMAE'. 
- -max_segs: how many 10s segments is will be used to compute embedding. default 30 (5mins).
- -use_trained: whether to used fine-tuned DAC and AudioMAE. The checkpoints can be found [here](https://drive.google.com/drive/folders/11Sg_RA1RnvCm5zdP6_MOMS6WArQljHAE?usp=sharing). 
- category: set your dataset path and output path here.


## Training for the three main tasks

```
python -m PianoJudge.scripts.ranking
python -m PianoJudge.scripts.difficulty
python -m PianoJudge.scripts.technique
```
Config can be found in their respective path, e.g. ```conf/ranking.yaml```.
- -encoder: 'Jukebox' or 'MERT' or 'DAC' or 'AudioMAE'. Note that the previous step must have the embeddings saved as we don't support on-the-fly calculation. 
- -dataset.num_classes: number of classes in the respective task.


## ICPC-2015 Prediction
The International Chopin Piano Competition 2015 data is curated by [this repository](https://github.com/cyrta/ICPC2015-dataset). Similarly, all performances can be fetched.
```
python -m PianoJudge.scripts.ranking mode=test
python -m PianoJudge.scripts.competition_rank
```

Setting mode=test will inference on all possible pairs and save to ```checkpoints/rank_test_*_prediction.csv```. The following scripts cleaned up the prediction. 
