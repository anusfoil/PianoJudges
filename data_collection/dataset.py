import os, sys, math, itertools
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from collections import OrderedDict, Counter

from .. import hook


NOVICE_PATH = '/import/c4dm-datasets/PianoJudge/novice/metadata.csv'
ADVANCED_PATH = '/import/c4dm-datasets/PianoJudge/advanced/metadata.csv'
VIRTUOSO_PATH = '/import/c4dm-datasets/ATEPP-audio/ATEPP-meta-audio.csv'
DIFFICULTY_PATH = '/import/c4dm-datasets/ATEPP/atepp_metadata_difficulty_henle_.csv'
ATEPP_DIR = '/import/c4dm-datasets/ATEPP-audio/'

ICPC_PATH = '/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/wav/metadata_.csv'
ICPC_RESULT_PATH = "/import/c4dm-datasets/ICPC2015-dataset/data/results.tsv"


class ExpertiseDataloader:
    def __init__(self, mode='train', split_ratio=0.8, pair_mode='all'):
        self.novice_data = pd.read_csv(NOVICE_PATH)
        self.advanced_data = pd.read_csv(ADVANCED_PATH)
        self.virtuoso_data = pd.read_csv(VIRTUOSO_PATH)

        self.pair_mode = pair_mode

        self.balance_data()
        self.pairs = self.create_pairs()

        # Split the data into train and test
        total_pairs = len(self.pairs)
        split_index = int(total_pairs * split_ratio)

        if mode == 'train':
            self.pairs = self.pairs[:split_index]
        elif mode == 'test':
            self.pairs = self.pairs[split_index:]


    def balance_data(self):
        # Find the size of the smallest group
        min_size = min(len(self.novice_data), len(self.advanced_data), len(self.virtuoso_data))

        # Randomly sample each group to match the size of the smallest group
        self.novice_data = self.novice_data.sample(n=min_size, random_state=42)
        self.advanced_data = self.advanced_data.sample(n=min_size, random_state=42)
        self.virtuoso_data = self.virtuoso_data.sample(n=min_size, random_state=42)


    def create_pairs(self):
        # Combine all data with labels: 0 for novice, 1 for advanced, 2 for virtuoso
        combined_data = [
            ('/import/c4dm-datasets/PianoJudge/novice/' + row['id'] + ".wav", 0) for _, row in self.novice_data.iterrows()
        ] + [
            ('/import/c4dm-datasets/PianoJudge/advanced/' + row['id'] + ".wav", 1) for _, row in self.advanced_data.iterrows()
        ] + [
            (ATEPP_DIR + row['audio_path'], 2) for _, row in self.virtuoso_data.iterrows()
        ]

        # Create all possible pairs
        if self.pair_mode == 'all':
            pairs = list(itertools.combinations(combined_data, 2))
            reversed_pairs = [(b, a) for a, b in pairs]
            all_pairs = pairs + reversed_pairs
        elif self.pair_mode == 'once':
            shuffled_data = random.sample(combined_data, len(combined_data))
            it = iter(shuffled_data)
            all_pairs = list(zip(it, it))

        # Assign labels to pairs: 1 if first is better, -1 if second is better, 0 if same level
        labeled_pairs = []
        for (path1, level1), (path2, level2) in all_pairs:
            label = 0
            if level1 > level2:
                label = 1
            elif level1 < level2:
                label = -1
            labeled_pairs.append(((path1, path2), label))

        return labeled_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (p1, p2), label = self.pairs[idx]
        
        return {
            "audio_path_1": p1,
            "audio_path_2": p2,
            "label": label
        }


class ICPCDataloader:
    def __init__(self, split_ratio=0.8, pair_mode='all'):

        self.metadata = pd.read_csv(ICPC_PATH)
        self.metadata = self.metadata[self.metadata['ranking_score'] != -1] # remove rows with no ranking score

        self.pair_mode = pair_mode
        self.pairs = self.create_pairs()


    def create_pairs(self):
        return_pairs = []
        label_counter = Counter()

        if self.pair_mode == 'all':
            pairs = list(itertools.combinations(self.metadata.iterrows(), 2))
            reversed_pairs = [(b, a) for a, b in pairs]
            all_pairs = pairs + reversed_pairs
        elif self.pair_mode == 'once':
            shuffled_data = random.sample(list(self.metadata.iterrows()), len(self.metadata))
            it = iter(shuffled_data)
            all_pairs = list(zip(it, it))

        for (idx1, row1), (idx2, row2) in all_pairs:
            path1 = '/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/wav/' + row1['id'] + ".wav"
            path2 = '/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/wav/' + row2['id'] + ".wav"
            score1 = row1['ranking_score']
            score2 = row2['ranking_score']

            # Determine the label based on ranking_score
            if score1 > score2:
                label = 1
            elif score1 < score2:
                label = -1
            else:
                label = 0

            label_counter[label] += 1
            return_pairs.append(((path1, path2), label))

        print(f"Label Distribution: {dict(label_counter)}")

        return return_pairs
    

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (p1, p2), label = self.pairs[idx]
        
        return {
            "audio_path_1": p1,
            "audio_path_2": p2,
            "label": label
        }


class DifficultyDataloader:
    def __init__(self, mode='train', split_ratio=0.8):

        self.metadata = pd.read_csv(DIFFICULTY_PATH)
        self.metadata = self.metadata[~self.metadata['difficulty_label'].isna()] # remove rows with no difficulty label

        # Split based on the piece / movement to avoid leaking
        piece_paths = self.metadata['piece_path'].unique()
        random.shuffle(piece_paths)
        total_pieces = len(piece_paths)
        split_index = int(total_pieces * split_ratio)

        if mode == 'train':
            train_pieces = piece_paths[:split_index]
            self.metadata = self.metadata[self.metadata['piece_path'].isin(train_pieces)]
        elif mode == 'test':
            test_pieces = piece_paths[split_index:]
            self.metadata = self.metadata[self.metadata['piece_path'].isin(test_pieces)] 


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        p = ATEPP_DIR + self.metadata.iloc[idx]['audio_path']
        label = int(self.metadata.iloc[idx]['difficulty_label'])
        return {
            "audio_path": p,
            "label": label
        }



if __name__ == "__main__":

    # Create dataloader
    dataloader = DifficultyDataloader()
    # dataloader = ICPCDatalocader()
    # dataloader = PerformanceDataloader()

    # Get pairs
    pairs = dataloader.get_data()

    # Example usage: print first 5 pairs
    for pair in pairs[:5]:
        print(pair)
