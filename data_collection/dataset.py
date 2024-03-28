import os, sys, math, itertools
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from collections import OrderedDict, Counter

import hook
import pandas as pd
from ..scripts.utils import *


NOVICE_PATH = '/import/c4dm-datasets/PianoJudge/novice/metadata.csv'
ADVANCED_PATH = '/import/c4dm-datasets/PianoJudge/advanced/metadata.csv'
VIRTUOSO_PATH = '/import/c4dm-datasets/ATEPP-audio/ATEPP-meta-audio.csv'
DIFFICULTYAT_PATH = '/import/c4dm-datasets/ATEPP/atepp_metadata_difficulty_henle_.csv'
DIFFICULTYMK_PATH = '/import/c4dm-datasets/PianoJudge/difficulty_mk/metadata.csv'
DIFFICULTYCP_PATH = '/import/c4dm-scratch-02/difficulty_cipi/CIPI_youtube_links.csv'
DIFFICULTYCP_DIR = '/import/c4dm-scratch-02/difficulty_cipi/'
TECHNIQUE_PATH = '/import/c4dm-datasets/PianoJudge/techniques/metadata.csv'
TECHNIQUE_DIR = '/import/c4dm-datasets/PianoJudge/techniques/'
ATEPP_DIR = '/import/c4dm-datasets/ATEPP-audio/'

ICPC_PATH = '/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/wav/metadata_.csv'
ICPC_RESULT_PATH = "/import/c4dm-datasets/ICPC2015-dataset/data/results.tsv"


class ExpertiseDataloader:
    def __init__(self, mode='train', split_ratio=0.8, pair_mode='all', num_classes=2):
        self.novice_data = pd.read_csv(NOVICE_PATH)
        self.advanced_data = pd.read_csv(ADVANCED_PATH)
        self.virtuoso_data = pd.read_csv(VIRTUOSO_PATH)

        # limit the duration of the piece to 5 minutes
        self.novice_data = self.novice_data[self.novice_data['duration'].astype(int) < 400]
        self.advanced_data = self.advanced_data[self.advanced_data['duration'].astype(int) < 400]
        self.virtuoso_data = self.virtuoso_data[self.virtuoso_data['audio_duration'].astype(int) < 300]

        self.pair_mode = pair_mode
        self.num_classes =num_classes
        self.balance_data()

        # Split the data into train and test (on audio not on pairs)
        min_len = min(len(self.novice_data), len(self.advanced_data), len(self.virtuoso_data))
        split_index = int(min_len * split_ratio)

        if mode == 'train':
            self.novice_data = self.novice_data[:split_index]
            self.advanced_data = self.advanced_data[:split_index]
            self.virtuoso_data = self.virtuoso_data[:split_index]
        elif mode == 'test':
            self.novice_data = self.novice_data[split_index:]
            self.advanced_data = self.advanced_data[split_index:]
            self.virtuoso_data = self.virtuoso_data[split_index:]


        if num_classes == 2 or num_classes == 4: # only comparing intra-groups
            novice_data = [('/import/c4dm-datasets/PianoJudge/novice/' + row['id'] + ".wav", 0) for _, row in self.novice_data.iterrows()]
            advanced_data = [('/import/c4dm-datasets/PianoJudge/advanced/' + row['id'] + ".wav", 1) for _, row in self.advanced_data.iterrows()]
            virtuoso_data = [(ATEPP_DIR + row['audio_path'], 2) for _, row in self.virtuoso_data.iterrows()]

            if self.pair_mode == 'all':
                self.pairs = self.create_all_pairs([novice_data, advanced_data, virtuoso_data])
            elif self.pair_mode == 'once':
                self.pairs = self.create_once_pairs([novice_data, advanced_data, virtuoso_data])
        elif num_classes == 3:
            self.pairs = self.create_pairs()

        random.shuffle(self.pairs) # if not shuffle, the train and test would have different pairs (and their inversion) - actually find there is no effect..



    def balance_data(self):
        # Find the size of the smallest group
        min_size = min(len(self.novice_data), len(self.advanced_data), len(self.virtuoso_data))

        # Randomly sample each group to match the size of the smallest group
        self.novice_data = self.novice_data.sample(n=min_size, random_state=42)
        self.advanced_data = self.advanced_data.sample(n=min_size, random_state=42)
        self.virtuoso_data = self.virtuoso_data.sample(n=min_size, random_state=42)


    # Function to create all possible pairs
    def create_all_pairs(self, groups):
        pairs = []
        for group1, group2 in [(groups[0], groups[1]), (groups[1], groups[2]), (groups[0], groups[2])]:
            for item1 in group1:
                for item2 in group2:
                    pairs.append(((item1[0], item2[0]), 0 if item1[1] > item2[1] else 1))
                    pairs.append(((item2[0], item1[0]), 1 if item1[1] > item2[1] else 0))
        return pairs

    # Function to create pairs such that each recording is included at least once
    def create_once_pairs(self, groups):
        pairs = []
        for group1, group2 in [(groups[0], groups[1]), (groups[1], groups[2]), (groups[0], groups[2])]:
            # Combine the groups into pairs and shuffle
            combined = zip(group1, group2)

            for (piece1, level1), (piece2, level2) in combined:
                if self.num_classes == 2:
                    pairs.append(((piece1, piece2), 0 if level1 > level2 else 1))
                    pairs.append(((piece2, piece1), 1 if level1 > level2 else 0))
                elif self.num_classes == 4:
                    level_diff = level2 - level1  # -2, -1, 1, 2
                    rank_tag = (level_diff + 2) if level_diff < 0 else (level_diff + 1) # -2, -1, 1, 2 -> 0, 1, 2, 3
                    pairs.append(((piece1, piece2), rank_tag))
                    pairs.append(((piece2, piece1), 3 - rank_tag))

        return pairs


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
            # reversed_pairs = [(b, a) for a, b in pairs]
            # all_pairs = pairs + reversed_pairs
            all_pairs = pairs
        elif self.pair_mode == 'once':
            shuffled_data = random.sample(combined_data, len(combined_data))   # note: the random seed would be fixed in each run
            it = iter(shuffled_data)
            all_pairs = list(zip(it, it))

        # Assign labels to pairs: 1 if first is better, -1 if second is better, 0 if same level
        labeled_pairs = []
        for (path1, level1), (path2, level2) in all_pairs:
            label = 1
            if level1 > level2:
                label = 0
            elif level1 < level2:
                label = 2
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
    def __init__(self, split_ratio=0.5, pair_mode='all', num_classes=2, mode='train'):

        self.metadata = pd.read_csv(ICPC_PATH)
        self.metadata = self.metadata[self.metadata['ranking_score'] != -1] # remove rows with no ranking score
        self.num_classes = num_classes

        split_index = int(len(self.metadata) * split_ratio)
        if mode == 'train':
            self.metadata = self.metadata[:split_index]
        elif mode == 'test':
            self.metadata = self.metadata[split_index:]

        self.pair_mode = pair_mode
        self.pairs = self.create_pairs()


    def create_pairs(self):
        return_pairs = []
        label_counter = Counter()

        if self.pair_mode == 'all':
            pairs = list(itertools.combinations(self.metadata.iterrows(), 2))
            all_pairs = random.sample(pairs, len(pairs)) # 1596

        elif self.pair_mode == 'once':
            shuffled_data = random.sample(list(self.metadata.iterrows()), len(self.metadata))
            it = iter(shuffled_data)
            all_pairs = list(zip(it, it)) # 28

        for (idx1, row1), (idx2, row2) in all_pairs:
            path1 = '/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/wav/' + row1['id'] + ".wav"
            path2 = '/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/wav/' + row2['id'] + ".wav"
            score1 = row1['ranking_score']
            score2 = row2['ranking_score']

            # Determine the label based on ranking_score
            if score1 > score2:
                label = 0
            elif score1 < score2:
                label = 1 if self.num_classes == 2 else 2
            else:
                label = 1
                if self.num_classes == 2:
                    continue

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


class DifficultyATDataloader:
    def __init__(self, mode='train', split_ratio=0.8, class_size=100, rs=42):
        # Read the metadata CSV file
        metadata = pd.read_csv(DIFFICULTYAT_PATH)
        metadata = metadata[~metadata['difficulty_label'].isna()]  # Remove rows with no difficulty label
        metadata = metadata[metadata['audio_duration'] < 400]

        # Sample for class balance regarding difficulty_label
        class_counts = metadata['difficulty_label'].value_counts()
        sampled_metadata = metadata.groupby('difficulty_label').apply(lambda x: x.sample(class_size, replace=True, random_state=rs))

        # Split based on the piece/movement to avoid leaking
        piece_paths = sampled_metadata['piece_path'].unique()
        random.Random(rs).shuffle(piece_paths)
        total_pieces = len(piece_paths)
        split_index = int(total_pieces * split_ratio)

        if mode == 'train':
            train_pieces = piece_paths[:split_index]
            self.metadata = sampled_metadata[sampled_metadata['piece_path'].isin(train_pieces)]
        elif mode == 'test':
            test_pieces = piece_paths[split_index:]
            self.metadata = sampled_metadata[sampled_metadata['piece_path'].isin(test_pieces)]


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        p = ATEPP_DIR + self.metadata.iloc[idx]['audio_path']
        label = int(self.metadata.iloc[idx]['difficulty_label'])
        return {
            "audio_path": p,
            "label": label
        }


class DifficultyMKDataloader:
    def __init__(self, mode='train', split_ratio=0.8, class_size=100, rs=42):
        # Read the metadata CSV file
        metadata = pd.read_csv(DIFFICULTYMK_PATH)

        # Sample for class balance regarding difficulty_label
        class_counts = metadata['difficulty_label'].value_counts()
        sampled_metadata = metadata.groupby('difficulty_label').apply(lambda x: x.sample(class_size, replace=True, random_state=rs))

        # Split based on the piece/movement to avoid leaking
        piece_paths = sampled_metadata['piece_path'].unique()
        random.Random(rs).shuffle(piece_paths)
        total_pieces = len(piece_paths)
        split_index = int(total_pieces * split_ratio)

        if mode == 'train':
            train_pieces = piece_paths[:split_index]
            self.metadata = sampled_metadata[sampled_metadata['piece_path'].isin(train_pieces)]
        elif mode == 'test':
            test_pieces = piece_paths[split_index:]
            self.metadata = sampled_metadata[sampled_metadata['piece_path'].isin(test_pieces)]


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        p = ATEPP_DIR + self.metadata.iloc[idx]['audio_path']
        label = int(self.metadata.iloc[idx]['difficulty_label'])
        return {
            "audio_path": p,
            "label": label
        }


class DifficultyCPDataloader:
    def __init__(self, mode='train', split_ratio=0.8, num_classes=9, rs=42):

        self.num_classes = num_classes # can be either 9 or 3

        # Read the metadata CSV file
        metadata = pd.read_csv(DIFFICULTYCP_PATH)
        # metadata = metadata.sample(frac=1, random_state=rs)

        # Sample for class balance regarding difficulty_label
        class_counts = metadata['henle'].value_counts()
        balanced_metadata = metadata.groupby('henle').apply(lambda x: x.sample(class_counts.max(), replace=True, random_state=rs))

        if mode == 'train': # we don't do cross-validation
            self.metadata = balanced_metadata[(balanced_metadata['split'] == 'train')]
        elif mode == 'test':
            self.metadata = balanced_metadata[balanced_metadata['split'] == 'test']

        self.metadata = self.metadata.sample(frac=1, random_state=rs)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        p = DIFFICULTYCP_DIR + self.metadata.iloc[idx]['audio_path']
        label = int(self.metadata.iloc[idx]['henle']) - 1
        if self.num_classes == 3:
            label = int(label / 3)
        return {
            "audio_path": p,
            "label": label
        }



class TechniqueDataloader:
    def __init__(self, mode='train', split_ratio=0.8, rs=42, label='multi'):
        # Read the metadata CSV file
        metadata = pd.read_csv(TECHNIQUE_PATH)
        metadata = metadata.sample(frac=1, random_state=rs)

        self.label = label
        self.label_columns = ['Scales', 'Arpeggios', 'Ornaments', 'Repeatednotes', 'Doublenotes', 'Octave', 'Staccato']

        split_index = int(len(metadata) * split_ratio)

        if mode == 'train':
            train_pieces = metadata[:split_index]
            self.metadata = train_pieces
        elif mode == 'test':
            test_pieces = metadata[split_index:]
            self.metadata =  test_pieces
        


    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        
        p = TECHNIQUE_DIR + self.metadata.iloc[idx]['id'] + '.wav'

        if self.label == 'multi':
            labels = list(self.metadata.iloc[idx][self.label_columns])
        elif self.label == 'single':
            # labels = np.argmax(self.metadata.iloc[idx][self.label_columns])
            labels = list(self.metadata.iloc[idx][self.label_columns])
        return {
            "audio_path": p,
            "label": labels
        }


if __name__ == "__main__":

    # Create dataloader
    dataloader = ExpertiseDataloader()
    # dataloader = DifficultyATDataloader()
    dataloader = DifficultyCPDataloader()
    # dataloader = ICPCDatalocader()
    dataloader = ExpertiseDataloader()

    # Get pairs
    pairs = dataloader.get_data()

    # Example usage: print first 5 pairs
    for pair in pairs[:5]:
        print(pair)
