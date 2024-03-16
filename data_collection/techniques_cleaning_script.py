import hook
import os, glob
import audioread
import pandas as pd


def groups_clean():
    groups_file = "/homes/hz009/Research/PianoJudge/data_collection/technique_groups.txt"

    with open(groups_file) as f:
        lines = f.readlines()

    metadata = pd.read_csv('/import/c4dm-datasets/PianoJudge/techniques/metadata.csv')
    metadata['technique'] = ''

    cur_flag = ''
    for line in lines:
        if line[2:].strip() in ['Scales', 'Arpeggios', 'Ornaments', 'Repeatednotes', 'Doublenotes', 'Octave', 'Staccato']:
            cur_flag = line[2:].strip()
            print(cur_flag)
        
        if '#' not in line:
            id = line.split('v=')[-1].strip()
            if 'shorts' in id:
                id = id.split('shorts/')[-1].strip()
            metadata.loc[metadata['id'] == id, 'technique'] = cur_flag
            print(f"{id} set technique to {cur_flag}")

    # sort the metadata by technique
    metadata = metadata.sort_values(by=['technique'])

    # put the technique column as the second column 
    cols = metadata.columns.tolist()
    # cols = cols[:1] + cols[-1:] + cols[1:-1]
    metadata = metadata[cols]

    metadata.to_csv('/import/c4dm-datasets/PianoJudge/techniques/metadata.csv', index=False)


    
def add_self_recorded():
    metadata = pd.read_csv('/import/c4dm-datasets/PianoJudge/techniques/metadata.csv')
    wav_paths = glob.glob("/import/c4dm-datasets/PianoJudge/techniques/*.wav")
    for wav_path in wav_paths:
        for technique in ['Scales', 'Arpeggios', 'Ornaments', 'Repeatednotes', 'Doublenotes', 'Octave', 'Staccato']:
            if technique in wav_path:
                wav_path_ = wav_path
                os.system(f"mv '{wav_path}' '{wav_path_}'")

                with audioread.audio_open(wav_path_) as f:
                    totalsec = f.duration       
                
                wav_path_ = wav_path_.replace("/import/c4dm-datasets/PianoJudge/techniques/", '')
                techniques = wav_path_.split("_")[0]
                metadata = metadata.append({'id': wav_path_[:-4], 'technique': techniques, "title": wav_path_, "duration": totalsec}, ignore_index=True)
                break
    
    metadata.to_csv('/import/c4dm-datasets/PianoJudge/techniques/metadata.csv', index=False)
                
    

def onehot_techniques():
    metadata = pd.read_csv('/import/c4dm-datasets/PianoJudge/difficulty_cipi/CIPI_youtube_links.csv')
    # metadata = pd.read_csv('/import/c4dm-datasets/PianoJudge/techniques/metadata.csv')

    import audioread
    from tqdm import tqdm
    tol_duration = 0
    # load all of the audios
    for _, row in tqdm(metadata.iterrows()):
        path = '/import/c4dm-datasets/PianoJudge/difficulty_cipi/' + row['audio_path'] 
        with audioread.audio_open(path) as f:
            tol_duration += f.duration
        
    hook()
    # one-hot encode the techniques

    # Techniques to one-hot encode
    techniques = ['Scales', 'Arpeggios', 'Ornaments', 'Repeatednotes', 'Doublenotes', 'Octave', 'Staccato']

    # One-hot encode the 'technique' column
    for technique in techniques:
        metadata[technique] = metadata['technique'].str.contains(technique).astype(int)

    # print count for each technique
    for technique in techniques:
        print(f"{technique}: {metadata[technique].sum()}")
    '''
    Scales: 48
    Arpeggios: 40
    Ornaments: 31
    Repeatednotes: 35
    Doublenotes: 36
    Octave: 35
    Staccato: 41
    '''

    # print the count of of techniques that have more than one one-hot encoding
    print(metadata[techniques].sum(axis=1).value_counts())
    

    hook()
    return

# add_self_recorded()
onehot_techniques()