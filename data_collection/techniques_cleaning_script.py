import hook
import pandas as pd

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

hook()

metadata.to_csv('/import/c4dm-datasets/PianoJudge/techniques/metadata.csv', index=False)


    