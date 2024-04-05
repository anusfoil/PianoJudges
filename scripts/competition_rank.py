import pandas as pd
from tqdm import tqdm
import hook

metadata = pd.read_csv("/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/wav/metadata_.csv")
pairs_prediction  = pd.read_csv("/homes/hz009/Research/PianoJudge/checkpoints/rank_test_dac_2_True_predictions.csv")

# map the paths in prediction table to names
pairs_prediction['path_1'] = pairs_prediction['path_1'].apply(lambda x: x.split('/')[-1][:-4])
pairs_prediction['path_2'] = pairs_prediction['path_2'].apply(lambda x: x.split('/')[-1][:-4])

pairs_prediction['candidate_1'] = pairs_prediction['path_1'].apply(lambda x: metadata[metadata['id'] == x]['title'].values[0].split(" – ")[0])
pairs_prediction['candidate_2'] = pairs_prediction['path_2'].apply(lambda x: metadata[metadata['id'] == x]['title'].values[0].split(" – ")[0])

pairs_prediction.to_csv("/homes/hz009/Research/PianoJudge/pairs_prediction.csv", index=False)


# test pairs consistency: if rank(a, b) = 1 then rank(b, a) = 0. if rank(a, b) = 1 and rank(b, c) = 1 then rank(a, c) = 1

# Define a function to check direct consistency
def check_direct_consistency(df):
    inconsistent_pairs = []
    consistent_pairs = []
    for index, row in tqdm(df.iterrows()):
        candidate1 = row['candidate_1']
        candidate2 = row['candidate_2']
        predicted_label = row['predicted_labels']
        
        # Check for the inverse pair in the dataframe
        inverse_label = df.loc[(df['candidate_1'] == candidate2) & (df['candidate_2'] == candidate1), 'predicted_labels'].values
        if len(inverse_label) > 0 and predicted_label == inverse_label[0]:
            inconsistent_pairs.append((candidate1, candidate2))
        if len(inverse_label) > 0 and predicted_label != inverse_label[0]:
            consistent_pairs.append((candidate1, candidate2))
    return inconsistent_pairs, consistent_pairs

# Define a function to check transitive consistency
def check_transitive_consistency(df):
    consistent_triplets = []
    inconsistent_triplets = []
    for index, row in tqdm(df.iterrows()):
        candidate1 = row['candidate_1']
        candidate2 = row['candidate_2']
        predicted_label = row['predicted_labels']
        
        # If rank(a, b) = 1, look for a b, c pair
        if predicted_label == 1:
            subsequent_pairs = df.loc[(df['candidate_1'] == candidate2) & (df['predicted_labels'] == 1)]
            
            for _, subsequent_row in subsequent_pairs.iterrows():
                candidate3 = subsequent_row['candidate_2']
                # Check if there exists an a, c pair with rank = 1
                if not df.loc[(df['candidate_1'] == candidate1) & (df['candidate_2'] == candidate3) & (df['predicted_labels'] == 1)].empty:
                    consistent_triplets.append((candidate1, candidate2, candidate3))
                    continue
                else:
                    inconsistent_triplets.append((candidate1, candidate2, candidate3))
    return inconsistent_triplets, consistent_triplets

def check_consistency():
    # Perform the checks
    inconsistent_pairs, consistent_pairs = check_direct_consistency(pairs_prediction)
    inconsistent_triplets, consistent_triplets = check_transitive_consistency(pairs_prediction)

    # Display results
    print("Inconsistent Pairs (Direct):", len(inconsistent_pairs), len(consistent_pairs))
    print("Inconsistent Triplets (Transitive):", len(inconsistent_triplets), len(consistent_triplets))




def select_candidates_for_stage(df, eligible_candidates, stage):
    # Filter for pairs where both candidates are eligible for the stage
    stage_df = df[(df['candidate_1'].isin(eligible_candidates)) & (df['candidate_2'].isin(eligible_candidates))]
    
    # Count how often each candidate was ranked first
    rank_counts = stage_df[stage_df['predicted_labels'] == 0]['candidate_1'].value_counts() + \
                  stage_df[stage_df['predicted_labels'] == 1]['candidate_2'].value_counts()
    rank_counts = rank_counts.fillna(0).astype(int)
    
    # Determine the number of candidates to select based on the stage
    num_to_select = len(eligible_candidates) // 2 if stage != 'Final' else 6
    
    # Select the top candidates based on rank count
    selected_candidates = rank_counts.nlargest(num_to_select).index.tolist()
    
    if stage == 'Stage I':
        hook()
    
    return selected_candidates

def get_compeitition_results():
    
    # Initialize eligible candidates for the preliminary round
    eligible_candidates = pd.unique(pairs_prediction[['candidate_1', 'candidate_2']].values.ravel('K'))

    # Initialize the results DataFrame
    results = pd.DataFrame({'Competitor': eligible_candidates})
    for stage in ['Prelim.', 'Stage I', 'Stage II', 'Stage III', 'Final']:
        results[stage] = 0

    # Loop through each stage
    stages = ['Prelim.', 'Stage I', 'Stage II', 'Stage III']
    for stage in stages:
        # Select candidates for the current stage
        passing_candidates = select_candidates_for_stage(pairs_prediction, eligible_candidates, stage)
        
        # Update the results DataFrame to indicate passing candidates
        results.loc[results['Competitor'].isin(passing_candidates), stage] = 1
        
        # Set the eligible candidates for the next stage
        eligible_candidates = passing_candidates

    # Select the final top 6 candidates
    final_candidates = select_candidates_for_stage(pairs_prediction, eligible_candidates, 'Final')
    results.loc[results['Competitor'].isin(final_candidates), 'Final'] = 1

    # Save the results to a CSV file
    results.to_csv('predicted_competition_results.csv', index=False)


if __name__ == "__main__":
    # check_consistency()
    get_compeitition_results()

hook()