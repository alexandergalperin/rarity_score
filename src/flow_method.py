import numpy as np
from tqdm import tqdm

def calculate_rarity_scores_flow(distances, n_next_hubs, decay=10):
    # Ensure distances is a numpy array for consistent processing
    distances = np.array(distances)
    
    # Compute flows based on distance with a decay parameter
    def compute_flows(distance):
        return np.exp(-decay * distance)

    # Sort the scores to get their sorted indices
    sorted_ids = np.argsort(distances, axis=1)

    # Iterative flow search
    inward_flow_results = np.zeros(len(distances))
    for id in tqdm(range(len(distances)), disable=True):  # disable tqdm in tests
        idx = sorted_ids[id][1:(n_next_hubs + 1)]
        inward_flow_results[id] += compute_flows(distances[id, idx]).sum()

    # Normalization and transformation
    rarity_score_flow = inward_flow_results
    min_score = np.min(rarity_score_flow)
    max_score = np.max(rarity_score_flow)
    rarity_score_flow = (rarity_score_flow - min_score) / (max_score - min_score)
    
    return rarity_score_flow
