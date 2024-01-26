import numpy as np
from tqdm import tqdm

# distances are the calculated distances of the data points
distances = []

# compute flows based on distance with a decay parameter
def compute_flows(distance, decay=10):
    return np.exp(-decay * distance)

# sort the scores to get their sorted indices
sorted_ids = np.argsort(distances)

# here you can set the threshold value n_next_hubs
n_next_hubs = 100

# iterative flow search
inward_flow_results = np.zeros(len(distances))
for id in tqdm(range(len(distances))):
    idx = sorted_ids[id][1:(n_next_hubs + 1)]
    inward_flow_results[id] += compute_flows(distances[id, idx]).sum()

# normalization and transformation
rarity_score_flow = inward_flow_results
min_score = min(rarity_score_flow)
max_score = max(rarity_score_flow)
rarity_score_flow = (rarity_score_flow - min_score) / (max_score - min_score)