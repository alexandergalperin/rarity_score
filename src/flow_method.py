import numpy as np

def calculate_rarity_scores_flow(distances, n_next_hubs, decay=10):
    """
    Calculates rarity scores based on distances, flow, and the number of next hubs.

    Args:
        distances: NumPy array with distances (shape: (n_samples, n_features)).
        n_next_hubs: Number of next hubs to consider.
        decay: Parameter to control the flow (default: 10).

    Returns:
        NumPy array with rarity scores (shape: (n_samples,)).
    """

    if len(distances) == 0:
        raise ValueError("Cannot calculate rarity scores with empty distances")

    for distance_array in distances:
        for distance_number in distance_array:
            if not isinstance(distance_number, (int,float, np.number)):
                raise TypeError("Distances must be numeric NumPy arrays")

    if n_next_hubs > len(distances):
        raise ValueError("n_next_hubs cannot be greater than the number of distances")

    # Compute flows based on distance with a decay parameter
    def compute_flows(distance):
        return np.exp(-decay * distance)

    # Sort the scores to get their sorted indices
    sorted_ids = np.argsort(distances, axis=1)

    # Iterative flow search
    inward_flow_results = np.zeros(len(distances))
    for id in range(len(distances)):
        idx = sorted_ids[id][1:(n_next_hubs + 1)]  # Corrected indexing
        inward_flow_results[id] += compute_flows(distances[id, idx]).sum()

    # Normalization and transformation
    rarity_score_flow = inward_flow_results
    min_score = np.min(rarity_score_flow)
    max_score = np.max(rarity_score_flow)

    # Check if max_score equals min_score to prevent division by zero
    if max_score == min_score:
        # If all scores are the same, rarity scores should be 0
        rarity_score_flow = [0 for _ in rarity_score_flow]
    else:
        rarity_score_flow = [(score - min_score) / (max_score - min_score) for score in rarity_score_flow]

    return rarity_score_flow

