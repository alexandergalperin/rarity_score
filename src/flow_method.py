import numpy as np

def calculate_rarity_scores_flow(distances, n_next_hubs, decay=10):
    """
    Berechnet Rarity-Scores basierend auf Entfernungen, Fluss und der Anzahl der nächsten Hubs.

    Args:
        distances: Numpy-Array mit Distanzen.
        n_next_hubs: Anzahl der nächsten Hubs zur Berücksichtigung.
        decay: Parameter zur Steuerung des Flusses (Standardwert: 10).

    Returns:
        Numpy-Array mit Rarity-Scores.
    """

    if not distances:
        raise ValueError("Cannot calculate rarity scores with empty distances")

    if not isinstance(distances, np.ndarray):
        raise TypeError("Distances must be numeric")

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
        idx = sorted_ids[id][1:(n_next_hubs + 1)]
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