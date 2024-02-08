import numpy as np

def calculate_rarity_scores(distances, n_neighbours):
    # Ensure distances is a numpy array for consistent processing
    distances = [np.array(distance_array) for distance_array in distances]

    if not distances:
        raise ValueError("Cannot calculate rarity scores with empty distances list")
    
    sorted_distances = [np.sort(distance_array)[:n_neighbours] for distance_array in distances]

    averages = [np.mean(score_array) for score_array in sorted_distances]

    min_score = min(averages)
    max_score = max(averages)

    # Check if max_score equals min_score to prevent division by zero
    if max_score == min_score:
        # If all scores are the same, rarity scores should be 0
        rarity_score = [0 for _ in averages]
    else:
        rarity_score = [(average - min_score) / (max_score - min_score) for average in averages]
    
    return rarity_score

