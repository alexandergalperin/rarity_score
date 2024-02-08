import numpy as np 

def calculate_rarity_scores(distances, n_neighbours):
    """
    Calculates rarity scores based on distances and the number of neighbors.

    Args:
        distances: List of distance matrices (NumPy arrays).
        n_neighbours: Number of neighbors to consider.

    Returns:
        List of rarity scores (NumPy arrays).
    """
    distances = [np.array(distance_array) for distance_array in distances]

    if not distances:
        raise ValueError("Cannot calculate rarity scores with empty distances")

    for distance_array in distances:
        for distance_number in distance_array:
            if not isinstance(distance_number, (int,float, np.number)):
                raise TypeError("Distances must be numeric NumPy arrays")

    if n_neighbours > len(distances[0]):
        raise ValueError("n_neighbours cannot be greater than the number of distances")

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



