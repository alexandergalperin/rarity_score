import numpy as np

def calculate_rarity_scores(distance_array, n_neighbours):
    """
    Calculates rarity scores based on distances and the number of neighbors.

    Args:
        distance_array: Distance matrix (NumPy array).
        n_neighbours: Number of neighbors to consider.

    Returns:
        Rarity scores (NumPy array).
    """

    if len(distance_array) == 0:
        raise ValueError("Cannot calculate rarity scores with an empty distance matrix")


    if not isinstance(distance_array, np.ndarray) or distance_array.ndim != 2:
        raise ValueError("distance_array must be a 2D NumPy array representing a distance matrix")
    
    # Ensure we don't try to find more neighbours than we have data points
    valid_n_neighbours = min(n_neighbours, len(distance_array) - 1)

    # Sort distances for each point and select the closest n_neighbours
    sorted_distances = np.sort(distance_array, axis=1)[:, 1:valid_n_neighbours+1]

    # Calculate the average distance to the n nearest neighbours for each point
    avg_distances = np.mean(sorted_distances, axis=1)

    # Normalize the rarity scores to be between 0 and 1
    min_score = np.min(avg_distances)
    max_score = np.max(avg_distances)

    if max_score == min_score:
        rarity_score = np.zeros(len(avg_distances))  # All points are equally rare/common
    else:
        rarity_score = (avg_distances - min_score) / (max_score - min_score)

    return rarity_score
