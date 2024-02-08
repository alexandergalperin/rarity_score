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
    distances = np.array(distances)

    if not distances:
        raise ValueError("Cannot calculate rarity scores with empty distances")

    # Verify that each distance matrix in the list is a numeric NumPy array
    for distance_array in distances:
        if not isinstance(distance_array, np.ndarray) or not np.issubdtype(distance_array.dtype, np.number):
            raise TypeError("Distances must be numeric NumPy arrays")

    rarity_scores = []
    for distance_array in distances:
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

        rarity_scores.append(rarity_score)

    return rarity_scores
