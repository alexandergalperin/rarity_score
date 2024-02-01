import numpy as np

def calculate_rarity_scores(distances, n_neighbours):
    # Ensure distances is a list of numpy arrays for consistent processing
    distances = [np.array(distance_array) for distance_array in distances]
    
    # sort the scores and take the first n_neighbours values
    sorted_distances = [np.sort(distance_array)[:n_neighbours] for distance_array in distances]

    # calculate the average for each array
    averages = [np.mean(score_array) for score_array in sorted_distances]

    # normalization and transformation
    min_score = min(averages)
    max_score = max(averages)
    rarity_score = [(average - min_score) / (max_score - min_score) for average in averages]
    
    return rarity_score
