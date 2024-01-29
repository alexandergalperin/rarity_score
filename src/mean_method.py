import numpy as np

# distances are the calculated distances of the data points
distances = []

# here you can set the threshold value n_neighbours
n_neighbours = 200

# sort the scores and take the first n_neighbours values
sorted_distances = [sorted(distance_array)[:n_neighbours] for distance_array in distances]

# calculate the average for each array
averages = [sum(score_array) / n_neighbours for score_array in sorted_distances]

# normalization and transformation
min_score = min(averages)
max_score = max(averages)
rarity_score = [(average - min_score) / (max_score - min_score) for average in averages]
