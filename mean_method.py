import numpy as np
from tqdm import tqdm

# distances are the calculated distances of the data points
distances = []

# here you can set the threshold value n_neighbours
n_neighbours = 100

# sort the scores and take the first n_neighbours values
sorted_distances = [sorted(distance_array)[:n_neighbours] for distance_array in distances]

# calculate the sum of values and the number of instances in each array
value_sums = [sum(distance_array) for distance_array in sorted_distances]
instance_counts = [len(distance_array) for distance_array in sorted_distances]

# calculate the average for each array
averages = [value_sum / instance_count for value_sum, instance_count in zip(value_sums, instance_counts)]

# normalization and transformation
rarity_score = averages
min_score = min(rarity_score)
max_score = max(rarity_score)
rarity_score = (rarity_score - min_score) / (max_score - min_score)