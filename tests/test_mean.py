import pytest
import numpy as np
from mean_method import calculate_rarity_scores  

@pytest.fixture
def euclidean_distances():
    # Generate a sample distance matrix using euclidean distance
    data = np.random.rand(10, 2)  # 10 data points in 2D space
    dist_matrix = np.sqrt(((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2).sum(axis=2))
    return dist_matrix

@pytest.fixture
def cosine_distances():
    # Generate a sample distance matrix using cosine distance
    data = np.random.rand(10, 2)  # 10 data points in 2D space
    norm_data = data / np.linalg.norm(data, axis=1, keepdims=True)
    dist_matrix = 1 - np.dot(norm_data, norm_data.T)
    return dist_matrix

def test_rarity_scores_with_euclidean(euclidean_distances):
    n_neighbours = 5
    rarity_scores = calculate_rarity_scores([euclidean_distances], n_neighbours)
    assert len(rarity_scores) == 1, "There should be one rarity score array"
    assert 0 <= rarity_scores[0].min() <= 1, "Rarity scores should be normalized between 0 and 1"

def test_rarity_scores_with_cosine(cosine_distances):
    n_neighbours = 5
    rarity_scores = calculate_rarity_scores([cosine_distances], n_neighbours)
    assert len(rarity_scores) == 1, "There should be one rarity score array"
    assert 0 <= rarity_scores[0].min() <= 1, "Rarity scores should be normalized between 0 and 1"
