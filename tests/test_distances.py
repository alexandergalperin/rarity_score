import pytest
import numpy as np
from mean_method import calculate_rarity_scores
from flow_method import calculate_rarity_scores_flow

@pytest.fixture 
def euclidean_distances():
    # Generate a sample distance matrix using Euclidean distance with guaranteed variance
    data = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    dist_matrix = np.sqrt(((data[:, np.newaxis, :] - data[np.newaxis, :, :]) ** 2).sum(axis=2))
    return dist_matrix

@pytest.fixture
def cosine_distances():
    # Generate a sample distance matrix using cosine distance with guaranteed variance
    data = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1], [2, 0], [0, 2]])
    norm_data = data / np.linalg.norm(data, axis=1, keepdims=True)
    dist_matrix = 1 - np.dot(norm_data, norm_data.T)
    return dist_matrix


def test_empty_distances_mean_method():
    n_neighbours = 5
    with pytest.raises(ValueError, match="Cannot calculate rarity scores with empty distances list"):
        calculate_rarity_scores([], n_neighbours)

def test_empty_distances_flow_method():
    n_next_hubs = 5
    with pytest.raises(ValueError, match="Cannot calculate rarity scores with empty distances array"):
        calculate_rarity_scores_flow([], n_next_hubs)

def test_rarity_scores_with_euclidean(euclidean_distances):
    n_neighbours = 5
    rarity_scores = calculate_rarity_scores([euclidean_distances], n_neighbours)
    assert len(rarity_scores) == 1, "There should be one rarity score array"
    assert 0 <= min(rarity_scores) <= 1, "Rarity scores should be normalized between 0 and 1"

def test_rarity_scores_with_cosine(cosine_distances):
    n_neighbours = 5
    rarity_scores = calculate_rarity_scores([cosine_distances], n_neighbours)
    assert len(rarity_scores) == 1, "There should be one rarity score array"
    assert 0 <= min(rarity_scores) <= 1, "Rarity scores should be normalized between 0 and 1"

def test_rarity_scores_flow_with_euclidean(euclidean_distances):
    n_next_hubs = 5
    rarity_scores = calculate_rarity_scores([euclidean_distances], n_next_hubs)
    assert len(rarity_scores) == 1, "There should be one rarity score array"
    assert 0 <= min(rarity_scores) <= 1, "Rarity scores should be normalized between 0 and 1"

def test_rarity_scores_flow_with_cosine(cosine_distances):
    n_next_hubs = 5
    rarity_scores = calculate_rarity_scores([cosine_distances], n_next_hubs)
    assert len(rarity_scores) == 1, "There should be one rarity score array"
    assert 0 <= min(rarity_scores) <= 1, "Rarity scores should be normalized between 0 and 1"
