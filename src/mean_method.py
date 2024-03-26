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

    if len(distances) == 0:
        raise ValueError("Cannot calculate rarity scores with empty distances")

    rarity_scores = []
    for distance_matrix in distances:
        # Stelle sicher, dass die Distanzmatrix ein 2D NumPy Array ist
        if distance_matrix.ndim != 2:
            raise ValueError("Each distance matrix must be a 2D NumPy array")

        # Sortiere die Distanzen für jeden Punkt und wähle die n nächsten Nachbarn
        # Hier verwenden wir `sorted` für jede Zeile und konvertieren zurück zu NumPy Arrays
        sorted_distances = np.array([sorted(row)[1:n_neighbours+1] for row in distance_matrix])

        # Berechne den Durchschnitt der Distanz zu den n nächsten Nachbarn für jeden Punkt
        avg_distances = np.mean(sorted_distances, axis=1)

        # Normalisiere die Seltenheitswerte, um sie zwischen 0 und 1 zu erhalten
        min_score = np.min(avg_distances)
        max_score = np.max(avg_distances)
        if max_score == min_score:
            rarity_score = np.zeros(len(avg_distances))  # Alle Punkte sind gleich selten/gewöhnlich
        else:
            rarity_score = (avg_distances - min_score) / (max_score - min_score)

        rarity_scores.append(rarity_score)

    return rarity_scores
