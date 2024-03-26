import numpy as np

def calculate_rarity_scores(distances, n_neighbours):
    """
    Berechnet die Seltenheitswerte auf Basis der Distanzen und der Anzahl der Nachbarn.

    Args:
        distances: Eine Liste von Distanzmatrizen (NumPy Arrays).
        n_neighbours: Anzahl der Nachbarn, die berücksichtigt werden sollen.

    Returns:
        Eine Liste von Seltenheitswerten (NumPy Arrays).
    """

    if len(distances) == 0:
        raise ValueError("Cannot calculate rarity scores with empty distances")

    rarity_scores = []
    for distance_matrix in distances:
        # Stelle sicher, dass distance_matrix 2D ist
        if distance_matrix.ndim != 2:
            print(distance_matrix , 'fail')
            raise ValueError("Each distance matrix must be a 2D NumPy array")

        # Sortiere jede Zeile der Distanzmatrix und wähle die nächsten n_neighbours aus
        sorted_distances = np.sort(distance_matrix, axis=1)[:, 1:n_neighbours+1]

        # Berechne die durchschnittliche Distanz zu den n nächsten Nachbarn für jeden Punkt
        avg_distances = np.mean(sorted_distances, axis=1)

        # Normalisiere die Seltenheitswerte, um sie zwischen 0 und 1 zu erhalten
        min_score = np.min(avg_distances)
        max_score = np.max(avg_distances)
        if max_score == min_score:
            rarity_score = np.zeros_like(avg_distances)  # Alle Punkte sind gleich selten/häufig
        else:
            rarity_score = (avg_distances - min_score) / (max_score - min_score)

        rarity_scores.append(rarity_score)

    return rarity_scores
