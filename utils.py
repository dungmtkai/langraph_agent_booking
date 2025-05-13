def euclidean_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate squared Euclidean distance between two points (no square root for efficiency)."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return dlat ** 2 + dlon ** 2