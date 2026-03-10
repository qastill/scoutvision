"""
teams.py — K-means team assignment based on jersey colors
"""
import numpy as np
from sklearn.cluster import KMeans


def assign_teams(players: list[dict]) -> tuple[list[dict], str, str]:
    """
    Use K-means clustering on jersey colors (RGB) to assign players to 2 teams.
    Returns updated players list and (teamA_name, teamB_name).
    """
    if not players:
        return players, "Tim A", "Tim B"

    # Collect jersey colors
    colors = np.array([p.get("jerseyColor", [128, 128, 128]) for p in players], dtype=float)

    if len(players) < 2:
        for p in players:
            p["team"] = "Tim A"
            p["teamColor"] = "#00E676"
        return players, "Tim A", "Tim B"

    n_clusters = min(2, len(players))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(colors)

    # Determine team names from cluster center colors
    centers = kmeans.cluster_centers_  # shape: (2, 3) RGB

    # Name teams by dominant color channel
    def color_name(rgb):
        r, g, b = rgb
        if r > g and r > b:
            return "Tim Merah"
        elif g > r and g > b:
            return "Tim Hijau"
        elif b > r and b > g:
            return "Tim Biru"
        elif r > 200 and g > 200 and b > 200:
            return "Tim Putih"
        else:
            return "Tim Gelap"

    team_names = [color_name(c) for c in centers]
    team_colors_hex = []
    for c in centers:
        r, g, b = int(c[0]), int(c[1]), int(c[2])
        team_colors_hex.append(f"#{r:02X}{g:02X}{b:02X}")

    # Fallback if both teams have same name
    if team_names[0] == team_names[1]:
        team_names = ["Tim A", "Tim B"]
        team_colors_hex = ["#00E676", "#18FFFF"]

    for i, p in enumerate(players):
        cluster = int(labels[i])
        p["team"] = team_names[cluster]
        p["teamColor"] = team_colors_hex[cluster]

    return players, team_names[0], team_names[1]


def get_dominant_color(image_crop, k: int = 3) -> list[int]:
    """
    Get dominant JERSEY color from an image crop, ignoring green grass background.
    Returns [R, G, B].
    """
    if image_crop is None or image_crop.size == 0:
        return [128, 128, 128]

    import cv2

    # Focus on upper 60% of crop (jersey area, avoid legs/grass)
    h = image_crop.shape[0]
    upper = image_crop[:max(int(h * 0.6), 4), :]

    small = cv2.resize(upper, (32, 32))

    # Convert to HSV to mask out grass (green)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Grass green: H 35-85, S > 40, V > 40
    grass_lo = np.array([35, 40, 40])
    grass_hi = np.array([85, 255, 255])
    grass_mask = cv2.inRange(hsv, grass_lo, grass_hi)
    non_grass = small[grass_mask == 0]  # pixels that are NOT grass

    # Fall back to all pixels if >80% was grass
    pixels = non_grass if len(non_grass) >= max(k, 8) else small.reshape(-1, 3)
    pixels = pixels.reshape(-1, 3).astype(float)

    if len(pixels) < k:
        return [int(pixels[:, 2].mean()), int(pixels[:, 1].mean()), int(pixels[:, 0].mean())]

    kmeans = KMeans(n_clusters=min(k, len(pixels)), random_state=0, n_init=3)
    kmeans.fit(pixels)

    # Most common cluster
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]

    # Convert BGR to RGB
    return [int(dominant[2]), int(dominant[1]), int(dominant[0])]
