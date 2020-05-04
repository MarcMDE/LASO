accions_veu = {
    "a1": ["stop", "parar", "pausa"],
    "a2": []
}

# BGR SEGMENTATION BASE COLORS
SEG_COLORS = {
    "wall": (0, 255, 0),
    "ground": (0, 255, 255),
    "hole": (0, 0, 0),
    "ball": (255, 0, 0),  # Single channel ball for processing speed (completely different channel from wall and ground)
    "start": (0, 0, 255),
    "end": (255, 255, 255)
}