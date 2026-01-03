from enum import IntEnum
from os import cpu_count

N_WORKERS: int = max([4, cpu_count()])

RENDER_DIST = 4
HIGHEST_LEVEL = 12
CHUNK_SIDE = 16
CHUNK_DIMS = tuple([
    CHUNK_SIDE + 2 for _ in range(3)
])  # Padding of 2 for neighbouring chunk data

HEURISTIC = [max([12 - i, 4]) for i in range(HIGHEST_LEVEL + 2)]

class ChunkState(IntEnum):
    NOT_GENERATED = 0
    TERRAIN_GENERATED = 1
    MESH_GENERATED = 2

FRONT  = 0
BACK   = 1
LEFT   = 2
RIGHT  = 3
TOP    = 4
BOTTOM = 5

FACES = [
    (FRONT,  (0,  0,  1)),
    (BACK,   (0,  0, -1)),
    (LEFT,  (-1,  0,  0)),
    (RIGHT,  (1,  0,  0)),
    (TOP,    (0,  1,  0)),
    (BOTTOM, (0, -1,  0)),
]

