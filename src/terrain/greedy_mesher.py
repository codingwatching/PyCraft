import numpy as np
from type_hints import PositionType
from constants import FACES, CHUNK_SIDE
from .mesh_data import ChunkMeshData

# TODO why is this here
BLOCKS = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1]
], dtype=np.uint8)

def greedy_mesher(
        terrain: np.ndarray, 
        position: PositionType, 
        level: int
    ) -> ChunkMeshData | None:
    scale_factor: int = 2 ** level
    width, height = [float(scale_factor) for _ in range(2)]
    solid = terrain[1:-1, 1:-1, 1:-1] > 0

    positions: list[np.ndarray] = []
    orientations: list[np.ndarray] = []
    tex_ids: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    heights: list[np.ndarray] = []

    for face, (dx, dy, dz) in FACES:
        neighbor = terrain[
            (1 + dx):(-1 + dx) or None,
            (1 + dy):(-1 + dy) or None,
            (1 + dz):(-1 + dz) or None
        ]
        visible_mask = solid & (neighbor == 0)

        if not np.any(visible_mask):
            continue

        vx, vy, vz = np.nonzero(visible_mask)
        wxv = (position[0] * CHUNK_SIDE + vx) * width
        wyv = (position[1] * CHUNK_SIDE + vy) * height
        wzv = (position[2] * CHUNK_SIDE + vz) * width

        positions.append(np.column_stack((wxv, wyv, wzv)))
        orientations.append(np.full(vx.shape[0], face, np.uint32))

        visible_blocks = terrain[1:-1, 1:-1, 1:-1][visible_mask]
        tex_ids.append(BLOCKS[visible_blocks, face])
        widths.append(np.full(vx.shape[0], width, np.float32))
        heights.append(np.full(vx.shape[0], height, np.float32))

    if not positions:
        return None

    mesh_data: ChunkMeshData = ChunkMeshData()
    mesh_data.position = np.concatenate(positions)
    mesh_data.orientation = np.concatenate(orientations)
    mesh_data.tex_id = np.concatenate(tex_ids)
    mesh_data.width = np.concatenate(widths)
    mesh_data.height = np.concatenate(heights)

    return mesh_data

