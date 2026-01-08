import numpy as np
from type_hints import PositionType
from constants import CHUNK_SIDE, FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM, FACES
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
    scale_factor = 2 ** level
    width, height = float(scale_factor), float(scale_factor)

    # interior solid blocks
    solid = terrain[1:-1, 1:-1, 1:-1] > 0

    masks: dict[int, np.ndarray] = {}
    for face_idx, (dx, dy, dz) in FACES:
        neighbor = terrain[
            (1 + dx):(-1 + dx) or None,
            (1 + dy):(-1 + dy) or None,
            (1 + dz):(-1 + dz) or None
        ]
        masks[face_idx] = solid & (neighbor == 0)

    positions = []
    orientations = []
    tex_ids = []
    widths = []
    heights = []

    for face_idx, mask in masks.items():
        if not np.any(mask):
            continue

        for z in range(mask.shape[2]):
            slice_mask = mask[:, :, z]
            if not np.any(slice_mask):
                continue

            vx, vy = np.nonzero(slice_mask)
            vz = np.full(vx.shape[0], z)

            wxv = (position[0] * CHUNK_SIDE + vx) * width
            wyv = (position[1] * CHUNK_SIDE + vy) * height
            wzv = (position[2] * CHUNK_SIDE + vz) * width

            positions.append(np.column_stack((wxv, wyv, wzv)))
            orientations.append(np.full(vx.shape[0], face_idx, np.uint32))

            visible_blocks = terrain[1:-1, 1:-1, 1:-1][:, :, z][slice_mask]
            tex_ids.append(BLOCKS[visible_blocks, face_idx])
            widths.append(np.full(vx.shape[0], width, np.float32))
            heights.append(np.full(vx.shape[0], height, np.float32))

    if not positions:
        return None

    mesh_data = ChunkMeshData()
    mesh_data.position = np.concatenate(positions)
    mesh_data.orientation = np.concatenate(orientations)
    mesh_data.tex_id = np.concatenate(tex_ids)
    mesh_data.width = np.concatenate(widths)
    mesh_data.height = np.concatenate(heights)

    return mesh_data

