from random import randint
from time import time
from noise import snoise2
import numpy as np
from typing import TypeAlias

CHUNK_SIDE = 16
CHUNK_HEIGHT = 64
CHUNK_DIMS = tuple([
    CHUNK_SIDE + 2,
    CHUNK_HEIGHT + 2,
    CHUNK_SIDE + 2
])  # Padding of 2 for neighbouring chunk data

# TODO: Use enums instead of whatever this is
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

PositionType: TypeAlias = tuple[int, int, int]

class ChunkMeshData:
    def __init__(self):
        self.position = []
        self.orientation = []
        self.tex_id = []

class Chunk:
    def __init__(self, position: PositionType):
        self.position: PositionType = position
        self.state: int = NOT_GENERATED

        self.terrain: np.typing.NDArray[np.uint8] = np.zeros(CHUNK_DIMS, dtype=np.uint8)
        self.meshdata: ChunkMeshData = ChunkMeshData()

    @property
    def id(self) -> str:
        return f"chunk_{self.position[0]}_{self.position[1]}_{self.position[2]}"

    def is_air(self, x: int, y: int, z: int) -> bool:
        return self.terrain[x, y, z] == 0

    def update_neighbour_terrain(self, world) -> None:
        neighbour_dirs = {
            (1, 0, 0): (slice(-1, None), slice(1, -1), slice(1, -1)),
            (-1, 0, 0): (slice(0, 1), slice(1, -1), slice(1, -1)),
            (0, 1, 0): (slice(1, -1), slice(-1, None), slice(1, -1)),
            (0, -1, 0): (slice(1, -1), slice(0, 1), slice(1, -1)),
            (0, 0, 1): (slice(1, -1), slice(1, -1), slice(-1, None)),
            (0, 0, -1): (slice(1, -1), slice(1, -1), slice(0, 1)),
        }

        center = (slice(1, -1), slice(1, -1), slice(1, -1))

        for (dx, dy, dz), dest_slice in neighbour_dirs.items():
            neighbor_pos = (
                self.position[0] + dx,
                self.position[1] + dy,
                self.position[2] + dz,
            )
            if world.chunk_exists(neighbor_pos):
                neighbor = world.chunks[neighbor_pos]
                if dx == 1:
                    source = (slice(1, 2),) + center[1:]
                elif dx == -1:
                    source = (slice(-2, -1),) + center[1:]
                elif dy == 1:
                    source = (center[0], slice(1, 2), center[2])
                elif dy == -1:
                    source = (center[0], slice(-2, -1), center[2])
                elif dz == 1:
                    source = center[:2] + (slice(1, 2),)
                elif dz == -1:
                    source = center[:2] + (slice(-2, -1),)

                self.terrain[dest_slice] = neighbor.terrain[source]

    def generate_terrain(self) -> None:
        x_coords = np.arange(CHUNK_SIDE) + self.position[0] * CHUNK_SIDE
        y_coords = np.arange(CHUNK_HEIGHT) + self.position[1] * CHUNK_HEIGHT
        z_coords = np.arange(CHUNK_SIDE) + self.position[2] * CHUNK_SIDE

        x_grid, z_grid = np.meshgrid(x_coords, z_coords, indexing='ij')
        heights = np.vectorize(lambda x, z: snoise2(x / 100, z / 100) * 10)(x_grid, z_grid)
        heights = heights.astype(np.float32)

        terrain = (y_coords[None, :, None] < heights[:, None, :]).astype(np.uint8)
        self.terrain[1:-1, 1:-1, 1:-1] = terrain
        self.state = TERRAIN_GENERATED

    def generate_mesh(self, world) -> bool:
        if not np.any(self.terrain):
            self.state = MESH_GENERATED
            return False

        self.update_neighbour_terrain(world)
        terrain = self.terrain

        xs, ys, zs = np.nonzero(terrain[1:-1, 1:-1, 1:-1])
        if xs.size == 0:
            self.meshdata.position = np.empty((0, 3), dtype=np.float32)
            self.meshdata.orientation = np.empty(0, dtype=np.uint32)
            self.meshdata.tex_id = np.empty(0, dtype=np.uint32)
            self.state = MESH_GENERATED
            return False

        xs += 1
        ys += 1
        zs += 1

        wx = self.position[0] * CHUNK_SIDE + (xs - 1) - CHUNK_SIDE // 2
        wy = self.position[1] * CHUNK_HEIGHT + (ys - 1) - CHUNK_HEIGHT // 2
        wz = self.position[2] * CHUNK_SIDE + (zs - 1) - CHUNK_SIDE // 2

        positions = []
        orientations = []
        tex_ids = []

        for face, (dx, dy, dz) in FACES:
            neighbor_mask = terrain[xs + dx, ys + dy, zs + dz] == 0
            if np.any(neighbor_mask):
                positions.append(np.column_stack((wx[neighbor_mask], wy[neighbor_mask], wz[neighbor_mask])))
                orientations.append(np.full(np.sum(neighbor_mask), face, dtype=np.uint32))
                tex_ids.append(np.random.randint(0, 4, np.sum(neighbor_mask), dtype=np.uint32))

        if positions:
            self.meshdata.position = np.vstack(positions).astype(np.float32)
            self.meshdata.orientation = np.concatenate(orientations)
            self.meshdata.tex_id = np.concatenate(tex_ids)
        else:
            self.meshdata.position = np.empty((0, 3), dtype=np.float32)
            self.meshdata.orientation = np.empty(0, dtype=np.uint32)
            self.meshdata.tex_id = np.empty(0, dtype=np.uint32)

        self.state = MESH_GENERATED
        return True

