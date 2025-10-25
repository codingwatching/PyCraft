from random import randint
from noise import snoise2
import numpy as np
from typing import TypeAlias

CHUNK_SIDE = 16
CHUNK_DIMS = tuple(
    CHUNK_SIDE + 2 for _ in range(3)
)  # Padding of 2 for neighbouring chunk data

# TODO: Use enums instead of whatever this is
NOT_GENERATED = 0
TERRAIN_GENERATED = 1
MESH_GENERATED = 2

PositionType: TypeAlias = tuple[int, int, int]

class ChunkMeshData:
    def __init__(self):
        self.position = []
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
        terrain = np.ones((CHUNK_SIDE, CHUNK_SIDE, CHUNK_SIDE), dtype=np.uint8)

        for x in range(CHUNK_SIDE):
            i = self.position[0] * CHUNK_SIDE + x
            for z in range(CHUNK_SIDE):
                k = self.position[2] * CHUNK_SIDE + z
                for y in range(CHUNK_SIDE):
                    j = self.position[1] * CHUNK_SIDE + y
                    height = snoise2(i / 100, k / 100) * 10
                    terrain[x, y, z] = j < height

        self.terrain[1:-1, 1:-1, 1:-1] = terrain
        self.state = TERRAIN_GENERATED

    def generate_mesh(self, world) -> None:
        if not np.any(self.terrain[1:-1, 1:-1, 1:-1]):
            self.state = MESH_GENERATED
            return

        self.update_neighbour_terrain(world)

        position = []
        tex_id = []
        for x in range(CHUNK_SIDE):
            for y in range(CHUNK_SIDE):
                for z in range(CHUNK_SIDE):
                    i, j, k = x+1, y+1, z+1
                    if self.terrain[i][j][k]:
                        tex_id.append(randint(0,2))
                        position.append([
                            self.position[0] * CHUNK_SIDE + x,
                            self.position[1] * CHUNK_SIDE + y,
                            self.position[2] * CHUNK_SIDE + z,
                        ])

        self.meshdata.position = position
        self.meshdata.tex_id = tex_id

        self.state = MESH_GENERATED

