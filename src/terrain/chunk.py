from __future__ import annotations
import logging
from multiprocessing import shared_memory as shm

import pyfastnoisesimd as fns
import numpy as np
from typing import TypeAlias, TypedDict
from core.mesh import instance_dtype

from constants import (
    CHUNK_DIMS,
    CHUNK_SIDE,
    FACES,
    ChunkState,
)

logger = logging.getLogger(__name__) 

# TODO make this configurable or sth
# seed = np.random.randint(2**31)
seed = 42
N_threads = 12
perlin = fns.Noise(seed=seed, numWorkers=N_threads)
perlin.frequency = 0.004
perlin.noiseType = fns.NoiseType.Perlin
perlin.fractal.octaves = 12
perlin.fractal.lacunarity = 128
perlin.fractal.gain = 42
perlin.perturb.perturbType = fns.PerturbType.NoPerturb

PositionType: TypeAlias = tuple[int, int, int]

class ChunkDict(TypedDict):
    id: str
    position: PositionType
    level: int
    state: ChunkState
    terrain: str
    n_terrain: int
    mesh: str
    n_mesh: int

# TODO why is this here
BLOCKS = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1]
], dtype=np.uint8)

class ChunkMeshData:
    position: np.ndarray
    orientation: np.ndarray
    tex_id: np.ndarray
    scale: np.ndarray

    def __init__(self):
        self.clear()

    def clear(self):
        self.position = np.array([])
        self.orientation = np.array([])
        self.tex_id = np.array([])
        self.scale = np.array([])

    def concatenate(self, others: list[ChunkMeshData]):
        self.position = np.concatenate(
            (self.position, *(data.position for data in others)),
            axis=0
        )
        self.orientation = np.concatenate(
            (self.orientation, *(data.orientation for data in others)),
            axis=0
        )
        self.tex_id = np.concatenate(
            (self.tex_id, *(data.tex_id for data in others)),
            axis=0
        )
        self.scale = np.concatenate(
            (self.scale, *(data.scale for data in others)),
            axis=0
        )

    def pack(self) -> np.ndarray:
        if self.position.size == 0:
            return np.empty(0, dtype=instance_dtype)

        n = self.position.shape[0]
        data = np.empty(n, dtype=instance_dtype)

        data["position"] = self.position.astype(np.float32, copy=False)
        data["orientation"] = self.orientation.astype(np.uint32, copy=False)
        data["tex_id"] = self.tex_id.astype(np.float32, copy=False)
        data["scale"] = self.scale.astype(np.uint32, copy=False)

        return data

    @staticmethod
    def unpack(packed_data: np.ndarray) -> ChunkMeshData:
        mesh_data = ChunkMeshData()

        if packed_data.size == 0:
            return mesh_data

        if packed_data.dtype != instance_dtype:
            raise TypeError(
                f"Invalid mesh dtype: {packed_data.dtype}, expected {instance_dtype}"
            )

        mesh_data.position = packed_data["position"]
        mesh_data.orientation = packed_data["orientation"]
        mesh_data.tex_id = packed_data["tex_id"]
        mesh_data.scale = packed_data["scale"]

        return mesh_data

class Chunk:
    position: PositionType
    state: ChunkState
    level: int
    scale: int
    terrain: np.typing.NDArray[np.uint8]
    mesh_data: ChunkMeshData

    def __init__(self, position: PositionType, level: int = 0):
        self.position = position
        self.state = ChunkState.NOT_GENERATED
        self.level = level
        self.scale = 2 ** level

        self.terrain = np.zeros(CHUNK_DIMS, dtype=np.uint8)
        self.mesh_data = ChunkMeshData()

        logger.debug(f"New chunk created: {self.id_string}")

    @property
    def id_string(self) -> str:
        return (
            "chunk_" +
            f"{self.position[0]}_{self.position[1]}_{self.position[2]}" + 
            f"_{self.level}"
        )

    @property
    def center_pos(self) -> tuple[float, float, float]:
        side = CHUNK_SIDE * self.scale
        cx = (self.position[0] + 0.5) * side
        cy = (self.position[1] + 0.5) * side
        cz = (self.position[2] + 0.5) * side
        return (cx, cy, cz)

    @staticmethod
    def from_dict(data: ChunkDict) -> Chunk:
        position, level = data["position"], data["level"]
        chunk = Chunk(position, level)

        if data["state"] == ChunkState.NOT_GENERATED:
            return chunk

        terrain_shm = shm.SharedMemory(data["terrain"])
        terrain = np.ndarray(
            (data["n_terrain"], ),
            buffer=terrain_shm.buf
        )
        chunk.terrain = terrain
        terrain_shm.close()

        if data["state"] == ChunkState.TERRAIN_GENERATED:
            return chunk

        mesh_shm = shm.SharedMemory(data["mesh"])
        packed_mesh = np.ndarray(
            (data["n_mesh"], ),
            buffer=mesh_shm.buf,
            dtype=instance_dtype
        )
        mesh_data = ChunkMeshData.unpack(packed_mesh)
        chunk.mesh_data = mesh_data
        mesh_shm.close()

        if data["state"] == ChunkState.MESH_GENERATED:
            return chunk

        raise Exception(f"Invalid chunk state: {data["state"]}")

    def to_dict(
        self, 
        terrain: str,
        n_terrain: int,
        mesh: str,
        n_mesh: int,
    ) -> ChunkDict:
        return {
            "id": self.id_string,
            "position": self.position,
            "level": self.level,
            "state": self.state,
            "terrain": terrain,
            "n_terrain": n_terrain,
            "mesh": mesh,
            "n_mesh": n_mesh,
        }

    def generate_terrain(self) -> None:
        logger.debug("Chunk " + 
            f"{self.id_string} Generating terrain (scale={self.scale})")

        start_x = (self.position[0] * CHUNK_SIDE - 1) * self.scale
        end_x   = ((self.position[0] + 1) * CHUNK_SIDE + 1) * self.scale
        world_x = np.linspace(start_x, end_x, CHUNK_DIMS[0])

        start_y = (self.position[1] * CHUNK_SIDE - 1) * self.scale
        end_y   = ((self.position[1] + 1) * CHUNK_SIDE + 1) * self.scale
        world_y = np.linspace(start_y, end_y, CHUNK_DIMS[1])

        start_z = (self.position[2] * CHUNK_SIDE - 1) * self.scale
        end_z   = ((self.position[2] + 1) * CHUNK_SIDE + 1) * self.scale
        world_z = np.linspace(start_z, end_z, CHUNK_DIMS[2])

        x_grid, z_grid = np.meshgrid(world_x, world_z, indexing='ij')
        x_grid = x_grid.flatten()
        z_grid = z_grid.flatten()

        n = len(x_grid)
        coords = fns.empty_coords(n)
        coords[0, :n] = x_grid
        coords[1, :n] = np.full(n, 0)
        coords[2, :n] = z_grid

        # todo maybe club these requests together across multiple chunks
        # i.e. let something like ChunkStorage handle them.
        heights = perlin \
            .genFromCoords(coords)[:n] \
            .reshape(CHUNK_SIDE + 2, CHUNK_SIDE + 2)
        height_field = heights * 128
        Y = world_y.reshape(1, -1, 1)
        mask = Y < height_field[:, None, :]

        terrain = np.zeros_like(mask, dtype=np.uint8)
        terrain[mask] = np.random \
            .randint(1, 3, size=np.count_nonzero(mask), dtype=np.uint8)

        self.terrain = terrain
        self.state = ChunkState.TERRAIN_GENERATED

    def generate_mesh(self):
        terrain = self.terrain
        self.mesh_data.clear()

        if not np.any(terrain):
            self.state = ChunkState.MESH_GENERATED
            return

        solid = terrain[1:-1, 1:-1, 1:-1] > 0
        if not np.any(solid):
            self.state = ChunkState.MESH_GENERATED
            return

        positions: list[np.ndarray] = []
        orientations: list[np.ndarray] = []
        tex_ids: list[np.ndarray] = []
        scales: list[np.ndarray] = []

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
            wxv = (self.position[0] * CHUNK_SIDE + vx) * self.scale
            wyv = (self.position[1] * CHUNK_SIDE + vy) * self.scale
            wzv = (self.position[2] * CHUNK_SIDE + vz) * self.scale

            positions.append(np.column_stack((wxv, wyv, wzv)))
            orientations.append(np.full(vx.shape[0], face, np.uint32))

            visible_blocks = terrain[1:-1, 1:-1, 1:-1][visible_mask]
            tex_ids.append(BLOCKS[visible_blocks, face])
            scales.append(np.full(vx.shape[0], self.scale, np.uint32))

        if not positions:
            self.state = ChunkState.MESH_GENERATED
            return

        self.mesh_data.position = np.concatenate(positions)
        self.mesh_data.orientation = np.concatenate(orientations)
        self.mesh_data.tex_id = np.concatenate(tex_ids)
        self.mesh_data.scale = np.concatenate(scales)

        print("B", self.mesh_data.tex_id)

        self.state = ChunkState.MESH_GENERATED
        return

