from __future__ import annotations
import logging
from multiprocessing import shared_memory as shm

import numpy as np
from typing import TypedDict
from core.mesh import instance_dtype
from .mesh_data import ChunkMeshData
from .greedy_mesher import greedy_mesher
from .generator import terrain_generator
from type_hints import PositionType

from constants import (
    CHUNK_DIMS,
    CHUNK_SIDE,
    ChunkState,
)

logger = logging.getLogger(__name__) 


class ChunkDict(TypedDict):
    id: str
    position: PositionType
    level: int
    state: ChunkState
    terrain: str
    n_terrain: int
    mesh: str
    n_mesh: int


class Chunk:
    position: PositionType
    state: ChunkState
    level: int
    terrain: np.typing.NDArray[np.uint8]
    mesh_data: ChunkMeshData

    def __init__(self, position: PositionType, level: int = 0):
        self.position = position
        self.state = ChunkState.NOT_GENERATED
        self.level = level

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
        side = CHUNK_SIDE * self.width
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
        logger.debug(f"Chunk {self.id_string} Generating terrain")
        self.terrain = terrain_generator(
            self.position,
            self.level
        )
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

        new_mesh = greedy_mesher(
            terrain,
            self.position,
            self.level
        )
        if new_mesh is not None:
            self.mesh_data = new_mesh

        self.state = ChunkState.MESH_GENERATED

