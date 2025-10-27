from __future__ import annotations

import logging
logger = logging.getLogger(__name__) 

from math import dist
from noise import snoise2
import numpy as np
from typing import TypeAlias
from functools import lru_cache

from type_hints import Position
from constants import NOT_GENERATED, CHUNK_DIMS, CHUNK_SIDE, MESH_GENERATED, TERRAIN_GENERATED, FACES
from constants import RENDER_DIST, BATCH_SIZE

@lru_cache()
def fractal_noise(x: float, z: float) -> float:
    return (
        snoise2(x / 100, z / 100) * 10 +
        snoise2(x / 1000, z / 1000) * 100 +
        snoise2(x / 10000, z / 10000) * 1000
    )

PositionType: TypeAlias = tuple[int, int, int]

BLOCKS = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1]
], dtype=np.uint8)

class ChunkMeshData:
    def __init__(self):
        self.position: list[list[float]] = []
        self.orientation: list[int] = []
        self.tex_id: list[float] = []

    def clear(self):
        self.position = []
        self.orientation = []
        self.tex_id = []

class Chunk:
    def __init__(self, position: PositionType):
        self.position: PositionType = position
        self.state: int = NOT_GENERATED

        self.terrain: np.typing.NDArray[np.uint8] = np.zeros(CHUNK_DIMS, dtype=np.uint8)
        self.meshdata: ChunkMeshData = ChunkMeshData()

        logger.debug(f"New chunk created: {self.id_string}")

    @property
    def id_string(self) -> str:
        return f"chunk_{self.position[0]}_{self.position[1]}_{self.position[2]}"

    def update_neighbour_terrain(self, storage: ChunkStorage) -> None:
        logger.debug(f"Chunk {self.id_string} updating neighbour terrain")

        neighbour_dirs: dict[Position, tuple[slice, slice, slice]] = {
            (1, 0, 0): (slice(-1, None), slice(1, -1), slice(1, -1)),
            (-1, 0, 0): (slice(0, 1), slice(1, -1), slice(1, -1)),
            (0, 1, 0): (slice(1, -1), slice(-1, None), slice(1, -1)),
            (0, -1, 0): (slice(1, -1), slice(0, 1), slice(1, -1)),
            (0, 0, 1): (slice(1, -1), slice(1, -1), slice(-1, None)),
            (0, 0, -1): (slice(1, -1), slice(1, -1), slice(0, 1)),
        }

        center: tuple[slice, slice, slice] = (slice(1, -1), slice(1, -1), slice(1, -1))

        for (dx, dy, dz), dest_slice in neighbour_dirs.items():
            neighbor_pos: Position = (
                self.position[0] + dx,
                self.position[1] + dy,
                self.position[2] + dz,
            )
            if storage.chunk_exists(neighbor_pos):
                neighbor = storage.chunks[neighbor_pos]

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
        logger.debug(f"Chunk {self.id_string} Generating terrain")

        x_coords = np.arange(CHUNK_SIDE) + self.position[0] * CHUNK_SIDE
        y_coords = np.arange(CHUNK_SIDE) + self.position[1] * CHUNK_SIDE
        z_coords = np.arange(CHUNK_SIDE) + self.position[2] * CHUNK_SIDE

        x_grid, z_grid = np.meshgrid(x_coords, z_coords, indexing='ij')
        heights = np.vectorize(lambda x, z: fractal_noise(x, z))(x_grid, z_grid)
        heights = heights.astype(np.float32)

        terrain_mask = (y_coords[None, :, None] < heights[:, None, :]).astype(np.uint8) == 0
        terrain = np.random.randint(1, 3, (CHUNK_SIDE, CHUNK_SIDE, CHUNK_SIDE))
        terrain[terrain_mask] = 0

        self.terrain[1:-1, 1:-1, 1:-1] = terrain
        self.state = TERRAIN_GENERATED

    def generate_mesh(self, storage: ChunkStorage) -> bool:
        logger.debug(f"Chunk {self.id_string} Generating mesh")

        if not np.any(self.terrain):
            self.meshdata.clear()
            self.state = MESH_GENERATED

            logger.debug(f"Chunk {self.id_string} mesh empty!")

            return False

        self.update_neighbour_terrain(storage)
        terrain = self.terrain

        xs, ys, zs = np.nonzero(terrain[1:-1, 1:-1, 1:-1])
        if xs.size == 0:
            self.meshdata.clear()
            self.state = MESH_GENERATED

            logger.debug(f"Chunk {self.id_string} mesh empty!")

            return False

        wx = self.position[0] * CHUNK_SIDE + xs
        wy = self.position[1] * CHUNK_SIDE + ys
        wz = self.position[2] * CHUNK_SIDE + zs

        xs += 1
        ys += 1
        zs += 1

        positions = []
        orientations = []
        tex_ids = []

        for face, (dx, dy, dz) in FACES:
            neighbor_mask = terrain[xs + dx, ys + dy, zs + dz] == 0
            if not np.any(neighbor_mask):
                continue

            positions.extend(np.column_stack((wx[neighbor_mask], wy[neighbor_mask], wz[neighbor_mask])).tolist())
            orientations.extend(np.full(np.sum(neighbor_mask), face, dtype=np.uint32).tolist())

            block_types = terrain[xs, ys, zs]
            visible_blocks = block_types[neighbor_mask]
            face_tex = BLOCKS[visible_blocks, face]
            tex_ids.extend(face_tex.tolist())

        if positions:
            self.meshdata.position = positions
            self.meshdata.orientation = orientations
            self.meshdata.tex_id = tex_ids

            logger.debug(f"Chunk {self.id_string} mesh generated.")
        else:
            self.meshdata.clear()
            logger.debug(f"Chunk {self.id_string} mesh empty!")

        self.state = MESH_GENERATED
        return True

class ChunkStorage:
    chunks: dict[Position, Chunk]
    cache: dict[Position, Chunk]
    build_queue: list[Position]
    rebuild_queue: list[Position]
    camera_chunk: Position | None
    changed: bool

    def __init__(self) -> None:
        self.chunks = {}
        self.cache = {}
        self.build_queue = []
        self.rebuild_queue = []
        self.camera_chunk = None
        self.changed = False

    def ensure_chunk(self, position: Position):
        if position in self.chunks:
            return
        if position in self.cache:
            self.uncache_chunk(position)
            return

        logger.debug(f"Ensuring chunk at {position}")

        chunk = Chunk(position)
        self.chunks[position] = chunk
        self.build_queue.append(position)

    def cache_chunk(self, position: Position):
        logger.debug(f"Caching chunk at {position}")

        self.cache[position] = self.chunks[position]
        del self.chunks[position]

    def uncache_chunk(self, position: Position):
        logger.debug(f"Uncaching chunk at {position}")

        self.chunks[position] = self.cache[position]
        del self.cache[position]

    def chunk_exists(self, position: Position) -> bool:
        return position in self.chunks

    def get_neighbours(self, position: Position) -> list[Chunk]:
        x, y, z = position
        directions = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        neighbours: list[Chunk] = []
        for dx, dy, dz in directions:
            neighbour_id = (x + dx, y + dy, z + dz)
            if neighbour_id in self.chunks:
                neighbours.append(self.chunks[neighbour_id])

        return neighbours

    def notify_neighbours(self, position: Position) -> None:
        logger.debug(f"Notifying neighbours of {position}")

        neighbours = self.get_neighbours(position)
        for chunk in neighbours:
            self.rebuild_queue.append(chunk.position)

    def sort_by_distance(self, positions: list[Position]) -> list[Position]:
        if self.camera_chunk is None:
            return positions
        return sorted(positions, key=lambda x: dist(x, self.camera_chunk))

    def build_chunk(self, position: Position):
        logger.debug(f"Building chunk {position}")

        if position not in self.chunks:
            self.ensure_chunk(position)

        self.chunks[position].generate_terrain()
        self.notify_neighbours(position)

        counts = self.chunks[position].generate_mesh(self)
        self.changed = counts

        return counts

    def rebuild_chunk(self, position: Position) -> None:
        logger.debug(f"Rebuilding chunk {position}")

        if position not in self.chunks:
            self.ensure_chunk(position)
        chunk = None

        if position in self.chunks:
            chunk = self.chunks[position]
        elif position in self.cache:
            chunk = self.cache[position]

        if chunk is None:
            return

        chunk.generate_mesh(self)
        self.changed = True

    def generate_mesh_data(self) -> tuple | None:
        logger.debug(f"Generating unified mesh data")

        position: list[list[float]] = []
        orientation: list[int] = []
        tex_id: list[float] = []

        for id in list(self.chunks.keys()):
            chunk = self.chunks[id]

            if chunk.state != MESH_GENERATED:
                continue

            data = chunk.meshdata
            position.extend(data.position)
            orientation.extend(data.orientation)
            tex_id.extend(data.tex_id)

        try:
            return (
                np.array(position, dtype = np.float32),
                np.array(orientation, dtype = np.float32),
                np.array(tex_id, dtype = np.float32)
            )
        except ValueError:
            return None

    def update(self, camera_chunk: Position):
        self.changed = False
        self.camera_chunk = camera_chunk

        required_chunks: set[Position] = set()
        for x in range(-RENDER_DIST, RENDER_DIST + 1):
            for y in range(-RENDER_DIST, RENDER_DIST + 1):
                for z in range(-RENDER_DIST, RENDER_DIST + 1):
                    translated_x = x + camera_chunk[0]
                    translated_y = y + camera_chunk[1]
                    translated_z = z + camera_chunk[2]
                    required_chunks.add((
                        translated_x, 
                        translated_y, 
                        translated_z
                    ))

        for required in required_chunks:
            self.ensure_chunk(required)

        to_delete: set[Position] = set(self.chunks.keys()) - required_chunks
        for position in to_delete:
            self.cache_chunk(position)

        to_delete = set()
        for position in self.cache:
            distance = dist(position, camera_chunk)
            if distance > RENDER_DIST * 4:
                to_delete.add(position)

        for position in to_delete:
            logger.debug(f"Deleting chunk {position}")
            del self.cache[position]
            
        self.build_queue = self.sort_by_distance(self.build_queue)
        self.rebuild_queue = self.sort_by_distance(self.rebuild_queue)

        count = 0
        while len(self.build_queue) > 0 and count < BATCH_SIZE:
            position = self.build_queue.pop(0)
            if self.build_chunk(position):
                count += 1

        while len(self.rebuild_queue) > 0:
            position = self.rebuild_queue.pop(0)
            self.rebuild_chunk(position)

