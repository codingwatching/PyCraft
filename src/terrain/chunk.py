from __future__ import annotations

import logging
logger = logging.getLogger(__name__) 

from noise import snoise2
import numpy as np
from typing import TypeAlias
from functools import lru_cache

from type_hints import Position
from constants import NOT_GENERATED, CHUNK_DIMS, CHUNK_SIDE, HIGHEST_LEVEL, MESH_GENERATED, TERRAIN_GENERATED, FACES
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
        self.scale: list[int] = []

    def clear(self):
        self.position = []
        self.orientation = []
        self.tex_id = []
        self.scale = []

class Chunk:
    def __init__(self, position: PositionType, level: int = HIGHEST_LEVEL):
        self.position: PositionType = position
        self.state: int = NOT_GENERATED
        self.level: int = level
        self.scale: int = 2 ** level

        self.terrain: np.typing.NDArray[np.uint8] = np.zeros(CHUNK_DIMS, dtype=np.uint8)
        self.meshdata: ChunkMeshData = ChunkMeshData()

        logger.debug(f"New chunk created: {self.id_string}")

    @property
    def id_string(self) -> str:
        return f"chunk_{self.position[0]}_{self.position[1]}_{self.position[2]}_{self.level}"

    def update_neighbour_terrain(self, storage: ChunkStorage) -> None:
        logger.debug(f"Chunk {self.id_string} updating neighbour terrain")
        raise NotImplementedError

    def generate_terrain(self) -> None:
        logger.debug(f"Chunk {self.id_string} Generating terrain (scale={self.scale})")

        # Compute world-space coordinates with padding
        x_coords = np.arange(CHUNK_DIMS[0]) + self.position[0] * CHUNK_SIDE - 1
        y_coords = np.arange(CHUNK_DIMS[1]) + self.position[1] * CHUNK_SIDE - 1
        z_coords = np.arange(CHUNK_DIMS[2]) + self.position[2] * CHUNK_SIDE - 1

        world_x = x_coords * self.scale
        world_y = y_coords * self.scale
        world_z = z_coords * self.scale

        # Heightmap generation
        x_grid, z_grid = np.meshgrid(world_x, world_z, indexing='ij')
        heights = np.vectorize(lambda x, z: fractal_noise(x, z))(x_grid, z_grid)
        heights = heights.astype(np.float32)

        # Allocate full terrain including padding
        terrain = np.zeros(CHUNK_DIMS, dtype=np.uint8)

        # Fill blocks below height
        for i, y in enumerate(world_y):
            mask = y < heights
            solid = np.random.randint(1, 3, mask.shape, dtype=np.uint8)
            terrain[:, i, :][mask] = solid[mask]

        self.terrain = terrain
        self.state = TERRAIN_GENERATED

    def generate_mesh(self) -> bool: # storage: Storage
        logger.debug(f"Chunk {self.id_string} Generating mesh (scale={self.scale})")

        if not np.any(self.terrain):
            self.meshdata.clear()
            self.state = MESH_GENERATED
            return False

        # self.update_neighbour_terrain(storage)
        terrain = self.terrain

        xs, ys, zs = np.nonzero(terrain[1:-1, 1:-1, 1:-1])
        if xs.size == 0:
            self.meshdata.clear()
            self.state = MESH_GENERATED
            return False

        # Base world offset (non-scaled grid)
        base_x = self.position[0] * CHUNK_SIDE
        base_y = self.position[1] * CHUNK_SIDE
        base_z = self.position[2] * CHUNK_SIDE

        # Compute world-space coordinates with scale applied
        wx = (base_x + xs) * self.scale
        wy = (base_y + ys) * self.scale
        wz = (base_z + zs) * self.scale

        xs += 1
        ys += 1
        zs += 1

        positions = []
        orientations = []
        tex_ids = []
        scale = []

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

            scale.extend(np.full(np.sum(neighbor_mask), self.scale, dtype=np.uint32).tolist())

        if positions:
            self.meshdata.position = positions
            self.meshdata.orientation = orientations
            self.meshdata.tex_id = tex_ids
            self.meshdata.scale = scale
            logger.debug(f"Chunk {self.id_string} mesh generated.")
        else:
            self.meshdata.clear()
            logger.debug(f"Chunk {self.id_string} mesh empty!")

        self.state = MESH_GENERATED
        return True

class OctreeNode:
    def __init__(self, position: Position, storage: ChunkStorage, level: int = HIGHEST_LEVEL):
        self.storage = storage
        self.position = position
        self.level = level
        self.leaf = None

        self.is_split = False
        self.children = {}
        self.create_leaf()

    @property
    def id_string(self) -> str:
        return f"octreenode_{self.position[0]}_{self.position[1]}_{self.position[2]}_{self.level}"

    def create_leaf(self):
        if self.leaf:
            return
        self.leaf = Chunk(self.position, self.level)
        self.storage.chunks[self.leaf.id_string] = self.leaf
        self.storage.build_queue.append(self.leaf.id_string)

    def destroy_leaf(self):
        if not self.leaf:
            return
        del self.storage.chunks[self.leaf.id_string]
        del self.leaf
        self.leaf = None

    def split(self):
        if self.level <= 0:
            return

        if self.is_split:
            return

        child_level = self.level - 1

        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    child_pos = (
                        self.position[0] * 2 + dx,
                        self.position[1] * 2 + dy,
                        self.position[2] * 2 + dz,
                    )

                    child_node = OctreeNode(child_pos, self.storage, child_level)
                    self.children[child_node.id_string] = child_node
        
        self.destroy_leaf()
        self.is_split = True

    def unsplit(self):
        self.is_split = False
        del self.children
        self.children = {}

    def update(self, camera_position: Position):
        scale = 2 ** self.level
        side = CHUNK_SIDE * scale

        center_x = (self.position[0] + 0.5) * side
        center_y = (self.position[1] + 0.5) * side
        center_z = (self.position[2] + 0.5) * side

        cx, cy, cz = camera_position

        distance = np.sqrt(
            (center_x - cx) ** 2 +
            (center_y - cy) ** 2 +
            (center_z - cz) ** 2
        )

        split_threshold = side * 2.5
        unsplit_threshold = side * 4.0

        if distance < split_threshold and self.level > 0:
            if not self.is_split:
                self.split()
        elif distance > unsplit_threshold:
            if self.is_split:
                self.unsplit()

        for child in self.children.values():
            child.update(camera_position)

    def __del__(self):
        self.destroy_leaf()

class ChunkStorage:
    chunks: dict[Position, Chunk]
    build_queue: list[str]
    changed: bool

    def __init__(self) -> None:
        self.nodes = {}
        self.chunks = {}
        self.build_queue = []
        self.camera_node = None
        self.changed = False

    def build_chunk(self, position: Position):
        if position not in self.chunks:
            return # might happen, completely normal

        logger.debug(f"Building chunk {self.chunks[position].id_string}")
        self.chunks[position].generate_terrain()
        counts = self.chunks[position].generate_mesh()
        self.changed = counts
        return counts

    def generate_mesh_data(self) -> tuple | None:
        logger.debug(f"Generating unified mesh data")
        # todo ask the lod for this data instead,
        # make it spit out the lowest leaves.
        # build the aggregate from that
        # then, the lod gets to decide which chunks to give,
        # and hence we can do things like keep the unsplit
        # leaf until the split nodes are done generating.

        position: list[list[float]] = []
        orientation: list[int] = []
        tex_id: list[float] = []
        scale: list[int] = []

        for id in list(self.chunks.keys()):
            chunk = self.chunks[id]

            if chunk.state != MESH_GENERATED:
                continue

            data = chunk.meshdata
            position.extend(data.position)
            orientation.extend(data.orientation)
            tex_id.extend(data.tex_id)
            scale.extend(data.scale)

        try:
            return (
                np.array(position, dtype = np.float32),
                np.array(orientation, dtype = np.float32),
                np.array(tex_id, dtype = np.float32),
                np.array(scale, dtype = np.float32)
            )
        except ValueError:
            return None

    def update(self, camera_position: Position):
        self.changed = False
        self.camera_node = tuple((camera_position[i] // CHUNK_SIDE) // (2 ** (HIGHEST_LEVEL)) for i in range(3))

        required_nodes: set[Position] = set()
        for x in range(-RENDER_DIST, RENDER_DIST + 1):
            for y in range(-RENDER_DIST, RENDER_DIST + 1):
                for z in range(-RENDER_DIST, RENDER_DIST + 1):
                    translated_x = x + self.camera_node[0]
                    translated_y = y + self.camera_node[1]
                    translated_z = z + self.camera_node[2]
                    required_nodes.add((
                        translated_x, 
                        translated_y, 
                        translated_z
                    ))

        for required in required_nodes:
            if required in self.nodes:
                continue
            node = OctreeNode(required, self, HIGHEST_LEVEL)
            self.nodes[required] = node

        to_delete: set[Position] = set(self.nodes.keys()) - required_nodes
        for position in to_delete:
            del self.nodes[position]

        # update nodes
        for node in self.nodes.values():
            node.update(camera_position)

        # todo sort by ABSOLUTE distance. not scaled.
        # And weight by level while you're at it.
        # self.build_queue = self.sort_by_distance(self.build_queue)
        # self.rebuild_queue = self.sort_by_distance(self.rebuild_queue)

        count = 0
        while len(self.build_queue) > 0 and count < BATCH_SIZE:
            position = self.build_queue.pop(0)
            if self.build_chunk(position):
                count += 1

