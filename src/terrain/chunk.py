from __future__ import annotations

import logging
logger = logging.getLogger(__name__) 

import pyfastnoisesimd as fns
import numpy as np
from typing import TypeAlias

from type_hints import Position
from constants import NOT_GENERATED, CHUNK_DIMS, CHUNK_SIDE, HIGHEST_LEVEL, MESH_GENERATED, TERRAIN_GENERATED, FACES, RENDER_DIST

seed = np.random.randint(2**31)
N_threads = 12
perlin = fns.Noise(seed=seed, numWorkers=N_threads)
perlin.frequency = 0.004
perlin.noiseType = fns.NoiseType.Perlin
perlin.fractal.octaves = 12
perlin.fractal.lacunarity = 128
perlin.fractal.gain = 42
perlin.perturb.perturbType = fns.PerturbType.NoPerturb

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

    def append(self, other: ChunkMeshData):
        self.position.extend(other.position)
        self.orientation.extend(other.orientation)
        self.tex_id.extend(other.tex_id)
        self.scale.extend(other.scale)

class Chunk:
    def __init__(self, position: PositionType, level: int = HIGHEST_LEVEL):
        self.position: PositionType = position
        self.state: int = NOT_GENERATED
        self.level: int = level
        self.scale: int = 2 ** level

        self.terrain: np.typing.NDArray[np.uint8] = np.zeros(CHUNK_DIMS, dtype=np.uint8)
        self.mesh_data: ChunkMeshData = ChunkMeshData()

        logger.debug(f"New chunk created: {self.id_string}")

    @property
    def id_string(self) -> str:
        return f"chunk_{self.position[0]}_{self.position[1]}_{self.position[2]}_{self.level}"

    @property
    def center_pos(self) -> tuple[float, float, float]:
        side = CHUNK_SIDE * self.scale
        cx = (self.position[0] + 0.5) * side
        cy = (self.position[1] + 0.5) * side
        cz = (self.position[2] + 0.5) * side
        return (cx, cy, cz)

    def generate_terrain(self) -> None:
        logger.debug(f"Chunk {self.id_string} Generating terrain (scale={self.scale})")

        x_coords = np.arange(CHUNK_DIMS[0]) + self.position[0] * CHUNK_SIDE - 1
        y_coords = np.arange(CHUNK_DIMS[1]) + self.position[1] * CHUNK_SIDE - 1
        z_coords = np.arange(CHUNK_DIMS[2]) + self.position[2] * CHUNK_SIDE - 1

        world_x = x_coords * self.scale
        world_y = y_coords * self.scale
        world_z = z_coords * self.scale

        x_grid, y_grid, z_grid = np.meshgrid(world_x, world_y, world_z, indexing='ij')
        x_grid = x_grid.flatten()
        y_grid = y_grid.flatten()
        z_grid = z_grid.flatten()

        n = len(x_grid)
        coords = fns.empty_coords(n)
        coords[0, :n] = x_grid
        coords[1, :n] = np.full(n, 0)
        coords[2, :n] = z_grid

        heights = perlin.genFromCoords(coords)[:n].reshape(CHUNK_SIDE + 2, CHUNK_SIDE + 2, CHUNK_SIDE + 2)
        terrain = np.zeros(CHUNK_DIMS, dtype=np.uint8)

        for i, y in enumerate(world_y):
            mask = y < heights[:, 0, :] * 42
            solid = np.random.randint(1, 3, mask.shape, dtype=np.uint8)
            terrain[:, i, :][mask] = solid[mask]

        self.terrain = terrain
        self.state = TERRAIN_GENERATED

    def generate_mesh(self) -> bool:
        logger.debug(f"Chunk {self.id_string} Generating mesh (scale={self.scale})")

        if not np.any(self.terrain):
            self.mesh_data.clear()
            self.state = MESH_GENERATED
            return False

        terrain = self.terrain

        xs, ys, zs = np.nonzero(terrain[1:-1, 1:-1, 1:-1])
        if xs.size == 0:
            self.mesh_data.clear()
            self.state = MESH_GENERATED
            return False

        base_x = self.position[0] * CHUNK_SIDE
        base_y = self.position[1] * CHUNK_SIDE
        base_z = self.position[2] * CHUNK_SIDE

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
            self.mesh_data.position = positions
            self.mesh_data.orientation = orientations
            self.mesh_data.tex_id = tex_ids
            self.mesh_data.scale = scale
            logger.debug(f"Chunk {self.id_string} mesh generated.")
        else:
            self.mesh_data.clear()
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
        
        self.is_split = True

    def unsplit(self):
        self.is_split = False
        # dont delete children for now, let them sleep
        self.children = {}

    def update(self, camera_position: Position):
        for child in self.children.values():
            child.update(camera_position)

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

        factor = max([int(7 - self.level / HIGHEST_LEVEL), 6])
        split_threshold = side * factor
        unsplit_threshold = side * (factor + 1)

        if distance < split_threshold and self.level > 0:
            if not self.is_split:
                self.split()
        elif distance > unsplit_threshold:
            if self.is_split:
                self.unsplit()

    def get_mesh_data(self) -> ChunkMeshData:
        if not self.is_split or self.level == 0:
            return self.leaf.mesh_data

        data = ChunkMeshData()
        for child in self.children.values():
            data.append(child.get_mesh_data())

            if not child.leaf.state == MESH_GENERATED:
                return self.leaf.mesh_data

        return data

    def dispose(self):
        self.destroy_leaf()
        for child in self.children.values():
            child.dispose()

    def __del__(self):
        self.dispose()

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
        final_data = ChunkMeshData()

        for node in self.nodes.values():
            data = node.get_mesh_data()
            final_data.append(data)

        try:
            return ( # todo data.to_numpy() cvt directly to the struct thing.
                np.array(final_data.position, dtype = np.float32),
                np.array(final_data.orientation, dtype = np.float32),
                np.array(final_data.tex_id, dtype = np.float32),
                np.array(final_data.scale, dtype = np.float32)
            )
        except ValueError:
            return None

    def update(self, camera_position: Position):
        self.changed = False
        self.camera_node = tuple((camera_position[i] // CHUNK_SIDE) // (2 ** (HIGHEST_LEVEL)) for i in range(3))

        if not self.nodes:
            logger.warning("Loading initial terrain, this may take a while...")

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

        for node in self.nodes.values():
            node.update(camera_position)

        def distance_to_camera(chunk_id: str) -> float:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                return float('inf')
            cx, cy, cz = chunk.center_pos
            camx, camy, camz = camera_position
            return np.sqrt((cx - camx)**2 + (cy - camy)**2 + (cz - camz)**2)

        self.build_queue.sort(key=distance_to_camera)

        while len(self.build_queue) > 0:
            position = self.build_queue.pop(0)
            self.build_chunk(position)
            self.changed = True

