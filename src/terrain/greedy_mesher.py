import numpy as np
from dataclasses import dataclass
from type_hints import PositionType
from constants import CHUNK_SIDE, TOP, BOTTOM, LEFT, RIGHT, FRONT, BACK
from .mesh_data import ChunkMeshData

# TODO why is this here
BLOCKS = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1]
], dtype=np.uint8)

@dataclass
class Face:
    index: tuple[int, int]
    material: int
    width: int
    height: int

    def __init__(
        self,
        index: tuple[int, int],
        material: int,
        width: int,
        height: int
    ) -> None:
        self.index = index
        self.material = material
        self.width = width
        self.height = height

def greedy_faces(mask: np.ndarray) -> list[Face]:
    faces: list[Face] = []
    used = np.zeros_like(mask, dtype=bool)

    for z in range(1, CHUNK_SIDE + 1):
        for x in range(1, CHUNK_SIDE + 1):
            mat = mask[z, x]
            if mat == 0 or used[z, x]:
                continue

            w = 1
            while x + w <= CHUNK_SIDE and mask[z, x + w] == mat and not used[z, x + w]:
                w += 1

            h = 1
            done = False
            while z + h <= CHUNK_SIDE and not done:
                for dx in range(w):
                    if mask[z + h, x + dx] != mat or used[z + h, x + dx]:
                        done = True
                        break
                if not done:
                    h += 1

            for dz in range(h):
                for dx in range(w):
                    used[z + dz, x + dx] = True

            faces.append(Face((z, x), mat, w, h))

    return faces

def greedy_mesher(terrain: np.ndarray, position: PositionType, level: int) -> ChunkMeshData | None:
    scale = 2 ** level

    # solid mask
    solid = terrain > 0
    solid_in = solid[1:-1, 1:-1, 1:-1]

    if not np.any(solid_in):
        return None

    positions = []
    orientations = []
    tex_ids = []
    widths = []
    heights = []

    for x in range(1, CHUNK_SIDE + 1):
        slice = terrain[x, :, :]
        solid_here  = solid[x, :, :]
        solid_right = solid[x + 1, :, :]
        solid_left = solid[x - 1, :, :]
        exposed_right = solid_here & (~solid_right)
        exposed_left = solid_here & (~solid_left)

        faces = greedy_faces(slice * exposed_right)
        for face in faces:
            y, z = face.index

            wx = position[0] * CHUNK_SIDE + x
            wy = position[1] * CHUNK_SIDE + y
            wz = position[2] * CHUNK_SIDE + z

            positions.append((wx, wy, wz))
            widths.append(face.width * scale)
            heights.append(face.height * scale)
            tex_ids.append(BLOCKS[face.material][RIGHT])
            orientations.append(RIGHT)

        faces = greedy_faces(slice * exposed_left)
        for face in faces:
            y, z = face.index

            wx = position[0] * CHUNK_SIDE + x
            wy = position[1] * CHUNK_SIDE + y
            wz = position[2] * CHUNK_SIDE + z

            positions.append((wx, wy, wz))
            widths.append(face.width * scale)
            heights.append(face.height * scale)
            tex_ids.append(BLOCKS[face.material][LEFT])
            orientations.append(LEFT)

    for y in range(1, CHUNK_SIDE + 1):
        slice = terrain[:, y, :].T
        solid_here  = solid[:, y, :].T
        solid_above = solid[:, y + 1, :].T
        solid_below = solid[:, y - 1, :].T
        exposed_top = solid_here & (~solid_above)
        exposed_bottom = solid_here & (~solid_below)

        faces = greedy_faces(slice * exposed_top)
        for face in faces:
            z, x = face.index

            wx = position[0] * CHUNK_SIDE + x
            wy = position[1] * CHUNK_SIDE + y
            wz = position[2] * CHUNK_SIDE + z

            positions.append((wx, wy, wz))
            widths.append(face.width * scale)
            heights.append(face.height * scale)
            tex_ids.append(BLOCKS[face.material][TOP])
            orientations.append(TOP)

        faces = greedy_faces(slice * exposed_bottom)
        for face in faces:
            z, x = face.index

            wx = position[0] * CHUNK_SIDE + x
            wy = position[1] * CHUNK_SIDE + y
            wz = position[2] * CHUNK_SIDE + z

            positions.append((wx, wy, wz))
            widths.append(face.width * scale)
            heights.append(face.height * scale)
            tex_ids.append(BLOCKS[face.material][BOTTOM])
            orientations.append(BOTTOM)

    for z in range(1, CHUNK_SIDE + 1):
        slice = terrain[:, :, z]
        solid_here  = solid[:, :, z]
        solid_front = solid[:, :, z + 1]
        solid_back = solid[:, :, z - 1]
        exposed_front = solid_here & (~solid_front)
        exposed_back = solid_here & (~solid_back)

        faces = greedy_faces(slice * exposed_front)
        for face in faces:
            x, y = face.index

            wx = position[0] * CHUNK_SIDE + x
            wy = position[1] * CHUNK_SIDE + y
            wz = position[2] * CHUNK_SIDE + z

            positions.append((wx, wy, wz))
            widths.append(face.width * scale)
            heights.append(face.height * scale)
            tex_ids.append(BLOCKS[face.material][FRONT])
            orientations.append(FRONT)

        faces = greedy_faces(slice * exposed_back)
        for face in faces:
            x, y = face.index

            wx = position[0] * CHUNK_SIDE + x
            wy = position[1] * CHUNK_SIDE + y
            wz = position[2] * CHUNK_SIDE + z

            positions.append((wx, wy, wz))
            widths.append(face.width * scale)
            heights.append(face.height * scale)
            tex_ids.append(BLOCKS[face.material][BACK])
            orientations.append(BACK)

    if not positions:
        return None

    mesh = ChunkMeshData()
    mesh.position = np.asarray(positions, np.float32)
    mesh.orientation = np.asarray(orientations, np.int32)
    mesh.tex_id = np.asarray(tex_ids, np.float32)
    mesh.width = np.asarray(widths, np.float32)
    mesh.height = np.asarray(heights, np.float32)

    return mesh

