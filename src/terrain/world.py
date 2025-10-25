from math import dist
import threading
import multiprocessing

import numpy as np

from core.mesh import Mesh, BufferData
from core.state import State

from .chunk import CHUNK_HEIGHT, CHUNK_SIDE, MESH_GENERATED, Chunk

RENDER_DIST = 8
RENDER_HEIGHT = 3
BATCH_SIZE = multiprocessing.cpu_count()


class ChunkStorage:
    def __init__(self) -> None:
        self.chunks = {}
        self.cache = {}
        self.rebuild_queue = []
        self.build_queue = []
        self.camera_chunk = None
        self.changed = False

    def ensure_chunk(self, position):
        if position in self.chunks:
            return
        if position in self.cache:
            self.uncache_chunk(position)
            return

        chunk = Chunk(position)
        self.chunks[position] = chunk
        self.build_queue.append(position)

    def cache_chunk(self, position):
        self.cache[position] = self.chunks[position]
        del self.chunks[position]

    def uncache_chunk(self, position):
        self.chunks[position] = self.cache[position]
        del self.cache[position]

    def chunk_exists(self, position) -> bool:
        return position in self.chunks

    def get_neighbours(self, id) -> list[Chunk]:
        x, y, z = id
        directions = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]

        neighbours = []
        for dx, dy, dz in directions:
            neighbour_id = (x + dx, y + dy, z + dz)
            if neighbour_id in self.chunks:
                neighbours.append(self.chunks[neighbour_id])

        return neighbours

    def notify_neighbours(self, id) -> None:
        neighbours = self.get_neighbours(id)
        for chunk in neighbours:
            self.rebuild_queue.append(chunk.position)

    def sort_by_distance(self, positions):
        return sorted(positions, key=lambda x: dist(x, self.camera_chunk))

    def build_chunk(self, position):
        if not position in self.chunks:
            self.ensure_chunk(position)

        self.chunks[position].generate_terrain()
        notify = self.chunks[position].generate_mesh(self)

        if notify:
            self.notify_neighbours(position)

        self.changed = True

    def rebuild_chunk(self, position) -> None:
        if not position in self.chunks:
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

    def generate_mesh_data(self):
        position = []
        orientation = []
        tex_id = []

        for id in list(self.chunks.keys()):
            chunk = self.chunks[id]

            if chunk.state != MESH_GENERATED:
                continue

            data = chunk.meshdata
            position.extend(data.position)
            orientation.extend(data.orientation)
            tex_id.extend(data.tex_id)

        try:
            position = np.array(position, dtype=np.float32)
            orientation = np.array(orientation, dtype=np.float32)
            tex_id = np.array(tex_id, dtype=np.float32)
            return (position, orientation, tex_id)
        except ValueError:
            return None

    def update(self, camera_chunk):
        self.changed = False
        self.camera_chunk = camera_chunk

        required_chunks = set()
        for x in range(-RENDER_DIST, RENDER_DIST + 1):
            for y in range(-RENDER_HEIGHT, RENDER_HEIGHT + 1):
                for z in range(-RENDER_DIST, RENDER_DIST + 1):
                    translated_x = x + camera_chunk[0]
                    translated_y = y + camera_chunk[1]
                    translated_z = z + camera_chunk[2]
                    required_chunks.add((translated_x, translated_y, translated_z))

        for required in required_chunks:
            self.ensure_chunk(required)

        to_delete = set(self.chunks.keys()) - required_chunks
        for position in to_delete:
            self.cache_chunk(position)

        to_delete = []
        for position in self.cache:
            distance = dist(position, camera_chunk)
            if distance > RENDER_DIST * 4:
                to_delete.append(position)

        for position in to_delete:
            del self.cache[position]
            
        self.build_queue = self.sort_by_distance(self.build_queue)
        self.rebuild_queue = self.sort_by_distance(self.rebuild_queue)

        count = 0
        threads = []
        while len(self.build_queue) > 0 and count < BATCH_SIZE:
            position = self.build_queue.pop(0)
            threads.append(threading.Thread(target=self.build_chunk, args=[position], daemon=True))
            threads[-1].start()
            count += 1

        while len(self.rebuild_queue) > 0:
            position = self.rebuild_queue.pop(0)
            threads.append(threading.Thread(target=self.rebuild_chunk, args=[position], daemon=True))
            threads[-1].start()

        for thread in threads:
            thread.join()

class ChunkHandler:
    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.namespace = self.manager.Namespace()
        self.namespace.mesh_data = None
        self.namespace.changed = False
        self.namespace.camera_chunk = (0, 0, 0)
        self.namespace.alive = True

        self.process = multiprocessing.Process(
            target=self.worker, args=(self.namespace,)
        )
        self.process.start()

    def worker(self, namespace):
        storage = ChunkStorage()

        while self.namespace.alive:
            storage.update(namespace.camera_chunk)

            if not storage.changed:
                continue

            namespace.mesh_data = storage.generate_mesh_data()
            namespace.changed = True

    @property
    def mesh_data(self) -> tuple[BufferData, BufferData]:
        self.namespace.changed = False
        return self.namespace.mesh_data

    @property
    def changed(self) -> bool:
        return self.namespace.changed

    def set_camera_chunk(self, position) -> None:
        self.namespace.camera_chunk = position

    def kill(self) -> None:
        self.namespace.alive = False
        self.process.terminate()


class World:
    def __init__(self, state: State) -> None:
        self.state: State = state
        self.handler = ChunkHandler()
        self.state.world = self
        self.mesh: Mesh = self.state.mesh_handler.new_mesh("world")

    def update(self) -> None:
        player_position = self.state.camera.position
        camera_chunk = (
            player_position[0] // CHUNK_SIDE,
            player_position[1] // CHUNK_HEIGHT,
            player_position[2] // CHUNK_SIDE
        )
        self.handler.set_camera_chunk(camera_chunk)

        if not self.handler.changed:
            return

        data = self.handler.mesh_data
        if data is None:
            return

        self.mesh.set_data(*data)

    def on_close(self) -> None:
        self.handler.kill()

