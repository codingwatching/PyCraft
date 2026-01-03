import logging
import numpy as np
import multiprocessing
from multiprocessing.managers import SyncManager
from multiprocessing import shared_memory as shm
from core.mesh import BufferData
from .chunk import Chunk, ChunkMeshData

logger = logging.getLogger(__name__) 

class ChunkBuilder:
    def __init__(self) -> None:
        return

    def step(self, namespace, pid: int) -> None:
        queue = namespace.queues[pid]
        if len(queue) == 0:
            return
        
        entry = queue.pop(0)
        data = namespace.chunks[entry]
        chunk = Chunk.from_dict(data)
        chunk.generate_terrain()
        chunk.generate_mesh()
        mesh_data, offsets = chunk.mesh.pack()

        logger.info(f"Worker {pid} generated chunk {chunk.id_string}")

        terrain = shm.SharedMemory(
            create=True, 
            size=chunk.terrain.nbytes
        )
        mesh = shm.SharedMemory(
            create=True, 
            size=mesh_data.nbytes
        )

        newdata = chunk.to_dict(
            terrain.name,
            len(chunk.terrain),
            mesh.name,
            len(mesh_data),
            offsets
        )
        namespace.chunks[entry] = newdata

class MeshBuilder:
    def __init__(self) -> None:
        return

    def step(self, namespace) -> None:
        # todo build mesh and manage shared memory
        return

class ChunkHandler:
    manager: SyncManager
    processes: list[multiprocessing.Process]

    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.namespace = self.manager.Namespace()
        self.namespace.chunks = self.manager.dict()
        self.namespace.terrain = self.manager.dict()
        self.namespace.meshes = self.manager.dict()
        self.namespace.queues = self.manager.dict()
        self.namespace.camera = (0, 0, 0)
        self.namespace.changed = False
        self.namespace.alive = True

        logger.info("Starting workers")

        self.processes = []
        
        for pid in range(4):
            builder = multiprocessing.Process(
                target=self.build_worker,
                args=(self.namespace, pid)
            )
            self.namespace.queues[pid] = self.manager.list()
            builder.start()

        logger.info("ChunkHandler instantiated.")

    @property
    def mesh_data(self) -> tuple[BufferData, BufferData, BufferData, BufferData]:
        self.namespace.changed = False
        return (BufferData(42), BufferData(42), BufferData(42), BufferData(42))

    @property
    def changed(self) -> bool:
        return self.namespace.changed

    def set_camera_position(self, position: list[float]) -> None:
        self.namespace.camera = position

    def build_worker(self, namespace, pid: int) -> None:
        logger.info(f"Starting build worker {pid}")
        builder = ChunkBuilder()

        while namespace.alive:
            builder.step(namespace, pid)

    def kill(self) -> None:
        logger.info("Terminating worker")

        self.namespace.alive = False
        # TODO terminate worker processes
        # TODO deallocate shared memory

