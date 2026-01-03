import logging
import numpy as np
import multiprocessing
from multiprocessing.managers import SyncManager
from multiprocessing import shared_memory as shm

from numpy._core.multiarray import dtype
from .chunk import Chunk, ChunkMeshData
from constants import ChunkState
from core.utils import deallocate_shared_memory
from constants import N_WORKERS
from core.mesh import instance_dtype

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
        mesh_data = chunk.mesh_data.pack()

        logger.info(f"Worker {pid} generated {chunk.id_string}")
        namespace.terrain_changed = True

        if len(mesh_data) == 0:
            return

        terrain = shm.SharedMemory(
            create=True, 
            size=chunk.terrain.nbytes
        )
        mesh = shm.SharedMemory(
            create=True, 
            size=mesh_data.nbytes,
        )

        newdata = chunk.to_dict(
            terrain.name,
            len(chunk.terrain),
            mesh.name,
            len(mesh_data),
        )
        namespace.chunks[entry] = newdata

class MeshBuilder:
    def __init__(self) -> None:
        return

    def step(self, namespace) -> None:
        # if not namespace.terrain_changed:
        # return

        print("asdf")

        combined_mesh = ChunkMeshData()
        chunk_meshes = []

        for chunk_id, chunk_data in namespace.chunks.items():
            if chunk_data["state"] == ChunkState.MESH_GENERATED:
                mesh_shm = shm.SharedMemory(chunk_data["mesh"])
                packed_mesh = np.ndarray(
                    (chunk_data["n_mesh"], ),
                    buffer=mesh_shm.buf,
                    dtype=instance_dtype
                )
                mesh_data = ChunkMeshData.unpack(packed_mesh)
                print("M", mesh_data.tex_id)
                chunk_meshes.append(mesh_data)
                mesh_shm.close()

        if chunk_meshes:
            combined_mesh.concatenate(chunk_meshes)

        combined_packed = combined_mesh.pack()

        deallocate_shared_memory(namespace.mesh_shm)

        if len(combined_packed) == 0:
            return

        new_mesh_shm = shm.SharedMemory(
            create=True,
            size=combined_packed.nbytes
        )

        new_mesh_array = np.ndarray(
            (combined_packed.size,),
            dtype=combined_packed.dtype,
            buffer=new_mesh_shm.buf
        )
        new_mesh_array[:] = combined_packed

        namespace.mesh_shm = new_mesh_shm.name
        namespace.mesh_len = len(new_mesh_array)
        namespace.terrain_changed = False

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
        self.namespace.terrain_changed = False
        self.namespace.mesh_shm = None
        self.namespace.mesh_len = 0
        self.namespace.alive = True

        logger.info("Starting the following workers:")
        logger.info(f"\tChunkBuilder x{N_WORKERS - 2}")
        logger.info(f"\tMeshBuilder x1")
        logger.info(f"\tWorld x1")

        self.processes = []
        
        for pid in range(N_WORKERS - 2):
            builder = multiprocessing.Process(
                target=self.build_worker,
                args=(self.namespace, pid)
            )
            self.namespace.queues[pid] = self.manager.list()
            builder.start()
            self.processes.append(builder)

        mesher = multiprocessing.Process(
            target=self.mesh_worker,
            args=(self.namespace, )
        )
        mesher.start()
        self.processes.append(mesher)

        worker = multiprocessing.Process(
            target=self.world_worker,
            args=(self.namespace, )
        )
        worker.start()
        self.processes.append(worker)

        logger.info("ChunkHandler instantiated.")

    @property
    def mesh_data(self) -> tuple[str, int]:
        return self.namespace.mesh_shm, self.namespace.mesh_len

    def set_camera_position(self, position: list[float]) -> None:
        self.namespace.camera = position

    def build_worker(self, namespace, pid: int) -> None:
        logger.info(f"Starting build worker {pid}")
        builder = ChunkBuilder()

        while namespace.alive:
            builder.step(namespace, pid)

    def mesh_worker(self, namespace) -> None:
        logger.info(f"Starting mesh worker")
        mesher = MeshBuilder()

        while namespace.alive:
            mesher.step(namespace)

    def world_worker(self, namespace) -> None:
        logger.info("Starting world worker")
        
        # Create a singular chunk at origin
        origin_chunk = Chunk((0, 0, 0))
        chunk_data = origin_chunk.to_dict("", 0, "", 0)
        namespace.chunks[origin_chunk.id_string] = chunk_data
        
        # Add to the first worker's queue for processing
        if 0 in namespace.queues:
            namespace.queues[0].append(origin_chunk.id_string)
        
        while namespace.alive:
            pass
    
    def kill(self) -> None:
        logger.info("Terminating workers")

        self.namespace.alive = False
        
        # Terminate worker processes
        for process in self.processes:
            process.terminate()
            process.join()
        
        # Deallocate shared memory
        deallocate_shared_memory(self.namespace.mesh_shm)
        
        # Deallocate terrain and mesh shared memory for all chunks
        for chunk_id, chunk_data in self.namespace.chunks.items():
            deallocate_shared_memory(chunk_data.get("terrain"))
            deallocate_shared_memory(chunk_data.get("mesh"))

