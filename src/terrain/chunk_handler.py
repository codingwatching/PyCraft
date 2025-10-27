import logging
logger = logging.getLogger(__name__) 

from time import sleep
import multiprocessing
from multiprocessing.managers import SyncManager, NamespaceProxy
from core.mesh import BufferData
from .chunk import ChunkStorage
from type_hints import Position

class ChunkHandler:
    manager: SyncManager
    namespace: NamespaceProxy
    process: multiprocessing.Process

    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.namespace = self.manager.Namespace()
        self.namespace.mesh_data = None
        self.namespace.changed = False
        self.namespace.camera_chunk = (0, 0, 0)
        self.namespace.alive = True

        logger.info("Starting worker")

        self.process = multiprocessing.Process(
            target=self.worker, args=(self.namespace,)
        )
        self.process.start()

        logger.info("ChunkHandler instantiated.")

    def worker(self, namespace: NamespaceProxy) -> None:
        storage = ChunkStorage()

        logger.info("Worker: starting mainloop")
        while self.namespace.alive:
            storage.update(namespace.camera_chunk)

            if not storage.changed:
                sleep(1/60)
                continue

            namespace.mesh_data = storage.generate_mesh_data()
            namespace.changed = True

    @property
    def mesh_data(self) -> tuple[BufferData, BufferData, BufferData]:
        self.namespace.changed = False
        return self.namespace.mesh_data

    @property
    def changed(self) -> bool:
        return self.namespace.changed

    def set_camera_chunk(self, position: Position) -> None:
        self.namespace.camera_chunk = position

    def kill(self) -> None:
        logger.info("Terminating worker")

        self.namespace.alive = False
        self.process.terminate()

