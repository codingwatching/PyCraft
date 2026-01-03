from __future__ import annotations
from typing import TYPE_CHECKING
import logging
import numpy as np
from core.mesh import Mesh, instance_dtype
from .chunk_handler import ChunkHandler
from multiprocessing import shared_memory as shm
from .chunk import ChunkMeshData

logger = logging.getLogger(__name__) 

if TYPE_CHECKING:
    from core.window import State

class World:
    def __init__(self, state: State) -> None:
        self.state: State = state
        self.handler: ChunkHandler = ChunkHandler()
        self.state.world = self
        self.last_mesh = None

        if self.state.mesh_handler is not None:
            self.mesh: Mesh = self.state.mesh_handler.new_mesh("world")
        else:
            raise RuntimeError(
                "Tried to initialize World" + 
                "but MeshHandler wasn't registered"
            )

        logger.info("World instantiated.")

    def update(self) -> None:
        camera_position = [0.0, 0.0, 0.0]
        if self.state.camera is not None:
            camera_position = list(self.state.camera.position)
        self.handler.set_camera_position(camera_position)

        shm_name, length = self.handler.mesh_data
        if self.last_mesh == shm_name:
            return
        self.last_mesh = shm_name
            
        try:
            existing_shm = shm.SharedMemory(name=shm_name)
            packed_data = np.ndarray((length, ), dtype=instance_dtype, buffer=existing_shm.buf)
            mesh_data = ChunkMeshData.unpack(packed_data)
            
            self.mesh.set_data(
                mesh_data.position.astype(np.float32),
                mesh_data.orientation.astype(np.float32),
                mesh_data.tex_id.astype(np.float32),
                mesh_data.scale.astype(np.float32)
            )
            
            existing_shm.close()
        except FileNotFoundError:
            logger.warning(f"Shared memory {shm_name} not found")
        except Exception as e:
            logger.error(f"Error updating mesh: {e}")

    def on_close(self) -> None:
        logger.info("Sending kill signal to ChunkHandler")
        self.handler.kill()

