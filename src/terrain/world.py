from __future__ import annotations
from typing import TYPE_CHECKING
import logging
from core.mesh import Mesh
from .chunk_handler import ChunkHandler

logger = logging.getLogger(__name__) 

if TYPE_CHECKING:
    from core.window import State

class World:
    def __init__(self, state: State) -> None:
        self.state: State = state
        self.handler: ChunkHandler = ChunkHandler()
        self.state.world = self

        if self.state.mesh_handler is not None:
            self.mesh: Mesh = self.state.mesh_handler.new_mesh("world")
        else:
            raise RuntimeError(
                "Tried to initialize World" + 
                "but MeshHandler wasn't registered"
            )

        logger.info("World instantiated.")

    def update(self) -> None:
        camera_position: list[float] = (0, 0, 0)
        if self.state.camera is not None:
            camera_position: list[float] = self.state.camera.position
        self.handler.set_camera_position(camera_position)

        if not self.handler.changed:
            return

        data = self.handler.mesh_data
        self.mesh.set_data(*data)

    def on_close(self) -> None:
        logger.info("Sending kill signal to ChunkHandler")
        self.handler.kill()

