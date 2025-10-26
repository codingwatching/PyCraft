from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.window import State

from core.mesh import Mesh
from type_hints import Position
from .chunk_handler import ChunkHandler
from constants import CHUNK_SIDE

class World:
    def __init__(self, state: State) -> None:
        self.state: State = state
        self.handler: ChunkHandler = ChunkHandler()
        self.state.world = self

        if self.state.mesh_handler is not None:
            self.mesh: Mesh = self.state.mesh_handler.new_mesh("world")
        else:
            raise RuntimeError("[terrain.world.World] tried to initialize world but the mesh handler wasn't registered")

    def update(self) -> None:
        camera_chunk: Position = (0, 0, 0)
        if self.state.camera is not None:
            player_position: list[float] = self.state.camera.position
            camera_chunk = (
                int(player_position[0] // CHUNK_SIDE),
                int(player_position[1] // CHUNK_SIDE),
                int(player_position[2] // CHUNK_SIDE)
            )
        self.handler.set_camera_chunk(camera_chunk)

        if not self.handler.changed:
            return

        data = self.handler.mesh_data
        self.mesh.set_data(*data)

    def on_close(self) -> None:
        self.handler.kill()

