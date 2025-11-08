from __future__ import annotations
from typing import TYPE_CHECKING

try:
    from pyglm import glm
except ImportError:
    import glm

if TYPE_CHECKING:
    from .window import State


class Camera:
    def __init__(self, state: State):
        self.fov: float = 45.0
        self.aspect: float = 16 / 9
        self.near: float = 1.0
        self.far: float = 1000000000.0

        self.state: State = state
        self.position: list[float] = [0.0, 0.0, 0.0]
        self.rotation: list[float] = [0.0, 0.0, 0.0]

        self.state.camera = self

    def get_matrix(self) -> tuple[glm.mat4, glm.mat4]:
        matrix = glm.mat4(1.0)
        for i in range(3):
            thing = glm.vec3(0.0)
            thing[i] = 1.0
            matrix = glm.rotate(
                matrix,
                glm.radians(self.rotation[i]),
                thing
            )

        position = [-i for i in self.position]
        matrix = glm.translate(matrix, glm.vec3(position))

        size: tuple[int, int] = self.state.window.size
        width: int = size[0]
        height: int = size[1]
        self.aspect = 1
        if height != 0:
            self.aspect = width / height
        
        projection = glm.perspective(
            self.fov,
            self.aspect,
            self.near,
            self.far
        )
        
        return matrix, projection

