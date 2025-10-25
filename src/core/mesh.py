from typing import TypeAlias

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    glBindBuffer,
    glBufferData,
    glBufferSubData,
    glDeleteBuffers,
    glDisableVertexAttribArray,
    glDrawArraysInstanced,
    glEnableVertexAttribArray,
    glFlush,
    glGenBuffers,
    glVertexAttribPointer,
    glVertexAttribDivisor,
)

from .state import State

BufferData: TypeAlias = np.typing.NDArray[np.float32]

VERTEX = 2
UV = 3


class DisposableBuffer:
    def __init__(self, data: BufferData, type: int = VERTEX) -> None:
        self.data: BufferData = data
        self.type: int = type
        self.buffer: np.uint32 = glGenBuffers(1)
        self.ready: bool = False

    def send_to_gpu(self) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, self.buffer)
        glBufferData(GL_ARRAY_BUFFER, self.data.nbytes, None, GL_STATIC_DRAW)
        glBufferSubData(GL_ARRAY_BUFFER, 0, self.data.nbytes, self.data)
        glFlush()
        self.ready = True

    def __del__(self) -> None:
        glDeleteBuffers(1, self.buffer)
        del self.data

BufferList: TypeAlias = list[DisposableBuffer]

class Mesh:
    def __init__(
        self,
        state: State,
    ) -> None:
        self.buffers: BufferList = []
        self.state: State = state
        self.last_data_hash = None

    def get_latest_buffer(self) -> DisposableBuffer | None:
        latest = None
        for buffer in self.buffers:
            if not buffer.ready:
                continue
            latest = buffer
            break
        return latest

    def set_data(self, data: BufferData) -> None:
        data.flags.writeable = False
        data_hash = hash(data.tobytes())

        if self.last_data_hash == data_hash:
            return

        buffer = DisposableBuffer(data)
        self.buffers.insert(0, buffer)
        self.last_data_hash = data_hash

    def render(self) -> None:
        buffer = self.get_latest_buffer()
        if buffer is None:
            return

        glBindBuffer(GL_ARRAY_BUFFER, buffer.buffer)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,None)
        glVertexAttribDivisor(0,1)

        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, len(buffer.data))

        glDisableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_buffers(self) -> None:
        for buffer in self.buffers:
            if not buffer.ready:
                buffer.send_to_gpu()

        to_delete = []
        latest = self.get_latest_buffer()
        for buffer in self.buffers:
            if buffer == latest:
                continue
            if not buffer:
                continue
            to_delete.append(buffer)

        for buffer in to_delete:
            self.buffers.remove(buffer)
            del buffer

    def on_close(self) -> None:
        del self.buffers


MeshStore: TypeAlias = dict[str, Mesh]


class MeshHandler:
    def __init__(self, state: State) -> None:
        self.state: State = state
        self.meshes: MeshStore = {}

        if self.state.mesh_handler is not None:
            raise Exception(
                "[core.mesh.MeshHandler] Tried to create multiple instances of this class"
            )
        self.state.mesh_handler = self

    def new_mesh(self, id: str) -> Mesh:
        buffer = Mesh(self.state)
        self.meshes[id] = buffer
        return buffer

    def get_mesh(self, id: str) -> Mesh:
        return self.meshes[id]

    def remove_buffer(self, id: str) -> None:
        del self.meshes[id]

    def drawcall(self) -> None:
        try:
            for mesh in self.meshes:
                self.meshes[mesh].render()
        except RuntimeError:
            pass

    def update(self) -> None:
        try:
            for mesh in self.meshes:
                self.meshes[mesh].update_buffers()
        except RuntimeError:
            pass
        except IndexError:
            pass

    def on_close(self) -> None:
        try:
            for mesh in self.meshes:
                self.meshes[mesh].on_close()
        except RuntimeError:
            self.on_close()
        except KeyError:
            self.on_close()

