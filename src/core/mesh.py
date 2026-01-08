from __future__ import annotations
from typing import TYPE_CHECKING
import logging
logger = logging.getLogger(__name__) 

import ctypes
from typing import TypeAlias

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_FLOAT,
    GL_UNSIGNED_INT,
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

if TYPE_CHECKING:
    from .window import State

BufferData: TypeAlias = np.ndarray
instance_dtype = np.dtype([
    ("position", np.float32, 3),
    ("orientation", np.uint32),
    ("tex_id", np.float32),
    ("width", np.float32),
    ("height", np.float32),
])


class DisposableBuffer:
    def __init__(self, data: BufferData) -> None:
        self.data: BufferData = data
        self.buffer: np.uint32 = glGenBuffers(1)
        self.ready: bool = False

        logger.debug(f"Created DisposableBuffer with id {self.buffer}")

    def send_to_gpu(self) -> None:
        logger.debug(f"Sending to GPU: DisposableBuffer with id {self.buffer}")

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

    def get_latest_buffer(self) -> DisposableBuffer | None:
        latest = None
        for buffer in self.buffers:
            if not buffer.ready:
                continue
            latest = buffer
            break
        return latest

    def set_data(
        self, 
        position: BufferData, 
        orientation: BufferData, 
        tex_id: BufferData, 
        width: BufferData,
        height: BufferData
    ) -> None:
        if not (len(position) == len(tex_id) == len(orientation) == len(width) == len(height)):
            raise RuntimeError("buffer lengths don't match")

        if len(position) == 0:
            return

        data = np.zeros(len(position), dtype=instance_dtype)
        data['position'][:] = position
        data['orientation'][:] = orientation
        data['tex_id'][:] = tex_id
        data['width'][:] = width
        data['height'][:] = height

        logger.debug(f"Updating mesh data (length {len(data)})")
        buffer = DisposableBuffer(data)
        self.buffers.insert(0, buffer)

    def render(self) -> None:
        buffer = self.get_latest_buffer()
        if buffer is None:
            return

        glBindBuffer(GL_ARRAY_BUFFER, buffer.buffer)

        stride = buffer.data.strides[0]
        offset_pos = buffer.data.dtype.fields['position'][1]
        offset_ori = buffer.data.dtype.fields['orientation'][1]
        offset_tex = buffer.data.dtype.fields['tex_id'][1]
        offset_width = buffer.data.dtype.fields['width'][1]
        offset_height = buffer.data.dtype.fields['height'][1]

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0, 3, GL_FLOAT, False, 
            stride, ctypes.c_void_p(offset_pos)
        )
        glVertexAttribDivisor(0, 1)

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1, 1, GL_UNSIGNED_INT, False, 
            stride, ctypes.c_void_p(offset_ori)
        )
        glVertexAttribDivisor(1, 1)

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(
            2, 1, GL_FLOAT, False,
            stride, ctypes.c_void_p(offset_tex)
        )
        glVertexAttribDivisor(2, 1)

        glEnableVertexAttribArray(3)
        glVertexAttribPointer(
            3, 1, GL_FLOAT, False,
            stride, ctypes.c_void_p(offset_width)
        )
        glVertexAttribDivisor(3, 1)

        glEnableVertexAttribArray(4)
        glVertexAttribPointer(
            4, 1, GL_FLOAT, False,
            stride, ctypes.c_void_p(offset_height)
        )
        glVertexAttribDivisor(4, 1)

        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, len(buffer.data))

        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glDisableVertexAttribArray(3)
        glDisableVertexAttribArray(4)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def update_buffers(self) -> None:
        for buffer in self.buffers:
            if not buffer.ready:
                buffer.send_to_gpu()

        buffer_to_delete = None
        latest = self.get_latest_buffer()
        for buffer in self.buffers:
            if buffer == latest:
                continue
            if not buffer:
                continue

            buffer_to_delete = buffer
            break

        if buffer_to_delete is not None:
            self.buffers.remove(buffer_to_delete)
            del buffer_to_delete

    def on_close(self) -> None:
        del self.buffers


MeshStore: TypeAlias = dict[str, Mesh]


class MeshHandler:
    def __init__(self, state: State) -> None:
        self.state: State = state
        self.meshes: MeshStore = {}

        if self.state.mesh_handler is not None:
            raise Exception(
                "Tried to register multiple instances of MeshHandler"
            )
        self.state.mesh_handler = self

        logger.info("MeshHandler instantiated.")

    def new_mesh(self, id: str) -> Mesh:
        logger.info(f"Created new mesh with id {id}")

        buffer = Mesh(self.state)
        self.meshes[id] = buffer
        return buffer

    def get_mesh(self, id: str) -> Mesh:
        return self.meshes[id]

    def remove_mesh(self, id: str) -> None:
        logger.info(f"Removed mesh with id {id}")
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
        logger.info("Deallocating remaining buffers")
        try:
            for mesh in self.meshes:
                self.meshes[mesh].on_close()
        except RuntimeError:
            self.on_close()
        except KeyError:
            self.on_close()

