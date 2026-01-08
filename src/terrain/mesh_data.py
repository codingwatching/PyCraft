from __future__ import annotations
import numpy as np
from core.mesh import instance_dtype

class ChunkMeshData:
    position: np.ndarray
    orientation: np.ndarray
    tex_id: np.ndarray
    width: np.ndarray
    height: np.ndarray

    def __init__(self):
        self.clear()

    def clear(self):
        self.position = np.array([])
        self.orientation = np.array([])
        self.tex_id = np.array([])
        self.width = np.array([])
        self.height = np.array([])

    def concatenate(self, others: list[ChunkMeshData]):
        if self.position.size == 0:
            all_data = [data for data in others if data.position.size > 0]
            if all_data:
                self.position = np.concatenate([data.position for data in all_data], axis=0)
                self.orientation = np.concatenate([data.orientation for data in all_data], axis=0)
                self.tex_id = np.concatenate([data.tex_id for data in all_data], axis=0)
                self.width = np.concatenate([data.width for data in all_data], axis=0)
                self.height = np.concatenate([data.height for data in all_data], axis=0)
        else:
            self.position = np.concatenate(
                (self.position, *(data.position for data in others if data.position.size > 0)),
                axis=0
            )
            self.orientation = np.concatenate(
                (self.orientation, *(data.orientation for data in others if data.orientation.size > 0)),
                axis=0
            )
            self.tex_id = np.concatenate(
                (self.tex_id, *(data.tex_id for data in others if data.tex_id.size > 0)),
                axis=0
            )
            self.width = np.concatenate(
                (self.width, *(data.width for data in others if data.width.size > 0)),
                axis=0
            )
            self.height = np.concatenate(
                (self.height, *(data.height for data in others if data.height.size > 0)),
                axis=0
            )

    def pack(self) -> np.ndarray:
        if self.position.size == 0:
            return np.empty(0, dtype=instance_dtype)

        n = self.position.shape[0]
        data = np.empty(n, dtype=instance_dtype)

        data["position"] = self.position.astype(np.float32, copy=False)
        data["orientation"] = self.orientation.astype(np.uint32, copy=False)
        data["tex_id"] = self.tex_id.astype(np.float32, copy=False)
        data["width"] = self.width.astype(np.float32, copy=False)
        data["height"] = self.height.astype(np.float32, copy=False)

        return data

    @staticmethod
    def unpack(packed_data: np.ndarray) -> ChunkMeshData:
        mesh_data = ChunkMeshData()

        if packed_data.size == 0:
            return mesh_data

        if packed_data.dtype != instance_dtype:
            raise TypeError(
                f"Invalid mesh dtype: {packed_data.dtype}, expected {instance_dtype}"
            )

        mesh_data.position = packed_data["position"].astype(np.float32, copy=False)
        mesh_data.orientation = packed_data["orientation"].astype(np.uint32, copy=False)
        mesh_data.tex_id = packed_data["tex_id"].astype(np.float32, copy=False)
        mesh_data.width = packed_data["width"].astype(np.float32, copy=False)
        mesh_data.height = packed_data["height"].astype(np.float32, copy=False)

        return mesh_data

