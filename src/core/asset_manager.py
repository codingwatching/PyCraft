from __future__ import annotations
from typing import TYPE_CHECKING
import logging
logger = logging.getLogger(__name__)

import os
from time import perf_counter
from typing import Any

from OpenGL.raw.GL.VERSION.GL_1_1 import GL_NEAREST_MIPMAP_NEAREST
import numpy as np
from OpenGL.GL import (
    GL_FRAGMENT_SHADER,
    GL_NEAREST,
    GL_NEAREST_MIPMAP_NEAREST,
    GL_REPEAT,
    GL_RGBA8,
    GL_RGBA,
    GL_TEXTURE_2D_ARRAY,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    glBindTexture,
    glGenTextures,
    glTexImage3D,
    glTexParameteri,
    glUseProgram,
    glGenerateMipmap,
)
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram
from PIL import Image

if TYPE_CHECKING:
    from .window import State

ASSET_DIR = "./assets/"


class AssetManager:
    def __init__(self, state: State) -> None:
        self.state: State = state
        self.state.asset_manager = self

        self.shaders: dict[str, ShaderProgram] = {}
        self.texture: int = 0

    def load_assets(self, asset_dir: str = ASSET_DIR, name_prefix: str = "") -> None:
        logger.info("Loading assets")
        t0 = perf_counter()

        shader_dir = os.path.join(asset_dir, "shaders/")
        shader_pairs: set[str] = set()

        logger.info("Loading shaders")
        for file in os.listdir(shader_dir):
            name = file.split(".")[0]
            shader_pairs.add(name)

        for name in shader_pairs:
            logger.info(f"\t{name}")

            # todo error handling
            with open(os.path.join(shader_dir, name + ".vert")) as f:
                vert: np.uint32 = compileShader(f.read(), GL_VERTEX_SHADER)
            with open(os.path.join(shader_dir, name + ".frag")) as f:
                frag: np.uint32 = compileShader(f.read(), GL_FRAGMENT_SHADER)
            program: ShaderProgram = compileProgram(vert, frag)
            self.shaders[name_prefix + name] = program

        logger.info("Loading textures")
        self.texture: np.uint32 = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT)

        textures = []
        w, h = -1, -1

        texture_dir = os.path.join(asset_dir, "textures/")
        for file in os.listdir(texture_dir):
            logger.info(f"\t{file}")
            texture = Image.open(os.path.join(texture_dir, file)).convert("RGBA")
            w, h = texture.size
            data = np.array(texture)
            textures.append(data)

        if w < 0 or h < 0:
            return

        layer_data = np.stack(textures, axis=0)
        glTexImage3D(
            GL_TEXTURE_2D_ARRAY, 
            0, GL_RGBA8, 
            w, h, len(textures),
            0, GL_RGBA, 
            GL_UNSIGNED_BYTE, 
            layer_data
        )

        glGenerateMipmap(GL_TEXTURE_2D_ARRAY)
        logger.info(f"Asset load took {perf_counter() - t0}s")

    def use_shader(self, name: str) -> None:
        if name not in self.shaders:
            raise Exception(
                f"Tried to use shader {name} but it doesn't exist or isn't loaded yet"
            )
        glUseProgram(self.shaders[name])

    def get_shader_program(self, name: str) -> Any | None:
        if name not in self.shaders:
            raise Exception(
                f"Tried to use shader {name} but it doesn't exist or isn't loaded yet"
            )
        return self.shaders[name]

    def bind_texture(self) -> None:
        if self.texture is None:
            return
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture)

