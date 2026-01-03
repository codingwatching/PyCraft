from __future__ import annotations
from time import sleep
from typing import Any
from time import perf_counter_ns
import logging
import glfw
from OpenGL.GL import GL_TRUE, glEnable, GL_MULTISAMPLE, glViewport

from terrain.world import World
from .renderer import Renderer
from .asset_manager import AssetManager
from .mesh import MeshHandler
from .camera import Camera
from player import Player

logger = logging.getLogger(__name__) 


class State:
    def __init__(self, window: Window) -> None:
        self.frame: int = 0
        self.window: Window = window
        self.alive: bool = True
        self.shared_context_alive: bool = False
        self.asset_manager: AssetManager | None = None
        self.mesh_handler: MeshHandler | None = None
        self.camera: Camera | None = None
        self.player: Player | None = None
        self.world: World | None = None

        self.last_frame_time: int = perf_counter_ns()
        self.fps: float = 0.0
        self.dt: float = 0.0

    def on_drawcall(self) -> None:
        now: int = perf_counter_ns()
        self.dt = (now - self.last_frame_time) / 1_000_000_000
        self.last_frame_time = now

        self.fps = 1 / self.dt

        self.frame += 1

    def on_close(self) -> None:
        self.alive = False


class Window:
    def __init__(self) -> None:
        if not glfw.init():
            raise Exception(
                "Init failed: Could not initialize glfw"
            )

        glfw.window_hint(glfw.SAMPLES, 1)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        glEnable(GL_MULTISAMPLE)

        logger.info("Creating main window")
        self.window: Any = glfw.create_window(640, 480, "Voxl", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception(
                "Init failed: Could not create glfw window"
            )
        glfw.make_context_current(self.window)
        glfw.swap_interval(0) # vsync

        self.state: State = State(self)
        self.renderer: Renderer = Renderer(self.state)
        self.world: World = World(self.state)

        logger.info("Window instantiated.")

    def mainloop(self) -> None:
        logger.info("Begin mainloop")

        while not glfw.window_should_close(self.window):
            width, height = self.size
            sleep(1/60)
            glViewport(0, 0, width, height)

            if self.state.player is not None:
                self.state.player.drawcall()
            self.renderer.drawcall()
            self.state.on_drawcall()

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        logger.info("Terminated!")
        self.state.on_close()
        while self.state.shared_context_alive:
            pass
        glfw.terminate()

    @property
    def size(self) -> tuple[int, int]:
        return glfw.get_window_size(self.window)

