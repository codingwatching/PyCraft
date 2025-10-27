from __future__ import annotations
from typing import TYPE_CHECKING, Any
import logging
logger = logging.getLogger(__name__) 

import threading

if TYPE_CHECKING:
    from .window import State, Window

import glfw

class SharedContext:
    def __init__(self, state: State) -> None:
        self.state: State = state
        self.parent: Window = state.window
        self.thread: threading.Thread | None = None
        self.window: Any | None = None

        logger.info("SharedContext instantiated.")

    def start_thread(self) -> None:
        logger.info("Starting thread")
        if self.thread is not None:
            raise Exception(
                "Tried to start thread multiple times"
            )
        self.thread = threading.Thread(
            target=self.start,
        )
        self.thread.start()

    def start(self) -> None:
        logger.debug("Creating window")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(
            1, 1, "Shared Context", None, self.parent.window
        )
        self.state.shared_context_alive = True

        glfw.make_context_current(self.window)

        logger.info("Starting mainloop")
        while self.state.alive:
            self.step()
        
        logger.info("Cleaning up")
        if self.state.world:
            self.state.world.on_close()
        if self.state.mesh_handler:
            self.state.mesh_handler.on_close()

        glfw.destroy_window(self.window)
        self.state.shared_context_alive = False

    def step(self) -> None:
        if self.state.world:
            self.state.world.update()

        if self.state.mesh_handler:
            self.state.mesh_handler.update()

        if self.window is not None:
            glfw.swap_buffers(self.window)
        else:
            raise RuntimeError("[core.shared_context.SharedContext] tried to call self.step() but self.window is None")

        glfw.poll_events()

