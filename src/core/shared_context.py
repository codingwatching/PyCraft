from __future__ import annotations
from typing import TYPE_CHECKING, Any

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

    def start_thread(self) -> None:
        if self.thread is not None:
            raise Exception(
                "[core.shared_context.SharedContext] Tried to start thread multiple times"
            )
        self.thread = threading.Thread(
            target=self.start,
        )
        self.thread.start()

    def start(self) -> None:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(
            1, 1, "Shared Context", None, self.parent.window
        )
        self.state.shared_context_alive = True

        glfw.make_context_current(self.window)

        while self.state.alive:
            self.step()
        
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

