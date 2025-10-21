from time import perf_counter_ns
from typing import Any


class State:
    def __init__(self, window: Any) -> None:
        self.frame: int = 0
        self.window: Any = window
        self.alive: bool = True
        self.shared_context_alive: bool = False
        self.asset_manager: Any | None = None
        self.mesh_handler: Any | None = None
        self.camera: Any | None = None
        self.player: Any | None = None
        self.world: Any | None = None

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

