import threading
import random

from terrain.chunk import *
from terrain.block import *
from player.player import *
from constants import *
from core.util import *

class ChunkGenerationThread(threading.Thread):
    def __init__(self, parent):
        threading.Thread.__init__(self, daemon=True)
        self.parent = parent

    def run(self):
        while not glfw.window_should_close(self.parent.renderer.parent):
            if len(self.parent.to_generate) > 0:
                chunk = self.parent.chunks[self.parent.to_generate.pop()]
                chunk.generate()

class World:
    def __init__(self, parent):
        self.renderer = parent
        self.block_data = get_blocks(parent, self)
        self.blocks = self.block_data["blocks"]

        self.chunks = {}
        self.render_distance = 3
        self.seed = random.randint(0, 1000000)

        self.player = Player(self)
        self.to_generate = []
        self.generator_thread = ChunkGenerationThread(self)
        self.generator_thread.start()

        self.generate()

    def generate_chunk(self, position):
        vbo_id = self.renderer.create_vbo(encode_position(position))
        self.chunks[position] = Chunk(position, self, vbo_id)
        self.to_generate.append(position)

    def generate(self):
        for i in range(-self.render_distance, self.render_distance):
            for j in range(-self.render_distance, self.render_distance):
                self.generate_chunk((i, j))

    def block_exists(self, position):
        try:
            for i in self.chunks.values():
                if i.block_exists(position):
                    return True
            return False
        except:
            return False

    def drawcall(self):
        self.player.update()

        # INFGEN
        position = (round(self.player.pos[0] // CHUNK_SIZE), round(self.player.pos[2] // CHUNK_SIZE))
        positions = []
        for i in range(-self.render_distance-1 + position[0], self.render_distance+1 + position[0]):
            for j in range(-self.render_distance-1 + position[1], self.render_distance+1 + position[1]):
                if (i, j) not in self.chunks:
                    self.generate_chunk((i, j))
                    positions.append((i, j))

        for chunk in self.chunks.values():
            chunk._update(position)
