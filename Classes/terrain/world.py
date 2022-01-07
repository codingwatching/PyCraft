# imports
import pyglet
from pyglet.gl import *
import os
import random
from tqdm import trange
import opensimplex
import multiprocessing
import pyximport
pyximport.install()

# Inbuilt imports
from logger import *
import Classes as pycraft
from helpers.fast_func_executor import *

# all the block types
blocks_all = {}

# Function to load a texture
def load_texture(filename):
    """
    load_texture

    * Loads a texture from a file

    :filename: path of the file to load
    """
    try:
        tex = pyglet.image.load(filename).get_texture()
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        return pyglet.graphics.TextureGroup(tex)
    except:
        warn("Texture Loader", "Failed to load texture: " + filename)
        return None

class World:
    """
    World

    * Initializes the world
    """
    def __init__(self, parent):
        """
        World.__init__

        :parent: the parent window
        """
        self.parent = parent
        self.generator = pycraft.TerrainGenerator
        self.batch = pyglet.graphics.Batch()

        self.textures = {}
        self.block_types = {}
        self.liquid_types = {}
        self.structure_types = {}
        self.biomes = {}
        
        self.all_blocks = {}
        self._independent_blocks = {}
        self.all_liquids = {}
        self.all_chunks = {}

        self.seed = random.randint(0, 100000)
        self._noise = opensimplex.OpenSimplex(self.seed)

        self._load_textures()
        self._load_block_types()
        self._load_liquid_types()
        self._load_structures()
        self._load_biomes()

        self.render_distance = 5
        self.chunk_size = 8
        self.infgen_threshold = 0
        self.position = [0, 0]
        self._process_per_frame = 1 + round(multiprocessing.cpu_count() * 0.2)
        self.chunk_generation_delay = 1

        if self._process_per_frame <= 0:
            self._process_per_frame = 1

        info("World", "Processes per frame: " + str(self._process_per_frame))

        self._queue = []
        self._frame = 0
        self.generate()

        self.sky_color = (0.5, 0.7, 1)
        self.light_color = [5,5,5,5]
        self.daynight_min = 1
        self.daynight_max = 5
        self.light_change = 0

        # Enable fog
        glEnable(GL_FOG)
        glFogfv(GL_FOG_COLOR, (GLfloat * int(self.render_distance*16))(0.5, 0.69, 1.0, 10))
        glHint(GL_FOG_HINT, GL_DONT_CARE)
        glFogi(GL_FOG_MODE, GL_LINEAR)
        glFogf(GL_FOG_START, self.render_distance*4)
        glFogf(GL_FOG_END, self.render_distance*5)

        # Texture blending
        glEnable (GL_LINE_SMOOTH)
        glEnable (GL_BLEND)
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint (GL_LINE_SMOOTH_HINT, GL_DONT_CARE)

        # Lighting
        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT7, GL_AMBIENT, (GLfloat*4)(1,1,1,1))
        glEnable(GL_LIGHT7)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        self.cloud_generator = pycraft.CloudGenerator(self)

    def _load_textures(self):
        """
        _load_textures

        * Loads all the textures
        """
        log("Texture Loader", "Loading textures...")
        for i in os.listdir("assets/textures/block"):
            if i.endswith(".png"):
                self.textures[i.split(".")[0]] = load_texture("assets/textures/block"+"/"+i)
        log("Texture Loader", "Loaded " + str(len(self.textures)) + " textures")

    def _load_block_types(self):
        """
        _load_block_types

        * Loads all the block types
        """
        import importlib

        log("Block Loader", "Loading blocks...")
        for i in os.listdir("Classes/terrain/blocks"):
            if i.endswith(".py") and i != "__init__.py":
                self.block_types[i.split(".")[0]] = importlib.import_module("Classes.terrain.blocks." + i.split(".")[0]).Block(self)
        log("Block Loader", "Loaded " + str(len(self.block_types)) + " blocks")

    def _load_liquid_types(self):
        """
        _load_liquid_types

        * Loads all the liquid types
        """
        import importlib

        log("Liquid Loader", "Loading liquids...")
        for i in os.listdir("Classes/terrain/liquids"):
            if i.endswith(".py") and i != "__init__.py":
                self.liquid_types[i.split(".")[0]] = importlib.import_module("Classes.terrain.liquids." + i.split(".")[0]).Liquid(self)
        log("Liquid Loader", "Loaded " + str(len(self.liquid_types)) + " liquids")

    def _load_structures(self):
        """
        _load_structures

        * Loads all the structures
        """
        log("Structure Loader", "Loading structures...")
        self.structure_types = pycraft.load_structures(self)
        log("Structure Loader", "Loaded " + str(len(self.structure_types)) + " structures")

    def _load_biomes(self):
        """
        _load_biomes

        * Loads all the biomes
        """
        log("Biome Loader", "Loading biomes...")
        self.biomes = pycraft.load_biomes(self)
        log("Biome Loader", "Loaded " + str(len(self.biomes)) + " biomes")

    def generate(self):
        """
        generate

        * Generates the world
        """
        info("World", "Generating world...")
        for i in trange(-self.render_distance+1, self.render_distance):
            for j in range(-self.render_distance+1, self.render_distance):
                fast_exec(lambda: self.make_chunk((i, j)))

    def update(self):
        """
        update

        * Updates the world
        """
        # Updates the chunks
        for i in self.all_chunks:
            self.all_chunks[i].update()

        # Runs the queue
        if self._frame % 2 == 0:
            fast_exec(self._process_queue_item)

        # Updates the liquids
        for i in self.liquid_types:
            fast_exec(self.liquid_types[i].update)

        # INFGEN
        if self.parent.player.pos[0] / self.chunk_size > self.position[0] + self.infgen_threshold:
            self.add_row_x_plus()
        if self.parent.player.pos[0] / self.chunk_size < self.position[0] - self.infgen_threshold:
            self.add_row_x_minus()
        if self.parent.player.pos[2] / self.chunk_size > self.position[1] + self.infgen_threshold:
            self.add_row_z_plus()
        if self.parent.player.pos[2] / self.chunk_size < self.position[1] - self.infgen_threshold:
            self.add_row_z_minus()

        # Daynight cycle
        self._daynight_cycle()

    def _daynight_cycle(self):
        """
        _daynight_cycle

        * Updates the daynight cycle
        """
        if self.light_color[0] > self.daynight_max:
            self.light_change = -0.001
        elif self.light_color[0] < self.daynight_min:
            self.light_change = 0.001
        self.light_color[0] += self.light_change
        self.light_color[1] += self.light_change
        self.light_color[2] += self.light_change

    def draw(self):
        """
        draw

        * Draws the world
        """
        self._frame += 1
        glClearColor(*self.sky_color, 255)
        # Lights
        glLightfv(GL_LIGHT7, GL_AMBIENT, (GLfloat * 4)(*self.light_color))

        # Draw clouds and chunks
        self.cloud_generator.draw()
        self.batch.draw()

        if self.parent.player.pointing_at[0] is not None:
            self.draw_cube(self.parent.player.pointing_at[0][0], self.parent.player.pointing_at[0][1], self.parent.player.pointing_at[0][2], 1)

    @staticmethod
    def draw_cube(x, y, z, size):
        """
        draw_cube

        * Draws a cube.

        :x: The x coordinate of the cube.
        :y: The y coordinate of the cube.
        :z: The z coordinate of the cube.
        :size: The size of the cube.
        """
        X = x + size + 0.01
        Y = y + size + 0.01
        Z = z + size + 0.01
        x = x - 0.01
        y = y - 0.01
        z = z - 0.01
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v3f', (x, Y, Z,  X, Y, Z,  X, Y, z,  x, Y, z)), ('c3B', (255, 255, 255) * 4))
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v3f', (x, y, z,  X, y, z,  X, y, Z,  x, y, Z)), ('c3B', (255, 255, 255) * 4))
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v3f', (x, y, z,  x, y, Z,  x, Y, Z,  x, Y, z)), ('c3B', (255, 255, 255) * 4))
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v3f', (X, y, Z,  X, y, z,  X, Y, z,  X, Y, Z)), ('c3B', (255, 255, 255) * 4))
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v3f', (x, y, Z,  X, y, Z,  X, Y, Z,  x, Y, Z)), ('c3B', (255, 255, 255) * 4))
        pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v3f', (X, y, z,  x, y, z,  x, Y, z,  X, Y, z)), ('c3B', (255, 255, 255) * 4))

    def block_exists(self, position):
        """
        block_exists

        * Checks if a block exists at a position

        :position: the position to cdheck
        """
        return position in self.all_blocks

    def liquid_exists(self, position):
        """
        liquid_exists

        * Checks if a liquid exists at a position

        :position: the position to cdheck
        """
        return position in self.all_liquids

    def _process_liquid_instances(self):
        """
        _process_liquid_instances

        * Processes the liquid instances
        """
        for i in self.liquid_types:
            self.liquid_types[i]._process_preloads()

    def chunk_exists(self, position):
        """
        chunk_exists

        * Checks if a chunk exists at a position

        :position: the position to cdheck
        """
        return position in self.all_chunks

    def make_chunk(self, position):
        """
        make_chunk

        * Makes a chunk at a position

        :position: the position to make the chunk at
        """
        self.all_chunks[position] = pycraft.Chunk(self, {'x': position[0], 'z': position[1]})
        self._queue.append(position)

    def make_structure(self, position, structure_type, chunk):
        """
        make_structures

        * Makes all the structures at a position

        :position: the position to make the structures at
        """
        chunk.structures[position] = self.structure_types[structure_type](chunk, position)
        chunk.structures[position].generate()

    def _process_queue_item(self):
        """
        _process_queue_item

        * Processes an item in the queue
        """
        for i in range(self._process_per_frame):
            if len(self._queue) > 0:
                random_index = random.randint(0, len(self._queue) - 1)

                item = self._queue[random_index]
                pyglet.clock.schedule_once(lambda x: self.all_chunks[item].generate(), random.randint(0, self.chunk_generation_delay))
                self._queue.pop(random_index)

    def get_block(self, position):
        """
        get_block

        * Gets a block at a position

        :position: the position to get the block from
        """
        if position in self.all_blocks:
            return self.all_blocks[position]
        else:
            return None
        
    def add_block(self, position, block, chunk):
        """
        add_block

        * Adds a block at a position

        :position: the position to add the block to
        :block: the block to add
        """
        self.all_blocks[position] = block
        chunk.add_block(block, position)

    def add_liquid(self, position, liquid):
        """
        add_liquid

        * Adds a liquid at a position

        :position: the position to add the liquid to
        :liquid: the liquid to add
        """
        self.all_liquids[position] = liquid
        self.liquid_types[liquid].add_preloaded_instance(position)

    def remove_block(self, position):
        """
        remove_block

        * Removes a block at a position

        :position: the position to remove the block from
        """
        if tuple(position) in self.all_blocks:
            try:
                blocktype = self.all_blocks[tuple(position)][0]
                self.block_types[blocktype].remove(tuple(position))
            except TypeError:
                pass

    def add_row_x_minus(self):
        """
        add_row_x_minus

        * Adds a row of blocks to the world.
        """
        for z in range(-self.render_distance, self.render_distance):
            if not self.chunk_exists((self.position[0]-self.render_distance, self.position[1]+z)):
                self.make_chunk((self.position[0]-self.render_distance, self.position[1]+z))
        self.position[0] -= 1

    def add_row_x_plus(self):
        """
        add_row_x_plus

        * Adds a row of blocks to the world.
        """
        for z in range(-self.render_distance, self.render_distance):
            if not self.chunk_exists((self.position[0]+self.render_distance, self.position[1]+z)):
                self.make_chunk((self.position[0]+self.render_distance, self.position[1]+z))
        self.position[0] += 1

    def add_row_z_minus(self):
        """
        add_row_z_minus

        * Adds a row of blocks to the world.
        """
        for x in range(-self.render_distance, self.render_distance):
            if not self.chunk_exists((self.position[0]+x, self.position[1]-self.render_distance)):
                self.make_chunk((self.position[0]+x, self.position[1]-self.render_distance))
        self.position[1] -= 1

    def add_row_z_plus(self):
        """
        add_row_z_plus

        * Adds a row of blocks to the world.
        """
        for x in range(-self.render_distance, self.render_distance):
            if not self.chunk_exists((self.position[0]+x, self.position[1]+self.render_distance)):
                self.make_chunk((self.position[0]+x, self.position[1]+self.render_distance))
        self.position[1] += 1
