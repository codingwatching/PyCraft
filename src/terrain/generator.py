import numpy as np
import pyfastnoisesimd as fns
from type_hints import PositionType
from constants import CHUNK_SIDE, CHUNK_DIMS

seed = np.random.randint(2**31)
N_threads = 12
perlin = fns.Noise(seed=seed, numWorkers=N_threads)
perlin.frequency = 0.004
perlin.noiseType = fns.NoiseType.Perlin
perlin.fractal.octaves = 12
perlin.fractal.lacunarity = 128
perlin.fractal.gain = 42
perlin.perturb.perturbType = fns.PerturbType.NoPerturb

def terrain_generator(position: PositionType, level: int) -> np.ndarray:
    scale_factor: int = 2 ** level
    width, height = [float(scale_factor) for _ in range(2)]

    start_x = (position[0] * CHUNK_SIDE - 1) * width
    end_x   = ((position[0] + 1) * CHUNK_SIDE + 1) * width
    world_x = np.linspace(start_x, end_x, CHUNK_DIMS[0])

    start_y = (position[1] * CHUNK_SIDE - 1) * height
    end_y   = ((position[1] + 1) * CHUNK_SIDE + 1) * height
    world_y = np.linspace(start_y, end_y, CHUNK_DIMS[1])

    start_z = (position[2] * CHUNK_SIDE - 1) * width
    end_z   = ((position[2] + 1) * CHUNK_SIDE + 1) * width
    world_z = np.linspace(start_z, end_z, CHUNK_DIMS[2])

    x_grid, z_grid = np.meshgrid(world_x, world_z, indexing='ij')
    x_grid = x_grid.flatten()
    z_grid = z_grid.flatten()

    n = len(x_grid)
    coords = fns.empty_coords(n)
    coords[0, :n] = x_grid
    coords[1, :n] = np.full(n, 0)
    coords[2, :n] = z_grid

    heights = perlin \
        .genFromCoords(coords)[:n] \
        .reshape(CHUNK_SIDE + 2, CHUNK_SIDE + 2)
    height_field = heights * 32
    Y = world_y.reshape(1, -1, 1)
    mask = Y < height_field[:, None, :]

    terrain = np.zeros_like(mask, dtype=np.uint8)
    terrain[mask] = np.random \
        .randint(1, 3, size=np.count_nonzero(mask), dtype=np.uint8)

    return terrain

