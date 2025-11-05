import pyfastnoisesimd as fns
import numpy as np
shape = [512, 512, 512]
seed = np.random.randint(2**31)
N_threads = 4

perlin = fns.Noise(seed=seed, numWorkers=N_threads)
perlin.frequency = 0.02
perlin.noiseType = fns.NoiseType.Perlin
perlin.fractal.octaves = 4
perlin.fractal.lacunarity = 2.1
perlin.fractal.gain = 0.45
perlin.perturb.perturbType = fns.PerturbType.NoPerturb
result = perlin.genAsGrid(shape)

print(result)
