import psutil

DEV_MODE = False
DISABLE_CHUNK_CULLING = False
USING_GRAPHICS_DEBUGGER = False
CHUNK_SIZE = 16
CHUNK_GENERATORS = psutil.cpu_count(logical=False) // 2
VERTICES_SIZE = 256 * 16 * 16 * 8 * 24
TEXCOORDS_SIZE = 256 * 16 * 16 * 8 * 16
MIN_FPS = 30
FPS_SAMPLES = 1000
FOV = 60
MAX_DEBUG_LINES = 32