import logging
from multiprocessing import shared_memory as shm

logger = logging.getLogger(__name__)

def deallocate_shared_memory(name: str) -> None:
    """Safely deallocate shared memory by name.
    
    Args:
        name: The name of the shared memory block to deallocate
    """
    if name is None:
        return
        
    try:
        memory = shm.SharedMemory(name)
        memory.close()
        memory.unlink()
        logger.debug(f"Deallocated shared memory: {name}")
    except FileNotFoundError:
        logger.debug(f"Shared memory {name} not found, skipping deallocation")
    except Exception as e:
        logger.warning(f"Failed to deallocate shared memory {name}: {e}")
