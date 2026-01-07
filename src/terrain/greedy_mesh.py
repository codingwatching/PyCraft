from __future__ import annotations
import numpy as np
from typing import Tuple
from constants import FACES, CHUNK_SIDE


def greedy_mesh_terrain(terrain: np.ndarray, blocks: np.ndarray, chunk_pos: Tuple[int, int, int], chunk_width: float, chunk_height: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply binary greedy meshing to terrain data.
    
    Args:
        terrain: 3D array of voxel data
        blocks: Block type array for texture coordinates
        chunk_pos: Chunk position in world coordinates
        chunk_width: Chunk width scaling
        chunk_height: Chunk height scaling
        
    Returns:
        Tuple of (positions, orientations, tex_ids, widths, heights)
    """
    if not np.any(terrain):
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    
    solid = terrain[1:-1, 1:-1, 1:-1] > 0
    if not np.any(solid):
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    
    all_positions = []
    all_orientations = []
    all_tex_ids = []
    all_widths = []
    all_heights = []
    
    for face, (dx, dy, dz) in FACES:
        neighbor = terrain[
            (1 + dx):(-1 + dx) or None,
            (1 + dy):(-1 + dy) or None,
            (1 + dz):(-1 + dz) or None
        ]
        visible_mask = solid & (neighbor == 0)
        
        if not np.any(visible_mask):
            continue
        
        visible_blocks = terrain[1:-1, 1:-1, 1:-1][visible_mask]
        vx, vy, vz = np.nonzero(visible_mask)
        
        # Group by block type and apply greedy meshing
        for block_type in np.unique(visible_blocks):
            type_mask = (visible_blocks == block_type)
            type_vx = vx[type_mask]
            type_vy = vy[type_mask]
            type_vz = vz[type_mask]
            
            if len(type_vx) == 0:
                continue
                
            # Create a mask for just this block type
            type_grid = np.zeros_like(solid, dtype=bool)
            type_grid[type_vx, type_vy, type_vz] = True
            
            # Apply greedy meshing to this block type
            quads = _greedy_mesh_3d(type_grid, face, block_type)
            
            for quad_x, quad_y, quad_z, quad_w, quad_h in quads:
                # Create a position array for this quad
                if face in [0, 1]:  # FRONT/BACK faces
                    quad_positions = np.array([[quad_x, quad_y, quad_z]], dtype=np.float32)
                elif face in [2, 3]:  # LEFT/RIGHT faces
                    quad_positions = np.array([[quad_x, quad_y, quad_z]], dtype=np.float32)
                else:  # TOP/BOTTOM faces
                    quad_positions = np.array([[quad_x, quad_y, quad_z]], dtype=np.float32)
                
                all_positions.append(quad_positions)
                all_orientations.append(np.full(1, face, np.uint32))
                all_tex_ids.append(np.full(1, block_type, np.float32))
                all_widths.append(np.full(1, quad_w, np.float32))
                all_heights.append(np.full(1, quad_h, np.float32))
    
    if not all_positions:
        return (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    
    # Concatenate all face results
    positions = np.concatenate(all_positions)
    orientations = np.concatenate(all_orientations)
    tex_ids = np.concatenate(all_tex_ids)
    widths = np.concatenate(all_widths)
    heights = np.concatenate(all_heights)
    
    # Transform positions to world coordinates
    world_positions = np.zeros_like(positions)
    world_positions[:, 0] = (chunk_pos[0] * CHUNK_SIDE + positions[:, 0]) * chunk_width
    world_positions[:, 1] = (chunk_pos[1] * CHUNK_SIDE + positions[:, 1]) * chunk_height
    world_positions[:, 2] = (chunk_pos[2] * CHUNK_SIDE + positions[:, 2]) * chunk_width
    
    # Apply block type tex IDs from the original system
    tex_ids = blocks[tex_ids.astype(np.uint8), orientations.astype(np.uint8)]
    
    # Scale widths and heights
    widths *= chunk_width
    heights *= chunk_height
    
    return (world_positions, orientations, tex_ids, widths, heights)


def _greedy_mesh_3d(mask: np.ndarray, face: int, block_type: int) -> list:
    """
    Apply greedy meshing to a 3D mask for a specific face.
    
    Args:
        mask: 3D boolean array where True indicates visible voxels
        face: The face direction being meshed
        block_type: The block type ID
        
    Returns:
        List of (x, y, z, width, height) tuples for greedy meshed quads
    """
    quads = []
    height, width, depth = mask.shape
    
    # Determine which axes to work in based on face direction
    if face in [0, 1]:  # FRONT/BACK - work in X-Y planes
        for z in range(depth):
            if not np.any(mask[:, :, z]):
                continue
                
            visited = np.zeros((height, width), dtype=bool)
            
            for y in range(height):
                x = 0
                while x < width:
                    if mask[y, x, z] and not visited[y, x]:
                        # Find maximum width
                        max_width = 1
                        for k in range(x + 1, width):
                            if mask[y, k, z] and not visited[y, k]:
                                max_width += 1
                            else:
                                break
                        
                        # Find maximum height
                        max_height = 1
                        for h in range(y + 1, height):
                            can_extend = True
                            for w in range(max_width):
                                if x + w >= width or not mask[h, x + w, z] or visited[h, x + w]:
                                    can_extend = False
                                    break
                            if can_extend:
                                max_height += 1
                            else:
                                break
                        
                        # Mark visited
                        for h in range(max_height):
                            for w in range(max_width):
                                visited[y + h, x + w] = True
                        
                        quads.append((x, y, z, max_width, max_height))
                        x += max_width
                    else:
                        x += 1
                        
    elif face in [2, 3]:  # LEFT/RIGHT - work in Y-Z planes
        for x in range(width):
            if not np.any(mask[:, x, :]):
                continue
                
            visited = np.zeros((height, depth), dtype=bool)
            
            for y in range(height):
                z = 0
                while z < depth:
                    if mask[y, x, z] and not visited[y, z]:
                        # Find maximum width (in Z)
                        max_width = 1
                        for k in range(z + 1, depth):
                            if mask[y, x, k] and not visited[y, k]:
                                max_width += 1
                            else:
                                break
                        
                        # Find maximum height (in Y)
                        max_height = 1
                        for h in range(y + 1, height):
                            can_extend = True
                            for w in range(max_width):
                                if z + w >= depth or not mask[h, x, z + w] or visited[h, z + w]:
                                    can_extend = False
                                    break
                            if can_extend:
                                max_height += 1
                            else:
                                break
                        
                        # Mark visited
                        for h in range(max_height):
                            for w in range(max_width):
                                visited[y + h, z + w] = True
                        
                        quads.append((x, y, z, max_width, max_height))
                        z += max_width
                    else:
                        z += 1
                        
    else:  # TOP/BOTTOM - work in X-Z planes
        for y in range(height):
            if not np.any(mask[y, :, :]):
                continue
                
            visited = np.zeros((width, depth), dtype=bool)
            
            for x in range(width):
                z = 0
                while z < depth:
                    if mask[y, x, z] and not visited[x, z]:
                        # Find maximum width (in Z)
                        max_width = 1
                        for k in range(z + 1, depth):
                            if mask[y, x, k] and not visited[x, k]:
                                max_width += 1
                            else:
                                break
                        
                        # Find maximum height (in X)
                        max_height = 1
                        for h in range(x + 1, width):
                            can_extend = True
                            for w in range(max_width):
                                if z + w >= depth or not mask[y, h, z + w] or visited[h, z + w]:
                                    can_extend = False
                                    break
                            if can_extend:
                                max_height += 1
                            else:
                                break
                        
                        # Mark visited
                        for h in range(max_height):
                            for w in range(max_width):
                                visited[x + h, z + w] = True
                        
                        quads.append((x, y, z, max_width, max_height))
                        z += max_width
                    else:
                        z += 1
    
    return quads