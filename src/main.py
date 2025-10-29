from logs import setup_logging
import logging

from core.asset_manager import AssetManager
from core.window import Window
from player import Player

logger = logging.getLogger(__name__)

"""
next up: an lod system. yes, lod before greedy meshing.
- [x] "level" attr in the buffer/struct 
(^implemented "scale" instead, allows for more flexibility)
- [x] hierarchial octree lod implementation
- [ ] figure out the "hiding unseen faces" part again
- [ ] stitching different levels together
- [x] remove render distance? decided: KEEP
"""

if __name__ == "__main__":
    setup_logging()

    window: Window = Window()
    asset_manager: AssetManager = AssetManager(window.state)
    asset_manager.load_assets()
    player = Player(window.state)
    window.mainloop()

    logger.info("This is goodbye.")

