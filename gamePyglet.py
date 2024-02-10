import pyglet
import random
import numpy as np
from enum import Enum
from collections import namedtuple



# Define directions
class Direction(Enum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    
# Define point tuple
Point = namedtuple('Point', 'x, y')

# Define Constants
BLOCK_SIZE = 30 #px

# Define Colors

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class SnakeGame:
    """Simple snake game made to be controlled by
    an experimental AI model(s)
    """
    
    def __init__(self, width=800, height=600) -> None:
        """Initializes the game enviroment

        Args:
            width (int, optional): width of the game window. Defaults to 800.
            height (int, optional): height of the game window. Defaults to 600.
        """
        # Set screen dimentions
        self.SCREEN_WIDTH = width
        self.SCREEN_HEIGHT = height
        
        # Initialize the game window
        self.gameWindow = pyglet.window.Window(width=self.SCREEN_WIDTH, height=self.SCREEN_HEIGHT)
        
        