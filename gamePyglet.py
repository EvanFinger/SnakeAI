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
BLOCK_SIZE = 20 #px
FPS = 60

# Define Colors

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class SnakeGame(pyglet.window.Window):
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
        
        # Init high score counter
        self.h_score = 0
        
        # Initialize the game window
        super().__init__(
            width=self.SCREEN_WIDTH, height=self.SCREEN_HEIGHT,
            caption='SnakeGameAI', resizable=False
        )
        
        self.reset()
        
    ## PYGLET FUNCTIONS
     # Draws object on the game window
    def on_draw(self) -> None:
        self.clear()
        self.score_batch.draw()
        self.snake_batch.draw() # Draw Snake
        self.fruit_G.draw() # Draw Fruit
        
    # Event Listener
    def on_key_press(self, symbol, modifiers):
        
        if symbol == pyglet.window.key.W:
            self.changeDirection = Direction.NORTH
        elif symbol == pyglet.window.key.S:
            self.changeDirection = Direction.SOUTH
        elif symbol == pyglet.window.key.D:
            self.changeDirection = Direction.EAST
        elif symbol == pyglet.window.key.A:
            self.changeDirection = Direction.WEST
        
        return super().on_key_press(symbol, modifiers)
    
    
    def reset(self):
        """Resets the game to original state
        """
        
        # Initialize score coutner
        # Initialize snake head & body
        self.score = 0
        
        self.snakeHead = Point(BLOCK_SIZE * 5, BLOCK_SIZE * 5)
        
        self.snakeBody = [
            self.snakeHead,
            Point(self.snakeHead.x - BLOCK_SIZE, self.snakeHead.y),
            Point(self.snakeHead.x - (BLOCK_SIZE * 2), self.snakeHead.y)
        ]
        
        self.snakeBody_G = [] # List of pyglet rectangles representing snake
        self.snake_batch = pyglet.graphics.Batch() # Batch for snake body
        
        # Initialize the fruit
        self.fruit = None
        self._createFruit()
        self.fruit_G = pyglet.shapes.Rectangle(
            x=self.fruit.x,
            y=self.fruit.y,
            width=BLOCK_SIZE,
            height=BLOCK_SIZE,
            color=GREEN
            ) # Pyglet rectangle representing the fruit
        
        # Initialize the score label
        self.score_batch = pyglet.graphics.Batch()
        
        self.scoreLabel = pyglet.text.Label(
            '0',
            font_name='Times New Roman',
            font_size=64,
            x=BLOCK_SIZE * 2,
            y=BLOCK_SIZE * 4,
            color=(255, 255, 255, 100),
            batch=self.score_batch
        )
        
        self.h_scoreLabel = pyglet.text.Label(
            '0',
            font_name='Times New Roman',
            font_size=32,
            x=BLOCK_SIZE * 2,
            y=BLOCK_SIZE * 2,
            color=(255, 223, 94, 100),
            batch=self.score_batch
        )
        
        # Set snake's initial direction of movement
        self.direction = Direction.EAST
        self.changeDirection = Direction.EAST
        
        # Initialize backend variables
        self.frameIteration = 0
        """Keeps track of the current frame number
        """
        
    def playStep(self, action=None):
        """Plays the next frame of the game

        Args:
            action ([int, int, int], optional): The action provided by the agent from the model. Defaults to None.

        Returns:
            reward (int): The reward for completing the current step
            gameOver (bool): Indication of the game ending
            score (int): The score after completing the current step
        """
        # Update frame iteration
        self.frameIteration += 1
        
        # AI Input
        self._agentInput(action)
        
        # Move the Snake
        self._moveSnake() # updates the head's position
        self.snakeBody.insert(0, self.snakeHead)
        
        #Init reward value
        reward = 0
        
        # Check for game over conditions
        game_over = False
        if self.find_collision() or self.frameIteration > 100 * len(self.snakeBody):
            # When collision occurs or snake does not find food for long time
            game_over = True
            reward = -10
            return reward, game_over, self.score 
        
        # Place new food if eaten, or just move the snake
        if self._growSnake():
            reward = 10
            
        # Update high score
        if self.score > self.h_score:
            self.h_score = self.score
            
        # Update graphics
        self._updateScoreLabel()
        self._updateFruit_G()
        self._updateSnake_G()
        
        
        # Return key parameters
        return reward, game_over, self.score
    
    def find_collision(self, pt=None) -> bool:
        """Finds any collisions occurring between the snake head and the walls/body

        Args:
            pt (Point, optional): Point to check for collisions against. Defaults to None.

        Returns:
            bool: whether or not a collision was found
        """
        if pt is None:
            pt = self.snakeHead
        # Collides with wall
        if pt.x < 0 or pt.x > self.SCREEN_WIDTH - BLOCK_SIZE or pt.y < 0 or pt.y > self.SCREEN_HEIGHT - BLOCK_SIZE:
            return True
        # Collides with body
        if pt in self.snakeBody[1:]:
            return True
        return False
    
    ## PRIVATE ##
    
    def _createFruit(self):
        """Create a fruit at a random position on the window, outside of the snake's body
        """
        x = random.randint(0, (self.SCREEN_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.SCREEN_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.fruit = Point(x,y)
        
        # Prevent fruit from being place inside the snake
        if self.fruit in self.snakeBody:
            self._createFruit()
            
    def _gameOver(self):
        """
        Resets the game when the snake dies
        """
        self.reset()
        
    def _updateScoreLabel(self):
        """Shows the score as well as updates the score every frame

        Args:
            color (pygame.Color): The color of the text label
        """
        snakeScore = str(self.score)
        snakeHighScore = str(self.h_score)
        # Update the text in the score label
        self.scoreLabel.text = snakeScore
        self.h_scoreLabel.text = snakeHighScore
    
    def _growSnake(self) -> bool:
        """
        Check for snake and fruit collision and spawns new fruit
        """
        
        if self.snakeHead.x == self.fruit.x and self.snakeHead.y == self.fruit.y:
            self.score += 1
            self._createFruit()
            return True
        else:
            self.snakeBody.pop()
            return False
    
    def _moveSnake(self):
        """
        Moves the snake every fram based on which direction it is facing.
        """
        # Edit snake direction based on any direction changes
        
        if self.changeDirection == Direction.NORTH and self.direction != Direction.SOUTH:
            self.direction = Direction.NORTH
        elif self.changeDirection == Direction.SOUTH and self.direction != Direction.NORTH:
            self.direction = Direction.SOUTH
        elif self.changeDirection == Direction.EAST and self.direction != Direction.WEST:
            self.direction = Direction.EAST
        elif self.changeDirection == Direction.WEST and self.direction != Direction.EAST:
            self.direction = Direction.WEST
            
        # Edit snake position based on direction
        
        ## unpack x&y from snake head
        x = self.snakeHead.x
        y = self.snakeHead.y
        
        ## NOTE: Pyglet has reveres y-coord, so north is +y
        
        if self.direction == Direction.NORTH:
            y = y + BLOCK_SIZE # Moves snake up one block
        elif self.direction == Direction.SOUTH:
            y = y - BLOCK_SIZE # Moves snake down one block
        elif self.direction == Direction.WEST:
            x = x - BLOCK_SIZE # Moves snake left one block
        elif self.direction == Direction.EAST:
            x = x + BLOCK_SIZE # Moves snake right one block
            
        ## repack x&y into snake head
        self.snakeHead = Point(x,y)
        
    def _updateSnake_G(self):
        """
        Adds snake body segments to the snake body graphic 
        """
        self.snakeBody_G = []
        for segment in self.snakeBody:
            self.snakeBody_G.append(
                pyglet.shapes.Rectangle(
                    segment.x, segment.y,
                    BLOCK_SIZE, BLOCK_SIZE,
                    RED, batch=self.snake_batch
                )
            )
            
    def _updateFruit_G(self):
        """
        Updates the fruit's position eery frame
        """
        self.fruit_G.x = self.fruit.x
        self.fruit_G.y = self.fruit.y
        
    def _agentInput(self, action):
        """Alternative to handleInputs. Used by the agent to allow the AI to controll the snake.

        Args:
            action ([int, int, int]): array indicating which direction to turn [straight, right, left]
        """
        # Can move straight, turn right, or turn left
        
        direction_queue = [Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH]
        q_index = direction_queue.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            self.changeDirection = direction_queue[q_index]
        elif np.array_equal(action, [0, 1, 0]):
            next_q_index = (q_index + 1) % 4 # Iterate right w/ no overflow
            self.changeDirection = direction_queue[next_q_index] # right turn E -> S -> W -> N
        elif np.array_equal(action, [0, 0, 1]):
            next_q_index = (q_index - 1) % 4 # Iterate left w/ no overflow
            self.changeDirection = direction_queue[next_q_index] # left turn E -> N -> W -> S
        