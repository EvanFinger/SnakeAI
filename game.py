import pygame
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
BLOCK_SIZE = 10 #px

# Define colors

BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)
BLUE = pygame.Color(0, 0, 255)

class Game:
    """
    Simple snake game made specially for use with an AI player. 
    """
        
    def __init__(self, width=800, height=600) -> None:
        """Initializes the game enviroment

        Args:
            width (int, optional): width of the game screen. Defaults to 800.
            height (int, optional): height of the game screen. Defaults to 600.
        """
        pygame.init()
        # Set screen dimentions
        self.SCREEN_WIDTH = width
        self.SCREEN_HEIGHT = height
        
        # Initialize screen
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("SnakeAI")
        self.FPS = 600
        
        # Create screen and fps clock
        
        self.fpsClock = pygame.time.Clock()
        
        self.reset()
        
        
    def reset(self):
        """Resets the game to its beginning state
        """
        # Create Score & snake position
        self.score = 0
        
        self.snakeHead = Point(BLOCK_SIZE * 5, BLOCK_SIZE * 5)
        
        self.snakeBody = [
            self.snakeHead,
            Point(self.snakeHead.x - BLOCK_SIZE, self.snakeHead.y),
            Point(self.snakeHead.x - (BLOCK_SIZE * 2), self.snakeHead.y)
        ]
        
        # Define initial snake direction
        self.direction = Direction.EAST
        self.changeDirection = self.direction
        
        # Define initial game params
        self.score = 0
        self.fruit = None
        self._createFruit()
        
        # Keep track of frames
        self.frameIteration = 0
        
    def _createFruit(self):
        """Create a fruit at a random position on the window, outside of the snake's body
        """
        x = random.randint(0, (self.SCREEN_WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.SCREEN_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.fruit = Point(x,y)
        
        # Prevent fruit from being place inside the snake
        if self.fruit in self.snakeBody:
            self._createFruit()
            
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
        
        # User Input
        if action == None:
            self._handleUserInput()
        else:
            self._agentInput(action)
        
        # Move the Snake
        self._moveSnake() # updates the head's position
        self.snakeBody.insert(0, self.snakeHead)
        
        #Init reward value
        reward = 0
        
        # Check for game over conditions
        game_over = False
        if self.find_collision() or self.frameIteration > 200 * len(self.snakeBody):
            # When collision occurs or snake does not find food for long time
            game_over = True
            reward = -10
            return reward, game_over, self.score 
        
        # Place new food if eaten, or just move the snake
        if self._growSnake():
            reward = 10
        
        # Update the ui
        self.render()
        self.fpsClock.tick(self.FPS)
        
        # Return key parameters
        return reward, game_over, self.score
    
    def _gameOver(self):
        """
        Resets the game when the snake dies
        """
        
        self.reset()
    
    def _showScore(self, color):
        """Shows the score as well as updates the score every frame

        Args:
            color (pygame.Color): The color of the text label
        """
        snakeScore = str(self.score)
        scoreFont = pygame.font.Font('Roboto-Thin.ttf', 24)
        # Create text display for score
        self.scoreLabel = scoreFont.render(snakeScore, True, color)
        
        # Create rect object to nest the score
        self.labelRect = self.scoreLabel.get_rect()
        self.labelRect.left = 0
        self.labelRect.top = 0
        
        # Draw score to the display
        self.screen.blit(self.scoreLabel, self.labelRect)
    
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

    def _drawSnake(self):
        """
        Draws the snake from the snake body iterable
        """

        for segment in self.snakeBody:
            pygame.draw.rect(
                self.screen, 
                RED, 
                pygame.Rect(segment.x, segment.y, BLOCK_SIZE, BLOCK_SIZE)
                )    
            
    def _drawFruit(self):
        """
        Draws the fruit based on the fruit position
        """

        pygame.draw.rect(
            self.screen, 
            GREEN, 
            pygame.Rect(self.fruit.x, self.fruit.y, BLOCK_SIZE, BLOCK_SIZE)
            )

    def render(self):
        """
        Updates the snake game and the display.
        """
        self.screen.fill(BLACK)
        
        self._drawSnake()
        self._drawFruit()
        self._showScore(WHITE)
        pygame.display.update()      
    
    def _handleUserInput(self):
        """
        Handles any input comands to the game (USER INPUT - NOT AI).
        """
        ## Move Player
        key = pygame.key.get_pressed()
        if key[pygame.K_a]:
            self.changeDirection = Direction.WEST
        elif key[pygame.K_d]:
            self.changeDirection = Direction.EAST
        elif key[pygame.K_s]:
            self.changeDirection = Direction.SOUTH
        elif key[pygame.K_w]:
            self.changeDirection = Direction.NORTH
            
        ## Close Window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
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
        
        if self.direction == Direction.NORTH:
            y = y - BLOCK_SIZE # Moves snake up one block
        elif self.direction == Direction.SOUTH:
            y = y + BLOCK_SIZE # Moves snake down one block
        elif self.direction == Direction.WEST:
            x = x - BLOCK_SIZE # Moves snake left one block
        elif self.direction == Direction.EAST:
            x = x + BLOCK_SIZE # Moves snake right one block
            
        ## repack x&y into snake head
        self.snakeHead = Point(x,y)
    
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
    
        