import torch
import random
import numpy as np
from collections import deque
from game import Game, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEM = 100_000
BATCH_SIZE = 1000
LEARN_RATE = 0.001   

class Agent:
    
    def __init__(self) -> None:
        """Initializes the agent's default parameters
        """
        self.num_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (<1)
        self.memory = deque(maxlen=MAX_MEM) # popLeft() when MAX_MEM exceeded
        self.model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, LEARN_RATE, self.gamma)
    
    def getState(self, game):
        """Gets the game's current state

        Args:
            game (Game): The instance of the snake game that will be analyzed

        Returns:
            [bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool]: state array conveying game state data.
            [
            Danger Straight, Danger Right, Danger Left,
            Moving West, Moving East, Moving North, Moving South, 
            Fruit West, Fruit East, Fruit North, Fruit South
            ]
        """
        snake_head = game.snakeBody[0]
        head_w = Point(snake_head.x - BLOCK_SIZE, snake_head.y)
        head_e = Point(snake_head.x + BLOCK_SIZE, snake_head.y)
        head_n = Point(snake_head.x, snake_head.y - BLOCK_SIZE)
        head_s = Point(snake_head.x, snake_head.y + BLOCK_SIZE)
        
        direction_w = game.direction == Direction.WEST
        direction_e = game.direction == Direction.EAST
        direction_n = game.direction == Direction.NORTH
        direction_s = game.direction == Direction.SOUTH
        
        state = [
            # Danger Ahead (Straight)
            (direction_w and game.find_collision(head_w)) or
            (direction_e and game.find_collision(head_e)) or
            (direction_n and game.find_collision(head_n)) or
            (direction_s and game.find_collision(head_s)),
            
            # Danger Right
            (direction_s and game.find_collision(head_w)) or
            (direction_n and game.find_collision(head_e)) or
            (direction_e and game.find_collision(head_n)) or
            (direction_w and game.find_collision(head_s)),
            
            # Danger Left
            (direction_n and game.find_collision(head_w)) or
            (direction_s and game.find_collision(head_e)) or 
            (direction_w and game.find_collision(head_n)) or
            (direction_e and game.find_collision(head_s)),
            
            # Direction of movement
            direction_w,
            direction_e,
            direction_n,
            direction_s,
            
            # Food relative location
            game.fruit.x < game.snakeHead.x, # food west
            game.fruit.x > game.snakeHead.x, # food east
            game.fruit.y < game.snakeHead.y, # food north
            game.fruit.y > game.snakeHead.y # food south
        ]
        
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, game_over):
        """Stores the current state, action(move), reward, the next state, and game over into memory

        Args:
            state (state array): the current state of the game
            action ([int, int, int]): the action took this step
            reward (int): the reward for the action of this step
            next_state (state array): the next state of the game
            game_over (bool): whether or not the game ended
        """
        # Store all parameters as one tuple in the memory
        self.memory.append((state, action, reward, next_state, game_over)) # pops left if max mem is exceeded
    
    def trainLongMem(self):
        """Trains the model after game over. Uses a batch of memory rather than just one step
        """
        if len(self.memory) > BATCH_SIZE:
            # Generate a random batch of memory samples
            miniSample = random.sample(self.memory, BATCH_SIZE) 
        else:
            # Use whole memory if not enough samples for full batch
            miniSample = self.memory
        
        # Unpack all the states, actions, rewards, etc. from miniSample, and repack
        # in groups using built in zip function
        states, actions, rewards, next_states, game_overs = zip(*miniSample)
        # Pass the new packages to the trainer
        self.trainer.trainStep(states, actions, rewards, next_states, game_overs)
    
    def trainShortMem(self, state, action, reward, next_state, game_over):
        """Trains the model after every step.

        Args:
            state (state array): the current state of the game
            action ([int, int, int]): the action took this step
            reward (int): the reward for the action of this step
            next_state (state array): the next state of the game
            game_over (bool): whether or not the game ended
        """
        self.trainer.trainStep(state, action, reward, next_state, game_over)
    
    def getAction(self, state):
        """Determines the move for the current step in the game. Either random, or predicted
        based on the number of games played

        Args:
            state (state array): the current state of the game

        Returns:
            [int, int, int]: The action to be performed this step
        """
        # random moves: tradeoff exploration / exploitation
        # Random moves when still exploring/learning. (exploration)
        # Less random moves as the model gets better and better. (exploitation)
        self.epsilon = 80 - self.num_games # play around with this hardcode
        nextMove = [0, 0, 0]
        # As num games increases, if statement will be True less
        # If true, does random move
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            nextMove[move] = 1
        # If False, does predicted move from model
        else:
            state0 = torch.tensor(state, dtype=torch.float) # converts state array to torch tensor object
            prediction = self.model(state0) # calls the model's forward function (gets model's prediction)
            move = torch.argmax(prediction).item() 
            nextMove[move] = 1
        
        return nextMove
    
def train():
    """Training loop for the agent and model
    """
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = 0
    agent = Agent()
    game = Game()
    
    # Training Loop
    while True:
        # get old state
        oldState = agent.getState(game)
        
        # get the next move based on the state
        nextMove = agent.getAction(oldState)
        
        # perform the move and get the new game state
        reward, gameOver, score = game.playStep(nextMove) 
        newState = agent.getState(game)
        
        # train short mem (1 step)
        agent.trainShortMem(oldState, nextMove, reward, newState, gameOver)
        
        # remember
        agent.remember(oldState, nextMove, reward, newState, gameOver)
        
        if gameOver:
            # train long memory and plot results of game
            game.reset()
            agent.num_games += 1
            agent.trainLongMem()
            
            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.num_games, 'Score', score, 'Record:', record)
            
            plotScores.append(score)
            totalScore += score
            meanScore = totalScore / agent.num_games
            plotMeanScores.append(meanScore)
            
            plot(plotScores, plotMeanScores)
            
    
if __name__ == '__main__':
    train()
