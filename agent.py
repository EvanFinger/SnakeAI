import torch
import random
import numpy as np
from collections import deque
from gamePyglet import SnakeGame, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEM = 100_000
BATCH_SIZE = 1000
LEARN_RATE = 0.01   
RANDOMNESS = 100

class Agent:
    
    def __init__(self, model_filename=None) -> None:
        """Initializes the agent's default parameters

        Args:
            loaded_model (str, optional): File name for a model to load from the model/ dir. Defaults to None.
        """
        self.num_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (<1)
        self.memory = deque(maxlen=MAX_MEM) # popLeft() when MAX_MEM exceeded
        
        self.randmove = 0 # num of rand moves (testing remove later)
        
        # Check if model should be loaded
        if model_filename:
            self.model = torch.load(model_filename)
            self.loaded_model = True
        else:
            self.model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
            self.loaded_model = False
        # Create trainer
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
        
        self.epsilon = RANDOMNESS - self.num_games
        
        nextMove = [0, 0, 0]
        # As num games increases, if statement will be True less
        # If true, does random move
        if random.randint(0, RANDOMNESS * 2 ) < self.epsilon:
            self.randmove += 1
            move = random.randint(0, 2)
            nextMove[move] = 1
        # If False, does predicted move from model
        else:
            state0 = torch.tensor(state, dtype=torch.float) # converts state array to torch tensor object
            prediction = self.model(state0) # calls the model's forward function (gets model's prediction)
            move = torch.argmax(prediction).item() 
            nextMove[move] = 1
        
        return nextMove
        
    # POSSIBLY REMOVE
    def _calcualteEpsilon(self) -> int:
        
        epsilon = RANDOMNESS
        
        # Calculate any decreases for epsilon (randomness)
        decrease_epsilon = self.num_games
        if self.loaded_model: 
            decrease_epsilon += RANDOMNESS
        
        # Increases randomness as model stops improving
        increase_epsilon = self.numGamesUnimproved
        # Limit the dynamic randomness
        if increase_epsilon > RANDOMNESS * 0.01:
            increase_epsilon = RANDOMNESS * 0.01
        
        if decrease_epsilon < epsilon:
            epsilon -= decrease_epsilon
            epsilon += increase_epsilon
        else:
            epsilon = 0 + increase_epsilon
        
        return epsilon
        
        
class AgentTrainer():
    
    def __init__(self, agent:Agent, game:SnakeGame) -> None:
        """Initializes an agent trainer, which takes an agent and trains its model.

        Args:
            agent (Agent): Agent in which the model will be trained
            game (SnakeGame): Game where the agent is learning to play
        """
        # Init Variables for the Trainer
        self.agent = agent
        self.game = game
        self.plotScores = []
        self.plotMeanScores = []
        self.totalScore = 0
        self.record = 0
        
    def train(self):
        """ Trains the agent and model for the current frame
        """
        

        # get old state
        oldState = self.agent.getState(self.game)
        
        # get the next move based on the state
        nextMove = self.agent.getAction(oldState)
        
        # perform the move and get the new game state
        reward, gameOver, score = self.game.playStep(nextMove) 
        newState = self.agent.getState(self.game)
        
        # train short mem (1 step)
        self.agent.trainShortMem(oldState, nextMove, reward, newState, gameOver)
        
        # remember
        self.agent.remember(oldState, nextMove, reward, newState, gameOver)
        
        if gameOver:
            # train long memory and plot results of game
            self.game.reset()
            self.agent.num_games += 1
            self.agent.trainLongMem()
            
            # automaticall save the model as it gets better scores
            # also manages the dynamic randomness
            if score > self.record:
                record = score
                self.agent.model.save()
            
            
            # plot results in pyplot (only working n pycharm)
            self.plotScores.append(score)
            self.totalScore += score
            meanScore = self.totalScore / self.agent.num_games
            self.plotMeanScores.append(meanScore)
            
            print('Game', self.agent.num_games, 'Score', score, 'Record:', self.record, 'Average:', meanScore, 'Epsi:', self.agent.epsilon, "Rands:", self.agent.randmove)\
            #reset the rand move
            self.agent.randmove = 0
            # plot(plotScores, plotMeanScores)

    # Print out model data-----------------------REMOVE LATER
    # for param_tensor in agent.model.state_dict():
    #     print(param_tensor, "\t", agent.model.state_dict()[param_tensor].size())  
    # for var_name in agent.trainer.optimizer.state_dict():
    #     print(var_name, "\t", agent.trainer.optimizer.state_dict()[var_name])
             
# Run the program from "python agent.py" command
