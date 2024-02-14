from modelManager import AskLoadModel
from gamePyglet import SnakeGame
from agent import Agent, AgentTrainer
import pyglet
from pyglet import shapes
import time


if  __name__ == "__main__":
    
    path = AskLoadModel()
    
    tagent = Agent()
    tgame = SnakeGame()
    ttrainer = AgentTrainer(tagent, tgame)
        
    # add all functions to run during training
    def update(dt):
        ttrainer.train()
    
    pyglet.clock.schedule_interval(update, 1/20)
    pyglet.app.run()
    
    













