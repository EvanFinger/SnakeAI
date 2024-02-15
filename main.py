from modelManager import AskLoadModel
from gamePyglet import SnakeGame
from agent import Agent, AgentTrainer
import pyglet
from pyglet import shapes
import torch

if  __name__ == "__main__":
    
    path, load = AskLoadModel()
    
    print(path)
    
    tagent = Agent(load, path)
    input()
    tgame = SnakeGame()
    ttrainer = AgentTrainer(tagent, tgame, path)
        
    # add all functions to run during training
    def update(dt):
        ttrainer.train()
    
    pyglet.clock.schedule_interval(update, 1/100)
    pyglet.app.run()
    
    













