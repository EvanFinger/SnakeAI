import game
from gamePyglet import SnakeGame
import pyglet
from pyglet import shapes
import time


if  __name__ == "__main__":

    tgame = SnakeGame()
    
    # add all functions to run during training
    def update(dt):
        reward, gameOver, score = tgame.playStep()
        if gameOver:
            tgame._gameOver()
    
    pyglet.clock.schedule_interval(update, 1/20)
    pyglet.app.run()
    













