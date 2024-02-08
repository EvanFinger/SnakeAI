import game



if  __name__ == "__main__":
    snakeGame = game.Game()

    run = True
    while run:
    
        game_over, score = snakeGame.playStep()
    
        if game_over:
            break













