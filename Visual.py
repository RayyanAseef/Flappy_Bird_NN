import pygame
from game import Flappy_Bird_Game
from NN import Agent

clock = pygame.time.Clock()
pygame.init() 
screen_width, screen_height = 1200, 600

screen = pygame.display.set_mode((screen_width, screen_height)) 
game = Flappy_Bird_Game(screen_width, screen_height, False)

running = True
dead = False

# Specify the path where you want to save the model
model_path = "flappy_bird_model2_v3.pth"
agent = Agent(model_path=model_path)

record = 0
wait = 11

while running:
    clock.tick(50)
    
    if dead:
        if game.score > record:
            record = game.score
            print(f"New record {record} at game {agent.n_games}")
        game.reset()

    screen.fill((0, 0, 255))
    
    action = [1, 0]
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT: 
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if wait > 10:
                    action = [0, 1]

    wait += 1
    if wait > 10:
        state = game.get_state()
        action = agent.get_action(state)

    if action[1] == 1:
        wait = 0

    reward = game.input(action)
    dead = game.isDead()

    game.draw(screen)
    pygame.display.update()


