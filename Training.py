import torch
import random
from collections import deque
from game import Flappy_Bird_Game
from NN import Agent

def train_flappy_bird(num_episodes, model_path=None):
    game = Flappy_Bird_Game(1200, 600, False)
    agent = Agent(model_path=model_path)

    agent.n_games = 0
    record = 0

    for episode in range(num_episodes):
        game.reset()
        state = game.get_state()
        dead = False

        wait = 20

        while not dead:
            action = agent.get_action(state)
            if action[1] == 1: wait = 0
            reward = game.input(action)
            while True:
                if wait > 10:
                    next_state = game.get_state()
                    break
                else:
                    game.input([1, 0])
                    wait+=1
            
            dead = game.isDead()

            agent.remember(state, action, reward, next_state, dead)
            state = next_state

        agent.reinforcement_training(1000)

        if game.score > record:
            record = game.score
            print(f"New record {record} at episode {episode + 1}")
            agent.save_model(model_path)

        agent.n_games += 1

    print(f"Training completed. Highest score achieved: {record}")

# Example usage
if __name__ == "__main__":
    model_path = "flappy_bird_model2_v2.pth"
    num_episodes = 10000  # Set the number of episodes for training
    train_flappy_bird(num_episodes, model_path)