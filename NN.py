import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from collections import deque
from game import Flappy_Bird_Game
import random, math

import os

class Agent:
    def __init__(self, model_path=None):
        self.model = Model(5, 64, 2).to(torch.device("cuda:0"))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.loss_function = nn.MSELoss()  # Using MSELoss for Q-learning

        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.n_games = 0
        self.epsilon = self.epsilon_start

        self.memory = deque(maxlen=10000)
        
        if model_path and os.path.isfile(model_path):
            self.load_model(model_path)

    def get_action(self, state):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.n_games / self.epsilon_decay)
        move = [0, 0]

        if random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float).to(torch.device("cuda:0"))
            with torch.no_grad():
                action = torch.argmax(self.model(state)).item()
        else:
            action = random.randint(0, 1)

        move[action] = 1

        return move

    def remember(self, state, action, reward, next_state, dead_state):
        self.memory.append((state, action, reward, next_state, dead_state))

    def train_step(self, batch):
        states, actions, rewards, next_states, dead_states = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float).to(torch.device("cuda:0"))
        actions = torch.tensor(actions, dtype=torch.int64).to(torch.device("cuda:0"))
        rewards = torch.tensor(rewards, dtype=torch.float).to(torch.device("cuda:0"))
        next_states = torch.tensor(next_states, dtype=torch.float).to(torch.device("cuda:0"))
        dead_states = torch.tensor(dead_states, dtype=torch.bool).to(torch.device("cuda:0"))

        pred = self.model(states)
        target = pred.clone()

        for i in range(len(states)):
            q_new = rewards[i]
            if not dead_states[i]:
                q_new += self.gamma * torch.max(self.model(next_states[i])).item()

            target[i][actions[i].argmax()] = q_new

        self.optimizer.zero_grad()
        loss = self.loss_function(pred, target)
        loss.backward()
        self.optimizer.step()

    def reinforcement_training(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        self.train_step(batch)
    
    def save_model(self, model_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'n_games': self.n_games
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_games = checkpoint.get('n_games', 0)
        self.model.train()
        print(f"Model loaded from {model_path}")

# Model Class remains the same as before
class Model(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.layer1 = nn.Linear(n_input, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer3(x)
        return x