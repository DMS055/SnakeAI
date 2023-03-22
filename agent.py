import torch
import random
from termcolor import cprint
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness factor
        self.gamma = 0 # Discount Rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if exceeded
        # TODO: model, trainer
    
    def get_state(self, game):
        pass
    
    def remember(self, state, action, reward, next_state, done):
        pass
    
    def train_long_memory(self):
        pass
    
    def train_short_memory(self, state, action, reward, next_state, done):
        pass
    
    def get_action(self, state):
        pass

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get Current State
        state_old = agent.get_state(game)
        
        # Get Move Based On Current State
        final_move = agent.get_action(state_old)
        
        # Move And Get New State
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Train Short Memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # Train Long Memory (Expirience Replay), Then Plot Results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                # TODO: agent.model.save() 
                
            cprint("Game:", attrs=["bold", "reverse"])
            cprint(agent.n_games + "\n", attrs=["bold", "underline"])
            cprint("Score:", attrs=["bold", "reverse"])
            cprint(score + "\n", attrs=["bold", "underline"])
            cprint("Current Record:", attrs=["bold", "reverse"])
            cprint(record, attrs=["bold", "underline"])
            

if __name__ == '__main__':
    train()
        
        