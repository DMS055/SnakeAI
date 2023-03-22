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
        head = game.snake[0]

        # Clok-wise directions and angles
        cw_dirs = [
            Direction.RIGHT == game.direction,
            Direction.DOWN == game.direction,
            Direction.LEFT == game.direction,
            Direction.UP == game.direction
        ]
        cw_angs = np.array([0, np.pi/2, np.pi, -np.pi/2])

        # Position - in front: 0, on right: 1, on left: -1; BLOCK_SIZE = 20
        def getPoint(pos): return Point(
            head.x + 20*np.cos(cw_angs[(cw_dirs.index(True)+pos) % 4]),
            head.y + 20*np.sin(cw_angs[(cw_dirs.index(True)+pos) % 4]))

        state = [
            # Danger
            game.is_collision(getPoint(0)),
            game.is_collision(getPoint(1)),
            game.is_collision(getPoint(-1)),

            # Move direction
            cw_dirs[2],
            cw_dirs[0],
            cw_dirs[3],
            cw_dirs[1],

            # Food location
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
        ]

        return np.array(state, dtype=int)
    
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
        # Get current state
        state_old = agent.get_state(game)
        
        # Get move based on current state
        final_move = agent.get_action(state_old)
        
        # Move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # Train long memory (Expirience Replay), then plot results
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
            
            # TODO: Plot
            

if __name__ == '__main__':
    train()
        
        