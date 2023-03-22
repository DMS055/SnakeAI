import torch
import random
from termcolor import cprint
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNET, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # Learning rate

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness factor
        self.gamma = 0.9 # Discount rate !IMPORTANT: must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if exceeded
        self.model = Linear_QNET(11, 256, 3) # [state size, {can be changed} hidden size, action]
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
    
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
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Returns a list of tuples
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # Random moves: Tradeoff Exploration / Exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # Execute forward()
            move.torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

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
                agent.model.save() 
                
            cprint("Game:", attrs=["bold", "reverse"])
            cprint(agent.n_games + "\n", attrs=["bold", "underline"])
            cprint("Score:", attrs=["bold", "reverse"])
            cprint(score + "\n", attrs=["bold", "underline"])
            cprint("Current Record:", attrs=["bold", "reverse"])
            cprint(record, attrs=["bold", "underline"])
            
            # TODO: Plot
            

if __name__ == '__main__':
    train()
        
        