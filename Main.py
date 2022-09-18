from Agents import *
from Game import Game

n_games = 100
n_hidden_layers = 5
encoding_scheme = 2
opponent_index = 0  # 0: None, 1: Random, 2: One_step, 3: AB_agent

# DQN Agent
path = 'strat_v13.pt'
agent1 = DQN_Agent(load_path=path, n_hidden_layers=n_hidden_layers, encoding_scheme=encoding_scheme)

# Opponent
opponent = [None, Random(), One_step_ahead(), AB_agent()]
agent2 = opponent[opponent_index]

# instantiating game, and running for n_games
agents = [agent1, agent2]
game = Game(rows=6, cols=7, inarow=4, agents=agents)
scores = game.play_n_games(n_games // 2)

agents = [agent2, agent1]
game.agents = agents
x = game.play_n_games(n_games // 2)

scores.extend([[a, b] for [b, a] in x])
print_summary(scores, dqn=1)
