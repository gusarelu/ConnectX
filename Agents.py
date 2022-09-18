import random
import torch
import numpy as np
from Game import Game
from DQN import DQN
from Functions import *
from statistics import mean
from timeit import default_timer as timer
from datetime import timedelta


class Leftmost:
    def __call__(self, game):
        valid_moves = get_valid_moves(game.board)
        return valid_moves[0]


class Random:
    def __init__(self, valid=True):
        self.valid = valid

    def __call__(self, game):
        if self.valid:
            valid_moves = get_valid_moves(game.board)
            return random.choice(valid_moves)
        else:
            columns = [i for i in range(game.cols)]
            return random.choice(columns)


class One_step_ahead:
    def __call__(self, game):
        columns = get_winning_moves(game.board, game.inarow, game.mark)
        if columns:
            return random.choice(columns)
        columns = get_blocking_moves(game.board, game.inarow, game.mark)
        if columns:
            return random.choice(columns)
        columns = get_valid_moves(game.board)
        return random.choice(columns)


class N_steps_look_ahead_agent:
    def __init__(self, max_steps=3):
        self.max_steps = max_steps

    def __call__(self, game):
        columns = get_winning_moves(game.board, game.inarow, game.mark)
        if columns:  # if there is a winning move
            return random.choice(columns)
        columns = get_blocking_moves(game.board, game.inarow, game.mark)
        if columns:  # if there is a blocking move
            return random.choice(columns)
        columns = n_look_ahead(board=game.board, mark=game.mark, max_steps=self.max_steps, inarow=game.inarow)
        if columns:  # if there are max_steps ahead (i.e. step t+max is calculable). this wont be true when close to a draw
            return random.choice(columns)
        valid_moves = get_valid_moves()  # if all of the above fails. only gets called when close to a draw
        return random.choice(valid_moves)


class AB_agent():
    def __init__(self, max_steps=3):
        self.max_steps = max_steps

    def __call__(self, game):
        winning_moves = get_winning_moves(game.board, game.inarow, game.mark)
        if winning_moves:  # if there is a winning move
            return random.choice(winning_moves)
        blocking_moves = get_blocking_moves(game.board, game.inarow, game.mark)
        if blocking_moves:  # if there is a blocking move
            return random.choice(blocking_moves)
        enabling_moves = get_enabling_moves(game.board, game.inarow, game.mark)
        columns = alpha_beta(board=game.board, mark=game.mark, max_steps=self.max_steps, inarow=game.inarow)
        if columns:  # if there are max_steps ahead (i.e. step t+max is calculable). this wont be true when close to a draw
            revised_columns = [i for i in columns if i not in enabling_moves]
            if revised_columns:  # if there are non-enabling moves from alpha-beta pruning suggestions
                return random.choice(revised_columns)
        valid_moves = get_valid_moves(game.board)  # if all of the above fails. only should get called when close to a draw
        revised_valid_moves = [i for i in valid_moves if i not in enabling_moves]
        if revised_valid_moves:  # if there are non-enabling moves
            return random.choice(revised_valid_moves)
        return random.choice(valid_moves)


class DQN_Agent:
    def __init__(self, rows=6, cols=7, inarow=4, n_hidden_layers=2, n_hidden=None, load_path='', encoding_scheme=1):
        self.rows = rows
        self.cols = cols
        self.inarow = inarow
        self.encoding_scheme = encoding_scheme
        self.training_mode = False
        input_size = encoding_scheme * rows * cols
        if load_path:
            self.pol_net = torch.load(load_path)
        else:
            self.pol_net = DQN(input_size=input_size, output_size=cols, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden)
        self.tar_net = DQN(input_size=input_size, output_size=cols, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden)
        self.tar_net.load_state_dict(self.pol_net.state_dict())  # copying policy net to target net
        self.pol_net.eval()
        self.tar_net.eval()

    def format_board(self, board, mark):
    # formats the gameboard into a torch array so it can be fed to the DQN
        board = torch.from_numpy(board).flatten()
        if self.encoding_scheme == 1:
            discs_player = (board == mark).float()  # vector containing 1 if player disc, 0 otherwise
            discs_opponent = (board == 3-mark).float()  # vector containing 1 if opponent disc, 0 otherwise
            state = (discs_player - discs_opponent).unsqueeze(dim=0)
            return state  # size: (1, rows*cols)
        if self.encoding_scheme == 2:
            discs_player = (board == mark).float()  # vector containing 1 if player disc, 0 otherwise
            discs_opponent = (board == 3-mark).float()  # vector containing 1 if opponent disc, 0 otherwise
            state = torch.cat([discs_player, discs_opponent], 0).unsqueeze(dim=0)
            return state  # size: (1, 2*rows*cols)
    def __call__(self, game):
        if self.training_mode and (random.random() < self.epsilon):  # allowing exploration
            return random.randrange(self.cols)  # returns a random column, which may be invalid
        winning_column = get_winning_moves(game.board, game.inarow, game.mark, first=True)
        if winning_column:
            return winning_column
        x = self.format_board(game.board, game.mark)
        out = self.pol_net(x)  # Q-values (one per column)
        valid_moves = get_valid_moves(game.board)
        for index in range(self.cols):
            if index not in valid_moves:
                out[0, index] = out.min().item() - 1  # reducing the Q-value of invalid moves
        return out.argmax().item()  # selects the column with the highest Q-value

    def reinforce(self, batch, criterion, optimizer, gamma=0.95):
    # discussion on why the appropriate gamma depends on: win reward, draw reward, and max turns. hard coded for now.
        output = torch.zeros(len(batch), 1)
        target = torch.zeros(len(batch), 1)
        for i, experience in enumerate(batch):
            old_state = experience[0]
            action = experience[1]
            new_state = experience[2]
            reward = experience[3]
            gameover = experience[4]
            if gameover:
                Qhat = 0
            else:
                Qhat = self.tar_net(new_state).max().item()
            output[i] = self.pol_net(old_state)[0, action]
            target[i] = reward + gamma * Qhat
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, n_episodes=5000, epsilon_start=1, epsilon_min=0.05, epsilon_scale=10000, opponent=Random(), buffer_size=4200, batch_size=32, lr=0.01, save_path='', progress_path='progress.csv'):
        self.pol_net.train()
        start = timer()
        time1 = start
        self.training_mode = True
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_scale = epsilon_scale
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(params=self.pol_net.parameters(), lr=lr)
        buffer = []
        update_count = 0
        game = Game(rows=self.rows, cols=self.cols, inarow=self.inarow, agents=[self, opponent])  # the opponent can even be the DQN agent itself, or two completely different agents
        performance_list = np.zeros((n_episodes//1000, 3))
        for count_episode in range(n_episodes):
            game.reset()
            old_state = [None, None]
            new_state = [None, None]
            action = [None, None]
            reward = [None, None]
            while not game.gameover:
                player_index = game.mark - 1
                opponent_index = 1 - player_index
                player = game.agents[player_index]
                new_state[player_index] = self.format_board(board=game.board, mark=game.mark)
                if game.turn > 2:  # can only save a full experience once both players have made at least 1 turn each
                    experience = (old_state[player_index], action[player_index], new_state[player_index], reward[player_index], game.gameover)
                    if len(buffer) < buffer_size:
                        buffer.append(experience)
                    else:
                        batch = random.sample(population=buffer, k=batch_size)
                        self.reinforce(batch=batch, criterion=criterion, optimizer=optimizer)
                        buffer[update_count % buffer_size] = experience
                        update_count += 1
                        if update_count % 100 == 0:
                            self.tar_net.load_state_dict(self.pol_net.state_dict())  # updating target net
                old_state[player_index] = new_state[player_index].clone()
                action[player_index] = player(game)
                reward[player_index] = 0
                if game.drop(action[player_index]):  # if invalid move
                    game.gameover = True
                    reward[player_index] = -1000
                    reward[opponent_index] = 0
                elif game.check_win():
                    game.gameover = True
                    reward[player_index] = 100
                    reward[opponent_index] = -100
                elif game.check_draw():
                    game.gameover = True
                    reward[player_index] = 1
                    reward[opponent_index] = 1
                game.next_turn()
            # save last 2 experiences and reinforce:
            for i in range(2):
                experience = (old_state[i], action[i], new_state[i], reward[i], game.gameover)
                if len(buffer) < buffer_size:
                    buffer.append(experience)
                else:
                    batch = random.sample(population=buffer, k=batch_size)
                    self.reinforce(batch=batch, criterion=criterion, optimizer=optimizer)
                    buffer[update_count % buffer_size] = experience
                    update_count += 1
                    if update_count % 100 == 0:
                        self.tar_net.load_state_dict(self.pol_net.state_dict())  # updating target net
            if (count_episode+1) % 100 == 0:
                end = timer()
                print(f'Training progress: {100*(count_episode+1)/n_episodes:.2f}%, Episode: {(count_episode+1)}, Last epsilon: {self.epsilon:.2f}, Interval: {timedelta(seconds=end-time1)}, Time elapsed: {timedelta(seconds=end-start)}.')
                time1 = end
                if ((count_episode + 1) % 1000 == 0):
                    # saving strategy file
                    if save_path:
                        torch.save(self.pol_net, save_path)
                        print(f'Policy network saved to {save_path} at Episode {(count_episode+1)}.')
                    # testing and saving performance
                    if progress_path:
                        index = ((count_episode + 1) // 1000) - 1
                        performance_list[index, 0] = count_episode + 1
                        performance_list[index, 1] = test_play(self, Random())
                        performance_list[index, 2] = test_play(self, One_step_ahead())
                        np.savetxt(progress_path, performance_list, delimiter=",")
                        print(f'{progress_path} saved.')
            self.epsilon = max(self.epsilon_min, self.epsilon - 1 / self.epsilon_scale)  # updating epsilon
            game.agents[0], game.agents[1] = game.agents[1], game.agents[0]  # alternating order of players
        # end of n_episodes loop
        self.training_mode = False
        self.pol_net.eval()
        print('Training finished.')


if __name__ == '__main__':
    config = 1
    n_episodes = 100000
    opponent = Random()
    if config == 1:
        print('Config 1')
        load_path = 'strat_v13.pt'
        save_path = 'strat_v13_2.pt'
        progress_path = 'progress13.csv'
        n_hidden_layers = 5
        encoding_scheme = 2
    if config == 2:
        print('Config 2')
        load_path = ''
        save_path = 'strat_v12_1.pt'
        progress_path = 'progress2_1.csv'
        n_hidden_layers = 3
    agent = DQN_Agent(load_path=load_path, n_hidden_layers=n_hidden_layers, encoding_scheme=encoding_scheme)
    agent.train(n_episodes=n_episodes, save_path=save_path, progress_path=progress_path, opponent=opponent, epsilon_scale=(n_episodes/2), epsilon_start=1, batch_size=32)
