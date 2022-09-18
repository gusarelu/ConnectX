import torch

class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers=2, n_hidden_units=None):
        super(DQN, self).__init__()
        assert n_hidden_layers >= 1
        if n_hidden_units == None:
            n_hidden_units = input_size
        self.layers = torch.nn.ModuleList([torch.nn.Linear(input_size, n_hidden_units)])
        self.layers.extend([torch.nn.Linear(n_hidden_units, n_hidden_units) for i in range(n_hidden_layers - 1)])
        self.layers.append(torch.nn.Linear(n_hidden_units, output_size))
        self.dropout = torch.nn.Dropout()

    def forward(self, x):
        # assumes input x already formatted as a torch array
        for i in range(len(self.layers)-1):
            x = self.layers[i](x).relu()
            x = self.dropout(x)
        out = self.layers[-1](x)
        return out