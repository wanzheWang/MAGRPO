import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RNNGaussianAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            # DNN actor path: keep the same hidden layout as critic (fc1 -> fc2).
            self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.fc_mu = nn.Linear(args.hidden_dim, args.n_actions)

        init_log_std = float(getattr(args, "init_log_std", -0.5))
        self.log_std = nn.Parameter(th.full((args.n_actions,), init_log_std))
        self.log_std_min = float(getattr(args, "log_std_min", -20.0))
        self.log_std_max = float(getattr(args, "log_std_max", 2.0))

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.args.hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        if self.args.use_rnn:
            hidden_in = hidden_state.reshape(-1, self.args.hidden_dim)
            hidden = self.rnn(x, hidden_in)
        else:
            hidden = F.relu(self.fc2(x))

        mu = self.fc_mu(hidden)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max).expand_as(mu)
        return (mu, log_std), hidden
