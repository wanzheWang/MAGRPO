import torch as th
import torch.nn as nn
import torch.nn.functional as F


class TransformerGaussianAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(args.hidden_dim)

        nhead = int(getattr(args, "transformer_nhead", 4))
        n_layers = int(getattr(args, "transformer_layers", 2))
        ff_mult = int(getattr(args, "transformer_ff_mult", 4))
        dropout = float(getattr(args, "transformer_dropout", 0.0))

        if self.hidden_dim % nhead != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by transformer_nhead ({nhead})"
            )
        if n_layers < 1:
            raise ValueError("transformer_layers must be >= 1")

        self.fc_in = nn.Linear(input_shape, self.hidden_dim)
        self.mem_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=nhead,
            dim_feedforward=ff_mult * self.hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.fc_mu = nn.Linear(self.hidden_dim, args.n_actions)

        init_log_std = float(getattr(args, "init_log_std", -0.5))
        self.log_std = nn.Parameter(th.full((args.n_actions,), init_log_std))
        self.log_std_min = float(getattr(args, "log_std_min", -20.0))
        self.log_std_max = float(getattr(args, "log_std_max", 2.0))

    def init_hidden(self):
        return self.fc_in.weight.new_zeros(1, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.gelu(self.fc_in(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        mem = self.mem_proj(h_in)

        # Two-token recurrent transformer update:
        # token 0 = memory, token 1 = current input embedding.
        seq = th.stack([mem, x], dim=1)
        seq_out = self.encoder(seq)

        h_next = self.norm(seq_out[:, 0, :])
        act_feat = self.norm(seq_out[:, 1, :])

        mu = self.fc_mu(act_feat)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max).expand_as(mu)
        return (mu, log_std), h_next

