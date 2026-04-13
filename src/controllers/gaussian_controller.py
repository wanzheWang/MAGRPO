import numpy as np
import torch as th

from modules.agents import REGISTRY as agent_REGISTRY


class GaussianMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.action_bound = float(getattr(args, "action_bound", None) or args.env_args.get("action_bound", 1.0))

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        with th.no_grad():
            dist_params = self.forward(ep_batch, t_ep, test_mode=test_mode)
            mu = dist_params["mu"][bs]
            log_std = dist_params["log_std"][bs]

            if test_mode:
                actions = th.tanh(mu) * self.action_bound
            else:
                std = th.exp(log_std)
                pre_tanh = mu + std * th.randn_like(mu)
                actions = th.tanh(pre_tanh) * self.action_bound
            actions = th.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
            if not th.isfinite(actions).all():
                raise FloatingPointError("Non-finite actions detected in GaussianMAC.")

        return actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        (mu, log_std), self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        mu = mu.view(ep_batch.batch_size, self.n_agents, -1)
        log_std = log_std.view(ep_batch.batch_size, self.n_agents, -1)
        return {"mu": mu, "log_std": log_std}

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, self.n_agents, -1
        )

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])

        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            action_vshape = scheme["actions"]["vshape"]
            if isinstance(action_vshape, int):
                input_shape += action_vshape
            else:
                input_shape += int(np.prod(action_vshape))
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
