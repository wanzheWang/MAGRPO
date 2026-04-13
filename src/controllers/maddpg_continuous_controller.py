import torch as th

from modules.agents import REGISTRY as agent_REGISTRY


class MADDPGContinuousMAC:
    # Parameter-sharing multi-agent controller for continuous-action MADDPG.
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.action_bound = float(
            getattr(args, "action_bound", None)
            or args.env_args.get("action_bound", 1.0)
        )
        self.exploration_noise = float(getattr(args, "maddpg_exploration_noise", 0.1))

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env=0, bs=slice(None), test_mode=False):
        with th.no_grad():
            actions = self.forward(ep_batch, t_ep)[bs]
            if test_mode:
                return actions
            noise = th.randn_like(actions) * self.exploration_noise
            noisy_actions = actions + noise
            noisy_actions = noisy_actions.clamp(-self.action_bound, self.action_bound)
            return th.nan_to_num(noisy_actions, nan=0.0, posinf=0.0, neginf=0.0)

    def target_actions(self, ep_batch, t_ep):
        return self.forward(ep_batch, t_ep)

    def forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        (mu, _log_std), self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        mu = mu.view(ep_batch.batch_size, self.n_agents, -1)
        return th.tanh(mu) * self.action_bound

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
                input_shape += int(th.tensor(action_vshape).prod().item())
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
