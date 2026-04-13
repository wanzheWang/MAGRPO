# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy
import math
from typing import Tuple

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_resigtry
from utils.lr_schedules import build_lr_scheduler, get_current_lr, step_lr_scheduler


class PPOContinuousLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.action_bound = float(getattr(args, "action_bound", None) or args.env_args.get("action_bound", 1.0))
        self.log_prob_eps = float(getattr(args, "log_prob_eps", 1e-6))
        self.nan_guard_eps = float(getattr(args, "nan_guard_eps", 1e-8))
        self.gae_lambda = float(getattr(args, "gae_lambda", 0.95))
        self.norm_advantage = bool(getattr(args, "norm_advantage", False))
        self.adv_norm_eps = float(getattr(args, "adv_norm_eps", 1e-8))
        self.adv_clip = float(getattr(args, "adv_clip", 0.0))
        self.log_ratio_clip = float(getattr(args, "log_ratio_clip", 0.0))
        self.max_abs_pg_loss_for_update = float(
            getattr(args, "max_abs_pg_loss_for_update", 0.0)
        )
        self.max_grad_norm_for_update = float(
            getattr(args, "max_grad_norm_for_update", 0.0)
        )
        self.entropy_coef_start = float(getattr(args, "entropy_coef", 0.0))
        self.entropy_coef_end = float(
            getattr(args, "entropy_coef_end", self.entropy_coef_start)
        )
        self.entropy_anneal_steps = int(
            getattr(args, "entropy_anneal_steps", getattr(args, "t_max", 1))
        )

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.agent_lr_scheduler = build_lr_scheduler(self.agent_optimiser, args)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)
        self.critic_lr_scheduler = build_lr_scheduler(self.critic_optimiser, args)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=device)
        self.mini_batch_size = int(getattr(args, "mini_batch_size", args.batch_size))
        if self.mini_batch_size <= 0:
            raise ValueError("mini_batch_size must be > 0")
        if self.adv_clip < 0.0:
            raise ValueError("adv_clip must be >= 0")
        if self.log_ratio_clip < 0.0:
            raise ValueError("log_ratio_clip must be >= 0")
        if self.max_abs_pg_loss_for_update < 0.0:
            raise ValueError("max_abs_pg_loss_for_update must be >= 0")
        if self.max_grad_norm_for_update < 0.0:
            raise ValueError("max_grad_norm_for_update must be >= 0")

    def _atanh(self, x: th.Tensor) -> th.Tensor:
        return 0.5 * (th.log1p(x) - th.log1p(-x))

    def _tanh_gaussian_log_prob(
        self, actions: th.Tensor, mu: th.Tensor, log_std: th.Tensor
    ) -> th.Tensor:
        # actions, mu, log_std: (..., action_dim)
        action_scale = th.tensor(self.action_bound, device=actions.device, dtype=actions.dtype)
        scaled_actions = (actions / action_scale).clamp(
            -1.0 + self.log_prob_eps, 1.0 - self.log_prob_eps
        )
        pre_tanh = self._atanh(scaled_actions)

        std = th.exp(log_std)
        var = (std * std).clamp_min(self.nan_guard_eps)
        log_prob_u = -0.5 * (
            ((pre_tanh - mu) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_prob_u = log_prob_u.sum(dim=-1)

        correction = th.log(action_scale) * actions.size(-1) + th.log(
            1.0 - scaled_actions * scaled_actions + self.log_prob_eps
        ).sum(dim=-1)
        return th.nan_to_num(log_prob_u - correction, nan=0.0, posinf=0.0, neginf=0.0)

    def _gaussian_entropy(self, log_std: th.Tensor) -> th.Tensor:
        # Entropy of N(mu, std) (pre-tanh); shape (..., action_dim) -> (...,)
        ent = 0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std
        return ent.sum(dim=-1)

    def _current_entropy_coef(self, t_env: int) -> float:
        if self.entropy_anneal_steps <= 0:
            return self.entropy_coef_end
        frac = min(max(float(t_env) / float(self.entropy_anneal_steps), 0.0), 1.0)
        return self.entropy_coef_start + frac * (
            self.entropy_coef_end - self.entropy_coef_start
        )

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        entropy_coef = self._current_entropy_coef(t_env)
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]  # (bs, T, n_agents, action_dim)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(
                self.rew_ms.var.clamp_min(self.nan_guard_eps)
            )
        rewards = th.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        if self.args.common_reward:
            assert rewards.size(2) == 1, "Expected singular agent dimension for common rewards"
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        old_mus = []
        old_log_stds = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            params = self.old_mac.forward(batch, t=t)
            old_mus.append(params["mu"])
            old_log_stds.append(params["log_std"])
        old_mu = th.stack(old_mus, dim=1)
        old_log_std = th.stack(old_log_stds, dim=1)

        old_log_pi_taken = self._tanh_gaussian_log_prob(actions, old_mu, old_log_std)
        critic_train_stats = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }
        actor_pg_losses = []
        actor_grad_norms = []
        actor_log_std_means = []
        actor_entropy_means = []
        actor_adv_means = []
        actor_nonfinite_updates = 0
        actor_large_grad_skips = 0
        critic_nonfinite_updates = 0

        for _ in range(self.args.epochs):
            ep_indices = th.randperm(batch.batch_size, device=actions.device)
            for start in range(0, batch.batch_size, self.mini_batch_size):
                mb_ep_idx = ep_indices[start : start + self.mini_batch_size]
                mb_batch = batch[mb_ep_idx.tolist(), :]
                mb_actions = actions[mb_ep_idx]
                mb_rewards = rewards[mb_ep_idx]
                mb_mask = mask[mb_ep_idx]
                mb_critic_mask = critic_mask[mb_ep_idx]
                mb_old_log_pi_taken = old_log_pi_taken[mb_ep_idx]

                mus = []
                log_stds = []
                self.mac.init_hidden(mb_batch.batch_size)
                for t in range(mb_batch.max_seq_length - 1):
                    params = self.mac.forward(mb_batch, t=t)
                    mus.append(params["mu"])
                    log_stds.append(params["log_std"])
                mu = th.stack(mus, dim=1)
                log_std = th.stack(log_stds, dim=1)

                log_pi_taken = self._tanh_gaussian_log_prob(mb_actions, mu, log_std)
                entropy = self._gaussian_entropy(log_std)

                mb_terminated = terminated[mb_ep_idx]
                advantages, mb_critic_stats, mb_critic_nonfinite_updates = self.train_critic_sequential(
                    self.critic,
                    self.target_critic,
                    mb_batch,
                    mb_rewards,
                    mb_terminated,
                    mb_critic_mask,
                )
                critic_nonfinite_updates += mb_critic_nonfinite_updates
                advantages = advantages.detach()
                if self.norm_advantage:
                    valid_adv = advantages[mb_mask > 0]
                    if valid_adv.numel() > 1:
                        adv_mean = valid_adv.mean()
                        adv_std = valid_adv.std(unbiased=False).clamp_min(
                            self.adv_norm_eps
                        )
                        advantages = (advantages - adv_mean) / adv_std
                if self.adv_clip > 0.0:
                    advantages = advantages.clamp(-self.adv_clip, self.adv_clip)

                log_ratio = log_pi_taken - mb_old_log_pi_taken.detach()
                if self.log_ratio_clip > 0.0:
                    log_ratio = log_ratio.clamp(
                        -self.log_ratio_clip, self.log_ratio_clip
                    )
                ratios = th.exp(log_ratio)
                ratios = th.nan_to_num(ratios, nan=1.0, posinf=1.0, neginf=1.0)
                surr1 = ratios * advantages
                surr2 = (
                    th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                    * advantages
                )

                mb_mask_sum = mb_mask.sum().clamp_min(self.nan_guard_eps)
                pg_loss = (
                    -(
                        (th.min(surr1, surr2) + entropy_coef * entropy)
                        * mb_mask
                    ).sum()
                    / mb_mask_sum
                )
                pg_loss = th.nan_to_num(pg_loss, nan=0.0, posinf=0.0, neginf=0.0)
                pg_loss_value = float(pg_loss.item())
                if (not math.isfinite(pg_loss_value)) or (
                    self.max_abs_pg_loss_for_update > 0.0
                    and abs(pg_loss_value) > self.max_abs_pg_loss_for_update
                ):
                    actor_nonfinite_updates += 1
                    self.agent_optimiser.zero_grad(set_to_none=True)
                    step_lr_scheduler(self.agent_lr_scheduler, float("inf"))
                    continue

                self.agent_optimiser.zero_grad()
                pg_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.agent_params, self.args.grad_norm_clip
                )
                grad_norm_value = float(grad_norm.item())
                if not math.isfinite(grad_norm_value):
                    actor_nonfinite_updates += 1
                    self.agent_optimiser.zero_grad(set_to_none=True)
                    step_lr_scheduler(self.agent_lr_scheduler, float("inf"))
                    continue
                if (
                    self.max_grad_norm_for_update > 0.0
                    and grad_norm_value > self.max_grad_norm_for_update
                ):
                    actor_large_grad_skips += 1
                    self.agent_optimiser.zero_grad(set_to_none=True)
                    step_lr_scheduler(self.agent_lr_scheduler, float("inf"))
                    continue
                self.agent_optimiser.step()
                step_lr_scheduler(self.agent_lr_scheduler, pg_loss_value)

                for key, values in mb_critic_stats.items():
                    critic_train_stats[key].extend(values)
                actor_pg_losses.append(pg_loss_value)
                actor_grad_norms.append(grad_norm_value)
                actor_log_std_means.append(
                    float(th.nan_to_num(log_std.mean(), nan=0.0, posinf=0.0, neginf=0.0).item())
                )
                entropy_denom = mb_mask.sum().clamp_min(self.nan_guard_eps).item()
                actor_entropy_means.append(
                    float(
                        th.nan_to_num((entropy * mb_mask).sum(), nan=0.0, posinf=0.0, neginf=0.0).item()
                        / entropy_denom
                    )
                )
                adv_denom = mb_mask.sum().clamp_min(self.nan_guard_eps).item()
                actor_adv_means.append(
                    float(
                        th.nan_to_num((advantages * mb_mask).sum(), nan=0.0, posinf=0.0, neginf=0.0).item()
                        / adv_denom
                    )
                )

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = max(1, len(critic_train_stats["critic_loss"]))
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat(
                "advantage_mean",
                sum(actor_adv_means) / max(1, len(actor_adv_means)),
                t_env,
            )
            self.logger.log_stat(
                "pg_loss", sum(actor_pg_losses) / max(1, len(actor_pg_losses)), t_env
            )
            self.logger.log_stat(
                "agent_grad_norm",
                sum(actor_grad_norms) / max(1, len(actor_grad_norms)),
                t_env,
            )
            self.logger.log_stat("agent_lr", get_current_lr(self.agent_optimiser), t_env)
            self.logger.log_stat("critic_lr", get_current_lr(self.critic_optimiser), t_env)
            self.logger.log_stat(
                "log_std_mean",
                sum(actor_log_std_means) / max(1, len(actor_log_std_means)),
                t_env,
            )
            self.logger.log_stat(
                "entropy_mean",
                sum(actor_entropy_means) / max(1, len(actor_entropy_means)),
                t_env,
            )
            self.logger.log_stat("entropy_coef", entropy_coef, t_env)
            self.logger.log_stat(
                "actor_nonfinite_updates", actor_nonfinite_updates, t_env
            )
            self.logger.log_stat(
                "actor_large_grad_skips", actor_large_grad_skips, t_env
            )
            self.logger.log_stat(
                "critic_nonfinite_updates", critic_nonfinite_updates, t_env
            )
            self.log_stats_t = t_env

    def train_critic_sequential(
        self, critic, target_critic, batch, rewards, terminated, mask
    ):
        with th.no_grad():
            values = critic(batch).squeeze(3)
            values_for_gae = values
            if self.args.standardise_returns:
                values_for_gae = values * th.sqrt(
                    self.ret_ms.var.clamp_min(self.nan_guard_eps)
                ) + self.ret_ms.mean
            advantages, target_returns = self._gae_returns(
                rewards, terminated, mask, values_for_gae, self.args.gamma, self.gae_lambda
            )
            advantages = th.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
            target_returns = th.nan_to_num(
                target_returns, nan=0.0, posinf=0.0, neginf=0.0
            )

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var.clamp_min(self.nan_guard_eps)
            )
        target_returns = th.nan_to_num(target_returns, nan=0.0, posinf=0.0, neginf=0.0)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }
        nonfinite_updates = 0

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        mask_sum = mask.sum().clamp_min(self.nan_guard_eps)
        loss = (masked_td_error**2).sum() / mask_sum
        loss = th.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            nonfinite_updates += 1
            step_lr_scheduler(self.critic_lr_scheduler, float("inf"))
            grad_norm_value = 0.0
        else:
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.critic_params, self.args.grad_norm_clip
            )
            grad_norm_value = float(grad_norm.item())
            if math.isfinite(grad_norm_value):
                self.critic_optimiser.step()
                step_lr_scheduler(self.critic_lr_scheduler, loss_value)
            else:
                nonfinite_updates += 1
                self.critic_optimiser.zero_grad(set_to_none=True)
                step_lr_scheduler(self.critic_lr_scheduler, float("inf"))
                grad_norm_value = 0.0

        running_log["critic_loss"].append(loss_value if math.isfinite(loss_value) else 0.0)
        running_log["critic_grad_norm"].append(grad_norm_value)
        mask_elems = mask.sum().clamp_min(self.nan_guard_eps).item()
        running_log["td_error_abs"].append(
            float(
                th.nan_to_num(masked_td_error.abs().sum(), nan=0.0, posinf=0.0, neginf=0.0).item()
                / mask_elems
            )
        )
        running_log["q_taken_mean"].append(
            float(
                th.nan_to_num((v * mask).sum(), nan=0.0, posinf=0.0, neginf=0.0).item()
                / mask_elems
            )
        )
        running_log["target_mean"].append(
            float(
                th.nan_to_num((target_returns * mask).sum(), nan=0.0, posinf=0.0, neginf=0.0).item()
                / mask_elems
            )
        )

        return advantages * mask, running_log, nonfinite_updates

    def _gae_returns(
        self,
        rewards: th.Tensor,
        terminated: th.Tensor,
        mask: th.Tensor,
        values: th.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[th.Tensor, th.Tensor]:
        # rewards/mask: (bs, T, n_agents), values: (bs, T+1, n_agents)
        term = terminated
        if term.size(2) == 1 and rewards.size(2) != 1:
            term = term.expand(-1, -1, rewards.size(2))
        term = term.float()

        advantages = th.zeros_like(rewards)
        gae = th.zeros_like(rewards[:, 0])

        for t in reversed(range(rewards.size(1))):
            nonterminal = 1.0 - term[:, t]
            delta = rewards[:, t] + gamma * values[:, t + 1] * nonterminal - values[:, t]
            delta = th.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            gae = delta + gamma * gae_lambda * nonterminal * gae
            gae = gae * mask[:, t]
            gae = th.nan_to_num(gae, nan=0.0, posinf=0.0, neginf=0.0)
            advantages[:, t] = gae

        returns = th.nan_to_num(
            advantages + values[:, :-1], nan=0.0, posinf=0.0, neginf=0.0
        )
        return advantages, returns

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** step * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** step * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage
            )
        )
