import copy
import math
from typing import Tuple

import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from modules.critics import REGISTRY as critic_resigtry
from utils.lr_schedules import build_lr_scheduler, get_current_lr, step_lr_scheduler


class MAGRPOContinuousLearner:
    # Two-stage learner:
    # 1) MAPPO-style warmup for a fixed number of env steps to build a strong reference policy.
    # 2) Pure GRPO updates (no critic updates) with group-relative advantages and optional KL-to-reference.
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.action_bound = float(
            getattr(args, "action_bound", None) or args.env_args.get("action_bound", 1.0)
        )
        self.log_prob_eps = float(getattr(args, "log_prob_eps", 1e-6))
        self.nan_guard_eps = float(getattr(args, "nan_guard_eps", 1e-8))
        self.group_norm_eps = float(getattr(args, "group_norm_eps", 1e-8))
        self.gae_lambda = float(getattr(args, "gae_lambda", 0.95))
        self.norm_advantage = bool(getattr(args, "norm_advantage", False))
        self.adv_norm_eps = float(getattr(args, "adv_norm_eps", 1e-8))
        self.entropy_coef_start = float(getattr(args, "entropy_coef", 0.0))
        self.entropy_coef_end = float(
            getattr(args, "entropy_coef_end", self.entropy_coef_start)
        )
        self.entropy_anneal_steps = int(
            getattr(args, "entropy_anneal_steps", getattr(args, "t_max", 1))
        )

        # GRPO controls
        self.grpo_warmup_steps = int(getattr(args, "grpo_warmup_steps", 0))
        self.grpo_group_size = int(getattr(args, "grpo_group_size", args.batch_size))
        self.grpo_kl_coef = float(getattr(args, "grpo_kl_coef", 0.0))
        self.grpo_gamma = float(getattr(args, "grpo_gamma", args.gamma))
        # Stability guards for pure-GRPO stage.
        self.log_ratio_clip = float(getattr(args, "log_ratio_clip", 10.0))
        self.grpo_kl_term_clip = float(getattr(args, "grpo_kl_term_clip", 10000.0))
        self.max_abs_pg_loss_for_update = float(
            getattr(args, "max_abs_pg_loss_for_update", 10000.0)
        )
        self.max_grad_norm_for_update = float(
            getattr(args, "max_grad_norm_for_update", 1000000.0)
        )
        if self.log_ratio_clip < 0.0:
            raise ValueError("log_ratio_clip must be >= 0")
        if self.grpo_kl_term_clip < 0.0:
            raise ValueError("grpo_kl_term_clip must be >= 0")
        if self.max_abs_pg_loss_for_update < 0.0:
            raise ValueError("max_abs_pg_loss_for_update must be >= 0")
        if self.max_grad_norm_for_update < 0.0:
            raise ValueError("max_grad_norm_for_update must be >= 0")

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.ref_mac = None
        self.ref_frozen = False
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
        self.agent_lr_scheduler = build_lr_scheduler(self.agent_optimiser, args)

        # Keep critic modules for warmup phase only.
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
        algo_name = str(getattr(args, "name", ""))
        m_antennas = int(args.env_args.get("m_antennas", 0))
        if "highm" in algo_name.lower() and m_antennas > 0 and m_antennas < 4:
            self.logger.console_logger.warning(
                "Config '%s' is tuned for m>=4, but current m_antennas=%d; "
                "this can increase instability risk.",
                algo_name,
                m_antennas,
            )

    def _atanh(self, x: th.Tensor) -> th.Tensor:
        return 0.5 * (th.log1p(x) - th.log1p(-x))

    def _tanh_gaussian_log_prob(
        self, actions: th.Tensor, mu: th.Tensor, log_std: th.Tensor
    ) -> th.Tensor:
        action_scale = th.tensor(
            self.action_bound, device=actions.device, dtype=actions.dtype
        )
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
        ent = 0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std
        return ent.sum(dim=-1)

    def _current_entropy_coef(self, t_env: int) -> float:
        if self.entropy_anneal_steps <= 0:
            return self.entropy_coef_end
        frac = min(max(float(t_env) / float(self.entropy_anneal_steps), 0.0), 1.0)
        return self.entropy_coef_start + frac * (
            self.entropy_coef_end - self.entropy_coef_start
        )

    def _discounted_returns(
        self, rewards: th.Tensor, terminated: th.Tensor, mask: th.Tensor, gamma: float
    ) -> th.Tensor:
        # rewards/mask: (bs, T, n_agents), terminated: (bs, T, 1 or n_agents)
        term = terminated
        if term.size(2) == 1 and rewards.size(2) != 1:
            term = term.expand(-1, -1, rewards.size(2))
        term = term.float()

        returns = th.zeros_like(rewards)
        running = th.zeros_like(rewards[:, 0])
        for t in reversed(range(rewards.size(1))):
            nonterminal = 1.0 - term[:, t]
            running = rewards[:, t] + gamma * running * nonterminal
            running = running * mask[:, t]
            returns[:, t] = running
        return th.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    def _group_relative_advantages(self, returns: th.Tensor, mask: th.Tensor) -> th.Tensor:
        # Group-wise normalize along batch axis only, preserving time/agent structure.
        bs = returns.size(0)
        group_size = max(1, self.grpo_group_size)
        advantages = th.zeros_like(returns)
        for start in range(0, bs, group_size):
            end = min(start + group_size, bs)
            group_returns = returns[start:end]
            group_mask = mask[start:end]
            denom = group_mask.sum(dim=0, keepdim=True).clamp_min(1.0)
            mean = (group_returns * group_mask).sum(dim=0, keepdim=True) / denom
            var = (
                ((group_returns - mean) ** 2) * group_mask
            ).sum(dim=0, keepdim=True) / denom
            std = th.sqrt(var + self.group_norm_eps)
            advantages[start:end] = (group_returns - mean) / std
        return advantages * mask

    def _trajectory_advantages(
        self, advantages: th.Tensor, mask: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        traj_mask = (mask.sum(dim=2) > 0).float()
        traj_denom = mask.sum(dim=2).clamp_min(1.0)
        traj_advantages = (advantages * mask).sum(dim=2) / traj_denom
        traj_advantages = th.nan_to_num(
            traj_advantages, nan=0.0, posinf=0.0, neginf=0.0
        )
        return traj_advantages, traj_mask

    def _raw_trajectory_reward_sums(
        self, rewards_raw: th.Tensor, terminated: th.Tensor, filled: th.Tensor
    ) -> th.Tensor:
        # Raw trajectory-level sum_t R_t using pre-standardization rewards.
        traj_mask = filled.float().clone()
        traj_mask[:, 1:] = traj_mask[:, 1:] * (1 - terminated[:, :-1].float())
        if traj_mask.size(2) == 1 and rewards_raw.size(2) != 1:
            traj_mask = traj_mask.expand(-1, -1, rewards_raw.size(2))

        masked_rewards = th.nan_to_num(
            rewards_raw.float() * traj_mask, nan=0.0, posinf=0.0, neginf=0.0
        )
        traj_reward_sum = masked_rewards.sum(dim=1)
        if traj_reward_sum.dim() == 2:
            if traj_reward_sum.size(1) == 1:
                traj_reward_sum = traj_reward_sum.squeeze(1)
            else:
                scalarisation = str(
                    getattr(self.args, "reward_scalarisation", "sum")
                ).lower()
                if scalarisation == "mean":
                    traj_reward_sum = traj_reward_sum.mean(dim=1)
                else:
                    traj_reward_sum = traj_reward_sum.sum(dim=1)

        traj_reward_sum = th.nan_to_num(
            traj_reward_sum, nan=0.0, posinf=0.0, neginf=0.0
        )
        return traj_reward_sum

    def _raw_trajectory_reward_sum_variance(
        self, rewards_raw: th.Tensor, terminated: th.Tensor, filled: th.Tensor
    ) -> th.Tensor:
        # Global Var(sum_t R_t) over all trajectories in the batch.
        traj_reward_sum = self._raw_trajectory_reward_sums(
            rewards_raw, terminated, filled
        )
        if traj_reward_sum.numel() <= 1:
            return th.zeros(
                (), device=rewards_raw.device, dtype=traj_reward_sum.dtype
            )

        traj_reward_mean = traj_reward_sum.mean()
        traj_reward_var = ((traj_reward_sum - traj_reward_mean) ** 2).mean()
        return th.nan_to_num(
            traj_reward_var, nan=0.0, posinf=0.0, neginf=0.0
        )

    def _raw_group_mean_trajectory_reward_sum_variance(
        self, rewards_raw: th.Tensor, terminated: th.Tensor, filled: th.Tensor
    ) -> th.Tensor:
        # Mean_g Var_{traj in g}(sum_t R_t), aligned with GRPO grouping along batch axis.
        traj_reward_sum = self._raw_trajectory_reward_sums(
            rewards_raw, terminated, filled
        )
        if traj_reward_sum.numel() <= 1:
            return th.zeros(
                (), device=rewards_raw.device, dtype=traj_reward_sum.dtype
            )

        group_size = max(1, self.grpo_group_size)
        group_vars = []
        for start in range(0, traj_reward_sum.size(0), group_size):
            end = min(start + group_size, traj_reward_sum.size(0))
            group_reward_sum = traj_reward_sum[start:end]
            if group_reward_sum.numel() <= 1:
                group_vars.append(
                    th.zeros(
                        (), device=group_reward_sum.device, dtype=group_reward_sum.dtype
                    )
                )
                continue
            group_mean = group_reward_sum.mean()
            group_var = ((group_reward_sum - group_mean) ** 2).mean()
            group_vars.append(
                th.nan_to_num(group_var, nan=0.0, posinf=0.0, neginf=0.0)
            )

        if not group_vars:
            return th.zeros(
                (), device=rewards_raw.device, dtype=traj_reward_sum.dtype
            )
        return th.nan_to_num(
            th.stack(group_vars).mean(), nan=0.0, posinf=0.0, neginf=0.0
        )

    def _log_normalized_advantages(
        self, advantages: th.Tensor, mask: th.Tensor, t_env: int
    ):
        if not self.logger.use_tb:
            return

        traj_advantages, traj_mask = self._trajectory_advantages(advantages, mask)
        step_digits = max(2, len(str(max(0, traj_advantages.size(1) - 1))))
        traj_digits = max(2, len(str(max(0, traj_advantages.size(0) - 1))))

        for traj_idx in range(traj_advantages.size(0)):
            for step_idx in range(traj_advantages.size(1)):
                if traj_mask[traj_idx, step_idx].item() <= 0:
                    continue
                self.logger.log_tb_scalar(
                    (
                        f"advantage_norm/step_{step_idx:0{step_digits}d}"
                        f"/traj_{traj_idx:0{traj_digits}d}"
                    ),
                    traj_advantages[traj_idx, step_idx].item(),
                    t_env,
                )

    def _compute_warmup_normalized_advantages(
        self,
        batch: EpisodeBatch,
        rewards: th.Tensor,
        terminated: th.Tensor,
        mask: th.Tensor,
    ) -> th.Tensor:
        with th.no_grad():
            values = self.critic(batch).squeeze(3)
            values_for_gae = values
            if self.args.standardise_returns:
                values_for_gae = values * th.sqrt(
                    self.ret_ms.var.clamp_min(self.nan_guard_eps)
                ) + self.ret_ms.mean

            advantages, _ = self._gae_returns(
                rewards,
                terminated,
                mask,
                values_for_gae,
                self.args.gamma,
                self.gae_lambda,
            )
            advantages = th.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
            valid_adv = advantages[mask > 0]
            if valid_adv.numel() > 1:
                adv_mean = valid_adv.mean()
                adv_std = valid_adv.std(unbiased=False).clamp_min(self.adv_norm_eps)
                advantages = (advantages - adv_mean) / adv_std

        return th.nan_to_num(advantages * mask, nan=0.0, posinf=0.0, neginf=0.0)

    def _forward_mac_params(self, mac, batch: EpisodeBatch):
        mus = []
        log_stds = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            params = mac.forward(batch, t=t)
            mus.append(params["mu"])
            log_stds.append(params["log_std"])
        return th.stack(mus, dim=1), th.stack(log_stds, dim=1)

    def _maybe_freeze_reference(self):
        if self.ref_frozen:
            return
        self.ref_mac = copy.deepcopy(self.mac)
        self.ref_mac.load_state(self.mac)
        for param in self.ref_mac.parameters():
            param.requires_grad = False
        self.ref_frozen = True

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        entropy_coef = self._current_entropy_coef(t_env)
        should_log_dense_adv = self.logger.use_tb and (
            t_env - self.log_stats_t >= self.args.learner_log_interval
        )
        rewards_raw = batch["reward"][:, :-1].float()
        rewards = rewards_raw.clone()
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        reward_sum_var_raw = self._raw_trajectory_reward_sum_variance(
            rewards_raw, batch["terminated"][:, :-1], batch["filled"][:, :-1]
        )
        reward_sum_var_group_mean_raw = (
            self._raw_group_mean_trajectory_reward_sum_variance(
                rewards_raw, batch["terminated"][:, :-1], batch["filled"][:, :-1]
            )
        )

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

        old_mu, old_log_std = self._forward_mac_params(self.old_mac, batch)
        old_log_pi_taken = self._tanh_gaussian_log_prob(actions, old_mu, old_log_std)

        returns_var_raw = None
        actor_nonfinite_updates = 0
        actor_large_grad_skips = 0
        dense_advantages = None
        if t_env < self.grpo_warmup_steps:
            if should_log_dense_adv:
                dense_advantages = self._compute_warmup_normalized_advantages(
                    batch, rewards, terminated, mask
                )
            pg_loss, grad_norm, log_std, entropy, adv_mean, adv_std, critic_train_stats = (
                self._warmup_mappo_update(
                    batch,
                    rewards,
                    actions,
                    terminated,
                    old_log_pi_taken,
                    mask,
                    critic_mask,
                    entropy_coef,
                )
            )
            self.critic_training_steps += 1
            kl_mean = th.tensor(0.0, device=actions.device, dtype=actions.dtype)
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
            mode = 0.0  # MAPPO warmup mode
        else:
            self._maybe_freeze_reference()
            (
                pg_loss,
                grad_norm,
                log_std,
                entropy,
                adv_mean,
                adv_std,
                returns_var_raw,
                kl_mean,
                actor_nonfinite_updates,
                actor_large_grad_skips,
                dense_advantages,
            ) = (
                self._pure_grpo_update(
                    batch,
                    rewards,
                    actions,
                    terminated,
                    old_log_pi_taken,
                    mask,
                    entropy_coef,
                )
            )
            critic_train_stats = None
            mode = 1.0  # pure GRPO mode

        self.old_mac.load_state(self.mac)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            if critic_train_stats is not None:
                ts_logged = max(1, len(critic_train_stats["critic_loss"]))
                for key in [
                    "critic_loss",
                    "critic_grad_norm",
                    "td_error_abs",
                    "q_taken_mean",
                    "target_mean",
                ]:
                    self.logger.log_stat(
                        key, sum(critic_train_stats[key]) / ts_logged, t_env
                    )
            self.logger.log_stat("magrpo_mode", mode, t_env)
            self.logger.log_stat("advantage_mean", adv_mean.item(), t_env)
            self.logger.log_stat("advantage_std", adv_std.item(), t_env)
            self.logger.log_stat(
                "reward_sum_var_raw", reward_sum_var_raw.item(), t_env
            )
            self.logger.log_stat(
                "reward_sum_var_group_mean_raw",
                reward_sum_var_group_mean_raw.item(),
                t_env,
            )
            if returns_var_raw is not None:
                self.logger.log_stat(
                    "returns_var_raw", returns_var_raw.item(), t_env
                )
            self.logger.log_stat(
                "pg_loss",
                th.nan_to_num(pg_loss, nan=0.0, posinf=0.0, neginf=0.0).item(),
                t_env,
            )
            self.logger.log_stat(
                "agent_grad_norm",
                th.nan_to_num(grad_norm, nan=0.0, posinf=0.0, neginf=0.0).item(),
                t_env,
            )
            self.logger.log_stat("agent_lr", get_current_lr(self.agent_optimiser), t_env)
            self.logger.log_stat("critic_lr", get_current_lr(self.critic_optimiser), t_env)
            self.logger.log_stat(
                "log_std_mean",
                th.nan_to_num(log_std.mean(), nan=0.0, posinf=0.0, neginf=0.0).item(),
                t_env,
            )
            mask_sum = mask.sum().clamp_min(self.nan_guard_eps).item()
            self.logger.log_stat(
                "entropy_mean",
                th.nan_to_num(
                    (entropy * mask).sum() / max(mask_sum, self.nan_guard_eps),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).item(),
                t_env,
            )
            self.logger.log_stat("entropy_coef", entropy_coef, t_env)
            self.logger.log_stat(
                "actor_nonfinite_updates", actor_nonfinite_updates, t_env
            )
            self.logger.log_stat(
                "actor_large_grad_skips", actor_large_grad_skips, t_env
            )
            if self.ref_frozen:
                self.logger.log_stat("grpo_kl_mean", kl_mean.item(), t_env)
            if dense_advantages is not None:
                self._log_normalized_advantages(dense_advantages, mask, t_env)
            self.log_stats_t = t_env

    def _warmup_mappo_update(
        self,
        batch: EpisodeBatch,
        rewards: th.Tensor,
        actions: th.Tensor,
        terminated: th.Tensor,
        old_log_pi_taken: th.Tensor,
        mask: th.Tensor,
        critic_mask: th.Tensor,
        entropy_coef: float,
    ):
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

        for _ in range(self.args.epochs):
            ep_indices = th.randperm(batch.batch_size, device=actions.device)
            for start in range(0, batch.batch_size, self.mini_batch_size):
                mb_ep_idx = ep_indices[start : start + self.mini_batch_size]
                mb_batch = batch[mb_ep_idx.tolist(), :]
                mb_actions = actions[mb_ep_idx]
                mb_rewards = rewards[mb_ep_idx]
                mb_terminated = terminated[mb_ep_idx]
                mb_mask = mask[mb_ep_idx]
                mb_critic_mask = critic_mask[mb_ep_idx]
                mb_old_log_pi_taken = old_log_pi_taken[mb_ep_idx]

                mu, log_std = self._forward_mac_params(self.mac, mb_batch)
                log_pi_taken = self._tanh_gaussian_log_prob(mb_actions, mu, log_std)
                entropy = self._gaussian_entropy(log_std)

                advantages, mb_critic_stats = self.train_critic_sequential(
                    self.critic,
                    self.target_critic,
                    mb_batch,
                    mb_rewards,
                    mb_terminated,
                    mb_critic_mask,
                )
                advantages = advantages.detach()
                if self.norm_advantage:
                    valid_adv = advantages[mb_mask > 0]
                    if valid_adv.numel() > 1:
                        adv_mean = valid_adv.mean()
                        adv_std = valid_adv.std(unbiased=False).clamp_min(
                            self.adv_norm_eps
                        )
                        advantages = (advantages - adv_mean) / adv_std

                ratios = th.exp(log_pi_taken - mb_old_log_pi_taken.detach())
                ratios = th.nan_to_num(ratios, nan=1.0, posinf=1.0, neginf=1.0)
                surr1 = ratios * advantages
                surr2 = (
                    th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                    * advantages
                )

                mb_mask_sum = mb_mask.sum().clamp_min(self.nan_guard_eps)
                pg_loss = -(
                    (th.min(surr1, surr2) + entropy_coef * entropy) * mb_mask
                ).sum() / mb_mask_sum
                pg_loss = th.nan_to_num(pg_loss, nan=0.0, posinf=0.0, neginf=0.0)

                self.agent_optimiser.zero_grad()
                pg_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(
                    self.agent_params, self.args.grad_norm_clip
                )
                self.agent_optimiser.step()
                step_lr_scheduler(self.agent_lr_scheduler, pg_loss.item())

                for key, values in mb_critic_stats.items():
                    critic_train_stats[key].extend(values)
                actor_pg_losses.append(pg_loss.item())
                actor_grad_norms.append(grad_norm.item())
                actor_log_std_means.append(log_std.mean().item())
                mb_mask_denom = mb_mask.sum().clamp_min(self.nan_guard_eps).item()
                actor_entropy_means.append(
                    (entropy * mb_mask).sum().item() / mb_mask_denom
                )
                actor_adv_means.append(
                    (advantages * mb_mask).sum().item() / mb_mask_denom
                )

        pg_loss = th.tensor(
            sum(actor_pg_losses) / max(1, len(actor_pg_losses)),
            device=actions.device,
        )
        grad_norm = th.tensor(
            sum(actor_grad_norms) / max(1, len(actor_grad_norms)),
            device=actions.device,
        )
        adv_mean = th.tensor(
            sum(actor_adv_means) / max(1, len(actor_adv_means)),
            device=actions.device,
        )
        adv_std = th.tensor(
            1.0, device=actions.device, dtype=actions.dtype
        )
        log_std = th.tensor(
            sum(actor_log_std_means) / max(1, len(actor_log_std_means)),
            device=actions.device,
        ).view(1)
        entropy = th.tensor(
            sum(actor_entropy_means) / max(1, len(actor_entropy_means)),
            device=actions.device,
        ).view(1, 1, 1)
        return pg_loss, grad_norm, log_std, entropy, adv_mean, adv_std, critic_train_stats

    def _pure_grpo_update(
        self,
        batch: EpisodeBatch,
        rewards: th.Tensor,
        actions: th.Tensor,
        terminated: th.Tensor,
        old_log_pi_taken: th.Tensor,
        mask: th.Tensor,
        entropy_coef: float,
    ):
        returns = self._discounted_returns(rewards, terminated, mask, self.grpo_gamma)
        returns = th.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        raw_advantages = self._group_relative_advantages(returns, mask).detach()
        raw_advantages = th.nan_to_num(
            raw_advantages, nan=0.0, posinf=0.0, neginf=0.0
        )
        mask_sum = mask.sum().clamp_min(self.nan_guard_eps)
        adv_mean = (raw_advantages * mask).sum() / mask_sum
        adv_var = ((raw_advantages - adv_mean) ** 2 * mask).sum() / mask_sum
        adv_std = th.sqrt(adv_var + self.group_norm_eps)

        # Variance over per-step returns from step 1 to current step (masked).
        returns_mean = (returns * mask).sum() / mask_sum
        returns_var = (((returns - returns_mean) ** 2) * mask).sum() / mask_sum
        returns_var = th.nan_to_num(
            returns_var, nan=0.0, posinf=0.0, neginf=0.0
        )

        ref_mu, ref_log_std = self._forward_mac_params(self.ref_mac, batch)
        ref_log_pi_taken = self._tanh_gaussian_log_prob(
            actions, ref_mu, ref_log_std
        ).detach()

        kl_mean = th.tensor(0.0, device=actions.device, dtype=actions.dtype)
        pg_loss = th.tensor(0.0, device=actions.device, dtype=actions.dtype)
        grad_norm = th.tensor(0.0, device=actions.device, dtype=actions.dtype)
        log_std = th.zeros(1, device=actions.device, dtype=actions.dtype)
        entropy = th.zeros_like(mask)
        actor_nonfinite_updates = 0
        actor_large_grad_skips = 0
        for _ in range(self.args.epochs):
            mu, log_std = self._forward_mac_params(self.mac, batch)
            log_std = th.nan_to_num(log_std, nan=0.0, posinf=0.0, neginf=0.0)
            log_pi_taken = self._tanh_gaussian_log_prob(actions, mu, log_std)
            entropy = self._gaussian_entropy(log_std)
            entropy = th.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)

            log_ratio = log_pi_taken - old_log_pi_taken.detach()
            if self.log_ratio_clip > 0.0:
                log_ratio = log_ratio.clamp(-self.log_ratio_clip, self.log_ratio_clip)
            ratios = th.exp(log_ratio)
            ratios = th.nan_to_num(ratios, nan=1.0, posinf=1.0, neginf=1.0)
            surr1 = ratios * raw_advantages
            surr2 = (
                th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * raw_advantages
            )
            # Use a non-negative local KL proxy so the penalty never rewards
            # moving away from the frozen reference policy when the log-prob gap flips sign.
            log_prob_gap = th.nan_to_num(
                log_pi_taken - ref_log_pi_taken, nan=0.0, posinf=0.0, neginf=0.0
            )
            kl_term = 0.5 * log_prob_gap.pow(2)
            if self.grpo_kl_term_clip > 0.0:
                kl_term = kl_term.clamp_max(self.grpo_kl_term_clip)
            kl_term = th.nan_to_num(kl_term, nan=0.0, posinf=0.0, neginf=0.0)
            kl_mean = th.nan_to_num(
                (kl_term * mask).sum() / mask_sum, nan=0.0, posinf=0.0, neginf=0.0
            )

            pg_loss = -(
                (
                    th.min(surr1, surr2)
                    + entropy_coef * entropy
                    - self.grpo_kl_coef * kl_term
                )
                * mask
            ).sum() / mask_sum
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

        return (
            pg_loss,
            grad_norm,
            log_std,
            entropy,
            adv_mean,
            adv_std,
            returns_var,
            kl_mean,
            actor_nonfinite_updates,
            actor_large_grad_skips,
            raw_advantages,
        )

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
                rewards,
                terminated,
                mask,
                values_for_gae,
                self.args.gamma,
                self.gae_lambda,
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

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        mask_sum = mask.sum().clamp_min(self.nan_guard_eps)
        loss = (masked_td_error ** 2).sum() / mask_sum
        loss = th.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        step_lr_scheduler(self.critic_lr_scheduler, loss.item())

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask_sum.item()
        running_log["td_error_abs"].append(
            th.nan_to_num(masked_td_error.abs().sum(), nan=0.0, posinf=0.0, neginf=0.0).item()
            / mask_elems
        )
        running_log["q_taken_mean"].append(
            th.nan_to_num((v * mask).sum(), nan=0.0, posinf=0.0, neginf=0.0).item()
            / mask_elems
        )
        running_log["target_mean"].append(
            th.nan_to_num((target_returns * mask).sum(), nan=0.0, posinf=0.0, neginf=0.0).item()
            / mask_elems
        )

        return advantages * mask, running_log

    def _gae_returns(
        self,
        rewards: th.Tensor,
        terminated: th.Tensor,
        mask: th.Tensor,
        values: th.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[th.Tensor, th.Tensor]:
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
            gae = th.nan_to_num(gae, nan=0.0, posinf=0.0, neginf=0.0)
            gae = gae * mask[:, t]
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
        if self.ref_mac is not None:
            self.ref_mac.cuda()

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
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage)
        )
