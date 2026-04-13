import math

import torch as th

from utils.lr_schedules import get_current_lr, step_lr_scheduler
from .ppo_continuous_learner import PPOContinuousLearner


class ActorCriticContinuousLearner(PPOContinuousLearner):
    """
    Continuous-action actor-critic learner (MAA2C-style):
    - Gaussian tanh policy
    - Single on-policy actor update per train call
    - Critic/advantage path reused from PPOContinuousLearner
    """

    def train(self, batch, t_env: int, episode_num: int):
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
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        mus = []
        log_stds = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            params = self.mac.forward(batch, t=t)
            mus.append(params["mu"])
            log_stds.append(params["log_std"])
        mu = th.stack(mus, dim=1)
        log_std = th.stack(log_stds, dim=1)

        log_pi_taken = self._tanh_gaussian_log_prob(actions, mu, log_std)
        entropy = self._gaussian_entropy(log_std)

        (
            advantages,
            critic_train_stats,
            critic_nonfinite_updates,
        ) = self.train_critic_sequential(
            self.critic,
            self.target_critic,
            batch,
            rewards,
            terminated,
            critic_mask,
        )
        advantages = advantages.detach()
        if self.norm_advantage:
            valid_adv = advantages[mask > 0]
            if valid_adv.numel() > 1:
                adv_mean = valid_adv.mean()
                adv_std = valid_adv.std(unbiased=False).clamp_min(self.adv_norm_eps)
                advantages = (advantages - adv_mean) / adv_std

        mask_sum = mask.sum().clamp_min(self.nan_guard_eps)
        pg_loss = -(
            (advantages * log_pi_taken + entropy_coef * entropy) * mask
        ).sum() / mask_sum
        pg_loss = th.nan_to_num(pg_loss, nan=0.0, posinf=0.0, neginf=0.0)
        pg_loss_value = float(pg_loss.item())

        actor_nonfinite_updates = 0
        grad_norm_value = 0.0
        if not math.isfinite(pg_loss_value):
            actor_nonfinite_updates = 1
            self.agent_optimiser.zero_grad(set_to_none=True)
            step_lr_scheduler(self.agent_lr_scheduler, float("inf"))
        else:
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            grad_norm_value = float(grad_norm.item())
            if math.isfinite(grad_norm_value):
                self.agent_optimiser.step()
                step_lr_scheduler(self.agent_lr_scheduler, pg_loss_value)
            else:
                actor_nonfinite_updates = 1
                self.agent_optimiser.zero_grad(set_to_none=True)
                step_lr_scheduler(self.agent_lr_scheduler, float("inf"))
                grad_norm_value = 0.0

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
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            entropy_denom = mask_sum.item()
            self.logger.log_stat(
                "advantage_mean",
                float(
                    th.nan_to_num((advantages * mask).sum(), nan=0.0, posinf=0.0, neginf=0.0).item()
                    / entropy_denom
                ),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss_value, t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm_value, t_env)
            self.logger.log_stat("agent_lr", get_current_lr(self.agent_optimiser), t_env)
            self.logger.log_stat("critic_lr", get_current_lr(self.critic_optimiser), t_env)
            self.logger.log_stat(
                "log_std_mean",
                float(
                    th.nan_to_num(log_std.mean(), nan=0.0, posinf=0.0, neginf=0.0).item()
                ),
                t_env,
            )
            self.logger.log_stat(
                "entropy_mean",
                float(
                    th.nan_to_num((entropy * mask).sum(), nan=0.0, posinf=0.0, neginf=0.0).item()
                    / entropy_denom
                ),
                t_env,
            )
            self.logger.log_stat("entropy_coef", entropy_coef, t_env)
            self.logger.log_stat(
                "actor_nonfinite_updates", actor_nonfinite_updates, t_env
            )
            self.logger.log_stat(
                "critic_nonfinite_updates", critic_nonfinite_updates, t_env
            )
            self.log_stats_t = t_env
