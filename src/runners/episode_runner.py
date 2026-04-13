from functools import partial
import json
import os

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        self.env = env_REGISTRY[self.args.env](
            **self.args.env_args,
            common_reward=self.args.common_reward,
            reward_scalarisation=self.args.reward_scalarisation,
        )
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000
        self._best_return_mean = {"": None, "test_": None}
        self._interval_best = {"": None, "test_": None}
        self._terminal_reward_only = bool(
            getattr(self.args, "env_args", {}).get("terminal_reward_only", False)
        )
        self._return_stat_base = (
            "terminal_reward" if self._terminal_reward_only else "return"
        )

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        for key, value in (env_info or {}).items():
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                cur_stats[key] = cur_stats.get(key, 0) + float(value)
                cur_stats[f"last_{key}"] = float(value)
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        if self._terminal_reward_only:
            episode_return_for_log = episode_return
        else:
            episode_return_for_log = episode_return / max(self.t, 1)

        export_prefix = "test_" if test_mode else ""
        if self._should_export(export_prefix):
            self._update_interval_best(
                export_prefix, episode_return_for_log, env_info, self.batch, env_idx=0
            )

        cur_returns.append(episode_return_for_log)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac, "action_selector") and hasattr(
                self.mac.action_selector, "epsilon"
            ):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        base = self._return_stat_base
        if self.args.common_reward:
            window_return_mean = float(np.mean(returns))
            window_return_std = float(np.std(returns))
            # Keep train as last-episode value; use test-window mean for test logging.
            return_ep = (
                window_return_mean if prefix == "test_" else float(returns[-1])
            )
            metric_for_best = return_ep
            self.logger.log_stat(prefix + f"{base}_ep", return_ep, self.t_env)
            self.logger.log_stat(
                prefix + f"{base}_window_mean", window_return_mean, self.t_env
            )
            self.logger.log_stat(
                prefix + f"{base}_window_std", window_return_std, self.t_env
            )
        else:
            returns_arr = np.array(returns)
            for i in range(self.args.n_agents):
                agent_window_mean = float(returns_arr[:, i].mean())
                agent_window_std = float(returns_arr[:, i].std())
                agent_return_ep = float(returns_arr[-1, i])
                self.logger.log_stat(
                    prefix + f"agent_{i}_{base}_ep",
                    agent_return_ep,
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_{base}_window_mean",
                    agent_window_mean,
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_{base}_window_std",
                    agent_window_std,
                    self.t_env,
                )
            total_returns = returns_arr.sum(axis=-1)
            window_return_mean = float(total_returns.mean())
            window_return_std = float(total_returns.std())
            return_ep = (
                window_return_mean
                if prefix == "test_"
                else float(total_returns[-1])
            )
            metric_for_best = return_ep
            self.logger.log_stat(prefix + f"total_{base}_ep", return_ep, self.t_env)
            self.logger.log_stat(
                prefix + f"total_{base}_window_mean", window_return_mean, self.t_env
            )
            self.logger.log_stat(
                prefix + f"total_{base}_window_std", window_return_std, self.t_env
            )
        self._maybe_export_best(prefix, metric_for_best)
        returns.clear()

        for k, v in stats.items():
            if (
                k != "n_episodes"
                and not str(k).startswith("sensing")
                and not str(k).startswith("last_")
            ):
                if k in {"sum_rate", "intra", "inter"} and f"last_{k}" in stats:
                    value_to_log = float(stats[f"last_{k}"])
                else:
                    value_to_log = v / stats["n_episodes"]
                self.logger.log_stat(
                    prefix + k + "_mean", value_to_log, self.t_env
                )
        stats.clear()
        self._reset_interval_best(prefix)

    def _scalar_return(self, value):
        if np.isscalar(value):
            return float(value)
        return float(np.asarray(value).sum())

    def _serialize_return(self, value):
        if np.isscalar(value):
            return float(value)
        return np.asarray(value).tolist()

    def _should_export(self, prefix):
        if not getattr(self.args, "export_best_point", False):
            return False
        mode = getattr(self.args, "export_best_point_mode", "train")
        if mode == "both":
            return True
        if mode == "train":
            return prefix == ""
        if mode == "test":
            return prefix == "test_"
        return False

    def _extract_snapshot(self, batch, env_idx):
        filled = batch["filled"][env_idx, :, 0].detach().to("cpu")
        valid = (filled > 0).nonzero(as_tuple=False)
        if valid.numel() == 0:
            return None
        last_state_idx = int(valid[-1].item())
        state = (
            batch["state"][env_idx, last_state_idx]
            .detach()
            .to("cpu")
            .numpy()
            .tolist()
        )
        obs = (
            batch["obs"][env_idx, last_state_idx]
            .detach()
            .to("cpu")
            .numpy()
            .tolist()
        )
        action_idx = max(last_state_idx - 1, 0)
        actions = (
            batch["actions"][env_idx, action_idx]
            .detach()
            .to("cpu")
            .numpy()
            .tolist()
        )
        return {
            "timestep": last_state_idx,
            "global_state": state,
            "local_observation": obs,
            "joint_action": actions,
        }

    def _update_interval_best(self, prefix, episode_return, env_info, batch, env_idx):
        snapshot = self._extract_snapshot(batch, env_idx)
        if snapshot is None:
            return
        scalar_return = self._scalar_return(episode_return)
        current = self._interval_best.get(prefix)
        if current is None or scalar_return > current["episode_return"]:
            self._interval_best[prefix] = {
                "episode_return": scalar_return,
                "episode_return_raw": self._serialize_return(episode_return),
                "snapshot": snapshot,
                "env_info": env_info or {},
                "t_env": self.t_env,
            }

    def _serialize_env_info(self, env_info):
        serialized = {}
        for key, value in (env_info or {}).items():
            try:
                serialized[key] = float(value)
            except (TypeError, ValueError):
                serialized[key] = value
        return serialized

    def _build_p1_constraints(self, env_info):
        serialized = self._serialize_env_info(env_info)
        keys = ["power_violation", "spacing_violation"]
        missing = [k for k in keys if k not in serialized]
        if missing:
            return {"ok": None, "missing": missing}
        tol = float(getattr(self.args, "export_best_point_violation_tol", 1e-6))
        constraints = {
            "ok": all(serialized[k] <= tol for k in keys),
            "violation_tol": tol,
            "power_violation": serialized["power_violation"],
            "spacing_violation": serialized["spacing_violation"],
        }
        for key in ("p_tx", "sum_rate", "violation"):
            if key in serialized:
                constraints[key] = serialized[key]
        env_args = getattr(self.args, "env_args", {})
        for key in ("p_max", "d_min"):
            if key in env_args:
                constraints[key] = float(env_args[key])
        return constraints

    def _export_best_point(self, prefix, payload):
        export_root = getattr(self.args, "export_best_point_dir", None)
        if not export_root:
            export_root = getattr(self.args, "local_results_path", "results")
        token = getattr(self.args, "unique_token", "run")
        export_dir = os.path.join(export_root, token)
        os.makedirs(export_dir, exist_ok=True)
        mode_label = "train" if prefix == "" else "test"
        export_path = os.path.join(export_dir, f"best_{mode_label}.json")
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        self.logger.console_logger.info(
            f"Exported best {mode_label} point to {export_path}"
        )

    def _maybe_export_best(self, prefix, metric_value):
        if not self._should_export(prefix):
            return
        interval = self._interval_best.get(prefix)
        if interval is None:
            return
        best_mean = self._best_return_mean.get(prefix)
        if best_mean is not None and metric_value <= best_mean:
            return
        self._best_return_mean[prefix] = float(metric_value)
        metric_key = (
            "terminal_reward_ep" if self._terminal_reward_only else "return_ep"
        )
        payload = {
            "mode": "train" if prefix == "" else "test",
            "t_env": interval["t_env"],
            metric_key: float(metric_value),
            "episode_return": interval["episode_return"],
            "episode_return_raw": interval["episode_return_raw"],
            "snapshot_timestep": interval["snapshot"]["timestep"],
            "global_state": interval["snapshot"]["global_state"],
            "local_observation": interval["snapshot"]["local_observation"],
            "joint_action": interval["snapshot"]["joint_action"],
            "p1_constraints": self._build_p1_constraints(interval["env_info"]),
            "env_info": self._serialize_env_info(interval["env_info"]),
        }
        self._export_best_point(prefix, payload)

    def _reset_interval_best(self, prefix):
        if prefix in self._interval_best:
            self._interval_best[prefix] = None
