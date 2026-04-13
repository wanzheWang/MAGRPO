from functools import partial
from multiprocessing import Pipe, Process
import json
import os

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        base_seed = int(self.args.env_args.get("seed", getattr(self.args, "seed", 1)))
        shared_channel_seed = int(self.args.env_args.get("channel_seed", base_seed))
        shared_user_position_seed = int(
            self.args.env_args.get("user_position_seed", base_seed)
        )
        for i in range(self.batch_size):
            # Keep per-env dynamics diverse while sharing channel/user-position randomness.
            env_args[i]["seed"] = base_seed + i
            env_args[i]["channel_seed"] = shared_channel_seed
            env_args[i]["user_position_seed"] = shared_user_position_seed
            env_args[i]["common_reward"] = self.args.common_reward
            env_args[i]["reward_scalarisation"] = self.args.reward_scalarisation
        self.ps = [
            Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
            )
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
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
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        if self.args.common_reward:
            episode_returns = [0 for _ in range(self.batch_size)]
        else:
            episode_returns = [
                np.zeros(self.args.n_agents) for _ in range(self.batch_size)
            ]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        terminated_env_infos = [None for _ in range(self.batch_size)]

        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                bs=envs_not_terminated,
                test_mode=test_mode,
            )
            cpu_actions = actions.detach().to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {"actions": actions.unsqueeze(1)}
            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[
                        idx
                    ]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            # Update envs_not_terminated
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {"reward": [], "terminated": []}
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                        terminated_env_infos[idx] = data["info"]
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        if self._terminal_reward_only:
            episode_returns_for_log = list(episode_returns)
        else:
            episode_returns_for_log = []
            for ret, ep_len in zip(episode_returns, episode_lengths):
                steps = max(int(ep_len), 1)
                episode_returns_for_log.append(ret / steps)

        export_prefix = "test_" if test_mode else ""
        if self._should_export(export_prefix):
            best_idx = int(
                np.argmax([self._scalar_return(ret) for ret in episode_returns_for_log])
            )
            best_return = episode_returns_for_log[best_idx]
            snapshot = self._extract_snapshot(self.batch, best_idx)
            if snapshot is not None:
                self._update_interval_best(
                    export_prefix,
                    best_return,
                    terminated_env_infos[best_idx],
                    snapshot,
                )

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        for env_info in final_env_infos:
            for key, value in (env_info or {}).items():
                if isinstance(value, (int, float, np.number)) and not isinstance(
                    value, bool
                ):
                    cur_stats[key] = cur_stats.get(key, 0) + float(value)
                    cur_stats[f"last_{key}"] = float(value)
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns_for_log)

        n_test_runs = (
            max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )
        if test_mode and (len(self.test_returns) == n_test_runs):
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

    def _update_interval_best(self, prefix, episode_return, env_info, snapshot):
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


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            _, reward, terminated, truncated, env_info = env.step(actions)
            terminated = terminated or truncated
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send(
                {
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                }
            )
        elif cmd == "reset":
            env.reset()
            remote.send(
                {
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs(),
                }
            )
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "render":
            env.render()
        elif cmd == "save_replay":
            env.save_replay()
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)
