"""FAS environment based on the paper's POMDP formulation.

Implements a field-response channel model with multipath.
"""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FASEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_bs: int = 2,
        k_users: int = 2,
        m_antennas: int = 4,
        user_weights: float | list | np.ndarray = 1.0,
        bs_height: float = 0.0,
        user_height: float = 1.0,
        user_dist_bounds: Tuple[float, float] = (3.0, 5.0),
        user_fixed_angles_deg: list[float] | tuple[float, ...] | None = None,
        user_angle_range_deg: (
            tuple[float, float]
            | list[float]
            | tuple[tuple[float, float], ...]
            | list[list[float]]
            | None
        ) = None,
        action_mode: str = "continuous",
        action_bound: float = 1.0,
        obs_mode: str = "paper",
        pos_action_mode: str = "absolute",
        p_max: float = 1.0,
        d_min: float = 0.1,
        area_bounds: Tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
        bs_center_distance: float | None = None,
        fixed_antenna_offsets_xy: Any = None,
        beta_penalty: float = 1.0,
        reward_mode: str = "paper",
        sum_rate_scale: float = 1.0,
        beta_penalty_mode: str = "static",
        beta_lagrange_lr: float = 0.05,
        beta_lagrange_target: float = 0.0,
        beta_lagrange_min: float = 0.0,
        beta_lagrange_max: float = 50.0,
        interference_scale: float = 1.0,
        noise_var: float = 1e-3,
        lambda_c: float = 1.0,
        resample_fading_each_step: bool = False,
        normalize_w: bool = True,
        num_beam_actions: int = 8,
        num_pos_actions: int = 5,
        delta_u_step: float = 0.05,
        disable_position_action: bool = False,
        l_paths: int = 3,
        path_loss_exp: float = 2.0,
        path_loss_exp_intra: float | None = None,
        path_loss_exp_inter: float | None = None,
        zeta_var: float = 1.0,
        zeta_resample_interval: int = 0,
        fixed_channel_on_reset: bool = False,
        fixed_positions_on_reset: bool = True,
        shared_user_positions: bool = True,
        terminal_reward_only: bool = False,
        shared_fixed_aoa_aod: bool = False,
        fixed_aoa_aod_on_reset: bool = False,
        user_position_resample_interval: int = 0,
        enforce_spacing_by_resample: bool = True,
        spacing_resample_max_tries: int = 64,
        rng_seed: int | None = None,
        channel_seed: int | None = None,
        user_position_seed: int | None = None,
        **kwargs: Any,
    ):
        self.n_agents = int(n_bs)
        self.k_users = int(k_users)
        self.m_antennas = int(m_antennas)
        self.user_weights = self._init_user_weights(user_weights)
        self.bs_height = float(bs_height)
        self.user_height = float(user_height)
        self.user_dist_bounds = tuple(float(x) for x in user_dist_bounds)
        self.user_fixed_angles_rad = None
        if user_fixed_angles_deg is not None:
            angles_deg = np.asarray(user_fixed_angles_deg, dtype=np.float32).reshape(-1)
            if angles_deg.shape[0] != self.k_users:
                raise ValueError(
                    "user_fixed_angles_deg must have length equal to k_users"
                )
            self.user_fixed_angles_rad = np.deg2rad(angles_deg).astype(np.float32)
        self.user_angle_range_rad = None
        self.user_angle_ranges_rad = None
        if user_angle_range_deg is not None:
            angle_bounds_deg = np.asarray(user_angle_range_deg, dtype=np.float32)
            if angle_bounds_deg.ndim == 1:
                angle_bounds_deg = angle_bounds_deg.reshape(-1)
                if angle_bounds_deg.shape[0] != 2:
                    raise ValueError(
                        "user_angle_range_deg must be (min_deg, max_deg) or shape (k_users, 2)"
                    )
                min_deg = float(angle_bounds_deg[0])
                max_deg = float(angle_bounds_deg[1])
                if max_deg < min_deg:
                    raise ValueError(
                        "user_angle_range_deg max must be >= min (in degrees)"
                    )
                self.user_angle_range_rad = (
                    np.deg2rad(min_deg).astype(np.float32),
                    np.deg2rad(max_deg).astype(np.float32),
                )
            elif angle_bounds_deg.ndim == 2 and angle_bounds_deg.shape == (
                self.k_users,
                2,
            ):
                mins = angle_bounds_deg[:, 0]
                maxs = angle_bounds_deg[:, 1]
                if np.any(maxs < mins):
                    raise ValueError(
                        "user_angle_range_deg per-user max must be >= min (in degrees)"
                    )
                self.user_angle_ranges_rad = np.deg2rad(angle_bounds_deg).astype(
                    np.float32
                )
            else:
                raise ValueError(
                    "user_angle_range_deg must be (min_deg, max_deg) or shape (k_users, 2)"
                )
        self.action_mode = str(action_mode).lower().strip()
        self.action_bound = float(action_bound)
        self.obs_mode = str(obs_mode).lower().strip()
        self.pos_action_mode = str(pos_action_mode).lower().strip()
        self.p_max = float(p_max)
        self.d_min = float(d_min)
        self.area_bounds = tuple(float(x) for x in area_bounds)
        self.bs_center_distance = (
            None if bs_center_distance is None else float(bs_center_distance)
        )
        self.fixed_antenna_offsets_xy = fixed_antenna_offsets_xy
        self.beta_penalty = float(beta_penalty)
        self.reward_mode = str(reward_mode).lower().strip()
        self.sum_rate_scale = float(sum_rate_scale)
        self.beta_penalty_mode = str(beta_penalty_mode).lower().strip()
        self.beta_lagrange_lr = float(beta_lagrange_lr)
        self.beta_lagrange_target = float(beta_lagrange_target)
        self.beta_lagrange_min = float(beta_lagrange_min)
        self.beta_lagrange_max = float(beta_lagrange_max)
        self.interference_scale = float(interference_scale)
        self.noise_var = float(noise_var)
        self.lambda_c = float(lambda_c)
        self.resample_fading_each_step = bool(resample_fading_each_step)
        self.normalize_w = bool(normalize_w)
        self.num_beam_actions = int(num_beam_actions)
        self.num_pos_actions = int(num_pos_actions)
        self.delta_u_step = float(delta_u_step)
        self.disable_position_action = bool(disable_position_action)
        self.l_paths = int(l_paths)
        self.path_loss_exp = float(path_loss_exp)
        self.path_loss_exp_intra = (
            self.path_loss_exp
            if path_loss_exp_intra is None
            else float(path_loss_exp_intra)
        )
        self.path_loss_exp_inter = (
            self.path_loss_exp
            if path_loss_exp_inter is None
            else float(path_loss_exp_inter)
        )
        self.zeta_var = float(zeta_var)
        self.zeta_resample_interval = int(zeta_resample_interval)
        self.fixed_channel_on_reset = bool(fixed_channel_on_reset)
        self.fixed_positions_on_reset = bool(fixed_positions_on_reset)
        self.shared_user_positions = bool(shared_user_positions)
        self.terminal_reward_only = bool(terminal_reward_only)
        self.shared_fixed_aoa_aod = bool(shared_fixed_aoa_aod)
        self.fixed_aoa_aod_on_reset = bool(fixed_aoa_aod_on_reset)
        self.user_position_resample_interval = int(user_position_resample_interval)
        self.enforce_spacing_by_resample = bool(enforce_spacing_by_resample)
        self.spacing_resample_max_tries = int(spacing_resample_max_tries)
        self.rng = np.random.default_rng(rng_seed)
        self.channel_seed = channel_seed
        self.channel_rng = np.random.default_rng(
            channel_seed if channel_seed is not None else rng_seed
        )
        self.user_position_seed = user_position_seed
        self.user_pos_rng = np.random.default_rng(
            user_position_seed if user_position_seed is not None else rng_seed
        )
        if self.action_mode not in {"continuous", "discrete"}:
            raise ValueError("action_mode must be 'continuous' or 'discrete'")
        if self.obs_mode not in {"paper", "extended"}:
            raise ValueError("obs_mode must be 'paper' or 'extended'")
        if self.pos_action_mode not in {"absolute", "delta"}:
            raise ValueError("pos_action_mode must be 'absolute' or 'delta'")
        if self.action_bound <= 0.0:
            raise ValueError("action_bound must be > 0")
        if self.bs_center_distance is not None and self.bs_center_distance <= 0.0:
            raise ValueError("bs_center_distance must be > 0 when provided")
        if self.beta_penalty_mode not in {"static", "lagrange"}:
            raise ValueError("beta_penalty_mode must be 'static' or 'lagrange'")
        if self.beta_lagrange_lr < 0.0:
            raise ValueError("beta_lagrange_lr must be >= 0")
        if self.beta_lagrange_min < 0.0:
            raise ValueError("beta_lagrange_min must be >= 0")
        if self.beta_lagrange_min > self.beta_lagrange_max:
            raise ValueError("beta_lagrange_min must be <= beta_lagrange_max")
        if self.interference_scale < 0.0:
            raise ValueError("interference_scale must be >= 0")
        if self.action_mode == "discrete" and self.num_pos_actions < 1:
            raise ValueError("num_pos_actions must be >= 1")
        if self.action_mode == "discrete" and self.num_beam_actions < 1:
            raise ValueError("num_beam_actions must be >= 1")
        if self.l_paths < 1:
            raise ValueError("l_paths must be >= 1")
        if self.reward_mode not in {"paper", "smooth"}:
            raise ValueError("reward_mode must be 'paper' or 'smooth'")
        if self.sum_rate_scale <= 0.0:
            raise ValueError("sum_rate_scale must be > 0")
        if self.path_loss_exp <= 0.0:
            raise ValueError("path_loss_exp must be > 0")
        if self.path_loss_exp_intra <= 0.0:
            raise ValueError("path_loss_exp_intra must be > 0")
        if self.path_loss_exp_inter <= 0.0:
            raise ValueError("path_loss_exp_inter must be > 0")
        if self.zeta_var < 0.0:
            raise ValueError("zeta_var must be >= 0")
        if self.zeta_resample_interval < 0:
            raise ValueError("zeta_resample_interval must be >= 0")
        if self.user_position_resample_interval < 0:
            raise ValueError("user_position_resample_interval must be >= 0")
        if self.spacing_resample_max_tries < 1:
            raise ValueError("spacing_resample_max_tries must be >= 1")
        if len(self.user_dist_bounds) != 2:
            raise ValueError("user_dist_bounds must be (min, max)")
        if self.user_dist_bounds[0] <= 0.0:
            raise ValueError("user_dist_bounds[0] must be > 0")
        if self.user_dist_bounds[0] > self.user_dist_bounds[1]:
            raise ValueError("user_dist_bounds[0] must be <= user_dist_bounds[1]")
        # user_dist_bounds are horizontal (XY) distance bounds from each BS center.

        self._step_count = 0
        self._beam_codebook = None
        self._pos_actions = None
        self._beam_action_dim = 0
        self._pos_action_dim = 0
        self._action_dim = 0
        if self.action_mode == "discrete":
            self._beam_codebook = self._build_beam_codebook()
            self._pos_actions = self._build_pos_actions()
        else:
            # Action per agent: {w_i,k(t+1), ΔU_i(t+1)} (Eq. (16)).
            # w_i,k is complex -> store real/imag, and ΔU_i is per-antenna 2D motion.
            self._beam_action_dim = 2 * self.k_users * self.m_antennas
            self._pos_action_dim = (
                0 if self.disable_position_action else 2 * self.m_antennas
            )
            self._action_dim = self._beam_action_dim + self._pos_action_dim

        self.bs_positions = self._init_bs_positions()
        self.fixed_antenna_offsets_xy = self._init_fixed_antenna_offsets(
            self.fixed_antenna_offsets_xy
        )
        self.user_positions = None
        self.antenna_positions = None
        self._fixed_user_positions = None
        self._fixed_antenna_positions = None
        self._episode_count = 0
        self.h = None
        self.h_hat = None
        self.gamma = None
        self.p_tx = None
        self.i_intra = None
        self.i_inter = None
        self._dir_cos_r = None
        self._dir_cos_t = None
        self._zeta_true = None
        self._fixed_channel_params = None
        self._fixed_shared_aoa_aod_params = None
        self._fixed_aoa_aod_params = None
        self.alpha_t_hat = None
        self.phi_t_hat = None
        self.alpha_r_hat = None
        self.phi_r_hat = None
        self.zeta_hat = None
        self._cached_zeta_true = None
        self._cached_zeta_hat = None
        self.w_actions = None

        self._last_sum_rate = 0.0
        self._last_violation = 0.0
        self._last_power_violation = 0.0
        self._last_spacing_violation = 0.0
        self._last_p_tx_mean = 0.0
        self._last_signal_mean = 0.0
        self._last_sinr_mean = 0.0
        self._last_intra_mean = 0.0
        self._last_inter_mean = 0.0
        self._last_beta_penalty = float(self.beta_penalty)
        self._last_beta_penalty_next = float(self.beta_penalty)

        self._ep_sum_rate = 0.0
        self._ep_violation = 0.0
        self._ep_power_violation = 0.0
        self._ep_spacing_violation = 0.0
        self._ep_p_tx = 0.0
        self._ep_signal = 0.0
        self._ep_sinr = 0.0
        self._ep_intra = 0.0
        self._ep_inter = 0.0

        obs_dim = self._obs_dim()
        # Global state size as in Eq. (13).
        self.state_size = self._state_dim() * self.n_agents
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Box(
                    low=-1.0e6, high=1.0e6, shape=(obs_dim,), dtype=np.float32
                )
                for _ in range(self.n_agents)
            )
        )
        if self.action_mode == "discrete":
            self.action_space = spaces.Tuple(
                tuple(
                    spaces.Discrete(self.num_beam_actions * self.num_pos_actions)
                    for _ in range(self.n_agents)
                )
            )
        else:
            low = -self.action_bound * np.ones((self._action_dim,), dtype=np.float32)
            high = self.action_bound * np.ones((self._action_dim,), dtype=np.float32)
            self.action_space = spaces.Tuple(
                tuple(
                    spaces.Box(low=low, high=high, dtype=np.float32)
                    for _ in range(self.n_agents)
                )
            )

    def _obs_dim(self) -> int:
        if self.obs_mode == "paper":
            # Paper definition:
            # {u_i,m, w_i,k, v_i,k, h_i,i,k, I_2,i,k, omega_i,k*log2(1+SINR_i,k)}
            u_dim = self.m_antennas * 3
            v_dim = self.k_users * 3
            w_dim = 2 * self.k_users * self.m_antennas
            hself_dim = 2 * self.k_users * self.m_antennas
            i2_dim = self.k_users
            weighted_rate_dim = self.k_users
            return (
                u_dim
                + v_dim
                + w_dim
                + hself_dim
                + i2_dim
                + weighted_rate_dim
            )

        # Extended (legacy): U_i (M*3), h_i,k (K*M complex -> 2*K*M),
        # gamma_i,k (K), P_tx (1), I_inter (1).
        return self.m_antennas * 3 + 2 * self.k_users * self.m_antennas + self.k_users + 2

    def _state_dim(self) -> int:
        # Paper definition:
        # {u_i,m, w_i,k, v_i,k, h_i,i,k, h_j!=i,i,k, omega_i,k*log2(1+SINR_i,k)}
        u_dim = self.m_antennas * 3
        v_dim = self.k_users * 3
        w_dim = 2 * self.k_users * self.m_antennas
        hself_dim = 2 * self.k_users * self.m_antennas
        hinter_dim = 2 * (self.n_agents - 1) * self.k_users * self.m_antennas
        weighted_rate_dim = self.k_users
        return (
            u_dim
            + v_dim
            + w_dim
            + hself_dim
            + hinter_dim
            + weighted_rate_dim
        )

    def _init_bs_positions(self) -> np.ndarray:
        # Place BSs on a circle for simple geometry.
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_agents, endpoint=False)
        if self.bs_center_distance is None:
            radius = 0.5 * (self.area_bounds[1] - self.area_bounds[0])
        elif self.n_agents <= 1:
            radius = 0.0
        elif self.n_agents == 2:
            radius = 0.5 * self.bs_center_distance
        else:
            # Adjacent BS spacing on the circle equals bs_center_distance.
            radius = self.bs_center_distance / (2.0 * np.sin(np.pi / self.n_agents))
        return np.stack(
            [radius * np.cos(angles), radius * np.sin(angles)], axis=1
        ).astype(np.float32)

    def _init_user_weights(self, user_weights: float | list | np.ndarray) -> np.ndarray:
        if isinstance(user_weights, (float, int)):
            weights = np.full((self.n_agents, self.k_users), float(user_weights))
        else:
            weights = np.array(user_weights, dtype=np.float32)
            if weights.ndim == 1 and weights.shape[0] == self.k_users:
                weights = np.tile(weights[None, :], (self.n_agents, 1))
            if weights.shape != (self.n_agents, self.k_users):
                raise ValueError(
                    "user_weights must be scalar, length K, or shape (N, K)"
                )
        return weights.astype(np.float32)

    def _init_fixed_antenna_offsets(self, offsets: Any) -> np.ndarray | None:
        if offsets is None:
            return None

        arr = np.asarray(offsets, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape != (2,):
                raise ValueError(
                    "fixed_antenna_offsets_xy must be shape (2,), (M, 2), or (N, M, 2)"
                )
            arr = arr.reshape(1, 2)

        if arr.ndim == 2:
            if arr.shape != (self.m_antennas, 2):
                raise ValueError(
                    "fixed_antenna_offsets_xy with ndim=2 must have shape (m_antennas, 2)"
                )
            arr = np.tile(arr[None, :, :], (self.n_agents, 1, 1))
        elif arr.ndim == 3:
            if arr.shape != (self.n_agents, self.m_antennas, 2):
                raise ValueError(
                    "fixed_antenna_offsets_xy with ndim=3 must have shape (n_bs, m_antennas, 2)"
                )
        else:
            raise ValueError(
                "fixed_antenna_offsets_xy must be shape (2,), (M, 2), or (N, M, 2)"
            )

        xmin, xmax, ymin, ymax = self.area_bounds
        if np.any(arr[..., 0] < xmin) or np.any(arr[..., 0] > xmax):
            raise ValueError("fixed_antenna_offsets_xy x offsets must lie within area_bounds")
        if np.any(arr[..., 1] < ymin) or np.any(arr[..., 1] > ymax):
            raise ValueError("fixed_antenna_offsets_xy y offsets must lie within area_bounds")
        return arr.astype(np.float32)

    def _build_beam_codebook(self) -> np.ndarray:
        # Codebook of beamforming matrices per action: (A, K, M).
        codebook = []
        for _ in range(self.num_beam_actions):
            w = self.rng.normal(size=(self.k_users, self.m_antennas)) + 1j * self.rng.normal(
                size=(self.k_users, self.m_antennas)
            )
            # Normalize to satisfy total power <= p_max.
            power = np.sum(np.abs(w) ** 2)
            if power > 0:
                w = w * np.sqrt(self.p_max / power)
            codebook.append(w)
        return np.array(codebook)

    def _build_pos_actions(self) -> np.ndarray:
        # Simple 2D motion options for the whole antenna array.
        step = self.delta_u_step
        if self.num_pos_actions == 1:
            return np.array([[0.0, 0.0]], dtype=np.float32)

        moves = [[0.0, 0.0]]
        for i in range(self.num_pos_actions - 1):
            angle = 2.0 * np.pi * i / max(self.num_pos_actions - 1, 1)
            moves.append([step * np.cos(angle), step * np.sin(angle)])
        return np.array(moves, dtype=np.float32)

    def _decode_action(self, action_idx: int) -> tuple[int, int]:
        beam_idx = action_idx // self.num_pos_actions
        pos_idx = action_idx % self.num_pos_actions
        return beam_idx, pos_idx

    def _decode_continuous_action(self, action: Any) -> tuple[np.ndarray, np.ndarray]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] != self._action_dim:
            raise ValueError(
                f"Expected action dim {self._action_dim}, got {action_arr.shape[0]}"
            )
        action_arr = np.clip(action_arr, -self.action_bound, self.action_bound)
        beam = action_arr[: self._beam_action_dim]
        pos = action_arr[self._beam_action_dim :]

        km = self.k_users * self.m_antennas
        w_real = beam[:km].reshape(self.k_users, self.m_antennas)
        w_imag = beam[km:].reshape(self.k_users, self.m_antennas)
        w = (w_real + 1j * w_imag).astype(np.complex64)
        if self.normalize_w:
            power = float(np.sum(np.abs(w) ** 2))
            if power > self.p_max and power > 1e-12:
                w = w * np.sqrt(self.p_max / power)

        if self._pos_action_dim == 0:
            zeros = np.zeros((self.m_antennas, 2), dtype=np.float32)
            return w, zeros

        pos_arr = pos.reshape(self.m_antennas, 2)
        if self.pos_action_mode == "delta":
            delta = pos_arr * self.delta_u_step
            return w, delta.astype(np.float32)

        # Absolute antenna offsets within allowable bounds (Eq. (13d)-(13e)).
        xmin, xmax, ymin, ymax = self.area_bounds
        scale = 1.0 / (2.0 * self.action_bound)
        offsets_x = xmin + (pos_arr[:, 0] + self.action_bound) * (xmax - xmin) * scale
        offsets_y = ymin + (pos_arr[:, 1] + self.action_bound) * (ymax - ymin) * scale
        offsets = np.stack([offsets_x, offsets_y], axis=-1).astype(np.float32)
        return w, offsets

    def _project_power_if_needed(self, w: np.ndarray) -> np.ndarray:
        power = float(np.sum(np.abs(w) ** 2))
        if power > self.p_max and power > 1e-12:
            w = w * np.sqrt(self.p_max / power)
        return w

    def _project_positions_to_bounds(self, i: int) -> None:
        # Strictly scale the whole antenna layout around BS i into XY bounds
        # using a single global factor, preserving relative geometry.
        bs_pos = self.bs_positions[i]
        offsets = self.antenna_positions[i, :, :2] - bs_pos[None, :]
        xmin, xmax, ymin, ymax = self.area_bounds

        scale = 1.0
        flat = offsets.reshape(-1)
        lows = np.tile(np.array([xmin, ymin], dtype=np.float32), self.m_antennas)
        highs = np.tile(np.array([xmax, ymax], dtype=np.float32), self.m_antennas)
        for v, low, high in zip(flat, lows, highs):
            if v > 0:
                if high < 0:
                    scale = 0.0
                    break
                scale = min(scale, float(high / v))
            elif v < 0:
                if low > 0:
                    scale = 0.0
                    break
                scale = min(scale, float(low / v))

        scale = float(np.clip(scale, 0.0, 1.0))
        offsets = offsets * scale
        self.antenna_positions[i, :, :2] = bs_pos[None, :] + offsets
        self.antenna_positions[i, :, 2] = self.bs_height

    def _sample_user_positions(self) -> np.ndarray:
        d_min, d_max = self.user_dist_bounds
        # Users are sampled by horizontal (XY) distance in annulus [d_min, d_max]
        # around each BS. Sampling radius^2 uniformly gives area-uniform points.
        r_min_sq = d_min * d_min
        r_max_sq = d_max * d_max
        if self.shared_user_positions:
            # Shared user geometry: all agents use the same user offsets
            # relative to their serving BS center.
            r_sq = self.user_pos_rng.uniform(r_min_sq, r_max_sq, size=(self.k_users,))
            radii = np.sqrt(np.maximum(r_sq, 0.0))
            if self.user_fixed_angles_rad is None:
                if self.user_angle_ranges_rad is not None:
                    angle_min = self.user_angle_ranges_rad[:, 0]
                    angle_max = self.user_angle_ranges_rad[:, 1]
                elif self.user_angle_range_rad is None:
                    angle_min, angle_max = 0.0, 2.0 * np.pi
                else:
                    angle_min, angle_max = self.user_angle_range_rad
                angles = self.user_pos_rng.uniform(
                    angle_min, angle_max, size=(self.k_users,)
                )
            else:
                angles = self.user_fixed_angles_rad
            xs = radii * np.cos(angles)
            ys = radii * np.sin(angles)
            base_offsets_xy = np.stack([xs, ys], axis=-1).astype(np.float32)
            offsets_xy = np.tile(base_offsets_xy[None, :, :], (self.n_agents, 1, 1))
        else:
            r_sq = self.user_pos_rng.uniform(
                r_min_sq, r_max_sq, size=(self.n_agents, self.k_users)
            )
            radii = np.sqrt(np.maximum(r_sq, 0.0))
            if self.user_fixed_angles_rad is None:
                if self.user_angle_ranges_rad is not None:
                    angle_min = self.user_angle_ranges_rad[:, 0][None, :]
                    angle_max = self.user_angle_ranges_rad[:, 1][None, :]
                elif self.user_angle_range_rad is None:
                    angle_min, angle_max = 0.0, 2.0 * np.pi
                else:
                    angle_min, angle_max = self.user_angle_range_rad
                angles = self.user_pos_rng.uniform(
                    angle_min, angle_max, size=(self.n_agents, self.k_users)
                )
            else:
                angles = np.tile(
                    self.user_fixed_angles_rad[None, :], (self.n_agents, 1)
                )
            xs = radii * np.cos(angles)
            ys = radii * np.sin(angles)
            offsets_xy = np.stack([xs, ys], axis=-1).astype(np.float32)
        xy = offsets_xy + self.bs_positions[:, None, :]
        z = np.full((self.n_agents, self.k_users, 1), self.user_height, dtype=np.float32)
        return np.concatenate([xy, z], axis=-1)

    def _sample_antenna_positions(self) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.area_bounds
        xs = self.rng.uniform(xmin, xmax, size=(self.n_agents, self.m_antennas))
        ys = self.rng.uniform(ymin, ymax, size=(self.n_agents, self.m_antennas))
        offsets_xy = np.stack([xs, ys], axis=-1).astype(np.float32)
        xy = offsets_xy + self.bs_positions[:, None, :]
        z = np.full((self.n_agents, self.m_antennas, 1), self.bs_height, dtype=np.float32)
        return np.concatenate([xy, z], axis=-1)

    def _build_fixed_antenna_positions(self) -> np.ndarray:
        if self.fixed_antenna_offsets_xy is None:
            raise ValueError("fixed_antenna_offsets_xy is not configured")
        xy = self.fixed_antenna_offsets_xy + self.bs_positions[:, None, :]
        z = np.full((self.n_agents, self.m_antennas, 1), self.bs_height, dtype=np.float32)
        return np.concatenate([xy, z], axis=-1)

    def _resample_or_reject_position_action(
        self, i: int, prev_positions: np.ndarray, max_tries: int
    ) -> None:
        """Reject invalid spacing actions; otherwise resample a feasible antenna layout."""
        if self._spacing_violation(i) <= 1e-9:
            return

        xmin, xmax, ymin, ymax = self.area_bounds
        bs_pos = self.bs_positions[i]
        best_positions = prev_positions[i].copy()
        best_violation = float(self._spacing_violation(i))
        found_feasible = False

        for _ in range(max_tries):
            xs = self.rng.uniform(xmin, xmax, size=(self.m_antennas,))
            ys = self.rng.uniform(ymin, ymax, size=(self.m_antennas,))
            cand_xy = np.stack([xs, ys], axis=-1).astype(np.float32) + bs_pos[None, :]
            cand_z = np.full((self.m_antennas, 1), self.bs_height, dtype=np.float32)
            self.antenna_positions[i] = np.concatenate([cand_xy, cand_z], axis=-1)
            self._project_positions_to_bounds(i)
            violation = float(self._spacing_violation(i))
            if violation < best_violation:
                best_violation = violation
                best_positions = self.antenna_positions[i].copy()
            if violation <= 1e-9:
                found_feasible = True
                break

        if not found_feasible:
            # Reject: fallback to best sampled candidate; if none improved, keep previous.
            self.antenna_positions[i] = best_positions
            self._project_positions_to_bounds(i)

    def _angles_to_dir_cos(self, alpha: np.ndarray, phi: np.ndarray) -> np.ndarray:
        u = np.cos(alpha) * np.cos(phi)
        v = np.cos(alpha) * np.sin(phi)
        w = np.sin(alpha)
        return np.stack([u, v, w], axis=-1).astype(np.float32)

    def _user_to_bs_distance(self, user_pos: np.ndarray, bs_index: int) -> float:
        """3D distance d between user position and BS center position."""
        bs_pos = self.bs_positions[bs_index]
        bs_pos_3d = np.array([bs_pos[0], bs_pos[1], self.bs_height], dtype=np.float32)
        return float(np.linalg.norm(user_pos - bs_pos_3d) + 1e-6)

    def _sample_channel_estimates(self, resample_errors: bool = False) -> None:
        if self.fixed_channel_on_reset and self._fixed_channel_params is not None:
            fixed = self._fixed_channel_params
            self.alpha_r_hat = fixed["alpha_r_hat"].copy()
            self.phi_r_hat = fixed["phi_r_hat"].copy()
            self.alpha_t_hat = fixed["alpha_t_hat"].copy()
            self.phi_t_hat = fixed["phi_t_hat"].copy()
            self._dir_cos_r = fixed["dir_cos_r"].copy()
            self._dir_cos_t = fixed["dir_cos_t"].copy()
            self._zeta_true = fixed["zeta_true"].copy()
            self.zeta_hat = fixed["zeta_hat"].copy()
        else:
            shape = (self.n_agents, self.n_agents, self.k_users, self.l_paths)
            shape_t = (
                self.n_agents,
                self.n_agents,
                self.k_users,
                self.l_paths,
                self.m_antennas,
            )
            if self.fixed_aoa_aod_on_reset and self._fixed_aoa_aod_params is not None:
                fixed_angles = self._fixed_aoa_aod_params
                alpha_r_true = fixed_angles["alpha_r_true"].copy()
                phi_r_true = fixed_angles["phi_r_true"].copy()
                alpha_t_true = fixed_angles["alpha_t_true"].copy()
                phi_t_true = fixed_angles["phi_t_true"].copy()
            elif self.shared_fixed_aoa_aod:
                # Sample AoA/AoD once (seed-controlled) and share across all users/cells.
                if self._fixed_shared_aoa_aod_params is None:
                    alpha_r_base = self.channel_rng.uniform(
                        0.0, np.pi / 2.0, size=(self.l_paths,)
                    ).astype(np.float32)
                    phi_r_base = self.channel_rng.uniform(
                        0.0, 2.0 * np.pi, size=(self.l_paths,)
                    ).astype(np.float32)
                    alpha_t_base = self.channel_rng.uniform(
                        0.0, np.pi / 2.0, size=(self.l_paths, self.m_antennas)
                    ).astype(np.float32)
                    phi_t_base = self.channel_rng.uniform(
                        0.0, 2.0 * np.pi, size=(self.l_paths, self.m_antennas)
                    ).astype(np.float32)
                    self._fixed_shared_aoa_aod_params = {
                        "alpha_r_base": alpha_r_base,
                        "phi_r_base": phi_r_base,
                        "alpha_t_base": alpha_t_base,
                        "phi_t_base": phi_t_base,
                    }
                fixed_angles = self._fixed_shared_aoa_aod_params
                alpha_r_true = np.tile(
                    fixed_angles["alpha_r_base"][None, None, None, :],
                    (self.n_agents, self.n_agents, self.k_users, 1),
                )
                phi_r_true = np.tile(
                    fixed_angles["phi_r_base"][None, None, None, :],
                    (self.n_agents, self.n_agents, self.k_users, 1),
                )
                alpha_t_true = np.tile(
                    fixed_angles["alpha_t_base"][None, None, None, :, :],
                    (self.n_agents, self.n_agents, self.k_users, 1, 1),
                )
                phi_t_true = np.tile(
                    fixed_angles["phi_t_base"][None, None, None, :, :],
                    (self.n_agents, self.n_agents, self.k_users, 1, 1),
                )
            else:
                alpha_r_true = self.channel_rng.uniform(0.0, np.pi / 2.0, size=shape)
                phi_r_true = self.channel_rng.uniform(0.0, 2.0 * np.pi, size=shape)
                alpha_t_true = self.channel_rng.uniform(0.0, np.pi / 2.0, size=shape_t)
                phi_t_true = self.channel_rng.uniform(0.0, 2.0 * np.pi, size=shape_t)

            if self.fixed_aoa_aod_on_reset and self._fixed_aoa_aod_params is None:
                self._fixed_aoa_aod_params = {
                    "alpha_r_true": alpha_r_true.astype(np.float32).copy(),
                    "phi_r_true": phi_r_true.astype(np.float32).copy(),
                    "alpha_t_true": alpha_t_true.astype(np.float32).copy(),
                    "phi_t_true": phi_t_true.astype(np.float32).copy(),
                }
            # AoA/AoD and multipath components are treated as perfectly known.
            self.alpha_r_hat = alpha_r_true.astype(np.float32)
            self.phi_r_hat = phi_r_true.astype(np.float32)
            self.alpha_t_hat = alpha_t_true.astype(np.float32)
            self.phi_t_hat = phi_t_true.astype(np.float32)
            self._dir_cos_r = self._angles_to_dir_cos(alpha_r_true, phi_r_true)
            self._dir_cos_t = self._angles_to_dir_cos(alpha_t_true, phi_t_true)

            reuse_cached_zeta = (
                self.zeta_resample_interval > 0
                and self._cached_zeta_true is not None
                and (self._episode_count % self.zeta_resample_interval != 0)
            )
            if reuse_cached_zeta:
                self._zeta_true = self._cached_zeta_true.copy()
                self.zeta_hat = self._cached_zeta_hat.copy()
            else:
                self.zeta_hat = np.zeros(shape, dtype=np.complex64)
                self._zeta_true = np.zeros(shape, dtype=np.complex64)
                for i in range(self.n_agents):
                    for j in range(self.n_agents):
                        for k in range(self.k_users):
                            user_pos = self.user_positions[i, k]
                            dist = self._user_to_bs_distance(user_pos, j)
                            if i == j:
                                alpha_ij = self.path_loss_exp_intra
                            else:
                                alpha_ij = self.path_loss_exp_inter
                            path_scale = dist ** (-alpha_ij)
                            zeta_mean = np.full(
                                (self.l_paths,), path_scale, dtype=np.float32
                            ).astype(np.complex64)
                            if self.zeta_var > 0.0:
                                sigma = np.sqrt(self.zeta_var / 2.0)
                                zeta_noise = (
                                    self.channel_rng.normal(size=(self.l_paths,))
                                    + 1j * self.channel_rng.normal(size=(self.l_paths,))
                                ) * sigma
                                zeta_true = (zeta_mean + zeta_noise).astype(np.complex64)
                            else:
                                zeta_true = zeta_mean
                            self._zeta_true[i, j, k] = zeta_true
                            self.zeta_hat[i, j, k] = zeta_true
                if self.zeta_resample_interval > 0:
                    self._cached_zeta_true = self._zeta_true.copy()
                    self._cached_zeta_hat = self.zeta_hat.copy()

            if self.fixed_channel_on_reset:
                self._fixed_channel_params = {
                    "alpha_r_hat": self.alpha_r_hat.copy(),
                    "phi_r_hat": self.phi_r_hat.copy(),
                    "alpha_t_hat": self.alpha_t_hat.copy(),
                    "phi_t_hat": self.phi_t_hat.copy(),
                    "dir_cos_r": self._dir_cos_r.copy(),
                    "dir_cos_t": self._dir_cos_t.copy(),
                    "zeta_true": self._zeta_true.copy(),
                    "zeta_hat": self.zeta_hat.copy(),
                }

    def _update_channels(self):
        # Field-response channel vectors with multipath and antenna-position dependence.
        if self._zeta_true is None or self._dir_cos_r is None or self._dir_cos_t is None:
            self._sample_channel_estimates()
        h = np.zeros(
            (self.n_agents, self.n_agents, self.k_users, self.m_antennas),
            dtype=np.complex64,
        )
        two_pi_over_lambda = 2.0 * np.pi / self.lambda_c
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                bs_pos = self.bs_positions[j]
                bs_ant_pos = self.antenna_positions[j]
                for k in range(self.k_users):
                    user_pos = self.user_positions[i, k]
                    beta = self._zeta_true[i, j, k]

                    dir_r = self._dir_cos_r[i, j, k]
                    psi_r = (
                        user_pos[0] * dir_r[:, 0]
                        + user_pos[1] * dir_r[:, 1]
                        + user_pos[2] * dir_r[:, 2]
                    )
                    f_vec = np.exp(1j * two_pi_over_lambda * psi_r)

                    dir_t = self._dir_cos_t[i, j, k]
                    psi_t = (
                        dir_t[:, :, 0] * bs_ant_pos[None, :, 0]
                        + dir_t[:, :, 1] * bs_ant_pos[None, :, 1]
                        + dir_t[:, :, 2] * bs_ant_pos[None, :, 2]
                    )
                    g_mat = np.exp(1j * two_pi_over_lambda * psi_t)
                    # Multipath power normalization: keep channel scale comparable
                    # when changing number of paths L.
                    h[i, j, k] = ((np.conjugate(f_vec) * beta) @ g_mat) / np.sqrt(
                        max(self.l_paths, 1)
                    )
        self.h = h

    def _update_estimated_channels(self):
        # No CSI error: estimated channel equals true channel.
        if self.h is None:
            self._update_channels()
        self.h_hat = self.h.astype(np.complex64)

    def _spacing_violation(self, i: int) -> float:
        # Penalize any pairwise antenna spacing violations.
        pos = self.antenna_positions[i]
        if self.m_antennas < 2:
            return 0.0
        violations = 0.0
        norm = max(self.d_min, 1e-9)
        for a in range(self.m_antennas - 1):
            for b in range(a + 1, self.m_antennas):
                dist = np.linalg.norm(pos[a] - pos[b])
                if dist < self.d_min:
                    violations += (self.d_min - dist) / norm
        return float(violations)

    def _compute_reward(self, w_actions: np.ndarray) -> float:
        # Compute SINR with intra-cell and inter-cell interference powers.
        self.gamma = np.zeros((self.n_agents, self.k_users), dtype=np.float32)
        self.p_tx = np.zeros((self.n_agents,), dtype=np.float32)
        self.i_intra = np.zeros((self.n_agents, self.k_users), dtype=np.float32)

        for i in range(self.n_agents):
            # Enforce power constraint by scaling, then measure power.
            w_actions[i] = self._project_power_if_needed(w_actions[i])
            w = w_actions[i]
            power = float(np.sum(np.abs(w) ** 2))
            self.p_tx[i] = power

        self.i_inter = np.zeros((self.n_agents, self.k_users), dtype=np.float32)
        step_sum_rate = 0.0
        step_violation = 0.0
        step_power_violation = 0.0
        step_spacing_violation = 0.0
        step_p_tx_mean = float(np.mean(self.p_tx))
        step_signal_sum = 0.0
        violation_flag = 0.0
        beta_used = float(self.beta_penalty)
        for i in range(self.n_agents):
            # Enforce antenna position box constraint by scaling into bounds.
            self._project_positions_to_bounds(i)
            w = w_actions[i]
            for k in range(self.k_users):
                # Use the true channel for SINR computation.
                h = self.h[i, i, k]
                signal = np.abs(np.dot(h, w[k])) ** 2
                step_signal_sum += float(signal)
                intra = 0.0
                for k2 in range(self.k_users):
                    if k2 == k:
                        continue
                    intra += np.abs(np.dot(h, w[k2])) ** 2
                self.i_intra[i, k] = float(intra)
                inter = 0.0
                for j in range(self.n_agents):
                    if j == i:
                        continue
                    for k2 in range(self.k_users):
                        h_cross = self.h[i, j, k]
                        inter += self.interference_scale * (
                            np.abs(np.dot(h_cross, w_actions[j, k2])) ** 2
                        )
                self.i_inter[i, k] = float(inter)
                denom = intra + inter + self.noise_var
                self.gamma[i, k] = float(signal / max(denom, 1e-9))
                self.gamma[i, k] = float(
                    np.nan_to_num(self.gamma[i, k], nan=0.0, posinf=0.0, neginf=0.0)
                )

            sum_rate = float(
                np.sum(self.user_weights[i] * np.log2(1.0 + self.gamma[i]))
            )
            sum_rate = float(np.nan_to_num(sum_rate, nan=0.0, posinf=0.0, neginf=0.0))
            sum_rate *= self.sum_rate_scale
            # Power is projected to feasible set, so no penalty contribution.
            power_violation = 0.0
            spacing_violation = self._spacing_violation(i)
            violation = spacing_violation
            if spacing_violation > 0.0:
                violation_flag = 1.0

            step_sum_rate += sum_rate
            step_violation += violation
            step_power_violation += power_violation
            step_spacing_violation += spacing_violation

        self._last_sum_rate = float(step_sum_rate)
        if self.reward_mode == "paper":
            reward = step_sum_rate - beta_used * violation_flag
            self._last_violation = float(violation_flag)
        else:
            reward = step_sum_rate - beta_used * step_violation
            self._last_violation = float(step_violation)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        self._last_power_violation = float(step_power_violation)
        self._last_spacing_violation = float(step_spacing_violation)
        self._last_p_tx_mean = float(step_p_tx_mean)
        denom_users = float(max(self.n_agents * self.k_users, 1))
        self._last_signal_mean = float(step_signal_sum / denom_users)
        self._last_sinr_mean = float(np.mean(self.gamma))
        self._last_intra_mean = float(np.mean(self.i_intra))
        self._last_inter_mean = float(np.mean(self.i_inter))
        self._last_beta_penalty = beta_used
        self._last_beta_penalty_next = beta_used
        if self.reward_mode == "smooth" and self.beta_penalty_mode == "lagrange":
            updated = beta_used + self.beta_lagrange_lr * (
                step_violation - self.beta_lagrange_target
            )
            self.beta_penalty = float(
                np.clip(updated, self.beta_lagrange_min, self.beta_lagrange_max)
            )
            self._last_beta_penalty_next = float(self.beta_penalty)

        return float(reward)

    def _build_obs(self, i: int) -> np.ndarray:
        if self.obs_mode == "paper":
            w = self.w_actions[i].reshape(-1)
            w_features = np.concatenate([w.real, w.imag], axis=0).astype(np.float32)
            if self.h is None:
                h_self = np.zeros(
                    (2 * self.k_users * self.m_antennas,), dtype=np.float32
                )
            else:
                h_self_vec = self.h[i, i].reshape(-1)
                h_self = np.concatenate(
                    [h_self_vec.real, h_self_vec.imag], axis=0
                ).astype(np.float32)
            u = self.antenna_positions[i].reshape(-1).astype(np.float32)
            v = self.user_positions[i].reshape(-1).astype(np.float32)
            if self.i_inter is None:
                i2 = np.zeros((self.k_users,), dtype=np.float32)
            else:
                i2 = self.i_inter[i].reshape(-1).astype(np.float32)
            if self.gamma is None:
                gamma_i = np.zeros((self.k_users,), dtype=np.float32)
            else:
                gamma_i = self.gamma[i].reshape(-1).astype(np.float32)
            weighted_rates = (
                self.user_weights[i].reshape(-1) * np.log2(1.0 + gamma_i)
            )
            weighted_rates = np.nan_to_num(
                weighted_rates, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            obs = np.concatenate(
                [u, w_features, v, h_self, i2, weighted_rates], axis=0
            )
            return obs.astype(np.float32)

        u = self.antenna_positions[i].reshape(-1)
        h_vec = self.h[i, i].reshape(-1)
        h_features = np.concatenate([h_vec.real, h_vec.imag], axis=0).astype(np.float32)
        gamma = self.gamma[i].reshape(-1)
        p_tx = np.array([self.p_tx[i]], dtype=np.float32)
        i_inter = np.array([float(np.mean(self.i_inter[i]))], dtype=np.float32)
        obs = np.concatenate([u, h_features, gamma, p_tx, i_inter], axis=0)
        return obs.astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if self.channel_seed is None:
                self.channel_rng = np.random.default_rng(seed)
            if self.user_position_seed is None:
                self.user_pos_rng = np.random.default_rng(seed)
        self._step_count = 0
        self._ep_sum_rate = 0.0
        self._ep_violation = 0.0
        self._ep_power_violation = 0.0
        self._ep_spacing_violation = 0.0
        self._ep_p_tx = 0.0
        self._ep_signal = 0.0
        self._ep_sinr = 0.0
        self._ep_intra = 0.0
        self._ep_inter = 0.0
        # User-position refresh policy:
        # interval == 0 => legacy behavior controlled by fixed_positions_on_reset.
        # interval > 0  => resample user positions every `interval` episodes.
        if self.user_position_resample_interval == 0:
            if self.fixed_positions_on_reset:
                if self._fixed_user_positions is None:
                    self._fixed_user_positions = self._sample_user_positions()
                self.user_positions = self._fixed_user_positions.copy()
            else:
                self.user_positions = self._sample_user_positions()
        else:
            if (self._fixed_user_positions is None) or (
                self._episode_count % self.user_position_resample_interval == 0
            ):
                self._fixed_user_positions = self._sample_user_positions()
            self.user_positions = self._fixed_user_positions.copy()

        # Explicit fixed antenna offsets override random position sampling.
        if self.fixed_antenna_offsets_xy is not None:
            self._fixed_antenna_positions = self._build_fixed_antenna_positions()
            self.antenna_positions = self._fixed_antenna_positions.copy()
        elif self.fixed_positions_on_reset:
            if self._fixed_antenna_positions is None:
                self._fixed_antenna_positions = self._sample_antenna_positions()
            self.antenna_positions = self._fixed_antenna_positions.copy()
        else:
            self.antenna_positions = self._sample_antenna_positions()
        self._sample_channel_estimates(resample_errors=True)
        self._update_channels()
        self._update_estimated_channels()
        if self.action_mode == "discrete":
            # Initialize with first beam action for stable shapes.
            w_actions = np.repeat(
                self._beam_codebook[0][None, ...], self.n_agents, axis=0
            )
        else:
            w_actions = self.rng.normal(
                size=(self.n_agents, self.k_users, self.m_antennas)
            ) + 1j * self.rng.normal(size=(self.n_agents, self.k_users, self.m_antennas))
            for i in range(self.n_agents):
                power = float(np.sum(np.abs(w_actions[i]) ** 2))
                if power > 0.0:
                    w_actions[i] = w_actions[i] * np.sqrt(self.p_max / power)
            w_actions = w_actions.astype(np.complex64)
        self.w_actions = w_actions
        self.gamma = np.zeros((self.n_agents, self.k_users), dtype=np.float32)
        self.p_tx = np.zeros((self.n_agents,), dtype=np.float32)
        self.i_intra = np.zeros((self.n_agents, self.k_users), dtype=np.float32)
        self.i_inter = np.zeros((self.n_agents, self.k_users), dtype=np.float32)
        _ = self._compute_reward(w_actions)

        obs = tuple(self._build_obs(i) for i in range(self.n_agents))
        info: dict[str, Any] = {}
        self._episode_count += 1
        return obs, info

    def step(self, actions):
        self._step_count += 1
        prev_positions = self.antenna_positions.copy()
        if self.action_mode == "discrete":
            actions = [int(a) for a in actions]

            # Apply position actions (whole-array translation).
            for i, action_idx in enumerate(actions):
                _, pos_idx = self._decode_action(action_idx)
                delta = self._pos_actions[pos_idx]
                self.antenna_positions[i, :, :2] += delta[None, :]
                self._project_positions_to_bounds(i)
            if self.enforce_spacing_by_resample:
                for i in range(self.n_agents):
                    self._resample_or_reject_position_action(
                        i, prev_positions, self.spacing_resample_max_tries
                    )

            # Update channels based on new antenna positions.
            if self.resample_fading_each_step:
                self._sample_channel_estimates(resample_errors=False)
            self._update_channels()
            self._update_estimated_channels()

            # Decode beamforming actions.
            w_actions = np.zeros(
                (self.n_agents, self.k_users, self.m_antennas), dtype=np.complex64
            )
            for i, action_idx in enumerate(actions):
                beam_idx, _ = self._decode_action(action_idx)
                beam_idx = int(np.clip(beam_idx, 0, self.num_beam_actions - 1))
                w_actions[i] = self._project_power_if_needed(self._beam_codebook[beam_idx])
        else:
            actions = list(actions)
            if len(actions) != self.n_agents:
                raise ValueError(
                    f"Expected {self.n_agents} actions, got {len(actions)}"
                )

            w_actions = np.zeros(
                (self.n_agents, self.k_users, self.m_antennas), dtype=np.complex64
            )
            pos_updates = np.zeros((self.n_agents, self.m_antennas, 2), dtype=np.float32)

            for i, action in enumerate(actions):
                action = np.nan_to_num(
                    np.asarray(action, dtype=np.float32),
                    nan=0.0,
                    posinf=self.action_bound,
                    neginf=-self.action_bound,
                )
                if not np.isfinite(action).all():
                    raise FloatingPointError(
                        f"Non-finite action detected for agent {i}."
                    )
                w, pos_update = self._decode_continuous_action(action)
                w_actions[i] = w
                pos_updates[i] = pos_update

            if self._pos_action_dim > 0:
                if self.pos_action_mode == "delta":
                    self.antenna_positions[:, :, :2] = (
                        self.antenna_positions[:, :, :2] + pos_updates
                    )
                    for i in range(self.n_agents):
                        self._project_positions_to_bounds(i)
                else:
                    # Absolute offsets within bounds around each BS.
                    for i in range(self.n_agents):
                        bs_pos = self.bs_positions[i]
                        self.antenna_positions[i, :, :2] = (
                            bs_pos[None, :] + pos_updates[i]
                        )
                        self._project_positions_to_bounds(i)
            if self.enforce_spacing_by_resample:
                for i in range(self.n_agents):
                    self._resample_or_reject_position_action(
                        i, prev_positions, self.spacing_resample_max_tries
                    )

            if self.resample_fading_each_step:
                self._sample_channel_estimates(resample_errors=False)
            self._update_channels()
            self._update_estimated_channels()

        self.w_actions = w_actions
        for i in range(self.n_agents):
            self.w_actions[i] = self._project_power_if_needed(self.w_actions[i])
        reward = self._compute_reward(w_actions)
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        if not np.isfinite(reward):
            raise FloatingPointError("Non-finite reward detected in FASEnv.")
        self._ep_sum_rate += self._last_sum_rate
        self._ep_violation += self._last_violation
        self._ep_power_violation += self._last_power_violation
        self._ep_spacing_violation += self._last_spacing_violation
        self._ep_p_tx += self._last_p_tx_mean
        self._ep_signal += self._last_signal_mean
        self._ep_sinr += self._last_sinr_mean
        self._ep_intra += self._last_intra_mean
        self._ep_inter += self._last_inter_mean
        obs = tuple(self._build_obs(i) for i in range(self.n_agents))
        terminated = False
        truncated = False
        inv_steps = 1.0 / max(self._step_count, 1)
        info = {
            "sum_rate": self._ep_sum_rate * inv_steps,
            "violation": self._ep_violation * inv_steps,
            "power_violation": self._ep_power_violation * inv_steps,
            "spacing_violation": self._ep_spacing_violation * inv_steps,
            "p_tx": self._ep_p_tx * inv_steps,
            "signal": self._ep_signal * inv_steps,
            "sinr": self._ep_sinr * inv_steps,
            "intra": self._ep_intra * inv_steps,
            "inter": self._ep_inter * inv_steps,
            "beta_penalty": self._last_beta_penalty,
            "beta_penalty_next": self._last_beta_penalty_next,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None

    def get_state(self) -> np.ndarray:
        if (
            self.antenna_positions is None
            or self.user_positions is None
            or self.w_actions is None
            or self.gamma is None
        ):
            return np.zeros((self.state_size,), dtype=np.float32)

        parts = []
        for i in range(self.n_agents):
            w = self.w_actions[i].reshape(-1)
            w_features = np.concatenate([w.real, w.imag], axis=0).astype(np.float32)
            if self.h is None:
                h_self = np.zeros(
                    (2 * self.k_users * self.m_antennas,), dtype=np.float32
                )
                h_inter = np.zeros(
                    (2 * (self.n_agents - 1) * self.k_users * self.m_antennas,),
                    dtype=np.float32,
                )
            else:
                h_self_vec = self.h[i, i].reshape(-1)
                h_self = np.concatenate(
                    [h_self_vec.real, h_self_vec.imag], axis=0
                ).astype(np.float32)
                inter_blocks = []
                for j in range(self.n_agents):
                    if j == i:
                        continue
                    inter_blocks.append(self.h[i, j].reshape(-1))
                if inter_blocks:
                    h_inter_vec = np.concatenate(inter_blocks, axis=0)
                    h_inter = np.concatenate(
                        [h_inter_vec.real, h_inter_vec.imag], axis=0
                    ).astype(np.float32)
                else:
                    h_inter = np.zeros((0,), dtype=np.float32)
            weighted_rates = (
                self.user_weights[i].reshape(-1) * np.log2(1.0 + self.gamma[i].reshape(-1))
            )
            weighted_rates = np.nan_to_num(
                weighted_rates, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
            u = self.antenna_positions[i].reshape(-1).astype(np.float32)
            v = self.user_positions[i].reshape(-1).astype(np.float32)
            parts.append(
                np.concatenate(
                    [u, w_features, v, h_self, h_inter, weighted_rates],
                    axis=0,
                )
            )

        return np.concatenate(parts, axis=0).astype(np.float32)

    def get_state_size(self) -> int:
        return int(self.state_size)
