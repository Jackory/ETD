import numpy as np
import torch as th

from gym import spaces
from gym.spaces import Dict
from typing import Generator, Optional, Union

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize

from src.algo.buffers.type_aliases import RolloutBufferSamples, ReplayBufferSamples
from src.utils.common_func import normalize_rewards
from src.utils.running_mean_std import RunningMeanStd


class PPORolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        discount: float = 0.99,
        n_envs: int = 1,
        features_dim: int = 0,
        dim_policy_traj: int = 0,
        dim_model_traj: int = 0,
        int_rew_coef: float = 1.0,
        ext_rew_coef: float = 1.0,
        int_rew_norm: int = 0,
        int_rew_clip: float = 0.0,
        int_rew_eps: float = 1e-8,
        adv_momentum: float = 0.0,
        adv_norm: int = 0,
        adv_eps: float = 1e-8,
        gru_layers: int = 1,
        int_rew_momentum: Optional[float] = None,
        use_status_predictor: int = 0,
        offpolicy_buffer_size: int = int(1e6),
    ):
        if isinstance(observation_space, Dict):
            observation_space = list(observation_space.values())[0]
        super(PPORolloutBuffer, self)\
            .__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.discount = discount
        self.int_rew_coef = int_rew_coef
        self.int_rew_norm = int_rew_norm
        self.int_rew_clip = int_rew_clip
        self.ext_rew_coef = ext_rew_coef
        self.features_dim = features_dim
        self.dim_policy_traj = dim_policy_traj
        self.dim_model_traj = dim_model_traj
        self.int_rew_eps = int_rew_eps
        self.adv_momentum = adv_momentum
        self.adv_mean = None
        self.int_rew_mean = None
        self.int_rew_std = None
        self.ir_mean_buffer = []
        self.ir_std_buffer = []
        self.use_status_predictor = use_status_predictor
        self.adv_norm = adv_norm
        self.adv_eps = adv_eps
        self.gru_layers = gru_layers
        self.int_rew_momentum = int_rew_momentum
        self.int_rew_stats = RunningMeanStd(momentum=self.int_rew_momentum)
        self.advantage_stats = RunningMeanStd(momentum=self.adv_momentum)

        self.generator_ready = False
        self.reset()
        self.offpolicy_buffer_size = offpolicy_buffer_size
        self.aux_observations = np.zeros((self.offpolicy_buffer_size,) + self.obs_shape, dtype=observation_space.dtype)
        self.aux_future_observations = np.zeros((offpolicy_buffer_size,) + self.obs_shape, dtype=observation_space.dtype)
        self.aux_pos = 0
        self.aux_full = False
        self.offpolicy = True

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.new_observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.last_policy_mems = np.zeros((self.buffer_size, self.n_envs, self.gru_layers, self.dim_policy_traj), dtype=np.float32)
        self.last_model_mems = np.zeros((self.buffer_size, self.n_envs, self.gru_layers, self.dim_model_traj), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.intrinsic_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        if self.use_status_predictor:
            self.curr_key_status = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
            self.curr_door_status = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
            self.curr_target_dists = np.zeros((self.buffer_size, self.n_envs, 3), dtype=np.float32)
        self.generator_ready = False
        super(PPORolloutBuffer, self).reset()

    def compute_intrinsic_rewards(self) -> None:
        # Normalize intrinsic rewards per rollout buffer
        self.int_rew_stats.update(self.intrinsic_rewards.reshape(-1))
        self.int_rew_mean = self.int_rew_stats.mean
        self.int_rew_std = self.int_rew_stats.std
        self.intrinsic_rewards = normalize_rewards(
            norm_type=self.int_rew_norm,
            rewards=self.intrinsic_rewards,
            mean=self.int_rew_mean,
            std=self.int_rew_std,
            eps=self.int_rew_eps,
        )

        # Rescale by IR coef
        self.intrinsic_rewards *= self.int_rew_coef

        # Clip after normalization
        if self.int_rew_clip > 0:
            self.intrinsic_rewards = np.clip(self.intrinsic_rewards, -self.int_rew_clip, self.int_rew_clip)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        # Rescale extrinisc rewards
        self.rewards *= self.ext_rew_coef

        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.intrinsic_rewards[step] + \
                    self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

        # Normalize advantages per rollout buffer
        if self.adv_norm:
            self.advantage_stats.update(self.advantages)
            self.adv_mean = self.advantage_stats.mean
            self.adv_std = self.advantage_stats.std

            # Standardization
            if self.adv_norm == 2:
                self.advantages = (self.advantages - self.adv_mean) / (self.adv_std + self.adv_eps)

            # Standardization without subtracting the mean value
            if self.adv_norm == 3:
                self.advantages = self.advantages / (self.adv_std + self.adv_eps)


    def compute_future_states(self):
        T, N = self.episode_dones.shape
        ranges = np.full((T, N), T, dtype=np.int32)  # 初始化 ranges 为 T
        last_one = np.full((N,), T, dtype=np.int32)  # 用 T 初始化上一个 1 的位置

        for t in reversed(range(T)):
            ranges[t] = last_one - t - 1
            last_one = np.where(self.episode_dones[t] == 1, t, last_one)


        seeds = np.random.rand(*ranges.shape)
        if self.discount == 0:
            intervals = np.ones_like(seeds, dtype=ranges.dtype)
        elif self.discount == 1:
            intervals = np.ceil(seeds * ranges).astype(int)
        else:
            intervals = np.log(1 - (1 - self.discount**ranges) * seeds) / np.log(self.discount)
        intervals = np.minimum(np.ceil(intervals).astype(int), ranges)
        
        self.future_observations = self.observations[intervals+np.arange(T)[:,None], np.arange(N)]
        assert self.future_observations.shape == self.observations.shape

    
    def add(
        self,
        obs: np.ndarray,
        new_obs: np.ndarray,
        last_policy_mem: th.Tensor,
        last_model_mem: th.Tensor,
        action: np.ndarray,
        reward: np.ndarray,
        intrinsic_reward: np.ndarray,
        episode_start: np.ndarray,
        episode_done: np.ndarray,
        value: th.Tensor,
        log_prob: Optional[th.Tensor],
        curr_key_status: Optional[np.ndarray],
        curr_door_status: Optional[np.ndarray],
        curr_target_dist: Optional[np.ndarray],
    ) -> None:
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.new_observations[self.pos] = np.array(new_obs).copy()
        self.last_policy_mems[self.pos] = last_policy_mem.clone().cpu().numpy()
        self.last_model_mems[self.pos] = last_model_mem.clone().cpu().numpy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.intrinsic_rewards[self.pos] = np.array(intrinsic_reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.episode_dones[self.pos] = np.array(episode_done).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        if self.use_status_predictor:
            self.curr_key_status[self.pos] = np.array(curr_key_status).copy()
            self.curr_door_status[self.pos] = np.array(curr_door_status).copy()
            self.curr_target_dists[self.pos] = np.array(curr_target_dist).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def prepare_data(self):
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "new_observations",
                "future_observations",
                "last_policy_mems",
                "last_model_mems",
                "episode_starts",
                "episode_dones",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]
            if self.use_status_predictor:
                _tensor_names += [
                    "curr_key_status",
                    "curr_door_status",
                    "curr_target_dists",
                ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            dims = len(self.observations.shape)
            # Minigrid only
            same_elements = np.all(self.future_observations == self.observations, axis=tuple(range(1,dims)))
            length = self.observations[~same_elements].shape[0]
            if self.offpolicy:
                if self.aux_pos + length > self.offpolicy_buffer_size:
                    trunc = self.offpolicy_buffer_size - self.aux_pos
                    self.aux_observations[self.aux_pos:] = self.observations[~same_elements][:trunc].copy()
                    self.aux_future_observations[self.aux_pos:] = self.future_observations[~same_elements][:trunc].copy()
                    self.aux_observations[:length-trunc] = self.observations[~same_elements][trunc:].copy()
                    self.aux_future_observations[:length-trunc] = self.future_observations[~same_elements][trunc:].copy()
                    self.aux_full = True
                    self.aux_pos = length - trunc
                else:
                    self.aux_observations[self.aux_pos:self.aux_pos + length] = self.observations[~same_elements].copy()
                    self.aux_future_observations[self.aux_pos:self.aux_pos + length] = self.future_observations[~same_elements].copy()
                    self.aux_pos = self.aux_pos + length
            else:
                self.aux_observations[:length] = self.observations[~same_elements].copy()
                self.aux_future_observations[:length] = self.future_observations[~same_elements].copy()
                self.aux_pos = length
            self.generator_ready = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        self.prepare_data()

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        indices = np.random.permutation(self.buffer_size * self.n_envs)
        offpolicy_indices = np.random.permutation(self.offpolicy_buffer_size)
        upper_bound = self.offpolicy_buffer_size if self.aux_full else self.aux_pos
        offpolicy_indices = np.random.randint(0, upper_bound, size=batch_size)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size]), self._get_offpolicy_samples(offpolicy_indices)
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.new_observations[batch_inds],
            self.future_observations[batch_inds],
            self.last_policy_mems[batch_inds],
            self.last_model_mems[batch_inds],
            self.episode_starts[batch_inds],
            self.episode_dones[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        if self.use_status_predictor:
            data += (
                self.curr_key_status[batch_inds].flatten(),
                self.curr_door_status[batch_inds].flatten(),
                self.curr_target_dists[batch_inds].flatten(),
            )

        samples = tuple(map(lambda x: self.to_torch(x, copy=False), data))
        if not self.use_status_predictor:
            samples += (None, None, None,)
        return RolloutBufferSamples(*samples)
    
    def _get_offpolicy_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        data = (
            self.aux_observations[batch_inds],
            self.aux_future_observations[batch_inds]
        )
        samples = tuple(map(lambda x: self.to_torch(x, copy=False), data))
        return ReplayBufferSamples(*samples)