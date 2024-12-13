
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.registration import register

register(
    id='HalfCheetahVelSparse-v3',
    entry_point='src.env.mujoco_custom.halfcheetah_vel_sparse:HalfCheetahVelSparseEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

class HalfCheetahVelSparseEnv(HalfCheetahEnv):
    def __init__(self, **kwargs):
        super(HalfCheetahVelSparseEnv, self).__init__(**kwargs)
        self.target_velocity = 1.0
        self.velocity_tolerance = 0.1

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        # forward_reward = self._forward_reward_weight * x_velocity
        if abs(x_velocity - 1.0) < 0.1:
            forward_reward = 1.0
        else:
            forward_reward = 0.0

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return observation, reward, done, info