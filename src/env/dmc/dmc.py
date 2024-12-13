import numpy as np
from dm_control import suite
from dm_env import specs
from numpy.typing import NDArray

from gym import spaces
from gym.core import Env


def convert_dm_control_to_gym_space(dm_control_space, dtype=None, **kwargs):
    """Convert dm_control space to gym space."""
    if isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(
            low=dm_control_space.minimum,
            high=dm_control_space.maximum,
            dtype=dtype or dm_control_space.dtype,
        )
        assert space.shape == dm_control_space.shape
        return space
    elif isinstance(dm_control_space, specs.Array) and not isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=dm_control_space.shape,
            dtype=dtype or dm_control_space.dtype,
        )
        return space
    elif isinstance(dm_control_space, dict):
        kwargs.update({key: convert_dm_control_to_gym_space(value, dtype=dtype) for key, value in dm_control_space.items()})
        space = spaces.Dict(kwargs)
        return space


class DMCEnv(Env):
    _spec = None

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, spec):
        # fuck new gym.
        spec.max_episode_steps = self._spec.max_episode_steps
        self._spec = spec

    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        environment_kwargs=None,
        visualize_reward=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        channels_first=True,
        from_pixels=False,
        gray_scale=False,
        warmstart=True,
        # info: https://github.com/deepmind/dm_control/issues/64
        no_gravity=False,
        non_newtonian=False,
        skip_start=None,  # useful in Manipulator for letting things settle
        space_dtype=None,  # default to float for consistency
    ):
        self.env = suite.load(
            domain_name,
            task_name,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": round(1.0 / self.env.control_timestep()),
        }

        self.from_pixels = from_pixels
        self.gray_scale = gray_scale
        self.channels_first = channels_first
        obs_spec = self.env.observation_spec()
        if from_pixels:
            color_dim = 1 if gray_scale else 3
            image_shape = [color_dim, width, height] if channels_first else [width, height, color_dim]
            self.observation_space = convert_dm_control_to_gym_space(
                obs_spec,
                dtype=space_dtype,
                pixels=spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            )
        else:
            self.observation_space = convert_dm_control_to_gym_space(obs_spec, dtype=space_dtype)
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec(), dtype=space_dtype)
        self.viewer = None

        self.render_kwargs = dict(
            height=height,
            width=width,
            camera_id=camera_id,
        )
        self.frame_skip = frame_skip
        if not warmstart:
            self.env.physics.data.qacc_warmstart[:] = 0
        self.no_gravity = no_gravity
        self.non_newtonian = non_newtonian

        if self.no_gravity:  # info: this removes gravity.
            self.turn_off_gravity()

        self.skip_start = skip_start

    def turn_off_gravity(self):
        # note: specifically for manipulator, lets the object fall.
        self.env.physisc.body_mass[:-2] = 0

    def seed(self, seed=None):
        self.action_space.seed(seed)
        return self.env.task.random.seed(seed)

    def set_state(self, state):
        # note: missing the goal positions.
        # self.env.physics.
        self.env.physics.set_state(state)
        self.step([0])

    def step(self, action):
        reward = 0

        for i in range(self.frame_skip):
            ts = self.env.step(action)
            if self.non_newtonian:  # zero velocity if non newtonian
                self.env.physics.data.qvel[:] = 0
            reward += ts.reward or 0
            done = ts.last()
            if done:
                break

        sim_state = self.env.physics.get_state().copy()

        obs = ts.observation
        if self.from_pixels:
            obs["pixels"] = self._get_obs_pixels()

        return obs, reward, done, dict(sim_state=sim_state)

    def _get_obs(self):
        return self.env.task.get_observation(self.env.physics)

    def _get_obs_pixels(self):
        img = self.render("gray" if self.gray_scale else "rgb", **self.render_kwargs)
        return img.transpose([2, 0, 1]) if self.channels_first else img

    def reset(self):
        obs = self.env.reset().observation
        for i in range(self.skip_start or 0):
            obs = self.env.step([0]).observation

        if self.from_pixels:
            obs["pixels"] = self._get_obs_pixels()

        return obs

    def render(self, mode="human", height=None, width=None, camera_id=0, **kwargs) -> NDArray:
        if kwargs.get("depth", None):
            print("is depth")

        if mode == "depth":
            kwargs["depth"] = True

        img = self.env.physics.render(
            width=self.render_kwargs["width"] if width is None else width,
            height=self.render_kwargs["height"] if height is None else height,
            camera_id=self.render_kwargs["camera_id"] if camera_id is None else camera_id,
            **kwargs,
        )
        if mode in ["rgb", "rgb_array", "human"]:
            return img.astype(np.uint8)
        elif mode == "depth":
            return img
        elif mode in ["gray", "grey"]:
            return img.mean(axis=-1, keepdims=True).astype(np.uint8)
        elif mode == "notebook":
            from IPython.display import display
            from PIL import Image

            pil_img = Image.fromarray(img, "RGB")
            display(pil_img)
            return img.astype(np.uint8)
        else:
            raise NotImplementedError(f"`{mode}` mode is not implemented")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()