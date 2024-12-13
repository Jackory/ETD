from gym import spaces
from gym.core import ObservationWrapper


class FlattenObservation(ObservationWrapper):
    """Observation wrapper that flattens the observation.

    Note: Different from the default gym wrapper by adding _get_obs function
    """

    def __init__(self, env, incude_original=False):
        super(FlattenObservation, self).__init__(env)
        self.__include_original = incude_original

        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return spaces.flatten(self.env.observation_space, observation)

    def _get_obs(self):
        obs = self.env.unwrapped._get_obs()
        return self.observation(obs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # add the original observations if asked
        if self.__include_original:
            info["observations_original"] = observation
        return self.observation(observation), reward, done, info


class ObservationByKey(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env, obs_key):
        super().__init__(env)
        self.obs_key = obs_key
        self.observation_space = env.observation_space[obs_key]

    def observation(self, observation):
        return observation[self.obs_key]
