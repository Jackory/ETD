from dm_control import suite

from gym.envs.registration import register, EnvSpec
from gym.core import ObservationWrapper
from gym import spaces
from gym.wrappers import TimeLimit
from src.env.dmc.flat import FlattenObservation, ObservationByKey
import gym


ALL_ENVS = []

def make(*args, **kwargs):
    return env.make(*args, **kwargs)

def make_env(
    id: str,
    flatten_obs=True,
    from_pixels=False,
    frame_skip=1,
    episode_frames=1000,
    **kwargs,
):
    domain_name = kwargs['domain_name']
    task_name = kwargs['task_name']
    max_episode_steps = episode_frames / frame_skip

    from src.env.dmc.dmc import DMCEnv

    env = DMCEnv(from_pixels=from_pixels, frame_skip=frame_skip, **kwargs)

    if id:
        env._spec = EnvSpec(
            id_requested=f"{domain_name.capitalize()}-{task_name}",
            max_episode_steps=max_episode_steps,
        )
    # This spec object gets picked up by the gym.EnvSpecs constructor
    # used in gym.registration.EnvSpec.make, L:93 to generate the spec
    
    if from_pixels:
        env = ObservationByKey(env, "pixels")
    elif flatten_obs:
        env = FlattenObservation(env)
    return env


for domain_name, task_name in suite.ALL_TASKS:
    ID = f"dmc/{domain_name.capitalize()}-{task_name}-v1"
    ALL_ENVS.append(ID)
    register(
        id=ID,
        entry_point="src.env.dmc:make_env",
        kwargs=dict(
            id=ID,
            domain_name=domain_name,
            task_name=task_name,
            channels_first=True,
            width=84,
            height=84,
            frame_skip=1,
        ),
    )

DMC_IS_REGISTERED = True
