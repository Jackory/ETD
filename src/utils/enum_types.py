from enum import Enum

import numpy as np
from torch import nn


class ModelType(Enum):
    NoModel = 0
    ICM = 1  # Forward + Inverse
    RND = 2
    NGU = 3
    NovelD = 4  # Inverse
    DEIR = 5
    PlainForward = 6
    PlainInverse = 7
    PlainDiscriminator = 8
    TDD = 9
    E3B = 10
    Count = 11

    @staticmethod
    def get_enum_model_type(model_type):
        if isinstance(model_type, ModelType):
            return model_type
        if isinstance(model_type, str):
            model_type = model_type.strip().lower()
            if model_type == "icm":
                return ModelType.ICM
            elif model_type == "rnd":
                return ModelType.RND
            elif model_type == "ngu":
                return ModelType.NGU
            elif model_type == "noveld":
                return ModelType.NovelD
            elif model_type == "deir":
                return ModelType.DEIR
            elif model_type == "plainforward":
                return ModelType.PlainForward
            elif model_type == "plaininverse":
                return ModelType.PlainInverse
            elif model_type == "plaindiscriminator":
                return ModelType.PlainDiscriminator
            elif model_type == "tdd":
                return ModelType.TDD
            elif model_type == "e3b":
                return ModelType.E3B
            elif model_type == "count":
                return ModelType.Count
            else:
                return ModelType.NoModel
        raise ValueError


class NormType(Enum):
    NoNorm = 0
    BatchNorm = 1
    LayerNorm = 2

    @staticmethod
    def get_enum_norm_type(norm_type):
        if isinstance(norm_type, NormType):
            return norm_type
        if isinstance(norm_type, str):
            norm_type = norm_type.strip().lower()
            if norm_type == 'batchnorm':
                return NormType.BatchNorm
            if norm_type == 'layernorm':
                return NormType.LayerNorm
            if norm_type == 'nonorm':
                return NormType.NoNorm
        raise ValueError

    @staticmethod
    def get_norm_layer_1d(norm_type, fetures_dim, momentum=0.1):
        norm_type = NormType.get_enum_norm_type(norm_type)
        if norm_type == NormType.BatchNorm:
            return nn.BatchNorm1d(fetures_dim, momentum=momentum)
        if norm_type == NormType.LayerNorm:
            return nn.LayerNorm(fetures_dim)
        if norm_type == NormType.NoNorm:
            return nn.Identity()
        raise NotImplementedError

    @staticmethod
    def get_norm_layer_2d(norm_type, n_channels, n_size, momentum=0.1):
        norm_type = NormType.get_enum_norm_type(norm_type)
        if norm_type == NormType.BatchNorm:
            return nn.BatchNorm2d(n_channels, momentum=momentum)
        if norm_type == NormType.LayerNorm:
            return nn.LayerNorm([n_channels, n_size, n_size])
        if norm_type == NormType.NoNorm:
            return nn.Identity()
        raise NotImplementedError


class EnvSrc(Enum):
    MiniGrid = 0
    ProcGen = 1
    MiniWorld = 2
    Crafter = 3
    MuJoCo = 4
    DMC = 5


    @staticmethod
    def get_enum_env_src(env_src):
        if isinstance(env_src, EnvSrc):
            return env_src
        if isinstance(env_src, str):
            env_src = env_src.strip().lower()
            if env_src == 'minigrid':
                return EnvSrc.MiniGrid
            if env_src == 'procgen':
                return EnvSrc.ProcGen
            if env_src == 'miniworld':
                return EnvSrc.MiniWorld
            if env_src == 'mujoco':
                return EnvSrc.MuJoCo
            if env_src == 'dmc':
                return EnvSrc.DMC
            if env_src == 'crafter':
                return EnvSrc.Crafter
        raise ValueError


class DecayType(Enum):
    NoDecay = 0
    LinearDecay = 1
    CosineDecay = 2

    @staticmethod
    def get_enum_decay_type(decay_type):
        if isinstance(decay_type, DecayType):
            return decay_type
        if isinstance(decay_type, str):
            decay_type = decay_type.strip().lower()
            if decay_type == 'linear':
                return DecayType.LinearDecay
            if decay_type == 'cos':
                return DecayType.CosineDecay
            if decay_type == 'none':
                return DecayType.NoDecay
        raise ValueError
    
    @staticmethod
    def get_decay_rate(decay_type, current_progress_remaining: float):
        """
        Compute the decay rate using the current decay_type and the current progress remaining (from 1 to 0).
        """
        decay_type = DecayType.get_enum_decay_type(decay_type)
        current_progress_remaining = np.clip(current_progress_remaining, 0, 1)
        if decay_type == DecayType.LinearDecay:
            return current_progress_remaining
        if decay_type == DecayType.CosineDecay:
            return (1 + np.cos(np.pi * (1 - current_progress_remaining))) / 2
        if decay_type == DecayType.NoDecay:
            return 1
