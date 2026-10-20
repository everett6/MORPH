"""Explicit architecture dispatch for canonical checkpoints and FP16 exports."""
from . import MorphConfig, MorphModel


def config_from_dict(values):
    if values.get('model_type')=='delta':
        from .delta_config import DeltaConfig
        return DeltaConfig(**values)
    return MorphConfig(**values)


def model_from_config(config):
    if getattr(config,'model_type',None)=='delta':
        from .model.delta import DeltaModel
        return DeltaModel(config)
    return MorphModel(config)
