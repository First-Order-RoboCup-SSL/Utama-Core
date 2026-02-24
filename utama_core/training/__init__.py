"""BenchMARL training framework for ASPAC RoboCup SSL validation."""

from utama_core.training.experiment import SSLExperimentConfig, create_experiment
from utama_core.training.task import SSLTask

__all__ = ["SSLTask", "SSLExperimentConfig", "create_experiment"]
