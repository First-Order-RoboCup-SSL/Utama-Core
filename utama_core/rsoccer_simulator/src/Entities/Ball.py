from dataclasses import dataclass

from numpy.random import normal

from utama_core.rsoccer_simulator.src.Utils.gaussian_noise import RsimGaussianNoise


@dataclass()
class Ball:
    x: float = None
    y: float = None
    z: float = None
    v_x: float = 0.0
    v_y: float = 0.0
    v_z: float = 0.0
