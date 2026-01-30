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
    
    def add_gaussian_noise(self, noise: RsimGaussianNoise):
        """
        When running in rsim, add Gaussian noise to ball with the given standard deviations.
        Mutates the Robot object in place.
        
        Args:
            noise (RsimGaussianNoise): The 3 parameters are for x (in m), y (in m), and orientation (in degrees) respectively.
                Defaults to 0 for each.
        """
        
        if noise.x_stddev:
            self.x += normal(scale=noise.x_stddev)
            
        if noise.y_stddev:
            self.y += normal(scale=noise.y_stddev)
            
        # No noise addition for z, since rSim is 2-D
