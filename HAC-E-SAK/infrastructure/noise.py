from abc import ABC, abstractmethod

import torch
import math


class ActionNoise(ABC):
    """
    Base class for noise applied to DDPG action outputs.
    """

    def __init__(self):
        super(ActionNoise, self).__init__()

    def reset(self) -> None:
        """
        Resetting noise at the end of a learning episode.
        """
        pass

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError()


class NormalActionNoise(ActionNoise):
    """
    Action noise from a Gaussian.
    :param mean: the mean value of the noise
    :param std: the scale of the noise (std here)
    """

    def __init__(self, mean: float, std: float, device: str = "cuda:0"):
        super().__init__()
        self._mu = mean
        self._sigma = std

        self.device = device

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_size = len(action)
        noise = torch.normal(
            self._mu * torch.ones(action_size), self._sigma * torch.ones(action_size),
        ).to(self.device)
        return action + noise

    def __repr__(self) -> str:
        return "NormalActionNoise(mu={}, sigma={})".format(self._mu, self._sigma)


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    Action noise from an Ornstein-Uhlenbeck process, approximating Brownian motion with friction.
    :param mean: the mean of the noise
    :param std: the scale of the noise
    :param theta: the rate of mean reversion
    :param dt: the timestep for the noise
    :param initial_noise: the initial value for the noise output, (if None: 0)
    """

    def __init__(
        self,
        mean: float,
        std: float,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: torch.Tensor = None,
        device: str = "cuda:0",
    ):
        super().__init__()
        self._theta = theta
        self._mu = mean
        self._sigma = std
        self._dt = dt
        self.initial_noise = initial_noise
        self.noise_prev = None
        self.reset()

        self.device = device

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:

        action_size = len(action)
        self._mu = self._mu * torch.ones(action_size)
        self._sigma = self._sigma * torch.ones(action_size)

        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * math.sqrt(self._dt) * torch.randn(self._mu.shape)
        ).to(self.device)
        self.noise_prev = noise
        return action + noise

    def reset(self) -> None:
        """
        Resetting the Ornstein-Uhlenbeck noise.
        """
        self.noise_prev = (
            self.initial_noise * torch.ones(self.action_size)
            if self.initial_noise is not None
            else torch.zeros_like(self._mu)
        ).to(self.device)

    def __repr__(self) -> str:
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(
            self._mu, self._sigma
        )


class AdaptiveParameterNoiseSpec(object):
    """
    An implementation of Adaptive *Parameter* Noise.
    :param init_std: the initial value for the standard deviation of the noise
    :param desired_action_std: the desired value for the standard deviation of the noise
    :param update_coeff: the update coefficient for the standard deviation of the noise
    """

    def __init__(
        self,
        init_std: float = 0.1,
        desired_action_std: float = 0.1,
        update_coeff: float = 1.01,
    ):
        self.init_std = init_std
        self.desired_action_std = desired_action_std
        self.update_coeff = update_coeff

        self.current_stddev = init_std

    def adapt(self, distance: float):
        """
        Update the standard deviation for the parameter noise.
        :param distance: the noise distance applied to the parameters
        """
        if distance > self.desired_action_std:
            # Decrease stddev.
            self.current_stddev /= self.update_coeff
        else:
            # Increase stddev.
            self.current_stddev *= self.update_coeff

    def get_stats(self):
        """
        Return the standard deviation for the parameter noise.
        :return: (dict) the stats of the noise
        """
        return {"param_noise_stddev": self.current_stddev}

    def __repr__(self):
        fmt = "AdaptiveParamNoiseSpec(init_std={}, desired_action_std={}, update_coeff={})"
        return fmt.format(self.init_std, self.desired_action_std, self.update_coeff)


# if __name__ == "__main__":
#     noise = OrnsteinUhlenbeckActionNoise(0, 1, 5)
#     action = noise(torch.ones(5), torch.zeros(5))
#     print(action)
