import torch.nn as nn
import torch
import torch.distributions as dist


class ReturnNormalizer:
    def __init__(self, decay=0.99, epsilon=1e-8):
        self.decay = decay
        self.epsilon = epsilon
        self.ema_return = 0.0
        self.ema_return_squared = 0.0

    def update(self, return_value):
        self.ema_return = self.decay * self.ema_return + \
            (1 - self.decay) * return_value.mean()
        self.ema_return_squared = self.decay * self.ema_return_squared + \
            (1 - self.decay) * (return_value ** 2).mean()

    def normalize(self, return_value):
        std_return = torch.sqrt(
            self.ema_return_squared - self.ema_return ** 2) + self.epsilon
        normalized_return = return_value / std_return
        return normalized_return


def compute_discounted_returns(rewards, gamma):
    discounted_returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_returns.insert(0, R)
    return torch.tensor(discounted_returns)


def compute_discounted_returns_batch(rewards_batch, gamma):
    discounted_returns_batch = [
        compute_discounted_returns(rewards, gamma) for rewards in rewards_batch
    ]
    return torch.stack(discounted_returns_batch)
