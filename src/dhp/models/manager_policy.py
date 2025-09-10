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


class ManagerPolicy(nn.Module):
    def __init__(self, latent_dim, code_size, num_codes):
        super(ManagerPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_codes * code_size)
        )
        self.num_codes = num_codes
        self.code_size = code_size

    def forward(self, s_t):
        logits = self.fc(s_t).view(-1, self.num_codes, self.code_size)
        return logits


class ManagerValueCritic(nn.Module):
    def __init__(self, latent_dim):
        super(ManagerValueCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, s_t):
        value = self.fc(s_t)
        return value


# ================================================================================================


def generate_rollout(world_model, start_state, horizon):
    states = [start_state]
    extrinsic_rewards = []
    exploration_bonuses = []

    for t in range(horizon):
        # TODO: Implement the forward pass of the world model to get the next state, extrinsic reward and exploration bonus; extr reward and exp bonus follow the equations 6 & 8
        next_state, extr_reward, exp_bonus = world_model.predict(states[-1])
        states.append(next_state)
        extrinsic_rewards.append(extr_reward)
        exploration_bonuses.append(exp_bonus)

    return states, extrinsic_rewards, exploration_bonuses


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

