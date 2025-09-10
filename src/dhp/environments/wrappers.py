import torch
from envs import PinPad  # Assuming this is your custom environment

class EnvWrapper():
    def __init__(self, env):
        self.env = env

    def step(self, action):
        action = action.item()
        obs = self.env.step({'action': action+1, 'reset': False})
        x_t = torch.Tensor(obs['image']).reshape(-1).unsqueeze(0)
        r_t = torch.Tensor([obs['reward']]).reshape(-1).unsqueeze(0)
        return x_t, r_t

    def reset(self):
        obs = self.env.step({'action': 0, 'reset': True})
        x_t = torch.Tensor(obs['image']).reshape(-1).unsqueeze(0)
        return x_t