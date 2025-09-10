from collections import deque
import random
import torch


class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)

    def add(self, xt, action, reward):
        """
        Add a new experience to the buffer
        args:
            xt: the current observation of the environment
            action: the action taken by the agent
            reward: the reward received by the agent
        """
        experience = (xt[0], action.item(), reward.item())
        self.buffer.append(experience)

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            raise ValueError(
                f"batch_size {batch_size} cannot be greater than the size of the buffer {len(self.buffer)}")

        start_index = random.randint(0, len(self.buffer) - batch_size)

        sampled_elements = [self.buffer[i]
                            for i in range(start_index, start_index + batch_size)]

        xts = torch.stack([item[0] for item in sampled_elements])

        actions = torch.tensor([item[1] for item in sampled_elements])
        num_classes = 4
        actions = torch.nn.functional.one_hot(
            actions, num_classes=num_classes).float()

        rewards = torch.tensor([item[2] for item in sampled_elements])

        return xts.unsqueeze(0), actions.unsqueeze(0), rewards.unsqueeze(0).unsqueeze(-1)

    def __len__(self):
        return len(self.buffer)
