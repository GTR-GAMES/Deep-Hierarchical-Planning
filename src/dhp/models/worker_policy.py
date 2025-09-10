import torch.nn as nn
import torch
import torch.distributions as dist
import torch.nn.functional as F


class WorkerPolicy(nn.Module):
    def __init__(self, latent_dim, goal_dim, action_dim):
        super(WorkerPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + goal_dim, 512),
            nn.LayerNorm(512),  # Added LayerNorm
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),  # Added LayerNorm
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, s_t, g):
        # Concatenate state and goal vectors
        x = torch.cat([s_t, g], dim=-1)
        return self.fc(x)


class WorkerValueCritic(nn.Module):
    def __init__(self, latent_dim):
        super(WorkerValueCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, 512),
            nn.LayerNorm(512),  # Added LayerNorm
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),  # Added LayerNorm
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, s_t, g):
        x = torch.cat([s_t, g], dim=-1)
        return self.fc(x)


def compute_max_cosine_reward(s_next, g):
    m = torch.maximum(torch.norm(g, dim=-1), torch.norm(s_next, dim=-1))
    normalized_g = g / m.unsqueeze(-1)
    normalized_s = s_next / m.unsqueeze(-1)
    return torch.sum(normalized_g * normalized_s, dim=-1)


# ================================================================================================


def train_worker(imagined_states, goals):
    """Trains worker on K-step segments"""
    # Split trajectories into K-step chunks
    num_segments = imagined_states.shape[1] // K
    segments = []

    for i in range(num_segments):
        start = i * K
        end = start + K
        segment = {
            'states': imagined_states[:, start:end],
            'goals': goals[:, i].unsqueeze(1).expand(-1, K, -1)
        }
        segments.append(segment)

    total_policy_loss = 0
    total_value_loss = 0

    for segment in segments:
        states = segment['states']  # [batch_size, K, latent_dim]
        goals = segment['goals']    # [batch_size, K, goal_dim]

        # Flatten for processing
        batch_size, seq_len = states.shape[:2]
        states_flat = states.reshape(-1, latent_dim)
        goals_flat = goals.reshape(-1, goal_dim)

        print(f"States: {states_flat.shape}, Goals: {goals_flat.shape}")

        # Get action logits and values
        action_logits = worker_policy(states_flat, goals_flat)
        values = worker_critic(states_flat, goals_flat).view(
            batch_size, seq_len)

        # Sample actions (for discrete action space)
        action_dist = dist.Categorical(logits=action_logits)
        actions = action_dist.sample().view(batch_size, seq_len)

        print(f"{actions.shape=}")

        # Compute rewards (using next states from world model)
        s_next = torch.randn_like(states)  # Mock next states
        rewards = compute_max_cosine_reward(
            s_next.view(-1, latent_dim),
            goals_flat
        ).view(batch_size, seq_len)

        print(f"{rewards.shape=}")

        with torch.no_grad():
            # Only use first K-1 steps for targets
            target_values = rewards[:, :-1] + gamma * values[:, 1:]
            advantages = target_values - values[:, :-1]

        # Policy loss (only on first K-1 steps)
        log_probs = action_dist.log_prob(
            actions.flatten()).view(batch_size, seq_len)
        policy_loss = -(log_probs[:, :-1] * advantages).mean()

        # Value loss (only on first K-1 steps)
        value_loss = F.mse_loss(values[:, :-1], target_values)

        # Update
        optimizer_worker.zero_grad()
        optimizer_critic.zero_grad()
        (policy_loss + value_loss).backward()
        optimizer_worker.step()
        optimizer_critic.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / len(segments), total_value_loss / len(segments)
