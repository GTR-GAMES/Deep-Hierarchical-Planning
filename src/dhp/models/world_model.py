import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(self, action_dim, obs_dim, latent_dim, hidden_dim):
        super(RSSM, self).__init__()

        # Representation Model (reprθ)
        # Representation Model (Posterior)
        self.repr_gru = nn.GRUCell(action_dim + obs_dim, latent_dim)
        self.repr_ln = nn.LayerNorm(latent_dim)
        self.repr_to_mean = nn.Linear(latent_dim, latent_dim)  # Mean
        self.repr_to_std = nn.Linear(latent_dim, latent_dim)   # Std

        # Dynamics Model (dynθ)
        self.dyn_gru = nn.GRUCell(action_dim, latent_dim)
        self.dyn_ln = nn.LayerNorm(latent_dim)
        self.dyn_to_mean = nn.Linear(latent_dim, latent_dim)   # Mean
        self.dyn_to_std = nn.Linear(latent_dim, latent_dim)    # Std

        # Decoder Model (recθ)
        self.decoder_model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

        # Reward Prediction Model (rewθ)
        self.reward_model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Predicting a scalar reward
        )

    def repr(self, prev_latent, action, obs):
        """
        Representation model: maps (obs, action) to latent state.
        """
        gru_input = torch.cat([action, obs], dim=-1)
        hidden = self.repr_gru(gru_input, prev_latent)
        hidden = self.repr_ln(hidden)

        # Compute mean and std
        mean = self.repr_to_mean(hidden)
        std = F.softplus(self.repr_to_std(hidden))  # Ensure std > 0
        return mean, std

    def dynamics(self, prev_latent, action):
        """
        Dynamics model: maps (prev_latent, action) to next latent state.
        """
        hidden = self.dyn_gru(action, prev_latent)
        hidden = self.dyn_ln(hidden)

        # Compute mean and std
        mean = self.dyn_to_mean(hidden)
        std = F.softplus(self.dyn_to_std(hidden))  # Ensure std > 0
        return mean, std

    def decode(self, latent):
        """
        Decoder model: maps latent state to observation space.
        """
        return self.decoder_model(latent)

    def reward(self, latent):
        """
        Reward predictor: maps latent state to predicted reward.
        """
        return self.reward_model(latent)

    def forward(self, obs, actions, initial_states=None):
        batch_size, seq_len = obs.shape[:2]
        latent_dim = self.repr_gru.hidden_size

        # Initialize latent states
        if initial_states is None:
            h_t = torch.zeros(batch_size, latent_dim).to(obs.device)
            s_t = torch.zeros(batch_size, latent_dim).to(obs.device)
        else:
            h_t, s_t = initial_states  # Carry over from previous sequence

        # Store predictions and distributions
        decoded_obs = []
        predicted_rewards = []
        post_means, post_stds = [], []  # Separate lists
        prior_means, prior_stds = [], []  # Separate lists
        s_t_posts = [] 

        for t in range(seq_len):
            # Prior (Dynamics)
            prior_mean, prior_std = self.dynamics(h_t, actions[:, t])
            s_t_prior = self.sample_gaussian(prior_mean, prior_std)

            # Posterior (Representation)
            post_mean, post_std = self.repr(s_t, actions[:, t], obs[:, t])
            s_t_post = self.sample_gaussian(post_mean, post_std)

            # Store components separately
            post_means.append(post_mean)
            post_stds.append(post_std)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            s_t_posts.append(s_t_post)

            # Decoding and reward
            decoded_obs.append(self.decode(s_t_post))
            predicted_rewards.append(self.reward(s_t_post))

            # Update states
            h_t = s_t_prior
            s_t = s_t_post

        # Stack separately
        decoded_obs = torch.stack(decoded_obs, dim=1)
        predicted_rewards = torch.stack(predicted_rewards, dim=1)
        posteriors = (torch.stack(post_means, dim=1),
                      torch.stack(post_stds, dim=1))
        priors = (torch.stack(prior_means, dim=1),
                  torch.stack(prior_stds, dim=1))
        
        s_t_posts = torch.stack(s_t_posts, dim=1)

        # Return final states for carry-over
        final_states = (h_t, s_t)

        return decoded_obs, predicted_rewards, posteriors, priors, final_states, s_t_posts

    def loss(self, decoded_obs, predicted_rewards, obs, rewards, posteriors, priors, beta=1.0):
        # Unpack distributions
        post_mean, post_std = posteriors
        prior_mean, prior_std = priors

        # Compute KL divergence
        kl = 0.5 * (
            (prior_std / post_std).log()
            + (post_std**2 + (post_mean - prior_mean)**2) / prior_std**2
            - 1
        ).sum(dim=-1).mean()

        # Other losses
        recon_loss = F.mse_loss(decoded_obs, obs)
        reward_loss = F.mse_loss(predicted_rewards, rewards)

        return recon_loss + reward_loss + beta * kl

    def sample_gaussian(self, mean, std):
        """
        Samples from a Gaussian distribution using the reparameterization trick.
        """
        eps = torch.randn_like(std)  # Noise ~ N(0, 1)
        return mean + std * eps

    def kl_loss(posterior_mean, posterior_std, prior_mean, prior_std):
        """
        Computes KL divergence between two Gaussians.
        """
        kl = 0.5 * (
            (prior_std / posterior_std).log()
            + (posterior_std**2 + (posterior_mean - prior_mean)**2) / prior_std**2
            - 1
        )
        return kl.sum(dim=-1).mean()

