import torch
import torch.nn as nn
import torch.distributions as dist
import os

from src.dhp.models import goal_autoencoder
from ..models import RSSM, ManagerPolicy, ManagerValueCritic, GoalAutoencoder, WorkerPolicy, WorkerValueCritic
from ..utils import ReplayBuffer, imagine_trajectory, compute_discounted_returns_batch, ReturnNormalizer
from ..utils.logger import Logger
from ..models import compute_max_cosine_reward

class HRLAgent:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        
        # Initialize dimensions from config
        self.latent_dim = config['latent_dim']
        self.hidden_dim = config['hidden_dim']
        self.action_dim = config['action_dim']
        self.goal_classes = config['goal_classes']
        self.goal_codes = config['goal_codes']
        self.K = config['K']
        self.E = config['E']
        self.horizon = config['horizon']
        self.gamma = config['gamma']
        self.w_extrinsic = config['w_extrinsic']
        self.w_exploration = config['w_exploration']
        
        # Initialize models
        self.world_model = RSSM(self.action_dim, config['obs_dim'], self.latent_dim, self.hidden_dim)
        self.manager_policy = ManagerPolicy(self.latent_dim, self.goal_classes, self.goal_codes)
        self.manager_normalizer_extrinsic = ReturnNormalizer()
        self.manager_normalizer_exploration = ReturnNormalizer()
        self.manager_critic_extrinsic = ManagerValueCritic(self.latent_dim)
        self.manager_critic_exploration = ManagerValueCritic(self.latent_dim)
        self.goal_autoencoder = GoalAutoencoder(self.latent_dim)
        self.worker_policy = WorkerPolicy(self.latent_dim, self.latent_dim, self.action_dim)
        self.worker_critic = WorkerValueCritic(self.latent_dim)
        
        # Initialize optimizers
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=config['world_model_lr'])
        self.goal_autoencoder_optimizer = torch.optim.Adam(self.goal_autoencoder.parameters(), lr=config['goal_autoencoder_lr'])
        self.manager_policy_optimizer = torch.optim.Adam(self.manager_policy.parameters(), lr=config['manager_policy_lr'])
        self.manager_critic_extrinsic_optimizer = torch.optim.Adam(
            self.manager_critic_extrinsic.parameters(), lr=config['manager_critic_lr'])
        self.manager_critic_exploration_optimizer = torch.optim.Adam(
            self.manager_critic_exploration.parameters(), lr=config['manager_critic_lr'])
        self.worker_policy_optimizer = torch.optim.Adam(self.worker_policy.parameters(), lr=config['worker_policy_lr'])
        self.worker_critic_optimizer = torch.optim.Adam(self.worker_critic.parameters(), lr=config['worker_critic_lr'])
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training state
        self.train_step = 0
        self.prev_s_t = torch.zeros(1, self.latent_dim)
        self.prev_a_t = torch.Tensor([0, 0, 0, 0]).unsqueeze(0)
        self.current_goal = None
        
    def act(self, x_t):
        # Acting phase
        post_mean, post_std = self.world_model.repr(self.prev_s_t, self.prev_a_t, x_t)
        s_t = self.world_model.sample_gaussian(post_mean, post_std)

        if self.train_step % self.K == 0:
            # Update internal goal
            logits = self.manager_policy(s_t)
            goal_distributions = dist.Categorical(logits=logits)
            selected_goal = goal_distributions.sample()
            one_hot_goal = torch.nn.functional.one_hot(
                selected_goal, num_classes=self.goal_classes)
            z = one_hot_goal.view(1, -1).float()
            self.current_goal = self.goal_autoencoder.decoder(z)

        # Sample action
        action_logits = self.worker_policy(s_t, self.current_goal)
        action_dist = dist.Categorical(logits=action_logits)
        a_t = action_dist.sample().view(1, -1)
        
        # Update previous state and action
        self.prev_s_t = s_t
        self.prev_a_t = torch.nn.functional.one_hot(a_t, num_classes=self.action_dim)[0]
        
        return a_t
    
    def update(self, x_t, a_t, r_t, next_x_t):
        # Add transition to replay buffer
        self.replay_buffer.add(x_t, a_t, r_t)
        
        # Learning phase
        if self.train_step % self.E == 0:
            self._learn()
        
        self.train_step += 1
        
    def _learn(self):
        if len(self.replay_buffer) < 16:
            return  # Not enough data to sample a batch
        
        # Draw Sequence Batch
        x, a, r = self.replay_buffer.sample(16)
        
        # Update World Model
        decoded_obs, pred_rewards, posteriors, priors, states, s_t_posts = self.world_model(x, a)
        world_model_loss = self.world_model.loss(decoded_obs, pred_rewards, x, r, posteriors, priors)
        
        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        self.world_model_optimizer.step()
        
        # Update Goal Autoencoder
        s_t_posts = s_t_posts.detach()
        logits, z, decoded = self.goal_autoencoder(s_t_posts)
        decoded = decoded.unsqueeze(0)
        loss = self.goal_autoencoder.loss(decoded, s_t_posts, logits)
        
        self.goal_autoencoder_optimizer.zero_grad()
        loss.backward()
        self.goal_autoencoder_optimizer.step()
        
        # Update Policies
        states, actions, goals, latent_goals = imagine_trajectory.imagine_trajectory(
            self.world_model, self.manager_policy, self.goal_autoencoder, 
            self.worker_policy, s_t_posts, horizon=self.horizon)
        
        # Predict extrinsic rewards
        extrinsic_rewards = self.world_model.reward(
            states).reshape(-1).unsqueeze(0).detach()

        # Compute exploration rewards
        encoded_states = self.goal_autoencoder.encoder(states)
        decoded_states = self.goal_autoencoder.decoder(encoded_states)
        exploration_rewards = torch.mean(
            (decoded_states - states) ** 2, dim=1).unsqueeze(0).detach()

        # Compute goal rewards
        goal_rewards = compute_max_cosine_reward(
            states, goals).unsqueeze(0).detach()

        # Abstract trajectory to update manager
        abstract_states = states[::self.K, :].detach()  # Select every K-th state
        abstract_extrinsic_return = []
        abstract_exploration_return = []
        # print(f"{states.shape=}")
        for t in range(0, states.shape[0], self.K):
            step_extrinsic_rewards = extrinsic_rewards[:, t:t+self.K]
            step_exploration_bonuses = exploration_rewards[:, t:t+self.K]

            extrinsic_discounted_returns = compute_discounted_returns_batch(
                step_extrinsic_rewards, self.gamma)
            exploration_discounted_returns = compute_discounted_returns_batch(
                step_exploration_bonuses, self.gamma)

            abstract_extrinsic_return.append(
                extrinsic_discounted_returns[:, 0])
            abstract_exploration_return.append(
                exploration_discounted_returns[:, 0])

        abstract_extrinsic_return = torch.stack(
            abstract_extrinsic_return, dim=1)
        abstract_exploration_return = torch.stack(
            abstract_exploration_return, dim=1)

        # Update normalizers
        self.manager_normalizer_extrinsic.update(abstract_extrinsic_return)
        self.manager_normalizer_exploration.update(abstract_exploration_return)

        # Normalize returns
        normalized_extrinsic = self.manager_normalizer_extrinsic.normalize(
            abstract_extrinsic_return)
        normalized_exploration = self.manager_normalizer_exploration.normalize(
            abstract_exploration_return)
        total_advantage = self.w_extrinsic * normalized_extrinsic + \
            self.w_exploration * normalized_exploration

        total_advantage = total_advantage.reshape(-1)

        # Policy update
        logits = self.manager_policy(abstract_states)
        goal_distributions = dist.Categorical(logits=logits)
        selected_goal = goal_distributions.sample()
        one_hot_goal = torch.nn.functional.one_hot(
            selected_goal, num_classes=self.goal_classes)
        sparse_goals = one_hot_goal.view(self.horizon, -1)
        log_prob = goal_distributions.log_prob(selected_goal)
        log_prob = log_prob.sum(dim=-1)
        loss_policy = - (log_prob * total_advantage).mean()
        self.manager_policy_optimizer.zero_grad()
        loss_policy.backward()
        self.manager_policy_optimizer.step()

        # Critic updates
        loss_critic_extrinsic = (self.manager_critic_extrinsic(
            abstract_states) - abstract_extrinsic_return).pow(2).mean()
        self.manager_critic_extrinsic_optimizer.zero_grad()
        loss_critic_extrinsic.backward()
        self.manager_critic_extrinsic_optimizer.step()

        loss_critic_exploration = (self.manager_critic_exploration(
            abstract_states) - abstract_exploration_return).pow(2).mean()
        self.manager_critic_exploration_optimizer.zero_grad()
        loss_critic_exploration.backward()
        self.manager_critic_exploration_optimizer.step()

        # Split trajectory to update worker
        states_segments = [states[i:i+self.K] for i in range(0, len(states), self.K)]
        goals_segments = [goals[i:i+self.K] for i in range(0, len(goals), self.K)]

        for i, _ in enumerate(states_segments):
            segment_states = states_segments[i].detach()
            segment_goals = goals_segments[i].detach()

            # Flatten for processing
            batch_size, seq_len = 1, len(segment_states)
            states_flat = segment_states.reshape(-1, self.latent_dim)
            goals_flat = segment_goals.reshape(-1, self.latent_dim)

            # Get action logits and values
            action_logits = self.worker_policy(states_flat, goals_flat)
            values = self.worker_critic(states_flat, goals_flat).reshape(1, -1)

            # Sample actions
            action_dist = dist.Categorical(logits=action_logits)
            actions = action_dist.sample()

            segment_rewards = goal_rewards[:, i*self.K:(i+1)*self.K].reshape(1, -1)

            with torch.no_grad():
                # Only use first K-1 steps for targets
                target_values = segment_rewards[:, :-1] + self.gamma * values[:, 1:]
                advantages = target_values - values[:, :-1]

            # Policy loss (only on first K-1 steps)
            log_probs = action_dist.log_prob(
                actions.flatten()).view(batch_size, seq_len)
            policy_loss = -(log_probs[:, :-1] * advantages).mean()

            # Value loss (only on first K-1 steps)
            value_loss = nn.functional.mse_loss(values[:, :-1], target_values)

            # Update worker
            self.worker_policy_optimizer.zero_grad()
            self.worker_critic_optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            self.worker_policy_optimizer.step()
            self.worker_critic_optimizer.step()

        # Log metrics if logger is available
        if self.logger:
            self.logger.log({
                "world_model_loss": world_model_loss.item(),
                "goal_autoencoder_loss": loss.item(),
                # Add other metrics here
            }, step=self.train_step)

    def _get_models_and_optimizers(self):
        """Automatically discover all models and optimizers in the agent"""
        models = {}
        optimizers = {}
        
        for attr_name in dir(self):
            attr_value = getattr(self, attr_name)
            
            # Find all nn.Module instances (models)
            if isinstance(attr_value, nn.Module):
                models[attr_name] = attr_value
            
            # Find all torch.optim.Optimizer instances
            elif isinstance(attr_value, torch.optim.Optimizer):
                optimizers[attr_name] = attr_value
        
        return models, optimizers
    
    def save_checkpoint(self, save_dir, episode, logger=None):
        """Save all models and optimizers to a checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        
        models, optimizers = self._get_models_and_optimizers()
        
        # Create checkpoint dictionary
        checkpoint = {
            'episode': episode,
            'train_step': self.train_step,
            'models': {name: model.state_dict() for name, model in models.items()},
            'optimizers': {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            'config': self.config,  # Save the config for reproducibility
            'training_state': {
                'prev_s_t': self.prev_s_t,
                'prev_a_t': self.prev_a_t,
                'current_goal': self.current_goal if hasattr(self, 'current_goal') else None,
            }
        }
        
        # Save individual models (optional, for inspection)
        for model_name, model in models.items():
            model_path = os.path.join(save_dir, f"{model_name}_ep_{episode}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved {model_name} to {model_path}")
            
            if logger:
                artifact_name = f"{model_name}-episode-{episode}"
                logger.save_model(model, model_path, artifact_name=artifact_name)
        
        # Save individual optimizers (optional)
        for opt_name, optimizer in optimizers.items():
            opt_path = os.path.join(save_dir, f"{opt_name}_ep_{episode}.pt")
            torch.save(optimizer.state_dict(), opt_path)
            
            if logger:
                artifact_name = f"{opt_name}-episode-{episode}"
                logger.save_model(optimizer, opt_path, artifact_name=artifact_name)
        
        # Save combined checkpoint (main file for loading)
        combined_path = os.path.join(save_dir, f"checkpoint_ep_{episode}.pt")
        torch.save(checkpoint, combined_path)
        print(f"Saved combined checkpoint to {combined_path}")
        
        if logger:
            logger.save_model(self, combined_path, artifact_name=f"full-checkpoint-episode-{episode}")
        
        return combined_path
    
    def load_checkpoint(self, checkpoint_path, load_optimizers=True, load_training_state=True):
        """Load models and optimizers from a checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        models, optimizers = self._get_models_and_optimizers()
        
        # Load model states
        for model_name in models.keys():
            if model_name in checkpoint['models']:
                models[model_name].load_state_dict(checkpoint['models'][model_name])
                print(f"Loaded {model_name}")
            else:
                print(f"Warning: {model_name} not found in checkpoint")
        
        # Load optimizer states
        if load_optimizers:
            for opt_name in optimizers.keys():
                if opt_name in checkpoint['optimizers']:
                    optimizers[opt_name].load_state_dict(checkpoint['optimizers'][opt_name])
                    print(f"Loaded {opt_name} optimizer")
                else:
                    print(f"Warning: {opt_name} optimizer not found in checkpoint")
        
        # Load training state
        if load_training_state and 'training_state' in checkpoint:
            training_state = checkpoint['training_state']
            self.prev_s_t = training_state.get('prev_s_t', self.prev_s_t)
            self.prev_a_t = training_state.get('prev_a_t', self.prev_a_t)
            if hasattr(self, 'current_goal') and 'current_goal' in training_state:
                self.current_goal = training_state['current_goal']
        
        # Load other states
        self.train_step = checkpoint.get('train_step', self.train_step)
        
        print(f"Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")
        return checkpoint.get('episode', 0)
    
    def get_model_info(self):
        """Get information about all models and optimizers in the agent"""
        models, optimizers = self._get_models_and_optimizers()
        
        info = {
            'models': {},
            'optimizers': {},
            'total_parameters': 0
        }
        
        for name, model in models.items():
            num_params = sum(p.numel() for p in model.parameters())
            info['models'][name] = {
                'type': type(model).__name__,
                'parameters': num_params,
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            info['total_parameters'] += num_params
        
        for name, optimizer in optimizers.items():
            info['optimizers'][name] = {
                'type': type(optimizer).__name__,
                'num_param_groups': len(optimizer.param_groups)
            }
        
        return info