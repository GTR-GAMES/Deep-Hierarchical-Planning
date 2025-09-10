import torch
import torch.nn as nn
import yaml
import sys
import os


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.dhp.utils.logger import Logger
from envs import PinPad
from src.dhp.environments.wrappers import EnvWrapper
from src.dhp.agents.hrl_agent import HRLAgent
from src.dhp.utils.yaml_parser import parse_yaml_with_types



def compute_max_cosine_reward(s_next, g):
    m = torch.maximum(torch.norm(g, dim=-1), torch.norm(s_next, dim=-1))
    normalized_g = g / m.unsqueeze(-1)
    normalized_s = s_next / m.unsqueeze(-1)
    return torch.sum(normalized_g * normalized_s, dim=-1)


def train(config):
    # Initialize environment
    env = EnvWrapper(PinPad(task=config['task'], length=config['length']))

    # Initialize logger
    logger = Logger(
        enabled=config['logging']['enabled'],
        project=config['logging']['project'],
        config=config,
        name=config['logging']['run_name']
    )

    # Training loop
    x_t = env.reset()
    converged = False
    episode_reward = 0
    episode = 0


    config['obs_dim'] = x_t.shape[-1]


    # Initialize agent
    agent = HRLAgent(config, logger=logger)

    model_info = agent.get_model_info()
    print("Model Information:")
    for name, info in model_info['models'].items():
        print(f"  {name}: {info['type']} with {info['parameters']:,} parameters")
    print(f"Total parameters: {model_info['total_parameters']:,}")

    while not converged and episode < config['training']['max_episodes']:
        # Acting
        a_t = agent.act(x_t)

        # Environment step
        next_x_t, r_t = env.step(a_t)
        episode_reward += r_t.item()

        # Learning
        agent.update(x_t, a_t, r_t, next_x_t)

        x_t = next_x_t

        # Check if episode ended
        if config['training']['max_steps'] and agent.train_step % config['training']['max_steps'] == 0:
            if logger:
                logger.log({"episode_reward": episode_reward}, step=episode)

            print(f"Episode {episode}, Reward: {episode_reward}")
            x_t = env.reset()
            episode_reward = 0
            episode += 1

            # Save model periodically
            if episode % config['training']['save_interval'] == 0:
                save_dir = "outputs/checkpoints"
                checkpoint_path = agent.save_checkpoint(save_dir, episode, logger)
                
                # Log model architecture to W&B
                if logger and logger.enabled:
                    model_info = agent.get_model_info()
                    logger.log({"model_info": model_info}, step=episode)

    if logger:
        logger.finish()


if __name__ == "__main__":
    # Load configuration
    # with open("configs/default.yaml", 'r') as f:
    #     config = yaml.safe_load(f)

    config = parse_yaml_with_types("configs/default.yaml")

    train(config)
