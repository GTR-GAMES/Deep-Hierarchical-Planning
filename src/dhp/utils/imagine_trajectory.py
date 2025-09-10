import torch
import torch.distributions as dist
import constants

goal_classes = constants.goal_classes
action_dim = constants.action_dim


# Returns states, actions, goals, latent goals (one hot)
def imagine_trajectory(world_model, manager_policy, goal_autoencoder, worker_policy, start_states, horizon=32):
    states = []
    actions = []
    goals = []
    latent_goals = []

    s_t = start_states.squeeze(0)[-1].unsqueeze(0)

    for t in range(horizon):
        states.append(s_t)
        if t % 8 == 0:
            # Sample new goal
            logits = manager_policy(s_t)
            goal_distributions = dist.Categorical(logits=logits)
            selected_goal = goal_distributions.sample()
            one_hot_goal = torch.nn.functional.one_hot(
                selected_goal, num_classes=goal_classes)
            z = one_hot_goal.view(-1, 64).float()
            # print(f"{selected_goal.shape=}")
            # print(f"{one_hot_goal.shape=}")
            # print(f"{z.shape=}")
            # print(f"{z=}")
            g = goal_autoencoder.decoder(z)

        # Sample action
        # print(f"{s_t.shape=}, {g.shape=}")
        action_logits = worker_policy(s_t, g)
        action_dist = dist.Categorical(logits=action_logits)
        a_t = action_dist.sample().view(1, -1)

        a_t = torch.nn.functional.one_hot(
            a_t, num_classes=action_dim)[0].float()

        mean, std = world_model.dynamics(s_t, a_t)

        s_t = world_model.sample_gaussian(mean, std)

        actions.append(a_t)
        goals.append(g)
        latent_goals.append(z)

    states = torch.cat(states)
    actions = torch.cat(actions)
    goals = torch.cat(goals)
    latent_goals = torch.cat(latent_goals)

    return states, actions, goals, latent_goals
