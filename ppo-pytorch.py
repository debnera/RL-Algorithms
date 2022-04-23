# %%

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import gym


# %%

# Env setup


def test_env():
    env = gym.make("CartPole-v1")
    env.reset()
    for _ in range(10):
        # env.render()  - Does not work in jupyter?
        obs, reward, done, info = env.step(env.action_space.sample())  # take a random action
        print(reward)
    env.close()


# %%


def eval(env, model=None):
    terminal_steps = []
    max_steps = 1000
    iterations = 100
    for i in range(iterations):
        env.reset()
        for step in range(max_steps):
            if model:
                action = model.get_action()
                obs, reward, done, info = env.step(action.cpu().numpy())
            else:
                obs, reward, done, info = env.step(env.action_space.sample())  # take a random action
            if reward == 0 or step == max_steps - 1:
                terminal_steps.append(step)
                break
    print(f"Mean reward: {np.mean(terminal_steps)}")


# %%


def test_eval():
    env = gym.make("CartPole-v1")
    eval(env)
    env.close()


# %%

# NN Architecture
# - ActorCritic
# - Shared weights vs not
class ActorCritic(torch.nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units, layers, discrete_actions=False):
        super().__init__()
        self.discrete_actions = discrete_actions
        self.actor = nn.Sequential(nn.Linear(input_shape, hidden_units), nn.ReLU())
        for i in range(layers):
            self.actor.append(nn.Linear(hidden_units, hidden_units))
            self.actor.append(nn.ReLU())

        self.critic = deepcopy(self.actor)
        self.critic.append(nn.Linear(hidden_units, 1))

        self.actor.append(nn.Linear(hidden_units, output_shape))
        self.actor.append(nn.Tanh())  # Why does actor need Tanh?

    def forward(self, x):
        values = self.critic(x)
        logits = self.actor(x)  # TODO: Do logits always result from Tanh?
        return values, logits

    def get_action(self, x, action=None):
        logits = self.actor(x)

        # TODO: Continuous vs discrete actions
        # As in, are logits probabilities or distributions?

        # Discrete action sampling:
        if self.discrete_actions:
            m = torch.distributions.Categorical(logits=logits)

        # Continuous action sampling
        else:
            std = 0  # TODO: Where does this come from and why?
            m = torch.distributions.Normal(logits, std)
        if action == None:
            action = m.sample()

        return action, m.log_prob(action)

    def get_value(self, x):
        values = self.critic(x)
        return values


# %%

# Test ActorCritic

# ac = ActorCritic(8, 2, 128, 2)
# print(ac)
# action = ac.get_action(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.float))
# print(action)

# %%

# Taken from cleanrl and ML-Collective
config = {
    "description": "cleanrl",
    "seed": 1,
    "torch_deterministic": True,
    "device": "cpu",
    "std_init": 0.05,
    "env_id": "MountainCar-v0",
    "num_workers": 8,  # rank (seed) / envs / N
    "num_epochs": 10,  # K number of
    "num_iterations": 30,  # number of times we collect a dataset or no. of update loops (300k-2mil total timesteps)
    "max_timesteps": 2048,  # T
    "epsilon": 0.2,  # clipping radius
    "lr": 3e-4,
    "gamma": 0.99,
    "batch_size": 512,
    "eval_actors": 4,  # not used
    "clip_value_loss": True,
    "gae": True,
    "gae_lambda": 0.95,
    "advantage_norm": True,
    "max_grad_norm": 0.5,
}


# %%

# PPO
class PPO:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config["env_id"])
        if type(self.env.action_space) is gym.spaces.discrete.Discrete:
            pass
            self.action_space = self.env.action_space.n
            self.discrete_action = True
        else:
            self.discrete_action = False
            self.action_space = self.env.action_space.shape[0]
        self.observation_space = self.env.observation_space.shape[0]
        self.agent = ActorCritic(
            self.observation_space,
            self.action_space,
            hidden_units=128,
            layers=2,
            discrete_actions=self.discrete_action,
        )
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config["lr"], eps=1e-5)

    def train(self):

        # Get config values (TODO: fetch from config)
        num_envs = 1
        iterations = 2
        buffer_size = 100
        steps_per_env = 100
        device = torch.device("cuda" if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")

        # Set up buffers
        obs = torch.zeros((steps_per_env, num_envs, self.observation_space)).to(device)
        actions = torch.zeros((steps_per_env, num_envs, self.action_space)).to(device)
        logprobs = torch.zeros((steps_per_env, num_envs)).to(device)
        rewards = torch.zeros((steps_per_env, num_envs)).to(device)
        dones = torch.zeros((steps_per_env, num_envs)).to(device)
        values = torch.zeros((steps_per_env, num_envs)).to(device)

        env_id = 0  # TODO: Implement multiple envs
        for i in range(iterations):
            done = torch.tensor(1.0).to(device)  # This is just to silence warning about possibly undefined variable
            observation = torch.tensor(self.env.reset()).to(device)  # Get initial observations
            for step in range(steps_per_env):
                # Collect experience
                obs[step, env_id] = observation
                with torch.no_grad():
                    action, log_prob = self.agent.get_action(observation)
                    value = self.agent.get_value(observation)
                observation, reward, done, info = self.env.step(action.cpu().numpy())
                observation = torch.tensor(observation).to(device)
                actions[step, env_id] = action
                rewards[step, env_id] = torch.tensor(reward).to(device)
                dones[step, env_id] = torch.tensor(done).to(device)
                if done:
                    # TODO: does env self-reset? Seems to do so on clean-rl
                    observation = torch.tensor(self.env.reset()).to(device)

            # Calculate advantages (future rewards - critic predictions --> critic loss)
            with torch.no_grad():  # TODO: Why no grad here?
                advantages = torch.zeros_like(rewards).to(device)
                final_done = torch.tensor(done, dtype=torch.float32).to(device)
                final_value = value
                for t in reversed(range(steps_per_env)):
                    if t == steps_per_env - 1:
                        next_not_done = 1.0 - final_done
                        next_value = final_value
                    else:
                        next_not_done = 1.0 - dones[t + 1, env_id]
                        next_value = values[t + 1, env_id]
                    advantages[t, env_id] = rewards[t] + self.config["gamma"] * next_not_done * next_value
                advantages = advantages - values  # Critic loss

            # Flatten batches (get rid of env-index and step index).
            # After flattening, all arrays should be in the same order, but no longer in step-wise order.
            # This is ok, because we already calculated the advantages based on future states.
            b_obs = obs.reshape((-1,) + self.env.observation_space.shape)
            b_log_probs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + (self.action_space,))  # TODO: Handle continuous actions
            b_advantages = advantages.reshape(-1)
            # b_returns = returns.reshape(-1)  # TODO: What is this?
            b_values = values.reshape(-1)

            # Optimize
            b_indices = np.arange(config["batch_size"])
            for epoch in config["num_epoch"]:
                np.random.shuffle(b_indices)
                batch_size = self.config["batch_size"]
                for start in range(0, steps_per_env, batch_size):
                    # Get batch from indices
                    end = start + batch_size
                    indices = b_indices[start:end]

                    # Get new predictions for current batch
                    # - calculate "ratio" between new and old predictions
                    # - TODO: Where is this in PPO paper? (ANS: original TRPO surrogate objective)
                    _, new_log_probs = self.agent.get_action(b_obs[indices], b_actions[indices])
                    new_values = self.agent.get_value(b_obs[indices])
                    ratio = torch.exp(new_log_probs - b_log_probs[indices])
                    print("succ")
                    return

                    #
                    # 3 Clipped Surrogate Objective
                    # TRPO uses the following:
                    # - L_CPI = Et [ratio] * Advantage_t
                    #       - Where ratio = pi_new(a|t) / pi_old(a|t)
                    #
                    # PPO paper proposes a slight modification (adds clipping):
                    # - L_CLIP = Et [min(ratio * advantage_t, clip(ratio) * advantage_t)]

                    # 4 Adaptive KL penalty?
                    # - PPO paper: "Alternative or addition to Clipped Surrogate Objective"

                    # Clean-RL uses "Monte-Carlo Approximation of KL Divergence"
                    # - taken from http://joschu.net/blog/kl-approx.html

                    pass

            # Evaluate maybe?
        # Final evaluate maybe?
        env_id.close()


# %%
# test_env()
# test_eval()
ppo = PPO(config)
ppo.train()

# %%


# Train
