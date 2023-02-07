import argparse
import os
import random
import time

import gym
import gym_yoli
import mlflow
import numpy as np
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical

def s2b(val):
    return val.lower() in ('yes', 'true', 't', '1')

def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="gym_yoli/YoliGame-v0", help="the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25_000, help="total timesteps of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x:  bool(s2b(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backend.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(s2b(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--mlflow-server", type=str, default="res56.itu.dk", help="the host for a mlflow server")
    parser.add_argument("--capture-video", type=lambda x: bool(s2b(x)), default=False, nargs="?", const=True,
        help="capture videos of the agent performances (stored in `videos`)")

    parser.add_argument("--num-envs", type=int, default=4, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(s2b(x)), default=True, nargs='?', const=True, help="toggle learning rate annealing for policy and value networks")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    return args

if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.mlflow_server is not None:
        mlflow.set_tracking_uri(f"http://{args.mlflow_server}:80")
    mlflow.set_experiment(args.exp_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        def make_env(gym_id, seed, idx, capture_video, run_name):
            def thunk():
                env = gym.make(gym_id)
                env = gym.wrappers.RecordEpisodeStatistics(env)
                if capture_video:
                    if idx == 0:
                        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
                env.seed(seed)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        def layer_init(layer, std=np.sqrt(2), bias_const=0.0) -> None:
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)
            return layer
        
        class Agent(nn.Module):
            def __init__(self, envs):
                super(Agent, self).__init__()
                self.critic = nn.Sequential(
                    layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 1), std=1.),
                )
                self.actor = nn.Sequential(
                    layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
                )

            def get_value(self, x):
                return self.critic(x)

            def get_action_and_value(self, x, action=None):
                logits = self.actor(x)
                probs = Categorical(logits=logits)
                if action is None:
                    action = probs.sample()
                return action, probs.log_prob(action), probs.entropy(), self.critic(x)

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        num_updates = args.total_timesteps // args.batch_size
        
        for update in range(1, num_updates + 1):
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                new_learning_rate = frac * args.learning_rate
                optimizer.param_groups[0]['lr'] = new_learning_rate
            
            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Execute the game and log data
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        mlflow.log_metric("episodic return", item["episode"]["r"], global_step)
                        mlflow.log_metric("episodic length", item["episode"]["l"], global_step)
                        
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps -1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                

        # mlflow.log_artifacts("videos")