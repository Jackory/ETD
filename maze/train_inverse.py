# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from envs.gridworld import gridworld_envs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pickle
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_id: str = "SpiralMaze"
    """the id of the environment"""
    add_noise: bool = False
    """if toggled, add random noise to the walls of the maze"""
    total_timesteps: int = int(1e5)
    logging_frequency: int = 100
    batch_size: int = 1024
    discount: float = 0
    "using for discount sampling"
    t: float = 0.1
    "temperature for global distance" 
    

def discounted_sampling(ranges, discount):
    assert 0 <= discount <= 1
    seeds = torch.rand(size=ranges.shape, device=ranges.device)
    if discount == 0:
        samples = torch.zeros_like(seeds, dtype=ranges.dtype)
    elif discount == 1:
        samples = torch.floor(seeds * ranges).int()
    else:
        samples = torch.log(1 - (1 - discount**ranges) * seeds) / np.log(discount)
        samples = torch.min(torch.floor(samples).long(), ranges - 1)
    return samples

    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        feat_dim = 50
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, feat_dim)
        self.forward_net = nn.Sequential(
            nn.Linear(feat_dim + 1, 128), nn.ReLU(),
            nn.Linear(128, feat_dim)
        )
        self.backward_net = nn.Sequential(
            nn.Linear(feat_dim*2, 128), nn.ReLU(),
            nn.Linear(128, 5), nn.Softmax(dim=-1)
        )
        

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.Tanh()(x)
        return x


def collect_data(env, n_samples=10000):
    data = []  # All trajectories
    traj = []  # Transitions in a single trajectory
    obs, info = env.reset()
    for _ in range(n_samples):
        action = env.action_space.sample()
        next_obs, rewards, terminations, truncations, infos = env.step(action)
        if terminations or truncations:
            # At the end of an episode
            data.append(traj)
            traj = []
            obs, info = env.reset()
        else:
            # In the middle of an episode
            traj.append({"obs": obs, "next_obs": next_obs, "action": action})
            obs = next_obs
    return data
    
    
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.add_noise:
        run_name = f"pixel_{args.env_id}Noise__{args.exp_name}__{args.seed}"
    else:
        run_name = f"pixel5000_{args.env_id}__{args.exp_name}__{args.seed}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    env = gridworld_envs.make("SpiralMaze", add_noise=args.add_noise)
    # Load or collect data

    args.data_path = f"runs/{run_name}/data.pkl"
    if os.path.exists(args.data_path):
        with open(args.data_path, "rb") as f:
            data = pickle.load(f)
        print("Use collected data")
    else:
        print("Collecting data ...")
        data = collect_data(env, n_samples=int(5000))
        print(f"Data collection done! # of data: 100000.")
        print(f"Saving to {args.data_path}")
        with open(args.data_path, "wb") as f:
            pickle.dump(data, f)
            
    obs = np.array([[step["obs"] for step in traj] for traj in data])
    obs = torch.from_numpy(obs)
    next_obs = np.array([[step["next_obs"] for step in traj] for traj in data])
    next_obs = torch.from_numpy(next_obs)
    actions = np.array([[step["action"] for step in traj] for traj in data])
    actions = torch.from_numpy(actions)
    n_trajs, n_steps = obs.shape[0], obs.shape[1]
    
    obs = obs.to(device).to(torch.float32)
    next_obs = next_obs.to(device).to(torch.float32)
    actions = actions.to(device).to(torch.float32)
    
    net = ConvNet().to(device)
    target_net = ConvNet().to(device)
    target_net.load_state_dict(net.state_dict())
    rep_optimizer = optim.Adam(net.parameters(), lr=1e-5)
    lag = torch.tensor([0.000], requires_grad=True, device=device)
    lag_optimizer = optim.Adam([lag], lr=1e-3)
    print("Training")
    loss_dict = {"loss": 0, "local_distance": 0, "global_distance": 0, "lag":0}  

    for i in range(args.total_timesteps):
        # Sample mini-batch data (positive pairs)
        traj_idx = torch.randint(n_trajs, (args.batch_size,), device=device)
        step_idx = torch.randint(n_steps, (args.batch_size,), device=device)
        intervals = discounted_sampling(n_steps - step_idx, args.discount)  # Sample from 0 ~ n-1
        pos_obs_i = obs[traj_idx, step_idx]
        pos_obs_j = next_obs[traj_idx, step_idx + intervals]
        action = actions[traj_idx, step_idx]

        # # Forward
        pos_rep_i = net(pos_obs_i)
        pos_rep_j = net(pos_obs_j)

        # next_obs_hat = net.forward_net(torch.cat([pos_rep_i, action.unsqueeze(-1)], dim=-1))
        action_hat = net.backward_net(torch.cat([pos_rep_i, pos_rep_j], dim=-1))
        # forward_loss = F.mse_loss(next_obs_hat, pos_rep_j)
        backward_loss = F.cross_entropy(action_hat, action.view(-1).long())
        
        rep_optimizer.zero_grad()
        backward_loss.backward()
        rep_optimizer.step()
        
    
        # Calculate gap to optimal & Logging
        
        if i % args.logging_frequency == 0:
            # torch.save(net.state_dict(), "metra_inverse.pt")
            writer.add_scalar("losses/loss", backward_loss.item(), i)
            print(f"Step {i} loss {backward_loss.item()}")
            
            # plot l2 distance to the goal
            n_states = env.task.maze.n_states
            pos_batch = env.task.maze.all_empty_grids()
            obs_batch = np.array([env.task.pos_to_obs(pos_batch[i]).agent.image for i in range(n_states)])
            goal_pos = np.array([1, 1])
            goal_obs = env.task.pos_to_obs(goal_pos)
            goal_state = goal_obs.agent.image
            
            with torch.no_grad():
                obs_batch = net(torch.from_numpy(obs_batch).to(device).to(torch.float32))
                # compute l2 ditance with goal
                obs_batch = (obs_batch).detach().cpu().numpy()
                goal_state = net(torch.from_numpy(goal_state).unsqueeze(0).to(device).to(torch.float32))
                goal_state = (goal_state).detach().cpu().numpy()
                
                l2_dists = np.sqrt(((obs_batch - goal_state)**2).sum(axis=-1))
            image_shape = goal_obs.agent.image.shape
            map_ = np.zeros(image_shape[:2], dtype=np.float32)
            map_[pos_batch[:, 0], pos_batch[:, 1]] = l2_dists
            plt.figure()
            plt.imshow(goal_obs.agent.image)
            plt.savefig('hardmaze_goal.png')
            plt.close()
            plt.figure(figsize=(6,6))
            im_ = plt.imshow(map_, interpolation='none', cmap='Blues')
            # plt.colorbar()
            # add the walls to the normalized distance plot
            walls = np.expand_dims(env.task.maze.render(), axis=-1)
            map_2 = im_.cmap(im_.norm(map_))
            map_2[:, :, :-1] = map_2[:, :, :-1] * (1 - walls) + 0.5 * walls
            map_2[:, :, -1:] = map_2[:, :, -1:] * (1 - walls) + 1.0 * walls
            map_2[goal_pos[0], goal_pos[1]] = [1, 0, 0, 1]
            plt.cla()
            plt.imshow(map_2, interpolation='none')
            plt.xticks([])
            plt.yticks([])
            plt.savefig("sprial_inverse.png", bbox_inches='tight', dpi=300)
            plt.close()