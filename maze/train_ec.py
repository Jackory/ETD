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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pickle

params = {
    "legend.fontsize": "large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
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
    method: str = "cmd1" # cmd1, cmd2
    env_id: str = "SpiralMaze"
    """the id of the environment"""
    add_noise: bool = False
    """if toggled, add random noise to the walls of the maze"""
    total_timesteps: int = int(50000)
    logging_frequency: int = 100
    batch_size: int = 512
    discount: float = 0.99
    repr_dim: int = 64
    n_samples: int = 50000
    reg_coef: float = 0


def discounted_sampling(ranges, discount):
    # k = 2 in EC https://arxiv.org/pdf/1810.02274
    samples = torch.bernoulli(torch.full_like(ranges, 0.5, dtype=torch.float32))
    samples = torch.min(torch.floor(samples).long(), ranges - 1)
    return samples


class QuasimetricNet(nn.Module):
    def __init__(self):
        super(QuasimetricNet, self).__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )

        self.quasimetric_hidden_dim = 256

        self.encoder = nn.Sequential(
            nn.Linear(64 * 17  * 17, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * self.quasimetric_hidden_dim)
        )

        self.lam = nn.Parameter(torch.tensor([1.0], requires_grad=True))


    def forward(self, obs: torch.Tensor, goal: torch.Tensor):
        # obs/goal: [N, H, W, C] -> [N, C, H, W] -> [N, D] -> [N, 1, D]/[1, N, D]
        obs = obs.permute(0, 3, 1, 2)
        goal = goal.permute(0, 3, 1, 2)
        state = self.img_encoder(obs)[:, None]
        goal = self.img_encoder(goal)[None]

        # Metric Residual Network (MRN) architecture (https://arxiv.org/pdf/2208.08133)
        x_sym, x_asym = torch.split(self.encoder(state), [self.quasimetric_hidden_dim] * 2, dim=-1)
        y_sym, y_asym = torch.split(self.encoder(goal), [self.quasimetric_hidden_dim] * 2, dim=-1)
        d_sym = torch.sqrt(torch.sum(torch.square(x_sym - y_sym) + 1e-8, dim=-1))
        d_asym = torch.max(torch.relu(x_asym - y_asym), dim=-1).values
        dist = d_sym + d_asym

        return dist


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

    # Parse arguments

    args = tyro.cli(Args)
    if args.add_noise:
        run_name = f"pixel_new{args.n_samples}{args.env_id}Noise__{args.exp_name}__{args.seed}"
    else:
        run_name = f"pixel_new{args.n_samples}{args.env_id}__{args.exp_name}__{args.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Set random seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Create environment

    env = gridworld_envs.make("SpiralMaze", add_noise=args.add_noise,  episode_len=50, start_pos='random')

    # Create network & optimizer

    quasimetric_net = QuasimetricNet().to(device)
    optimizer = optim.Adam([{'params': quasimetric_net.parameters(), 'lr': 3e-4}])

    # Create logger
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/{run_name}/{timestamp}"
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Load data

    args.data_path = f"runs/{run_name}/data.pkl"
    if os.path.exists(args.data_path):
        with open(args.data_path, "rb") as f:
            data = pickle.load(f)
        print("Use collected data")
    else:
        print("Collecting data ...")
        data = collect_data(env, n_samples=args.n_samples)
        print(f"Data collection done! # of data: {args.n_samples}.")
        print(f"Saving to {args.data_path}")
        with open(args.data_path, "wb") as f:
            pickle.dump(data, f)

    # Preprossing data

    obss = np.array([[step["obs"] for step in traj] for traj in data])
    obss = torch.from_numpy(obss)
    next_obss = np.array([[step["next_obs"] for step in traj] for traj in data])
    next_obss = torch.from_numpy(next_obss)
    actions = np.array([[step["action"] for step in traj] for traj in data])
    actions = torch.from_numpy(actions)
    n_trajs, n_steps = obss.shape[0], obss.shape[1]

    obss = obss.to(device).to(torch.float32)
    next_obss = next_obss.to(device).to(torch.float32)
    actions = actions.to(device).to(torch.float32)

    # Start training

    print("Training")
    metrics = {}
    start_time = time.time()
    for i in range(args.total_timesteps):

        # Sample mini-batch data (positive pairs)

        traj_idx = torch.randint(n_trajs, (args.batch_size,), device=device)
        step_idx = torch.randint(n_steps, (args.batch_size,), device=device)
        intervals = discounted_sampling(
            n_steps - step_idx, args.discount
        )  # Sample from 0 ~ n-1
        obs = obss[traj_idx, step_idx]
        goal = next_obss[traj_idx, step_idx + intervals]
        action = actions[traj_idx, step_idx]
        future_action = actions[traj_idx, step_idx + intervals]

        # Contrastive loss

        d_sg = quasimetric_net(obs, goal)   # [N, N]
        logits = d_sg               # logits[i, j] = d_sg[i, j] - c_g[j]
        I = torch.eye(args.batch_size, device=device)
        contrastive_loss = F.cross_entropy(logits, I).mean()
        logsumexp = torch.mean((torch.logsumexp(logits, axis=1)**2))
        # Backprop

        loss = contrastive_loss + args.reg_coef * logsumexp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate gap to optimal & Logging

        if i % args.logging_frequency == 0:
            metrics['contrastive/contrastive_loss'] = contrastive_loss.item()
            metrics['contrastive/categorical_accuracy'] = torch.mean((torch.argmax(logits, axis=1) ==  
                                                          torch.arange(args.batch_size,device=device)).float()).item()
            metrics['contrastive/logits_pos'] = torch.diag(logits).mean()
            metrics['contrastive/logits_neg'] = torch.mean(logits * (1 - I))
            metrics['contrastive/logits_logsumexp'] = torch.mean((torch.logsumexp(logits, axis=1)**2))
            metrics['quasimetric/d_sg_pos'] = torch.diag(d_sg).mean()
            metrics['quasimetric/d_sg_neg'] = torch.mean(d_sg * (1 - I))
            
            for k, v in metrics.items():
                writer.add_scalar(k, v, i)
            end_time = time.time()
            print(f"Step {i} contrastive_loss {contrastive_loss.item():.3f} time {end_time - start_time:.3f}")
            start_time = end_time
            # plot l2 distance to the goal
            n_states = env.task.maze.n_states
            pos_batch = env.task.maze.all_empty_grids()
            obs_batch = np.array([env.task.pos_to_obs(pos_batch[i]).agent.image for i in range(n_states)])
            goal_pos = np.array([1, 1])
            goal_obs = env.task.pos_to_obs(goal_pos)
            goal_batch = np.array([env.task.pos_to_obs(goal_pos).agent.image])

            with torch.no_grad():
                dists = -torch.log(quasimetric_net(
                    torch.from_numpy(obs_batch).to(device).to(torch.float32),
                    torch.from_numpy(goal_batch).to(device).to(torch.float32)
                )+1)    # shape: [n_states, 1]
            image_shape = goal_obs.agent.image.shape
            map_ = np.zeros(image_shape[:2], dtype=np.float32)
            map_[pos_batch[:, 0], pos_batch[:, 1]] = dists.detach().cpu().squeeze()
            plt.figure()
            plt.imshow(goal_obs.agent.image)
            plt.savefig("hardmaze_goal.png")
            plt.close()
            plt.figure(figsize=(6,6))
            im_ = plt.imshow(map_, interpolation="none", cmap="Blues")
            # cbar = plt.colorbar()
            # cbar.set_ticks([])
            # add the walls to the normalized distance plot
            walls = np.expand_dims(env.task.maze.render(), axis=-1)
            map_2 = im_.cmap(im_.norm(map_))
            map_2[:, :, :-1] = map_2[:, :, :-1] * (1 - walls) + 0.5 * walls
            map_2[:, :, -1:] = map_2[:, :, -1:] * (1 - walls) + 1.0 * walls
            map_2[goal_pos[0], goal_pos[1]] = [1, 0, 0, 1]
            plt.cla()
            plt.imshow(map_2, interpolation="none")
            plt.xticks([])
            plt.yticks([])
            plt.savefig("SpiralMaze_EC.png", bbox_inches="tight", dpi=300)
            plt.savefig(f"{run_dir}/SpiralMaze.pdf", bbox_inches="tight")
            plt.close()
