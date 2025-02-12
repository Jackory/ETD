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
    method: str = "cmd1" # cmd1
    env_id: str = "SpiralMaze"
    """the id of the environment"""
    add_noise: bool = False
    """if toggled, add random noise to the walls of the maze"""
    total_timesteps: int = int(5000)
    logging_frequency: int = 100
    batch_size: int = 512
    discount: float = 0.99
    repr_dim: int = 64
    n_samples: int = 50000
    logsumexp_coef: float = 0
    temperature = 1
    energy_fn: str = "mrn_pot"
    loss_fn: str = "infonce_symmetric"
    lr: float = 3e-4


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


def mrn_distance(x, y):
    eps = 1e-6
    d = x.shape[-1]
    x_prefix = x[..., :d // 2]
    x_suffix = x[..., d // 2:]
    y_prefix = y[..., :d // 2]
    y_suffix = y[..., d // 2:]
    max_component = torch.max(F.relu(x_prefix - y_prefix), axis=-1).values
    l2_component = torch.sqrt(torch.square(x_suffix - y_suffix).sum(axis=-1) + eps)
    return max_component + l2_component


class PotentialNet(nn.Module):
    def __init__(self):
        super(PotentialNet, self).__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )

        self.value = nn.Sequential(
            nn.Linear(64 * 17 * 17, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, obs: torch.Tensor):
        # obs: [N, H, W, C] -> [N, C, H, W] -> [N, 1]
        obs = obs.permute(0, 3, 1, 2)
        state = self.img_encoder(obs)
        value = self.value(state)
        return value


class S_Encoder(nn.Module):
    def __init__(self):
        super(S_Encoder, self).__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )
        self.value = nn.Sequential(
            nn.Linear(64 * 17 * 17, 1024),
            # nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            # nn.LayerNorm(1024),
            nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.LayerNorm(1024),
            nn.Linear(1024, 64),
        )

    def forward(self, obs: torch.Tensor):
        # obs: [N, H, W, C] -> [N, C, H, W] -> [N, 1]
        obs = obs.permute(0, 3, 1, 2)
        state = self.img_encoder(obs)
        value = self.value(state)
        return value


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
        run_name = f"CMD1_{args.n_samples}{args.env_id}Noise__{args.exp_name}__{args.seed}"
    else:
        run_name = f"CMD1_{args.n_samples}{args.env_id}__{args.exp_name}__{args.seed}"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Set random seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Create environment
    env = gridworld_envs.make(f"{args.env_id}", add_noise=args.add_noise,  episode_len=50, start_pos='random')

    # Create network & optimizer

    potential_net = PotentialNet().to(device)
    s_encoder = S_Encoder().to(device)
    g_encoder = S_Encoder().to(device)
    optimizer = optim.Adam([{'params': potential_net.parameters(), 'lr': args.lr},
                            {'params': s_encoder.parameters(), 'lr': args.lr},
                            {'params': g_encoder.parameters(), 'lr': args.lr}])

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

        c_g = potential_net(goal)           # [N, 1]
        phi_s = s_encoder(obs)
        phi_g = s_encoder(goal)
        
        if args.energy_fn == 'l2':
            logits = - torch.sqrt(((phi_s[:, None] - phi_g[None, :])**2).sum(dim=-1) + 1e-8)
        elif args.energy_fn == 'cos':
            s_norm = torch.linalg.norm(phi_s, axis=-1, keepdims=True)
            g_norm = torch.linalg.norm(phi_g, axis=-1, keepdims=True)
            phi_s_norm = phi_s / s_norm
            phi_g_norm = phi_g / g_norm
            
            phi_s_norm = phi_s_norm / args.temperature
            logits = torch.einsum("ik,jk->ij", phi_s_norm, phi_g_norm)
        elif args.energy_fn == 'mrn':
            logits = - mrn_distance(phi_s[:, None], phi_g[None, :])
        elif args.energy_fn == 'mrn_pot':
            logits = c_g.T - mrn_distance(phi_s[:, None], phi_g[None, :])
        elif args.energy_fn == 'dot':
            logits = torch.einsum("ik,jk->ij", phi_s, phi_g)

        I = torch.eye(args.batch_size, device=device)
        if args.loss_fn == 'infonce':
            contrastive_loss = F.cross_entropy(logits, I)
        elif args.loss_fn == 'infonce_backward':
            contrastive_loss = F.cross_entropy(logits.T, I)
        elif args.loss_fn == 'infonce_symmetric':
            contrastive_loss = (F.cross_entropy(logits, I) + F.cross_entropy(logits.T, I)) / 2
        elif args.loss_fn == 'dpo':
            positive = torch.diag(logits)
            diffs = positive[:, None] - logits
            contrastive_loss = -F.logsigmoid(diffs)

        contrastive_loss = torch.mean(contrastive_loss)
        logsumexp = torch.mean((torch.logsumexp(logits + 1e-6, axis=1)**2))
        # Backprop
        loss = contrastive_loss + args.logsumexp_coef * logsumexp
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
            metrics['contrastive/c_g_pos'] = torch.diag(c_g).mean()
            metrics['contrastive/c_g_neg'] = torch.mean(c_g * (1 - I))
            
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
            goal_batch = np.array([env.task.pos_to_obs(goal_pos).agent.image] * n_states)

            with torch.no_grad():
                s = torch.from_numpy(obs_batch).to(device).to(torch.float32)
                g = torch.from_numpy(goal_batch).to(device).to(torch.float32)
                phi_s = s_encoder(s)
                phi_g = s_encoder(g)
                
                if args.energy_fn == 'l2':
                    dists = ((phi_s[:, None] - phi_g[None, :])**2).sum(dim=-1).sqrt()
                elif args.energy_fn == 'cos':
                    s_norm = torch.linalg.norm(phi_s, axis=-1, keepdims=True)
                    g_norm = torch.linalg.norm(phi_g, axis=-1, keepdims=True)
                    phi_s_norm = phi_s / s_norm
                    phi_g_norm = phi_g / g_norm
                    
                    phi_s_norm = phi_s_norm / args.temperature
                    dists = - torch.einsum("ik,jk->ij", phi_s_norm, phi_g_norm)
                elif args.energy_fn == 'dot':
                    dists = - torch.einsum("ik,jk->ij", phi_s, phi_g)
                elif 'mrn' in args.energy_fn:
                    dists = mrn_distance(phi_s[:, None], phi_g[None, :])
                    
            image_shape = goal_obs.agent.image.shape
            map_ = np.zeros(image_shape[:2], dtype=np.float32)
            map_[pos_batch[:, 0], pos_batch[:, 1]] = torch.diag(dists).detach().cpu().squeeze()
            plt.figure()
            plt.imshow(goal_obs.agent.image)
            plt.savefig("hardmaze_goal.png")
            plt.close()
            plt.figure(figsize=(6,6))
            im_ = plt.imshow(map_, interpolation="none", cmap="Blues")
            cbar = plt.colorbar()
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
            if not args.add_noise:
                plt.savefig(f"{args.env_id}_{args.energy_fn}_{args.loss_fn}.png", bbox_inches="tight", dpi=300)
            else:
                plt.savefig(f"{args.env_id}Noise_{args.energy_fn}_{args.loss_fn}.png", bbox_inches="tight", dpi=300)
            plt.savefig(f"{run_dir}/{args.env_id}.pdf", bbox_inches="tight")
            plt.close()
