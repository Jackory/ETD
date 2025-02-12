from envs.gridworld import gridworld_envs
import numpy as np
env = gridworld_envs.make("InterlockedMaze")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.animation as animation
import torch
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

num = len(env.task.maze.all_empty_grids())

print(f"{num}")
# print()
temporal_distance, l2_distance = env.task.maze.compute_distances()
short_path = env.task.maze.get_shortest_path((1,1), env.task.goal_pos)

n_states = env.task.maze.n_states
pos_batch = env.task.maze.all_empty_grids()
obs_batch = np.array([env.task.pos_to_obs(pos_batch[i]).agent.position for i in range(n_states)])
# states_batch = np.array([obs_prepro(obs) for obs in obs_batch])
goal_pos = env.task.goal_pos
goal_obs = env.task.pos_to_obs(goal_pos)
# goal_state = goal_obs.agent.position
init_state = np.array([1,1])
init_state_index = env.task.maze._pos_indices[(1,1)]
# l2_dists = np.sqrt(((obs_batch - init_state)**2).sum(axis=-1))
# l2_dists = temporal_distance[init_state_index, :]
image_shape = goal_obs.agent.image.shape
print("image shape, ", image_shape[:2])
map_ = np.zeros(image_shape[:2], dtype=np.float32)
# breakpoint()
map_[pos_batch[:, 0], pos_batch[:, 1]] = temporal_distance[env.task.maze._pos_indices[(1,1)], :]
# for pos in short_path:
#     map_[pos[0], pos[1]] = 1.0
im_ = plt.imshow(map_, interpolation='none', cmap='Blues')
plt.colorbar()
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
plt.savefig("maze_large.png", bbox_inches='tight')