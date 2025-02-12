import numpy as np
from . import maze
from . import maze2d_single_goal
from .. import env_base


ONE_ROOM_MAZE = maze.Maze(maze.SquareRoomFactory(size=15))
ONE_ROOM_GOAL_POS = np.array([15, 15])
TWO_ROOM_MAZE = maze.Maze(maze.TwoRoomsFactory(size=15))
TWO_ROOM_GOAL_POS = np.array([9, 15])
HARD_MAZE = maze.Maze(maze.MazeFactoryBase(maze_str=maze.HARD_MAZE))
HARD_MAZE_GOAL_POS = np.array([10, 10])
HARD_MAZE_SMALL = maze.Maze(maze.MazeFactoryBase(maze_str=maze.HARD_MAZE_SMALL))
HARD_MAZE_SAMLL_POS = np.array([10, 1])
SPIRAL_MAZE = maze.Maze(maze.MazeFactoryBase(maze_str=maze.SPIRAL_MAZE))
SPIRAL_MAZE_POS = np.array([9, 7])
INTERLOCKED_MAZE = maze.Maze(maze.MazeFactoryBase(maze_str=maze.INTERLOCKED_MAZE))
INTERLOCKED_MAZE_POS = np.array([1, 1])


class OneRoomEnv(env_base.Environment):
    def __init__(self):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=ONE_ROOM_MAZE,
                episode_len=500,
                start_pos='first',
                use_stay_action=True,
                reward_type='neg',
                goal_pos=ONE_ROOM_GOAL_POS,
                end_at_goal=False)
        super().__init__(task)


class TwoRoomEnv(env_base.Environment):
    def __init__(self):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=TWO_ROOM_MAZE,
                episode_len=500,
                start_pos='first',
                use_stay_action=True,
                reward_type='pos',
                goal_pos=TWO_ROOM_GOAL_POS,
                end_at_goal=False)
        super().__init__(task)


class HardMazeEnv(env_base.Environment):
    def __init__(self):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=HARD_MAZE,
                episode_len=50,
                start_pos='random',
                use_stay_action=True,
                reward_type='neg',
                goal_pos=HARD_MAZE_GOAL_POS,
                end_at_goal=False)
        super().__init__(task)

class HardMazeSmallEnv(env_base.Environment):
    def __init__(self, add_noise=False, episode_len=50, end_at_goal=False, start_pos='random'):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=HARD_MAZE_SMALL,
                episode_len=episode_len,
                start_pos=start_pos,
                use_stay_action=False,
                reward_type='pos',
                goal_pos=HARD_MAZE_SAMLL_POS,
                end_at_goal=end_at_goal,
                add_noise=add_noise)
        super().__init__(task)

class SPIRALMAZEEnv(env_base.Environment):
    def __init__(self, add_noise=False, episode_len=500, end_at_goal=False, start_pos='random'):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=SPIRAL_MAZE,
                episode_len=episode_len,
                start_pos=start_pos,
                use_stay_action=False,
                reward_type='pos',
                goal_pos=SPIRAL_MAZE_POS,
                end_at_goal=end_at_goal,
                add_noise=add_noise)
        super().__init__(task)

class InterlockedMazeEnv(env_base.Environment):
    def __init__(self, add_noise=False, episode_len=500, end_at_goal=False, start_pos='random'):
        task = maze2d_single_goal.Maze2DSingleGoal(
                maze=INTERLOCKED_MAZE,
                episode_len=episode_len,
                start_pos=start_pos,
                use_stay_action=False,
                reward_type='pos',
                end_at_goal=end_at_goal,
                add_noise=add_noise,
                goal_pos=INTERLOCKED_MAZE_POS
                )
        super().__init__(task)


ENV_CLSS = {
    'OneRoom': OneRoomEnv,
    'TwoRoom': TwoRoomEnv,
    'HardMaze': HardMazeEnv,
    'HardMazeSmall': HardMazeSmallEnv,
    'SpiralMaze': SPIRALMAZEEnv,
    'InterlockedMaze': InterlockedMazeEnv
}


def make(env_id, **kwargs):
    return ENV_CLSS[env_id](**kwargs)

