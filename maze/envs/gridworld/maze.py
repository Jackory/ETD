import numpy as np
import scipy.sparse.csgraph as csgraph

DEFAULT_MAZE = '''
+-----+
|     |
|     |
|     |
|     |
|     |
+-----+
'''


HARD_MAZE = '''
+--------+-----+
|              |
|              |
+-----+  +-----+
|     |        |
|     |        |
|  +--+-  --+--+
|              |
|              |
|  +  +  +-----+
|  |  |  |     |
|  |  |  |     |
|  +--+  +---  |
|     |        |
|     |        |
+-----+--------+
'''

HARD_MAZE_SMALL = '''
+--------+-----+
|              |
+----+---+---+ |
|    |   |   | |
|  + | + | + | | 
|  | | | | | | |
|  | | | | | | |
|  | | | | | | |
|  | | | | | | |
|  | | | | | | |
|  | | | | | | |
|  | | | | | | |
|  | + | | | + |
|  |   |   |   |
|  |   |   |   |
+-----+----+---+
'''


SPIRAL_MAZE = '''
+---------------+
|               |
+-------------+ |
|             | |
| +---------+ | | 
| |         | | |
| | +-----+ | | |
| | |     | | | |
| | | +-+ | | | |
| | | |   | | | |
| | | +---+ | | |
| | |       | | |
| | +-------+ | |
| |           | |
| +-----------+ |
|               |
+---------------+
'''

SPIRAL_MAZE_LARGE = '''
+----------------------+
|                      |
|                      |
+-------------------+  |
|                   |  |
|                   |  |
|  +-------------+  |  | 
|  |             |  |  |
|  |             |  |  |
|  |  +-------+  |  |  |
|  |  |       |  |  |  |
|  |  |       |  |  |  |
|  |  |  +-+  |  |  |  |
|  |  |  +    |  |  |  |
|  |  |  +----+  |  |  |
|  |  |          |  |  |
|  |  |          |  |  |
|  |  +----------+  |  |
|  |                |  |
|  |                |  |
|  +----------------+  |
|                      |
|                      |
+----------------------+
'''

LOOP_MAZE = '''
+-------+-------+
|       |       |
|   +---+---+   |
|   |       |   |
+---+   +   +---+
|       |       |
|   +---+---+   |
|   |       |   |
+-------+-------+
'''

INTERLOCKED_MAZE = '''
+-------------+-------------+
|             |             |
|   +-----+   |   +     +   |
|   |     |   |   |     |   |
|   | +---+   +---+---+ |   |
|   | |               | |   |
+---+ |   +-------+   | +---+
|   | |   |       |   | |   |
|   | +---+   +---+   + |   |
|   |       |       |     | |
| +-+       +-------+-----+-+
|                           |
|   +-----+   +   +     +   |
|   |     |   |   |     |   |
|   |     +---+---+---+ |   |
|   |                 | |   |
| +-+     +-------+   | +---+
|   |     |       |   | |   |
|   |     +   +---+   + |   |
|   |                       |
+-------+-------+-------+---+
'''


class MazeFactoryBase:
    def __init__(self, maze_str=DEFAULT_MAZE):
        self._maze = self._parse_maze(maze_str)

    def _parse_maze(self, maze_source):
        width = 0
        height = 0
        maze_matrix = []
        for row in maze_source.strip().split('\n'):
            row_vector = row.strip()
            maze_matrix.append(row_vector)
            height += 1
            width = max(width, len(row_vector))
        maze_array = np.zeros([height, width], dtype=str)
        maze_array[:] = ' '
        for i, row in enumerate(maze_matrix):
            for j, val in enumerate(row):
                maze_array[i, j] = val
        return maze_array

    def get_maze(self):
        return self._maze


class SquareRoomFactory(MazeFactoryBase):
    """generate a square room with given size"""
    def __init__(self, size):
        maze_array = np.zeros([size+2, size+2], dtype=str)
        maze_array[:] = ' '
        maze_array[0] = '-'
        maze_array[-1] = '-'
        maze_array[:, 0] = '|'
        maze_array[:, -1] = '|'
        maze_array[0, 0] = '+'
        maze_array[0, -1] = '+'
        maze_array[-1, 0] = '+'
        maze_array[-1, -1] = '+'
        self._maze = maze_array


class FourRoomsFactory(MazeFactoryBase):
    """generate four rooms, each with the given size"""
    def __init__(self, size):
        maze_array = np.zeros([size*2+3, size*2+3], dtype=str)
        maze_array[:] = ' '
        wall_idx = [0, size+1, size*2+2]
        maze_array[wall_idx] = '-'
        maze_array[:, wall_idx] = '|'
        maze_array[wall_idx][:, wall_idx] = '+'
        door_idx = [int((size+1)/2), int((size+1)/2)+1, 
                int((size+1)/2)+size+1, int((size+1)/2)+size+2]
        maze_array[size+1, door_idx] = ' '
        maze_array[door_idx, size+1] = ' '
        self._maze = maze_array


class TwoRoomsFactory(MazeFactoryBase):
    def __init__(self, size):
        maze_array = np.zeros([size+2, size+2], dtype=str)
        maze_array[:] = ' '
        hwall_idx = [0, int((size+1)/2), size+1]
        vwall_idx = [0, size+1]
        maze_array[hwall_idx] = '-'
        maze_array[:, vwall_idx] = '|'
        maze_array[hwall_idx][:, vwall_idx] = '+'
        door_idx = [int((size+1)/2), int((size+1)/2)+1]
        maze_array[hwall_idx[1], door_idx] = ' '
        self._maze = maze_array


class Maze:
    def __init__(self, maze_factory):
        self._maze_factory = maze_factory
        # parse maze ...
        self._maze = None
        self._height = None
        self._width = None
        self._build_maze()
        self._all_empty_grids = np.argwhere(self._maze==' ')
        self._n_states = self._all_empty_grids.shape[0]
        self._pos_indices = {}
        for i, pos in enumerate(self._all_empty_grids):
            self._pos_indices[tuple(pos)] = i

    def _build_maze(self):
        self._maze = self._maze_factory.get_maze()
        self._height = self._maze.shape[0]
        self._width = self._maze.shape[1]

    def rebuild(self):
        self._build_maze()

    def __getitem__(self, key):
        return self._maze[key]

    def __setitem__(self, key, val):
        self._maze[key] = val

    def is_empty(self, pos):
        if (pos[0] >= 0 and pos[0] < self._height 
                and pos[1] >= 0 and pos[1] < self._width):
            return self._maze[tuple(pos)] == ' '
        else:
            return False
    
    @property
    def maze_array(self):
        return self._maze

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def n_states(self):
        return self._n_states

    def pos_index(self, pos):
        return self._pos_indices[tuple(pos)]

    def all_empty_grids(self):
        return np.argwhere(self._maze==' ')

    def random_empty_grids(self, k):
        '''Return k random empty positions.'''
        empty_grids = np.argwhere(self._maze==' ')
        selected = np.random.choice(
                np.arange(empty_grids.shape[0]),
                size=k,
                replace=False
                )
        return empty_grids[selected]

    def first_empty_grid(self):
        empty_grids = np.argwhere(self._maze==' ')
        assert empty_grids.shape[0] > 0
        return empty_grids[0]

    def render(self):
        # 0 for ground, 1 for wall
        return (self._maze!=' ').astype(np.float32)

    def compute_distances(self):
        graph = np.full((self._n_states, self._n_states), np.inf)
        for i, pos1 in enumerate(self._all_empty_grids):
            for j, pos2 in enumerate(self._all_empty_grids):
                if self.is_adjacent(pos1, pos2):
                    graph[i, j] = 1
        graph[np.arange(self._n_states), np.arange(self._n_states)] = 0

        shortest_paths = csgraph.floyd_warshall(graph)

        euclidean_distances = np.zeros_like(shortest_paths)
        for i in range(self._n_states):
            for j in range(self._n_states):
                pos1 = self._all_empty_grids[i]
                pos2 = self._all_empty_grids[j]
                euclidean_distances[i, j] = np.sqrt(np.sum((pos1 - pos2) ** 2))

        return shortest_paths, euclidean_distances

    def is_adjacent(self, pos1, pos2):
        # 检查两个位置是否相邻
        deltas = np.abs(pos1 - pos2)
        return np.all(deltas <= 1) and not np.all(deltas == 1)


    def get_shortest_path(self, start, end):
        shortest_paths, _ = self.compute_distances()
        start_idx = self.pos_index(start)
        end_idx = self.pos_index(end)
        path_length = shortest_paths[start_idx, end_idx]
        if np.isinf(path_length):
            return None  # No path exists
        
        # Reconstruct the path
        path = []
        current_idx = start_idx
        while current_idx != end_idx:
            path.append(self._all_empty_grids[current_idx])
            neighbors = np.where(shortest_paths[current_idx] == 1)[0]
            for neighbor in neighbors:
                if shortest_paths[neighbor, end_idx] < shortest_paths[current_idx, end_idx]:
                    current_idx = neighbor
                    break
        path.append(self._all_empty_grids[end_idx])
        return path