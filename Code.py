import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation
from IPython.display import HTML
from queue import PriorityQueue, deque

# Maze settings
maze_size = 20
maze = []
character = {'x': 1, 'y': 1}
end_point = {'x': 18, 'y': 13}

# Predefined maze
mazes = [
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
]

# Utility functions
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def solve_bfs(maze, start, end):
    queue = deque([(start['x'], start['y'])])
    came_from = { (start['x'], start['y']): None }
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    while queue:
        x, y = queue.popleft()

        if (x, y) == (end['x'], end['y']):
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in came_from and maze[ny][nx] == 0:
                queue.append((nx, ny))
                came_from[(nx, ny)] = (x, y)

    path = []
    current = (end['x'], end['y'])
    while current:
        path.append(current)
        current = came_from.get(current)

    path.reverse()
    return path

def solve_dijkstra(maze, start, end):
    queue = PriorityQueue()
    queue.put((0, start['x'], start['y']))
    came_from = { (start['x'], start['y']): None }
    cost_so_far = { (start['x'], start['y']): 0 }
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    while not queue.empty():
        _, x, y = queue.get()

        if (x, y) == (end['x'], end['y']):
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            new_cost = cost_so_far[(x, y)] + 1

            if (nx, ny) not in cost_so_far and maze[ny][nx] == 0:
                queue.put((new_cost, nx, ny))
                cost_so_far[(nx, ny)] = new_cost
                came_from[(nx, ny)] = (x, y)

    path = []
    current = (end['x'], end['y'])
    while current:
        path.append(current)
        current = came_from.get(current)

    path.reverse()
    return path

def solve_a_star(maze, start, end):
    queue = PriorityQueue()
    queue.put((0, start['x'], start['y']))
    came_from = { (start['x'], start['y']): None }
    cost_so_far = { (start['x'], start['y']): 0 }
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    while not queue.empty():
        _, x, y = queue.get()

        if (x, y) == (end['x'], end['y']):
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            new_cost = cost_so_far[(x, y)] + 1

            if (nx, ny) not in cost_so_far and maze[ny][nx] == 0:
                priority = new_cost + manhattan_distance(nx, ny, end['x'], end['y'])
                queue.put((priority, nx, ny))
                cost_so_far[(nx, ny)] = new_cost
                came_from[(nx, ny)] = (x, y)

    path = []
    current = (end['x'], end['y'])
    while current:
        path.append(current)
        current = came_from.get(current)

    path.reverse()
    return path

# Visualization function
def animate_path(path, maze):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])

    # Color the maze grid
    grid = np.array(maze)
    cmap = sns.color_palette("dark:gray", as_cmap=True)
    ax = sns.heatmap(grid, cmap=cmap, cbar=False, ax=ax, square=True, linewidths=0.5, linecolor='black')

    # Mark the start and end points
    ax.add_patch(plt.Rectangle((character['x'], character['y']), 1, 1, color="blue", alpha=0.5, label="Start"))
    ax.add_patch(plt.Rectangle((end_point['x'], end_point['y']), 1, 1, color="red", alpha=0.5, label="End"))

    # Animate the solution path
    def update(i):
        if i < len(path):
            x, y = path[i]
            ax.add_patch(plt.Rectangle((x, y), 1, 1, color="green", alpha=0.6))

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=100)
    return ani

# Run and animate different algorithms
maze = mazes[0]

# Choose the algorithm to use
# BFS
bfs_path = solve_bfs(maze, character, end_point)
bfs_animation = animate_path(bfs_path, maze)
HTML(bfs_animation.to_jshtml())

# Dijkstra's
dijkstra_path = solve_dijkstra(maze, character, end_point)
dijkstra_animation = animate_path(dijkstra_path, maze)
HTML(dijkstra_animation.to_jshtml())

# A*
a_star_path = solve_a_star(maze, character, end_point)
a_star_animation = animate_path(a_star_path, maze)
HTML(a_star_animation.to_jshtml())

