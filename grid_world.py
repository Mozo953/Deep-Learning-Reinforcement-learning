import random
import pygame
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# Define grid cell size and margin
CELL_SIZE = 50
MARGIN = 5

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREY = (140, 140, 140)
BROWN = (160, 82, 45)
from copy import deepcopy

import random
class GridWorld:
    def __init__(self, width, height, player_position=None, 
                 wall_positions=None, goal_positions=None, 
                 cliff_positions=None, marsh_positions=None):
        self.width = width
        self.height = height
        self.init_player_position = player_position if player_position else (0, 0) 
        self.player_position = deepcopy(self.init_player_position)

        self.wall_positions = wall_positions if wall_positions else []
        self.cliff_positions = cliff_positions if cliff_positions else []
        self.marsh_positions = marsh_positions if marsh_positions else []
        self.goal_positions = goal_positions if goal_positions else []
        self.actions = ["up", "down", "left", "right"]
    def move_player(self, direction):
        x, y = self.player_position
        if direction == "up":
            y = max(0, y - 1)
        elif direction == "down":
            y = min(self.height - 1, y + 1)
        elif direction == "left":
            x = max(0, x - 1)
        elif direction == "right":
            x = min(self.width - 1, x + 1)
        
        # Check if the new position is blocked
        if (x, y) in self.cliff_positions:
            self.player_position = self.init_player_position
        elif (x, y) not in self.wall_positions:
            self.player_position = (x, y)
        return self.player_position in self.goal_positions

    def print_grid(self):
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if (x, y) == self.player_position:
                    row += "P"
                elif (x, y) in self.goal_positions:
                    row += "G"
                elif (x, y) in self.wall_positions:
                    row += "B"
                elif (x, y) in self.marsh_positions:
                    row += "M"
                elif (x, y) in self.cliff_positions:
                    row += "C"
                else:
                    row += "."
            print(row)
    def is_game_over(self):
        return self.player_position in self.goal_positions


def draw_grid(screen, grid_world):
    for y in range(grid_world.height):
        for x in range(grid_world.width):
            color = WHITE
            if (x, y) == grid_world.init_player_position:
                color = GREY
            if (x, y) == grid_world.player_position:
                color = RED
            if (x, y) in grid_world.goal_positions:
                color = GREEN
            elif (x, y) in grid_world.wall_positions:
                color = BLUE
            elif (x, y) in grid_world.cliff_positions:
                color = YELLOW
            elif (x, y) in grid_world.marsh_positions:
                color = BROWN

            pygame.draw.rect(screen, color, [(MARGIN + CELL_SIZE) * x + MARGIN, (MARGIN + CELL_SIZE) * y + MARGIN, CELL_SIZE, CELL_SIZE])
    

class PygameGridWorld():
    def __init__(self):
        self.grid_world = None
        
        if self.grid_world is None:
            width, height = 5, 5
            self.grid_world = GridWorld(width, height, 
                            wall_positions = [(1, 1), (2, 2), (3, 3)], 
                            goal_positions = [(height-1, width-1)])

    def load_grid_world(self, map_name, datas):
        # Read the map
        map = datas[map_name]["map"]
        n=0 #dimension of the map
        wall=[]
        cliff=[]
        marsh=[]
        goal=[]
        
        for k in map:
            n=n+1
        for i in range(n):
            for j in range(n):
                if map[i][j] == 1:
                    wall.append((j, i))
                elif map[i][j] == 2:
                    cliff.append((j, i))
                elif map[i][j] == 3:
                    marsh.append((j, i))
                elif map[i][j] == 4:
                    goal.append((j, i))
        player_position = (datas[map_name]["start_place_x"], datas[map_name]["start_place_y"])
        height, width  = len(map), len(map[0])
        grid_world = GridWorld(width, height, player_position, wall, goal, cliff, marsh)
        self.grid_world = grid_world
        return grid_world

    def main(self, auto_close=True, grid_world=None, map_name=None, datas=None):
        if map is not None and datas is not None:
            grid_world = self.load_grid_world(map_name, datas)
        elif grid_world is None:
            width, height = 5, 5
            grid_world = GridWorld(width, height, 
                            wall_positions = [(1, 1), (2, 2), (3, 3)], 
                            goal_positions = [(height-1, width-1)])
        self.grid_world = grid_world

        directions = ["up", "down", "left", "right"]

        # Set up the display
        screen = pygame.display.set_mode([(CELL_SIZE + MARGIN) * grid_world.width + MARGIN, (CELL_SIZE + MARGIN) * grid_world.height + MARGIN])
        pygame.display.set_caption("GridWorld")

        # Main loop
        running = True
        clock = pygame.time.Clock()
        while running and not grid_world.is_game_over():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Random agent simulation
            if not grid_world.is_game_over():
                direction = random.choice(directions)
                grid_world.move_player(direction)

            # Draw the grid
            screen.fill(BLACK)
            draw_grid(screen, grid_world)

            # Update the display
            pygame.display.flip()

            # Limit the frame rate
            clock.tick(60)

        # Quit pygame
        if auto_close:
            pygame.quit()


import json
def load_data(dataset='datas.json'):
    with open(dataset, 'r') as file:
        datas=json.load(file)
    print("loaded maps :", datas.keys())
    return datas


def plot_grid_world(datas, map_name):
    # Define colors
    colors = {
        "BLACK": (0, 0, 0),
        "WHITE": (255, 255, 255), # passable road
        "GREEN": (0, 255, 0), # goal_positions
        "RED": (255, 0, 0), # player_position
        "BLUE": (0, 0, 255), # wall_positions
        "YELLOW": (255, 255, 0), # cliff_positions
        "GREY": (140, 140, 140), # init_player_position
        "BROWN": (160, 82, 45) # marsh_positions
    }
    pos_colors = {4:"GREEN", 3:"BROWN", 2:"YELLOW", 1:"BLUE", 0:"WHITE" }

    # Define a 10x10 grid
    grid_size = 10
    grid = datas[map_name]["map"] 
    init_player_pos = datas[map_name]['start_place_x'], \
                            datas[map_name]['start_place_y'] 
    # Plotting
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    for x in range(grid_size):
        for y in range(grid_size):
            color = colors[pos_colors[grid[x][y]]] if not (x==init_player_pos[1] and y==init_player_pos[0]) else colors["GREY"]
            ax.add_patch(plt.Rectangle((y, grid_size - x - 1), 1, 1, color=np.array(color)/255))

    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.grid(which='both', color='k', linewidth=2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.title(f"Visualization of {map_name}")
    plt.show()

def SimulationGridWorld(agent_class, params, 
                        episode, grid_world, auto_replay=True, auto_close=False, ticks=10):
   # Read the map 
    if grid_world is None:
        grid_world = GridWorld(10, 10, 
                        wall_positions=[ (1, 1), (2, 2), (3, 3), (4, 4), 
                                            (3, 2), (3, 1), 
                                            (5, 5), (5,0), 
                                            (6, 1), (6,2), (6,3),
                                            (8, 9), (8, 8),  (8, 7),(8, 6),
                                            (8, 5), (8, 4),(8, 3),(8, 2),],
                        goal_positions=[(9,9)])

    agent = agent_class(grid_world, 
                           gamma=params["gamma"], 
                           epsilon=params["epsilon"], 
                           alpha=params["alpha"])

    # Load the learned policy
    with open(f"learned_policy_{agent_class.__name__}_episode_{episode}.pkl", "rb") as f:
        agent.q_table = pkl.load(f)
        #for l in sorted(list(agent.q_table.items())):
        #    print(l)

    # Set up the display
    screen = pygame.display.set_mode([(CELL_SIZE + MARGIN) * grid_world.width + MARGIN, (CELL_SIZE + MARGIN) * grid_world.height + MARGIN])
    pygame.display.set_caption("GridWorld")

    # Main loop
    running = True
    clock = pygame.time.Clock()
    visited_positions = []
    simulation_steps = 0
    max_simulation_steps = 1000
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Use learned policy for agent simulation
        if not grid_world.is_game_over():
            state = grid_world.player_position
            action = agent.choose_action(state)  # why not taken from the policy and the state
            grid_world.move_player(action)
            visited_positions.append(grid_world.player_position)
            simulation_steps += 1

        # Draw the grid
        screen.fill(BLACK)
        draw_grid(screen, grid_world)

        # Draw visited positions with a changing color from dense to transparent
        for i, pos in enumerate(visited_positions):
            x, y = pos
            alpha = int(100 + 155 * (i / len(visited_positions)))
            color = (255, 200, 200, alpha)  # Light red with changing transparency
            temp_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(temp_surface, color, [0, 0, CELL_SIZE, CELL_SIZE])
            screen.blit(temp_surface, [(MARGIN + CELL_SIZE) * x + MARGIN, (MARGIN + CELL_SIZE) * y + MARGIN])

        # Update the display
        pygame.display.flip()

        # Limit the frame rate
        clock.tick(ticks)

        # Quit pygame
        if auto_replay and grid_world.is_game_over():
            visited_positions = []
            grid_world.player_position = grid_world.init_player_position
            simulation_steps = 0
        elif grid_world.is_game_over() or simulation_steps >= max_simulation_steps:
            running = False

    # Quit pygame
    if auto_close:
        pygame.quit()
