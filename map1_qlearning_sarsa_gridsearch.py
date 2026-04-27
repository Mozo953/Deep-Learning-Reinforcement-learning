import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from grid_world import GridWorld, load_data


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MAP_NAME = "map_1"
MAP_NAMES = ["map_1", "map_2", "map_7", "map_8", "map_12", "map_16"]
NUM_EPISODES = 300
MAX_STEPS = 200

FULL_GRID = {
    "alphas": [0.05, 0.1, 0.5],
    "gammas": [0.9, 0.99],
    "epsilons": [0.9, 0.5],
}

LIGHT_GRID = {
    "alphas": [0.05, 0.1],
    "gammas": [0.9, 0.99],
    "epsilons": [0.9],
}


def make_grid_world_from_datas(datas, map_name):
    grid = np.array(datas[map_name]["map"])
    wall_positions = []
    cliff_positions = []
    marsh_positions = []
    goal_positions = []

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            cell = grid[y, x]
            if cell == 1:
                wall_positions.append((x, y))
            elif cell == 2:
                cliff_positions.append((x, y))
            elif cell == 3:
                marsh_positions.append((x, y))
            elif cell == 4:
                goal_positions.append((x, y))

    start = (datas[map_name]["start_place_x"], datas[map_name]["start_place_y"])
    return GridWorld(
        width=grid.shape[1],
        height=grid.shape[0],
        player_position=start,
        wall_positions=wall_positions,
        goal_positions=goal_positions,
        cliff_positions=cliff_positions,
        marsh_positions=marsh_positions,
    )


def get_intended_position(state, action, grid_world):
    x, y = state
    if action == "up":
        y = max(0, y - 1)
    elif action == "down":
        y = min(grid_world.height - 1, y + 1)
    elif action == "left":
        x = max(0, x - 1)
    elif action == "right":
        x = min(grid_world.width - 1, x + 1)
    return (x, y)


def get_reward(
    grid_world,
    reached_goal,
    intended_position,
    reward_final=1000,
    reward_goal=None,
    reward_default=-1,
    reward_marsh=-10,
    reward_cliff=-100,
):
    if reward_goal is not None:
        reward_final = reward_goal
    if reached_goal:
        return reward_final
    if intended_position in grid_world.cliff_positions:
        return reward_cliff
    if intended_position in grid_world.marsh_positions:
        return reward_marsh
    return reward_default


def train_one_episode(agent, grid_world, max_steps=200, rewards_config=None):
    rewards_config = rewards_config or {}
    steps = 0
    total_reward = 0
    trajectory = [grid_world.player_position]

    if agent.__class__.__name__ == "SarsaAgent":
        state = grid_world.player_position
        action = agent.choose_action(state)

        while not grid_world.is_game_over() and steps < max_steps:
            intended_position = get_intended_position(state, action, grid_world)
            reached_goal = grid_world.move_player(action)
            next_state = grid_world.player_position
            reward = get_reward(
                grid_world, reached_goal, intended_position, **rewards_config
            )
            next_action = agent.choose_action(next_state)

            agent.learn(state, action, reward, next_state, next_action)
            state, action = next_state, next_action
            steps += 1
            total_reward += reward
            trajectory.append(next_state)
    else:
        while not grid_world.is_game_over() and steps < max_steps:
            state = grid_world.player_position
            action = agent.choose_action(state)
            intended_position = get_intended_position(state, action, grid_world)
            reached_goal = grid_world.move_player(action)
            next_state = grid_world.player_position
            reward = get_reward(
                grid_world, reached_goal, intended_position, **rewards_config
            )

            agent.learn(state, action, reward, next_state)
            steps += 1
            total_reward += reward
            trajectory.append(next_state)

    return steps, total_reward, trajectory


def map_contains_cell_type(datas, map_name, cell_value):
    grid = np.array(datas[map_name]["map"])
    return bool(np.any(grid == cell_value))


def train_agent_on_map(
    agent_class,
    datas,
    map_name,
    alpha,
    gamma,
    epsilon,
    num_episodes=300,
    max_steps=200,
    epsilon_min=0.01,
    epsilon_decay_rate=0.01,
    rewards_config=None,
):
    grid_world = make_grid_world_from_datas(datas, map_name)
    agent = agent_class(grid_world, alpha=alpha, gamma=gamma, epsilon=epsilon)
    steps_per_episode = []
    rewards_per_episode = []

    for _ in range(num_episodes):
        grid_world.player_position = grid_world.init_player_position
        steps, total_reward, _trajectory = train_one_episode(
            agent, grid_world, max_steps, rewards_config
        )
        steps_per_episode.append(steps)
        rewards_per_episode.append(total_reward)
        agent.epsilon = max(epsilon_min, agent.epsilon * np.exp(-epsilon_decay_rate))

    return {
        "agent": agent,
        "grid_world": grid_world,
        "steps": np.array(steps_per_episode),
        "rewards": np.array(rewards_per_episode),
    }


def greedy_trajectory(agent, grid_world, max_steps=200):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    grid_world.player_position = grid_world.init_player_position
    trajectory = [grid_world.player_position]

    for _ in range(max_steps):
        if grid_world.is_game_over():
            break
        state = grid_world.player_position
        action = agent.choose_action(state)
        grid_world.move_player(action)
        trajectory.append(grid_world.player_position)

    agent.epsilon = old_epsilon
    return trajectory


def run_grid_search(
    datas,
    map_name,
    reward_scenarios,
    q_learning_cls,
    sarsa_cls,
    grid_config=None,
):
    grid_config = grid_config or FULL_GRID
    alphas = grid_config["alphas"]
    gammas = grid_config["gammas"]
    epsilons = grid_config["epsilons"]
    results = []

    base_params = datas[map_name]["parameters"].copy()
    base_params.update({"episodes": NUM_EPISODES, "step_limit": MAX_STEPS})

    config_id = 0
    for alpha, gamma, epsilon in itertools.product(alphas, gammas, epsilons):
        config_id += 1
        for scenario_name, scenario_rewards in reward_scenarios.items():
            rewards_config = {
                "reward_final": base_params.get("reward_final", 1000),
                "reward_default": base_params.get("reward_default", -1),
                "reward_marsh": scenario_rewards["reward_marsh"],
                "reward_cliff": scenario_rewards["reward_cliff"],
            }
            for agent_class in [q_learning_cls, sarsa_cls]:
                run = train_agent_on_map(
                    agent_class,
                    datas,
                    map_name,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    num_episodes=NUM_EPISODES,
                    max_steps=MAX_STEPS,
                    epsilon_min=base_params.get("epsilon_min", 0.01),
                    epsilon_decay_rate=base_params.get("epsilon_decay_rate", 0.01),
                    rewards_config=rewards_config,
                )
                results.append(
                    {
                        "map_name": map_name,
                        "config_id": config_id,
                        "scenario": scenario_name,
                        "agent_name": agent_class.__name__,
                        "alpha": alpha,
                        "gamma": gamma,
                        "epsilon_initial": epsilon,
                        "mean_steps": float(np.mean(run["steps"])),
                        "mean_reward": float(np.mean(run["rewards"])),
                        "var_steps": float(np.var(run["steps"])),
                        "var_reward": float(np.var(run["rewards"])),
                        "run": run,
                    }
                )
    return results


def plot_grid_search_results(results, scenario_name, map_name):
    rows = [r for r in results if r["scenario"] == scenario_name]
    configs = sorted({r["config_id"] for r in rows})
    labels = [f"C{c}" for c in configs]
    metrics = [
        ("mean_steps", "Steps moyens"),
        ("mean_reward", "Reward moyen"),
        ("var_reward", "Variance reward"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        for agent_name, color in [
            ("QLearningAgent", "#2563eb"),
            ("SarsaAgent", "#dc2626"),
        ]:
            values = [
                next(
                    r[metric]
                    for r in rows
                    if r["config_id"] == c and r["agent_name"] == agent_name
                )
                for c in configs
            ]
            ax.plot(labels, values, marker="o", linewidth=2, label=agent_name, color=color)
        ax.set_title(title)
        ax.set_xlabel("Configuration")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Valeur")
    axes[-1].legend()
    fig.suptitle(f"Grid search sur {map_name} - {scenario_name}")
    plt.tight_layout()
    plt.show()

    print("Correspondance des configurations :")
    for c in configs:
        row = next(r for r in rows if r["config_id"] == c)
        print(
            f"C{c}: alpha={row['alpha']}, gamma={row['gamma']}, "
            f"epsilon={row['epsilon_initial']}"
        )


def summarize_best_results(results, metric="mean_reward"):
    best_rows = []
    scenarios = sorted({row["scenario"] for row in results})
    agent_names = sorted({row["agent_name"] for row in results})

    for scenario_name in scenarios:
        for agent_name in agent_names:
            rows = [
                row
                for row in results
                if row["scenario"] == scenario_name and row["agent_name"] == agent_name
            ]
            best = max(rows, key=lambda row: row[metric])
            best_rows.append(
                {
                    "scenario": scenario_name,
                    "agent_name": agent_name,
                    "config_id": best["config_id"],
                    "alpha": best["alpha"],
                    "gamma": best["gamma"],
                    "epsilon_initial": best["epsilon_initial"],
                    "mean_steps": round(best["mean_steps"], 2),
                    "mean_reward": round(best["mean_reward"], 2),
                    "var_steps": round(best["var_steps"], 2),
                    "var_reward": round(best["var_reward"], 2),
                }
            )
    return best_rows


def print_best_results_table(best_rows, map_name):
    print(f"\nResume des meilleures configurations pour {map_name}")
    print("-" * 90)
    print(
        f"{'Scenario':<20} {'Agent':<16} {'Cfg':<4} {'alpha':<6} "
        f"{'gamma':<6} {'eps':<5} {'steps':<10} {'reward':<10} {'var_r':<10}"
    )
    print("-" * 90)
    for row in best_rows:
        print(
            f"{row['scenario']:<20} {row['agent_name']:<16} C{row['config_id']:<3} "
            f"{row['alpha']:<6} {row['gamma']:<6} {row['epsilon_initial']:<5} "
            f"{row['mean_steps']:<10} {row['mean_reward']:<10} {row['var_reward']:<10}"
        )


def run_map_pipeline(
    q_learning_cls,
    sarsa_cls,
    map_name=MAP_NAME,
    grid_config=None,
    show_plots=True,
    show_3d=True,
):
    datas = load_data()
    reward_scenarios = {
        "cas simple": {"reward_cliff": -1, "reward_marsh": 0},
        "cas risque": {"reward_cliff": -100, "reward_marsh": -10},
    }

    results = run_grid_search(
        datas,
        map_name,
        reward_scenarios,
        q_learning_cls=q_learning_cls,
        sarsa_cls=sarsa_cls,
        grid_config=grid_config,
    )

    if show_plots:
        plot_grid_search_results(results, "cas simple", map_name)
        plot_grid_search_results(results, "cas risque", map_name)

    best_rows = summarize_best_results(results)
    print_best_results_table(best_rows, map_name)

    if show_3d:
        best_q_risk = max(
            [
                row
                for row in results
                if row["scenario"] == "cas risque"
                and row["agent_name"] == "QLearningAgent"
            ],
            key=lambda row: row["mean_reward"],
        )
        best_run = best_q_risk["run"]
        final_trajectory = greedy_trajectory(
            best_run["agent"], best_run["grid_world"], max_steps=MAX_STEPS
        )
        plot_trajectory_and_values_3d(
            best_run["agent"],
            best_run["grid_world"],
            final_trajectory,
            title=f"{map_name} - Q-Learning C{best_q_risk['config_id']} cas risque",
        )

    return results, best_rows


def run_selected_maps(
    q_learning_cls,
    sarsa_cls,
    map_names=None,
    grid_config=None,
    show_plots=False,
):
    datas = load_data()
    map_names = map_names or MAP_NAMES
    all_results = []
    all_best_rows = []

    for map_name in map_names:
        print(f"\n===== Pipeline sur {map_name} =====")
        print(
            f"Marais: {map_contains_cell_type(datas, map_name, 3)} | "
            f"Falaises: {map_contains_cell_type(datas, map_name, 2)}"
        )
        results, best_rows = run_map_pipeline(
            q_learning_cls,
            sarsa_cls,
            map_name=map_name,
            grid_config=grid_config,
            show_plots=show_plots,
            show_3d=False,
        )
        all_results.extend(results)
        for row in best_rows:
            row_with_map = dict(row)
            row_with_map["map_name"] = map_name
            all_best_rows.append(row_with_map)

    return all_results, all_best_rows


def print_report_ready_summary(all_best_rows):
    print("\nTableau final pret pour le rapport")
    print("=" * 110)
    print(
        f"{'Map':<8} {'Scenario':<20} {'Agent':<16} {'Cfg':<4} "
        f"{'alpha':<6} {'gamma':<6} {'eps':<5} {'steps':<10} {'reward':<10}"
    )
    print("=" * 110)
    for row in all_best_rows:
        print(
            f"{row['map_name']:<8} {row['scenario']:<20} {row['agent_name']:<16} "
            f"C{row['config_id']:<3} {row['alpha']:<6} {row['gamma']:<6} "
            f"{row['epsilon_initial']:<5} {row['mean_steps']:<10} {row['mean_reward']:<10}"
        )


def state_values(agent, grid_world):
    values = np.zeros((grid_world.height, grid_world.width), dtype=float)
    for y in range(grid_world.height):
        for x in range(grid_world.width):
            state = (x, y)
            if state in grid_world.wall_positions:
                values[y, x] = np.nan
            else:
                values[y, x] = max(agent.get_q_value(state, a) for a in agent.actions)
    return values


def plot_trajectory_and_values_3d(agent, grid_world, trajectory, title):
    grid = np.zeros((grid_world.height, grid_world.width), dtype=int)
    for x, y in grid_world.wall_positions:
        grid[y, x] = 1
    for x, y in grid_world.cliff_positions:
        grid[y, x] = 2
    for x, y in grid_world.marsh_positions:
        grid[y, x] = 3
    for x, y in grid_world.goal_positions:
        grid[y, x] = 4

    sx, sy = grid_world.init_player_position
    display_grid = grid.copy()
    display_grid[sy, sx] = 5

    colors = ["#f8fafc", "#2563eb", "#facc15", "#92400e", "#22c55e", "#9ca3af"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    values = state_values(agent, grid_world)

    fig = plt.figure(figsize=(17, 7))
    ax_map = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")

    ax_map.imshow(display_grid, cmap=cmap, norm=norm)
    ax_map.set_xticks(np.arange(grid_world.width))
    ax_map.set_yticks(np.arange(grid_world.height))
    ax_map.set_xticks(np.arange(-0.5, grid_world.width, 1), minor=True)
    ax_map.set_yticks(np.arange(-0.5, grid_world.height, 1), minor=True)
    ax_map.grid(which="minor", color="#111827", linewidth=1.0)
    ax_map.tick_params(which="minor", bottom=False, left=False)
    ax_map.set_title(f"Trajectoire finale - {title}")

    if trajectory:
        xs = [pos[0] for pos in trajectory]
        ys = [pos[1] for pos in trajectory]
        ax_map.plot(xs, ys, color="#ef4444", linewidth=2.5, marker="o", markersize=4)

    ax_map.text(sx, sy, "S", ha="center", va="center", fontweight="bold")
    for gx, gy in grid_world.goal_positions:
        ax_map.text(gx, gy, "G", ha="center", va="center", fontweight="bold")

    xs = []
    ys = []
    zs = []
    dx = []
    dy = []
    dz = []
    bar_colors = []

    for y in range(grid_world.height):
        for x in range(grid_world.width):
            if (x, y) in grid_world.wall_positions or np.isnan(values[y, x]):
                continue
            xs.append(x - 0.4)
            ys.append(y - 0.4)
            zs.append(0)
            dx.append(0.8)
            dy.append(0.8)
            dz.append(values[y, x])
            bar_colors.append(colors[grid[y, x]] if grid[y, x] < len(colors) else colors[0])

    ax_3d.bar3d(xs, ys, zs, dx, dy, dz, shade=True, color=bar_colors, alpha=0.9)
    ax_3d.set_title(f"Valeur V(s)=max_a Q(s,a) - {title}")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("V(s)")
    ax_3d.view_init(elev=30, azim=-55)

    plt.tight_layout()
    plt.show()


def run_experiment(q_learning_cls, sarsa_cls):
    return run_map_pipeline(
        q_learning_cls,
        sarsa_cls,
        map_name=MAP_NAME,
        grid_config=FULL_GRID,
        show_plots=True,
        show_3d=True,
    )
