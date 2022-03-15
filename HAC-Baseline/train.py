import torch
import gym
import asset
from datetime import datetime
import numpy as np
from HAC import HAC, ExplorationTechnique
from logger import Logger
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LARGE_CONST = 10000.0

def train(args):

    #################### Hyperparameters below ####################
    # Debug parameters
    random_seed = args.seed
    render = args.render
    test_mode = args.test

    # Generic parameters
    env_name = "Ant-v3" #"Humanoid-v2"
    save_episode = args.save_episode  # Save every n episodes
    eval_episode = args.eval_episode  # Run eval every n episodes
    max_episodes = args.max_episodes  # Number of training episodes

    # HAC parameters
    k_level = args.k_level  # num of levels in hierarchy
    H = args.horizon  # time horizon to achieve subgoal
    lamda = args.lamda  # subgoal testing parameter

    # DDPG parameters:
    gamma = args.gamma  # discount factor for future rewards
    n_iter = args.n_iter  # update policy n_iter times in one DDPG update
    batch_size = args.batch_size  # num of transitions sampled from replay buffer
    lr = args.lr

    # Exploration parameters
    exploration_technique = ExplorationTechnique(
        args.exploration_technique
    )  # How to explore
    sync = args.sync  # Whether to synchronize exploration across all levels
    #################### Hyperparameters above ####################

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Calculate the range of the action space (min and max)
    print("Action Dim", action_dim)
    print("Obs Dim", state_dim)
    print("Action Space Low", env.action_space.low)
    print("Obs Space", env.observation_space.low)

    action_bounds_high_np, action_bounds_low_np = env.action_space.high, env.action_space.low
    action_bounds = torch.FloatTensor(action_bounds_high_np.reshape(1, -1)).to(device)
    action_offset = np.mean([action_bounds_high_np, action_bounds_low_np], axis=0)
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    action_clip_low = np.array(action_bounds_low_np)
    action_clip_high = np.array(action_bounds_high_np)

    # Calculate the range of the state space (min and max)
    state_bounds_high_np, state_bounds_low_np = np.array(env.observation_space.high), np.array(env.observation_space.low)
    if np.all(np.isinf(state_bounds_high_np)):
        state_bounds_high_np = np.full_like(state_bounds_high_np, LARGE_CONST)
        state_bounds_low_np = np.full_like(state_bounds_low_np, -LARGE_CONST)
    state_bounds = torch.FloatTensor(state_bounds_high_np.reshape(1, -1)).to(device)
    state_offset = np.mean([state_bounds_high_np, state_bounds_low_np], axis=0)
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    state_clip_low = np.array(state_bounds_low_np)
    state_clip_high = np.array(state_bounds_high_np)

    # Noise to apply to explorations in states and actions
    exploration_action_noise = np.array([0.1])
    exploration_state_noise = np.array([0.02, 0.01])

    # Final goal state (car up on hill, with some velocity)
    # goal_state = np.array([0.48, 0.04])
    # threshold = np.array(
    #     [0.01, 0.02]
    # )  # Threshold for whether or not the current state matches goal
    goal_state = np.random.rand(*state_bounds_high_np.shape)
    threshold = np.ones_like(goal_state) * 0.01
    # save trained models
    directory = f"./preTrained/{env_name}/{k_level}level/"

    filename = f"HAC_{env_name}_{exploration_technique.value}_{'sync' if sync else 'async'}{'_solved' if test_mode else ''}"
    #########################################################
    logger = Logger(
        log_dir=f"logs/{env_name}_{exploration_technique.value}_{'sync' if sync else 'async'}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # creating HAC agent and setting parameters
    agent = HAC(
        k_level,
        H,
        state_dim,
        action_dim,
        render,
        threshold,
        action_bounds,
        action_offset,
        state_bounds,
        state_offset,
        lr,
        exploration_technique,
        sync,
    )

    agent.set_parameters(
        lamda,
        gamma,
        action_clip_low,
        action_clip_high,
        state_clip_low,
        state_clip_high,
        exploration_action_noise,
        exploration_state_noise,
    )

    # If testing, then load the solved file
    if test_mode:
        agent.load(directory, filename)

    for i_episode in range(max_episodes):

        # Check if we should use eval mode
        eval_mode = i_episode % eval_episode == 0

        # Reset logging variables and environment
        agent.reward = 0
        agent.timestep = 0
        state = env.reset()

        # Run episode in environment
        last_state, _done = agent.run_HAC(
            env, k_level - 1, state, goal_state, eval_mode or test_mode, False,
        )

        if (not test_mode) and agent.check_goal(last_state, goal_state, threshold):
            print("################ Solved! ################ ")
            name = filename + "_solved"
            agent.save(directory, name)

        # update all levels
        agent.update(n_iter, batch_size)

        if i_episode % save_episode == 0 and not test_mode:
            agent.save(directory, filename)

        print("Episode: {}\t Reward: {}".format(i_episode, agent.reward))

        prefix = None
        if test_mode:
            prefix = "Test"
        elif eval_mode:
            prefix = "Eval"
        else:
            prefix = "Train"

        logger.log_scalar(agent.timestep, f"{prefix} Num Steps", i_episode)
        logger.log_scalar(agent.reward, f"{prefix} Reward", i_episode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--test", action="store_true")

    ap.add_argument("--save_episode", type=int, default=10)
    ap.add_argument("--eval_episode", type=int, default=10)
    ap.add_argument("--max_episodes", type=int, default=5000)

    ap.add_argument("--k_level", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--lamda", type=float, default=0.3)

    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--n_iter", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.001)

    ap.add_argument(
        "--exploration_technique", type=str, default=ExplorationTechnique.SURPRISE
    )
    ap.add_argument("--sync", action="store_true")

    args = ap.parse_args()
    train(args)