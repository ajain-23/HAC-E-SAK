import torch
import gym
import asset
from datetime import datetime
import numpy as np
from HAC import HAC
from logger import Logger
import argparse
from enum import Enum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExplorationTechnique(Enum):
    OU = "ou"
    NORMAL = "normal"
    SURPRISE = "surprise"


def train(args):

    #################### Hyperparameters below ####################
    # Debug parameters
    random_seed = args.seed
    render = args.render
    test_mode = args.test

    # Generic parameters
    env_name = "MountainCarContinuous-h-v1"
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
    #################### Hyperparameters above ####################

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Calculate the range of the action space (min and max)
    action_bounds = env.action_space.high[0]
    action_offset = np.array([0.0])
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    action_clip_low = np.array([-1.0 * action_bounds])
    action_clip_high = np.array([action_bounds])

    # Calculate the range of the state space (min and max)
    state_bounds_np = np.array([0.9, 0.07])
    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset = np.array([-0.3, 0.0])
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    state_clip_low = np.array([-1.2, -0.07])
    state_clip_high = np.array([0.6, 0.07])

    # Noise to apply to explorations in states and actions
    exploration_action_noise = np.array([0.1])
    exploration_state_noise = np.array([0.02, 0.01])

    # Final goal state (car up on hill, with some velocity)
    goal_state = np.array([0.48, 0.04])
    threshold = np.array(
        [0.01, 0.02]
    )  # Threshold for whether or not the current state matches goal

    # save trained models
    directory = f"./preTrained/{env_name}/{k_level}level/"

    filename = (
        f"HAC_{env_name}_{exploration_technique.value}{'_solved' if test_mode else ''}"
    )
    #########################################################

    logger = Logger(
        log_dir=f"logs/{env_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
            env,
            k_level - 1,
            state,
            goal_state,
            eval_mode or test_mode,
            False,  # TODO(JS): Are these parameters to HAC correct?
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

        logger.log_scalar(agent.reward, f"{prefix} Reward", i_episode)
        logger.log_scalar(agent.timestep, f"{prefix} Num Steps", i_episode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--test", action="store_true")

    ap.add_argument("--save_episode", type=int, default=10)
    ap.add_argument("--eval_episode", type=int, default=10)
    ap.add_argument("--max_episodes", type=int, default=500)

    ap.add_argument("--k_level", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--lamda", type=float, default=0.3)

    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--n_iter", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.001)

    ap.add_argument(
        "--exploration_technique", type=str, default=ExplorationTechnique.SURPRISE
    )

    args = ap.parse_args()
    train(args)
