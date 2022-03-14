import torch
import numpy as np
from DDPG import DDPG
from utils import ReplayBuffer
from noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from exploration_policy import ExplorationPolicy
from enum import Enum

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExplorationTechnique(Enum):
    OU = "ou"
    NORMAL = "normal"
    SURPRISE = "surprise"


class HAC:
    def __init__(
        self,
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
    ):

        # adding lowest level
        self.HAC = [DDPG(state_dim, action_dim, action_bounds, action_offset, lr, H)]
        self.replay_buffer = [ReplayBuffer()]

        # adding remaining levels
        for _ in range(k_level - 1):
            self.HAC.append(
                DDPG(state_dim, state_dim, state_bounds, state_offset, lr, H)
            )
            self.replay_buffer.append(ReplayBuffer())

        # set some parameters
        self.k_level = k_level
        self.H = H
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.threshold = threshold
        self.render = render
        self.lr = lr

        self.exploration_technique = exploration_technique
        self.sync = sync

        self.action_min = 0
        self.action_max = 0
        self.state_min = 0
        self.state_max = 0

        # logging parameters
        self.goals = [None] * self.k_level
        self.reward = 0
        self.timestep = 0

    def set_parameters(
        self,
        lamda,
        gamma,
        action_clip_low,
        action_clip_high,
        state_clip_low,
        state_clip_high,
        exploration_action_noise,
        exploration_state_noise,
    ):

        self.lamda = lamda
        self.gamma = gamma
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high
        self.exploration_action_noise = exploration_action_noise
        self.exploration_state_noise = exploration_state_noise

        self.explorers = [
            ExplorationPolicy(
                self.state_dim,
                self.action_dim,
                torch.FloatTensor(action_clip_low),
                torch.FloatTensor(action_clip_high),
                torch.FloatTensor(state_clip_low),
                torch.FloatTensor(state_clip_high),
                self.lr,
            )
        ]

        for _ in range(self.k_level - 1):
            self.explorers.append(
                ExplorationPolicy(
                    self.state_dim,
                    self.state_dim,
                    torch.FloatTensor(state_clip_low),
                    torch.FloatTensor(state_clip_high),
                    torch.FloatTensor(state_clip_low),
                    torch.FloatTensor(state_clip_high),
                    self.lr,
                )
            )

    def check_goal(self, state, goal, threshold):
        for i in range(self.state_dim):
            if abs(state[i] - goal[i]) > threshold[i]:
                return False
        return True

    def run_HAC(
        self, env, i_level, state, goal, is_subgoal_test, is_subgoal_exploration
    ):
        next_state = None
        done = None
        goal_transitions = []

        # logging updates
        self.goals[i_level] = goal

        # H attempts
        for _ in range(self.H):
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test
            is_next_subgoal_exploration = is_subgoal_exploration

            action = self.HAC[i_level].select_action(state, goal)

            #   <================ high level policy ================>
            if i_level > 0:
                # add noise or take random action if not subgoal testing
                if not is_subgoal_test:
                    if is_subgoal_exploration:
                        if self.exploration_technique == ExplorationTechnique.NORMAL:
                            action += NormalActionNoise(
                                0, self.exploration_state_noise
                            )()
                        elif self.exploration_technique == ExplorationTechnique.OU:
                            action += OrnsteinUhlenbeckActionNoise(
                                np.array([0]), self.exploration_state_noise
                            )()
                        elif (
                            self.exploration_technique == ExplorationTechnique.SURPRISE
                        ):

                            action = (
                                self.explorers[i_level]
                                .get_action(
                                    torch.FloatTensor(state).to(device),
                                    torch.FloatTensor(action).to(device),
                                )
                                .cpu()
                                .numpy()
                            )

                        action = action.clip(self.state_clip_low, self.state_clip_high)
                    else:
                        action = np.random.uniform(
                            self.state_clip_low, self.state_clip_high
                        )

                # Determine whether to test subgoal (action)
                if np.random.random_sample() < self.lamda:
                    is_next_subgoal_test = True

                # Pass subgoal to lower level
                next_state, done = self.run_HAC(
                    env,
                    i_level - 1,
                    state,
                    action,
                    is_next_subgoal_test,
                    is_next_subgoal_exploration
                    if self.sync
                    else np.random.random_sample() > 0.2,
                )

                # if subgoal was tested but not achieved, add subgoal testing transition
                if is_next_subgoal_test and not self.check_goal(
                    action, next_state, self.threshold
                ):
                    self.replay_buffer[i_level].add(
                        (state, action, -self.H, next_state, goal, 0.0, float(done))
                    )

                # for hindsight action transition
                action = next_state

            #   <================ low level policy ================>
            else:
                # add noise or take random action if not subgoal testing
                if not is_subgoal_test:
                    if np.random.random_sample() > 0.2:
                        if self.exploration_technique == ExplorationTechnique.NORMAL:
                            action += NormalActionNoise(
                                0, self.exploration_action_noise
                            )()
                        elif self.exploration_technique == ExplorationTechnique.OU:
                            action += OrnsteinUhlenbeckActionNoise(
                                np.array([0]), self.exploration_action_noise
                            )()
                        elif (
                            self.exploration_technique == ExplorationTechnique.SURPRISE
                        ):
                            action = (
                                self.explorers[i_level]
                                .get_action(
                                    torch.FloatTensor(state).to(device),
                                    torch.FloatTensor(action).to(device),
                                )
                                .cpu()
                                .numpy()
                            )
                        action = action.clip(
                            self.action_clip_low, self.action_clip_high
                        )
                    else:
                        action = np.random.uniform(
                            self.action_clip_low, self.action_clip_high
                        )

                # take primitive action
                next_state, rew, done, _ = env.step(action)

                if self.render:

                    # env.render() ##########

                    if self.k_level == 2:
                        env.unwrapped.render_goal(self.goals[0], self.goals[1])
                    elif self.k_level == 3:
                        env.unwrapped.render_goal_2(
                            self.goals[0], self.goals[1], self.goals[2]
                        )

                # this is for logging
                self.reward += rew
                self.timestep += 1

            #   <================ finish one step/transition ================>

            # check if goal is achieved
            goal_achieved = self.check_goal(next_state, goal, self.threshold)

            # hindsight action transition
            if goal_achieved:
                self.replay_buffer[i_level].add(
                    (state, action, 0.0, next_state, goal, 0.0, float(done))
                )
            else:
                self.replay_buffer[i_level].add(
                    (state, action, -1.0, next_state, goal, self.gamma, float(done))
                )

            # copy for goal transition
            goal_transitions.append(
                [state, action, -1.0, next_state, None, self.gamma, float(done)]
            )

            state = next_state

            if done or goal_achieved:
                break

        #   <================ finish H attempts ================>

        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1][2] = 0.0
        goal_transitions[-1][5] = 0.0
        for transition in goal_transitions:
            # last state is goal for all transitions
            transition[4] = next_state
            self.replay_buffer[i_level].add(tuple(transition))

        return next_state, done

    def update(self, n_iter, batch_size):
        for i in range(self.k_level):
            self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)
            if self.exploration_technique == ExplorationTechnique.SURPRISE:
                states, actions, _, next_states, _, _, _ = self.replay_buffer[i].sample(
                    batch_size
                )
                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                self.explorers[i].update(states, actions, next_states)

    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name + "_level_{}".format(i))

    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name + "_level_{}".format(i))

