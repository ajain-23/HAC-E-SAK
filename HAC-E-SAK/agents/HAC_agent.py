from typing import Callable, List, Tuple
import torch
from collections import namedtuple
from enum import Enum
import random
from tqdm import tqdm
from pathlib import Path
import logging

from isaacgymenvs.tasks.base.vec_task import VecTask

from HACKerMan.policies.DDPG_policy import DDPGPolicy
from HACKerMan.policies.exploration_policy import ExplorationPolicy
from HACKerMan.infrastructure.replay_buffer import ReplayBuffer
from HACKerMan.infrastructure.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)


HACParams = namedtuple(
    "HACParams",
    [
        "lambda_",
        "gamma",
        "k",
        "horizon",
        "lr",
        "action_std",
        "state_std",
        "goal_threshold",
        "batch_size",
        "num_update_steps",
    ],
)

ExplorationParams = namedtuple(
    "ExplorationParams",
    ["exploration_strategy", "exploration_frequency", "noise_mean", "noise_std"],
)


class HACMiniAgent:
    PolicyMode = Enum("PolicyMode", "EXPLORE_RANDOM EXPLORE_HACKATHON SUBGOAL_TEST")

    def __init__(
        self,
        agent_id: int,
        action_size: int,
        goal_state: torch.Tensor,
        params: HACParams,
        exploration_params: ExplorationParams,
        device: str,
    ):
        self.agent_id = agent_id
        self.action_size = action_size
        self.goal_state = goal_state
        self.device = device

        # Store HAC hyperparameters
        self.lambda_ = params.lambda_  # Lambda is a reserved keyword in Python
        self.gamma = params.gamma
        self.k = params.k
        self.horizon = params.horizon
        self.lr = params.lr
        self.action_std = params.action_std
        self.state_std = params.state_std
        self.goal_threshold = params.goal_threshold
        self.batch_size = params.batch_size
        self.num_update_steps = params.num_update_steps

        # Store exploration hyperparameters
        self.exploration_strategy = exploration_params.exploration_strategy
        self.exploration_frequency = exploration_params.exploration_frequency
        self.noise_mean = exploration_params.noise_mean
        self.noise_std = exploration_params.noise_std

        # State for maintaining progress in actuation hierarchy
        self.training_modes = None
        self.step_counts = None
        self.subgoals = None
        self.prev_subgoals = None
        self.experiences = None

        # Fill in the above state variables
        self.reset()

    def reset(self):
        logging.debug(f"\t** RESET TRIGGERED BY ENV!! **")
        # How each layer in the hierarchy should decide between explore and exploit
        self.training_modes = [
            HACMiniAgent.PolicyMode.EXPLORE_RANDOM for _ in range(self.k)
        ]

        # How many steps each layer has taken
        self.step_counts = [self.horizon for _ in range(self.k)]

        # What subgoals each layer is trying to achieve
        self.subgoals = [None for _ in range(self.k)]
        # Save environment goal state as highest subgoal
        self.subgoals[-1] = self.goal_state

        # Previous subgoals for transition calculation
        self.prev_subgoals = [None for _ in range(self.k)]

        # Short-horizon saved experiences for HER
        self.experiences = [[] for _ in range(self.k)]

    def get_action_modifier(
        self, exploration_policy: ExplorationPolicy, training_mode: PolicyMode
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

        exploration_level = 0
        if training_mode == HACMiniAgent.PolicyMode.SUBGOAL_TEST:
            # Absolutely no modification on the action at all!
            exploration_level = 0
        elif training_mode == HACMiniAgent.PolicyMode.EXPLORE_RANDOM:
            # Sometimes go to overdrive, but usually stay at regular slight noise
            exploration_level = 2 if random.random() < self.exploration_frequency else 1
        elif training_mode == HACMiniAgent.PolicyMode.EXPLORE_HACKATHON:
            # Always go to overdrive
            exploration_level = 2

        if exploration_level == 0:
            return lambda s, a: a
        elif exploration_level == 1:
            if self.exploration_strategy == "Normal":
                return NormalActionNoise(self.noise_mean, self.noise_std)
            elif self.exploration_strategy == "OU":
                return OrnsteinUhlenbeckActionNoise(self.noise_mean, self.noise_std)
            elif self.exploration_strategy == "Surprise":
                return exploration_policy.get_action
            else:
                raise Exception(
                    f"Unrecognized exploration strategy {self.exploration_strategy}"
                )
        elif exploration_level == 2:
            return lambda s, a: torch.rand(size=a.size())

    def get_action(
        self,
        policies: List[DDPGPolicy],
        exploration_policies: List[ExplorationPolicy],
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:

        # Begin at the lowest level
        current_level = 0
        logging.debug(f"\tGetting action...")
        logging.debug(f"\tStarting step counts: {self.step_counts}")
        # Proceed upwards through all fully-expended levels to find first level with steps remaining
        while self.step_counts[current_level] >= self.horizon:
            logging.debug(
                f"\t\tCurrent Level {current_level} is saturated. Moving up one level..."
            )
            self.step_counts[current_level] = 0  # Reset step counts

            # Increment current level, unless we are already at the highest level
            if current_level == self.k - 1:
                logging.debug(
                    f"\t\t\tReached highest level, still saturated. Giving highest level more steps."
                )
                break
            else:
                current_level += 1
        logging.debug(
            f"\tFirst level with steps remaining is {current_level}, with {self.step_counts[current_level]} steps spent (1 about to be spent) out of {self.horizon}."
        )
        # At this point, the current level is capable of producing the next subgoal
        # Proceed downwards from current level to produce subgoals
        while current_level > 0:
            logging.debug(
                f"\t\tPerforming managerial actions at Level: {current_level}"
            )
            # Save previous subgoal for transition updates later
            self.prev_subgoals[current_level - 1] = self.subgoals[current_level - 1]

            # Fetch subgoal of lower level by using current level's policy
            deterministic_subgoal = policies[current_level].get_actions(
                state, self.subgoals[current_level]
            )

            # Modify subgoal for exploration, depending on current policies and modes
            subgoal_modifier = self.get_action_modifier(
                exploration_policies[current_level], self.training_modes[current_level],
            )
            subgoal = subgoal_modifier(state, deterministic_subgoal)

            # Update subgoal for lower level
            self.subgoals[current_level - 1] = subgoal

            # With some probability (or if already subgoal testing),
            # assign subsequent lower level to subgoal testing
            if (
                self.training_modes[current_level]
                == HACMiniAgent.PolicyMode.SUBGOAL_TEST
                or random.random() < self.lambda_
            ):
                self.training_modes[
                    current_level - 1
                ] = HACMiniAgent.PolicyMode.SUBGOAL_TEST
                logging.debug(
                    f"\t\t\tChanged mode to subgoal test for level {current_level - 1}"
                )
            self.step_counts[current_level] += 1
            logging.debug(
                f"\t\tIncreasing level {current_level}'s step counts to {self.step_counts[current_level]}."
            )
            current_level -= 1

        assert (
            current_level == 0
        ), f"\tAfter downwards pass, current level should be 0 but is instead {current_level}"

        # At the bottom level, produce a deterministic environment action to return
        deterministic_action = policies[0].get_actions(state, self.subgoals[0])

        # Modify environment action for exploration, depending on current policies and modes
        action_modifier = self.get_action_modifier(
            exploration_policies[0], self.training_modes[current_level],
        )
        action = action_modifier(state, deterministic_action)

        # Update step counts at the lowest layer
        self.step_counts[0] += 1
        logging.debug(f"\tIncreasing BASE level's step counts to {self.step_counts[0]}")

        # Check special case where initial prev_subgoals were all None
        if self.prev_subgoals[0] is None:
            logging.debug("\t\tSPECIAL CASE: Initializing prev subgoals!")
            self.prev_subgoals = [sg.clone() for sg in self.subgoals]

        logging.debug(f"\tEnding step counts: {self.step_counts}")
        return action, False

    def record_hindsight_action(
        self,
        buffer: ReplayBuffer,
        state: torch.Tensor,
        next_state: torch.Tensor,
        goal: torch.Tensor,
        gamma: float,
        done: bool,
        reached_goal: bool,
    ):
        # In Hindsight Action transitions, we pretend that the action we gave the lower-level agent is exactly the next state it achieved
        buffer.add_transitions(
            states=state,
            actions=next_state,
            rewards=torch.Tensor([0.0 if reached_goal else -1.0]).to(self.device),
            next_states=next_state,
            goals=goal,
            gammas=torch.Tensor([0.0 if reached_goal else gamma]).to(self.device),
            dones=torch.Tensor([done]).to(self.device),
        )

    def record_hindsight_goal(
        self,
        buffer: ReplayBuffer,
        experience: List[List],
        pretend_goal_state: torch.Tensor,
    ):
        assert (
            len(experience) <= self.horizon
        ), f"Experience has length {len(experience)} but should never be longer than {self.horizon}!"

        # In Hindsight Goal transitions, we pretend that the state finally achieved at the end of the experience was the goal all along

        # Set last transition's reward and discount to 0, since the pretend goal was "achieved"
        experience[-1][2] = torch.Tensor([0.0]).to(self.device)
        experience[-1][5] = torch.Tensor([0.0]).to(self.device)

        for transition in experience:
            # Fill in the TBD goal with the pretend goal
            transition[4] = pretend_goal_state

            # Unpack transition and add to buffer
            buffer.add_transitions(*transition)

        experience.clear()

    def record_transition(
        self,
        buffers: List[ReplayBuffer],
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        logging.debug("\tRecording transition...")
        logging.debug(f"\tRecording step counts: {self.step_counts}")

        # Update levels from bottom to top based on the received transition
        current_level = 0
        while current_level < self.k and (
            current_level == 0 or self.step_counts[current_level - 1] == 1
        ):
            logging.debug(f"\t\tRecording transition's impact on level {current_level}")
            # Extract the buffer for this level
            buffer = buffers[current_level]

            # Extract the goal for this level
            goal = self.prev_subgoals[current_level].to(self.device)

            # Check if the current state satisfies our goal criterion
            reached_goal = torch.norm(goal - next_state) < self.goal_threshold
            logging.debug(
                f"\t\tAt level {current_level}, reached_goal is: {reached_goal}"
            )

            # Hindsight Actions only occur at higher levels
            if current_level > 0:
                logging.debug(
                    f"\t\t\tAt level {current_level}, we are adding a hindsight action!"
                )
                # Add the current transition as a Hindsight Action
                self.record_hindsight_action(
                    buffer=buffer,
                    state=state,
                    next_state=next_state,
                    goal=goal,
                    gamma=self.gamma,
                    done=done,
                    reached_goal=reached_goal,
                )

            experience = self.experiences[current_level]

            # If the experience is complete, then add the experience as a Hindsight Goal
            if done or reached_goal or self.step_counts[current_level] == 1:
                logging.debug(
                    f"\t\t\tConsidering experience of length {len(experience)} at level {current_level}"
                )
                logging.debug(
                    f"\t\t\tCaused by: Done: {done} Reached Goal: {reached_goal} Step Counts reset: {self.step_counts[current_level]}"
                )

                if len(experience) > 0:
                    self.training_modes[
                        current_level
                    ] = HACMiniAgent.PolicyMode.EXPLORE_RANDOM

                    logging.debug(
                        f"\t\t\t\tRecording hindsight goal at level {current_level}"
                    )
                    self.record_hindsight_goal(
                        buffer=buffer,
                        experience=experience,
                        pretend_goal_state=next_state,
                    )

                    # If this transition represents a failed subgoal test, add the penalty to buffer
                    if (
                        current_level > 0
                        and self.training_modes[current_level - 1]
                        == HACMiniAgent.PolicyMode.SUBGOAL_TEST
                    ):
                        logging.debug(
                            f"\t\t\t\t\tConsidering the subgoal at level {current_level}"
                        )
                        if (
                            torch.norm(
                                self.prev_subgoals[current_level - 1] - next_state
                            )
                            >= self.goal_threshold
                        ):
                            logging.debug(
                                f"\t\t\t\t\t\tSubgoal FAILED for level {current_level}"
                            )
                            buffer.add_transitions(
                                states=state,
                                actions=self.prev_subgoals[current_level - 1],
                                rewards=torch.Tensor([-1 * self.horizon]).to(
                                    self.device
                                ),
                                next_states=next_state,
                                goals=goal,
                                gammas=torch.Tensor([0.0]).to(self.device),
                                dones=torch.Tensor([done]).to(self.device),
                            )

                else:
                    logging.debug(
                        f"\t\t\tSPECIAL CASE: The experience was empty, should only happen one time at this level {current_level}!"
                    )

            # Append transition to the current HER experience, which has goal TBD
            experience.append(
                [
                    state,
                    action if current_level == 0 else next_state,
                    torch.Tensor([-1.0]).to(
                        self.device
                    ),  # Haven't reached the TBD goal yet, so assume -1.0
                    next_state,
                    None,  # Goal will be decided based on where experience ends
                    torch.Tensor([self.gamma]).to(self.device),
                    torch.Tensor([done]).to(self.device),
                ]
            )
            logging.debug(
                f"\t\tAt level {current_level}, the experience is now {len(experience)} long"
            )

            current_level += 1


class HACCoordinator:
    def __init__(
        self,
        env: VecTask,
        initial_state: torch.Tensor,
        goal_state: torch.Tensor,
        params: HACParams,
        exploration_params: ExplorationParams,
        device: str = "cuda:0",
    ):
        self.env = env
        state_size = self.env.num_obs
        action_size = self.env.num_acts

        action_min, action_max = (
            -1 * torch.ones(action_size),
            torch.ones(action_size),
        )

        STATE_MAX_SIZE = (
            1e4  # Arbitrarily high max size, since actual environment is infinite
        )
        state_min, state_max = (
            -STATE_MAX_SIZE * torch.ones(state_size),
            STATE_MAX_SIZE * torch.ones(state_size),
        )

        # Store hyperparameters
        self.lambda_ = params.lambda_  # Lambda is a reserved keyword in Python
        self.gamma = params.gamma
        self.k = params.k
        horizon = params.horizon
        lr = params.lr
        self.action_std = params.action_std
        self.state_std = params.state_std
        self.goal_threshold = params.goal_threshold
        self.batch_size = params.batch_size
        self.num_update_steps = params.num_update_steps

        self.env = env
        self.device = device

        self.states = initial_state["obs"]

        self.agents = [
            HACMiniAgent(
                agent_id=i,
                action_size=action_size,
                goal_state=goal_state,
                params=params,
                exploration_params=exploration_params,
                device=device,
            )
            for i in range(self.env.num_environments)
        ]

        # For each of exploitation policies, exploration policies, and replay buffers,
        # the lowest level has an action size of the true action dimension of the environment,
        # while all higher levels output state subgoals as 'actions'
        self.policies = [
            # Lowest layer will output actions as actual environment actions
            DDPGPolicy(
                state_size=state_size,
                action_size=action_size,
                goal_size=state_size,
                action_min=action_min,
                action_max=action_max,
                state_min=state_min,
                state_max=state_max,
                learning_rate=lr,
                horizon=horizon,
                device=self.device,
            )
        ] + [
            # The (k-1) remaining layers will output actions as environment
            # states that are subgoals for lower layers
            DDPGPolicy(
                state_size=state_size,
                action_size=state_size,
                goal_size=state_size,
                # Note that since actions are environment states, so too must the action
                # min and max be environment state min and max
                action_min=state_min,
                action_max=state_max,
                state_min=state_min,
                state_max=state_max,
                learning_rate=lr,
                horizon=horizon,
                device=self.device,
            )
            for _ in range(1, self.k)
        ]

        self.exploration_policies = [
            ExplorationPolicy(
                state_size=state_size,
                action_size=action_size,
                action_min=action_min,
                action_max=action_max,
                state_min=state_min,
                state_max=state_max,
                learning_rate=lr,
                device=self.device,
            )
        ] + [
            ExplorationPolicy(
                state_size=state_size,
                # Actions are states for all higher levels
                action_size=state_size,
                action_min=state_min,
                action_max=state_max,
                state_min=state_min,
                state_max=state_max,
                learning_rate=lr,
                device=self.device,
            )
            for _ in range(1, self.k)
        ]

        self.replay_buffers = [
            ReplayBuffer(
                state_size=state_size, action_size=action_size, goal_size=state_size
            )
        ] + [
            ReplayBuffer(
                state_size=state_size,
                action_size=state_size,  # Actions are states for all higher levels
                goal_size=state_size,
            )
            for _ in range(1, self.k)
        ]

    def train(self, num_env_steps_per_episode: int):

        # Compute total episode environment reward
        reward = 0

        # Train for the requested number of environment steps
        for _ in tqdm(range(num_env_steps_per_episode)):

            # Collect action from each mini agent
            actions = torch.zeros((self.env.num_environments, self.env.num_acts)).to(
                self.device
            )
            agent_dones = torch.zeros((self.env.num_environments, 1)).to(
                self.device
            )  # NOTE: Not actually used since we can't drive an individual cubicle reset

            for i, agent in enumerate(self.agents):
                action, agent_done = agent.get_action(
                    policies=self.policies,
                    exploration_policies=self.exploration_policies,
                    state=self.states[i],
                )

                actions[i] = action
                agent_dones[i] = agent_done

            # After all actions are batched, send them to the environment
            next_states_dict, _rewards, dones, _ = self.env.step(actions)
            reward += _rewards.mean()
            # Unpack all state observations from the single-entry dictionary
            next_states = next_states_dict["obs"]

            # Record all necessary transitions in the appropriate replay buffers
            for i, agent in enumerate(self.agents):
                agent.record_transition(
                    buffers=self.replay_buffers,
                    state=self.states[i],
                    action=actions[i],
                    next_state=next_states[i],
                    done=dones[i],
                )

            # Reset all agents whose environments are done
            # for agent, done in zip(self.agents, dones):
            #     if done:
            #         agent.reset()

            self.states = next_states

            # input("Waiting for next step... Enter: ")

        print(f"Mean Environment Reward for Episode: {reward}")
        self.update(self.batch_size, self.num_update_steps)

    def update(self, batch_size: int, num_update_steps: int):
        # Update every level in the policy tree
        for i, (policy, exp_policy, buffer) in enumerate(
            zip(self.policies, self.exploration_policies, self.replay_buffers)
        ):
            if len(buffer) < batch_size:
                logging.debug(
                    f"Buffer at level {i} has only {len(buffer)} samples but requested {batch_size}"
                )
                continue

            # Potentially take multiple update steps per iteration
            for _ in range(num_update_steps):

                # Sample from replay buffer at this level
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    goals,
                    gammas,
                    dones,
                ) = buffer.sample(batch_size)

                # logging.debug(f"States: {states}")
                # logging.debug(f"Next states: {next_states}")

                assert not torch.any(torch.isnan(states)), f"States were NAN! {states}"
                assert not torch.any(
                    torch.isnan(actions)
                ), f"actions were NAN! {actions}"
                assert not torch.any(
                    torch.isnan(rewards)
                ), f"rewards were NAN! {rewards}"
                assert not torch.any(
                    torch.isnan(next_states)
                ), f"next_states were NAN! {next_states}"
                assert not torch.any(torch.isnan(goals)), f"goals were NAN! {goals}"
                assert not torch.any(torch.isnan(gammas)), f"gammas were NAN! {gammas}"
                assert not torch.any(torch.isnan(dones)), f"dones were NAN! {dones}"

                # Update policy with the sample
                policy.update(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    goals=goals,
                    gammas=gammas,
                    dones=dones,
                )

                # Update exploration policy
                exp_policy.update(
                    states=states, actions=actions, next_states=next_states
                )

    def save(self, save_location: Path) -> None:
        for i, policy in enumerate(self.policies):
            policy.save(save_location=save_location / f"level_{i}")

    def load(self, save_location: Path) -> None:
        for i, policy in enumerate(self.policies):
            policy.load(save_location=save_location / f"level_{i}")
