from typing import List, Tuple
import random
import torch


class ReplayBuffer:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        goal_size: int,
        max_length: int = 500000,
        device: str = "cuda:0",
    ):
        self.max_length = max_length
        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size

        self.states = torch.empty((0, state_size)).to(device)
        self.actions = torch.empty((0, action_size)).to(device)
        self.rewards = torch.empty((0, 1)).to(device)
        self.next_states = torch.empty((0, state_size)).to(device)
        self.goals = torch.empty((0, goal_size)).to(device)
        self.gammas = torch.empty((0, 1)).to(device)
        self.dones = torch.empty((0, 1)).to(device)

    def __len__(self):
        return self.states.size(0)

    def add_transitions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        goals: torch.Tensor,
        gammas: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        # Verify that the transitions are appropriately shaped
        # TODO(JS): This test isn't working!
        # batch_size = states.size(dim=0)
        # assert (
        #     actions.size(dim=0) == batch_size
        #     and rewards.size(dim=0) == batch_size
        #     and next_states.size(dim=0) == batch_size
        #     and goals.size(dim=0) == batch_size
        #     and gammas.size(dim=0) == batch_size
        #     and dones.size(dim=0) == batch_size
        # ), (
        #     f"Inconsistent batch size!\n "
        #     f"States: {states.size()}\n "
        #     f"Actions: {actions.size()}\n "
        #     f"Rewards: {rewards.size()}\n "
        #     f"Next States: {next_states.size()}\n "
        #     f"Goals: {goals.size()}\n "
        #     f"Gammas: {gammas.size()}\n "
        #     f"Dones: {dones.size()}"
        # )

        assert (
            states.size(-1) == self.state_size
        ), f"Transitions' current states size {states.size} did not match expected size {self.state_size} in relevant dimension!"
        assert (
            actions.size(-1) == self.action_size
        ), f"Transitions' actions size {actions.size} did not match expected size {self.action_size} in relevant dimension!"
        assert (
            rewards.size(-1) == 1
        ), f"Transitions' rewards size {rewards.size} did not match expected size {1} in relevant dimension!"
        assert (
            next_states.size(-1) == self.state_size
        ), f"Transitions' next states size {next_states.size} did not match expected size {self.state_size} in relevant dimension!"
        assert (
            goals.size(-1) == self.goal_size
        ), f"Transitions' goals size {goals.size} did not match expected size {self.goal_size} in relevant dimension!"
        assert (
            gammas.size(-1) == 1
        ), f"Transitions' gammas size {gammas.size} did not match expected size {1} in relevant dimension!"
        assert (
            dones.size(-1) == 1
        ), f"Transitions' dones size {dones.size} did not match expected size {1} in relevant dimension!"

        assert not any(torch.isnan(states)), f"States were NAN! {states}"
        assert not any(torch.isnan(actions)), f"actions were NAN! {actions}"
        assert not any(torch.isnan(rewards)), f"rewards were NAN! {rewards}"
        assert not any(torch.isnan(next_states)), f"next_states were NAN! {next_states}"
        assert not any(torch.isnan(goals)), f"goals were NAN! {goals}"

        self.states = torch.vstack((self.states, states))[-self.max_length :]
        self.actions = torch.vstack((self.actions, actions))[-self.max_length :]
        self.rewards = torch.vstack((self.rewards, rewards))[-self.max_length :]
        self.next_states = torch.vstack((self.next_states, next_states))[
            -self.max_length :
        ]
        self.goals = torch.vstack((self.goals, goals))[-self.max_length :]
        self.gammas = torch.vstack((self.gammas, gammas))[-self.max_length :]
        self.dones = torch.vstack((self.dones, dones))[-self.max_length :]

    def sample(
        self, batch_size=1
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        assert (
            len(self) >= batch_size
        ), f"Cannot sample batch of size {batch_size} when buffer has length of only {len(self)}!"
        rand_indices = random.sample(list(range(self.states.size(0))), k=batch_size)
        return (
            self.states[rand_indices],
            self.actions[rand_indices],
            self.rewards[rand_indices],
            self.next_states[rand_indices],
            self.goals[rand_indices],
            self.gammas[rand_indices],
            self.dones[rand_indices],
        )


# if __name__ == "__main__":
#     buffer = ReplayBuffer(20, 5, 20)
#     states = torch.ones(10, 20).to('cuda:0')
#     actions = torch.ones(10, 5).to('cuda:0')
#     rewards = torch.ones(10, 1).to('cuda:0')
#     next_states = torch.ones(10, 20).to('cuda:0')
#     goals = torch.ones(10, 20).to('cuda:0')
#     gammas = torch.ones(10, 1).to('cuda:0')
#     dones = torch.ones(10, 1).to('cuda:0')

#     buffer.add_transitions(states, actions, rewards, next_states, goals, gammas, dones)
#     print(buffer.sample())
