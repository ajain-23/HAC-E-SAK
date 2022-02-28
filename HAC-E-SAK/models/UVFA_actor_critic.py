import torch
import torch.nn as nn


class UVFAActor(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        goal_size: int,
        action_min: torch.tensor,
        action_max: torch.tensor,
        hidden_layer_size: int = 64,
    ):
        super(UVFAActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size

        # Based on UVFA paper
        self.actor = nn.Sequential(
            # Current state and goal state are combined input to the first layer
            nn.Linear(self.state_size + self.goal_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, self.action_size),
            nn.Tanh(),
        )

        assert (
            action_min.size(dim=-1) == self.action_size
        ), f"Action minima with size {action_min.size} does not match expected size {self.action_size} in relevant dimension!"
        assert (
            action_max.size(dim=-1) == self.action_size
        ), f"Action maxima with size {action_max.size} does not match expected size {self.action_size} in relevant dimension!"

        # Parameters to linearly transform output action range from
        # tanh's limited output range of [-1, 1] to environment's actual
        # action range [action_min, action_max]
        self.action_mean = (action_max + action_min) / 2
        self.action_range = (action_max - action_min) / 2

    def forward(self, state: torch.tensor, goal: torch.tensor) -> torch.tensor:
        assert (
            state.size(dim=-1) == self.state_size
        ), f"State with size {state.size} does not match expected size {self.state_size} in relevant dimension!"
        assert (
            goal.size(dim=-1) == self.goal_size
        ), f"Goal with size {goal.size} does not match expected size {self.goal_size} in relevant dimension!"

        # Concatenate state and goal for UVFA input
        combined_state_goal = torch.cat((state, goal), dim=-1)
        action_normalized = self.actor(combined_state_goal)

        # Scale the net's [-1, 1] action output to appropriate action range
        action_unnormalized = action_normalized * self.action_range + self.action_mean
        return action_unnormalized


class UVFACritic(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        goal_size: int,
        horizon: int,
        hidden_layer_size: int = 64,
    ):
        super(UVFACritic, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.horizon = horizon

        # Based on UVFA paper
        self.critic = nn.Sequential(
            nn.Linear(state_size + action_size + goal_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, goal: torch.Tensor
    ) -> float:
        assert (
            state.size(dim=-1) == self.state_size
        ), f"State with size {state.size} does not match expected size {self.state_size} in relevant dimension!"
        assert (
            action.size(dim=-1) == self.action_size
        ), f"Action with size {action.size} does not match expected size {self.action_size} in relevant dimension!"
        assert (
            goal.size(dim=-1) == self.goal_size
        ), f"Goal with size {goal.size} does not match expected size {self.goal_size} in relevant dimension!"

        combined_state_action_goal = torch.cat((state, action, goal), dim=-1)

        # Scale the net's score output from [0, 1] to [-H, 0]
        critic_score = -1 * self.horizon * self.critic(combined_state_action_goal)
        return critic_score
