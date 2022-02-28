from typing import Dict, Tuple, Iterable
import torch
from torch import optim
from pathlib import Path

from HACKerMan.models.UVFA_actor_critic import UVFAActor, UVFACritic


class DDPGPolicy:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        goal_size: int,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        state_min: torch.Tensor,
        state_max: torch.Tensor,
        learning_rate: float,
        horizon: int,
        device: str = "cuda:0",
    ):
        action_min = action_min.to(device)
        action_max = action_max.to(device)
        state_min = state_min.to(device)
        state_max = state_max.to(device)

        self.device = device

        self.actor = UVFAActor(
            state_size=state_size,
            action_size=action_size,
            goal_size=goal_size,
            action_min=action_min,
            action_max=action_max,
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = UVFACritic(
            state_size=state_size,
            action_size=action_size,
            goal_size=goal_size,
            horizon=horizon,
        ).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.loss = torch.nn.MSELoss()

    def get_actions(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return self.actor(states.to(self.device), goals.to(self.device)).detach()

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        goals: torch.Tensor,
        gammas: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        batch_size = states.size(dim=0)
        assert (
            actions.size(dim=0) == batch_size
            and rewards.size(dim=0) == batch_size
            and next_states.size(dim=0) == batch_size
            and goals.size(dim=0) == batch_size
            and gammas.size(dim=0) == batch_size
            and dones.size(dim=0) == batch_size
        ), (
            f"Inconsistent batch size!\n "
            f"States: {states.size()}\n "
            f"Actions: {actions.size()}\n "
            f"Rewards: {rewards.size()}\n "
            f"Next States: {next_states.size()}\n "
            f"Goals: {goals.size()}\n "
            f"Gammas: {gammas.size()}\n "
            f"Dones: {dones.size()}"
        )

        # Get critic's predictions for next states
        next_actions = self.actor(next_states, goals).detach()
        next_q_vals = self.critic(next_states, next_actions, goals).detach()
        discounted_q_vals = rewards + ((1 - dones) * gammas * next_q_vals)

        # After discount, critic's values for next states should match critic's values for current states
        critic_loss = self.loss(self.critic(states, actions, goals), discounted_q_vals)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Get actor's proposed actions at current state
        proposed_actions = self.actor(states, goals)

        # Evaluate proposed actions using updated critic
        actor_loss = -1 * self.critic(states, proposed_actions, goals).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def save(self, save_location: Path) -> None:
        torch.save(self.actor.state_dict(), save_location / "actor.pth")
        torch.save(self.critic.state_dict(), save_location / "critic.pth")

    def load(self, save_location: Path) -> None:
        self.actor.load_state_dict(torch.load(save_location / "actor.pth"))
        self.critic.load_state_dict(torch.load(save_location / "critic.pth"))
