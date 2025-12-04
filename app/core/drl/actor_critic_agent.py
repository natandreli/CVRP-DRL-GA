import random
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from app.core.drl.cvrp_environment import CVRPEnvironment
from app.core.drl.pointer_network import ActorCriticNetwork
from app.schemas import CVRPInstance, DRLConfig, Route, Solution


class ActorCriticAgent:
    """
    Actor-Critic agent for CVRP using Pointer Network.
    """

    def __init__(
        self,
        instance: CVRPInstance,
        config: DRLConfig,
        callback: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        """
        Initialize Actor-Critic agent.

        Args:
            instance (CVRPInstance): CVRP instance
            config (DRLConfig): DRL configuration
            callback (Optional[Callable[[int, float], None]]): Optional callback(episode, cost)
        """
        self.instance = instance
        self.config = config
        self.callback = callback
        self.env = CVRPEnvironment(instance)
        self.device = torch.device(config.device)
        self.customer_feature_dim = 5

        self.network = ActorCriticNetwork(
            customer_feature_dim=self.customer_feature_dim,
            embedding_dim=256,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
        ).to(self.device)

        # Separate optimizers for Actor and Critic
        actor_params = list(self.network.actor.parameters())
        critic_params = (
            list(self.network.critic_embedding.parameters())
            + list(self.network.critic_lstm.parameters())
            + list(self.network.critic_value.parameters())
        )

        self.optimizer_actor = torch.optim.Adam(
            actor_params, lr=config.learning_rate_actor
        )
        self.optimizer_critic = torch.optim.Adam(
            critic_params, lr=config.learning_rate_critic
        )

        # Exploration
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay

        # Training tracking
        self.episode_rewards = []
        self.episode_costs = []
        self.actor_losses = []
        self.critic_losses = []

    def _state_to_customer_features(self, state: np.ndarray) -> torch.Tensor:
        """
        Convert environment state to customer feature matrix.

        State structure: [current_x, current_y, capacity,
                         (visited, demand, distance, x, y) * num_customers]

        Args:
            state (np.ndarray): State from environment

        Returns:
            Customer features (torch.Tensor): [1, num_customers, 5]
        """
        num_customers = self.env.num_customers
        customer_data = state[3:].reshape(num_customers, 5)
        features = torch.FloatTensor(customer_data).unsqueeze(0).to(self.device)

        return features

    def _get_mask_from_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Reconstruct mask from state history for training stability.
        Index 0 of customer data is 'visited' (1.0 if visited, 0.0 otherwise).

        Args:
            state (np.ndarray): State from environment

        Returns:
            Mask tensor (torch.Tensor): [1, num_customers] - True for visited
        """
        num_customers = self.env.num_customers
        visited_status = state[3:].reshape(num_customers, 5)[:, 0]
        mask = torch.tensor(visited_status, dtype=torch.bool, device=self.device)
        return mask.unsqueeze(0)

    def _get_mask(self, valid_actions: list[int]) -> torch.Tensor:
        """
        Create mask for invalid actions.

        Args:
            valid_actions (list[int]): List of valid customer IDs

        Returns:
            Mask tensor (torch.Tensor): [1, num_customers] - True for invalid
        """
        num_customers = self.env.num_customers
        mask = torch.ones(1, num_customers, dtype=torch.bool, device=self.device)

        for action in valid_actions:
            mask[0, action - 1] = False  # Customer IDs start at 1

        return mask

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: list[int],
        hidden: tuple,
        explore: bool = True,
    ) -> tuple[int, torch.Tensor, torch.Tensor, tuple]:
        """
        Select action using Actor with epsilon-greedy.

        Args:
            state (np.ndarray): Current state
            valid_actions (list[int]): Valid customer IDs
            hidden (tuple): LSTM hidden state
            explore (bool): Whether to use epsilon-greedy

        Returns:
            tuple[int, torch.Tensor, torch.Tensor, tuple]:
                Selected action (customer ID), action probabilities,
                log probabilities, new hidden state
        """
        if not valid_actions:
            raise ValueError("No valid actions available")

        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            action = random.choice(valid_actions)
            customer_features = self._state_to_customer_features(state)
            mask = self._get_mask(valid_actions)
            decoder_input = torch.zeros(1, 1, 256, device=self.device)

            with torch.no_grad():
                probs, log_probs, hidden_new = self.network.forward_actor(
                    customer_features, decoder_input, hidden, mask
                )

            return action, probs, log_probs, hidden_new

        # Greedy: use policy
        customer_features = self._state_to_customer_features(state)
        mask = self._get_mask(valid_actions)
        decoder_input = torch.zeros(1, 1, 256, device=self.device)

        probs, log_probs, hidden_new = self.network.forward_actor(
            customer_features, decoder_input, hidden, mask
        )

        # Sample from policy
        if explore:
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = torch.argmax(probs, dim=1).item()

        action = action_idx + 1

        return action, probs, log_probs, hidden_new

    def train_step(
        self,
        states: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
    ) -> None:
        """
        Train Actor and Critic using collected episode trajectory.

        Args:
            states (list[np.ndarray]): List of states
            actions (list[int]): List of actions taken
            rewards (list[float]): List of rewards received
        """
        if len(states) == 0:
            return

        self.network.train()

        # Compute returns (discounted cumulative rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.config.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Train on trajectory
        total_actor_loss = 0
        # Train on trajectory
        total_actor_loss = 0
        total_critic_loss = 0

        hidden = self.network.init_hidden(1, self.device)

        for i, (state, action) in enumerate(zip(states, actions)):
            # Get features
            customer_features = self._state_to_customer_features(state)
            mask = self._get_mask_from_state(state)

            # Critic
            value = self.network.forward_critic(customer_features)
            advantage = returns[i] - value.squeeze()
            total_critic_loss += advantage.pow(2)

            # Actor
            decoder_input = torch.zeros(1, 1, 256, device=self.device)
            probs, log_probs, hidden = self.network.forward_actor(
                customer_features, decoder_input, hidden, mask
            )

            action_idx = action - 1
            log_prob = log_probs[0, action_idx]

            # Actor Loss with entropy bonus for exploration
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            actor_loss = -log_prob * advantage.detach() - 0.01 * entropy
            total_actor_loss += actor_loss

        # Backprop Actor
        avg_actor_loss = total_actor_loss / len(states)
        avg_critic_loss = total_critic_loss / len(states)
        self.optimizer_actor.zero_grad()
        avg_actor_loss.backward()  # No retain_graph needed anymore
        torch.nn.utils.clip_grad_norm_(self.network.actor.parameters(), 1.0)
        self.optimizer_actor.step()

        # Update Critic
        self.optimizer_critic.zero_grad()
        avg_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.network.critic_embedding.parameters())
            + list(self.network.critic_lstm.parameters())
            + list(self.network.critic_value.parameters()),
            1.0,
        )
        self.optimizer_critic.step()

        # Log metrics
        self.actor_losses.append(avg_actor_loss.item())
        self.critic_losses.append(avg_critic_loss.item())

    def train(self, episodes: Optional[int] = None) -> dict:
        """
        Train agent using Actor-Critic.

        Args:
            episodes (Optional[int]): Number of episodes (overrides config)

        Returns:
            dict: Training statistics
        """
        if episodes is None:
            episodes = self.config.episodes

        start_time = time.time()
        self.network.train()

        for episode in range(episodes):
            state = self.env.reset()
            hidden = self.network.init_hidden(1, self.device)

            episode_states = []
            episode_actions = []
            episode_rewards = []

            steps = 0

            while True:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break

                action, probs, log_probs, hidden = self.select_action(
                    state, valid_actions, hidden, explore=True
                )

                next_state, reward, done, info = self.env.step(action)

                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)

                state = next_state
                steps += 1

                if done:
                    break

            # Train on episode
            self.train_step(episode_states, episode_actions, episode_rewards)

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Track metrics
            total_reward = sum(episode_rewards)
            self.episode_rewards.append(total_reward)
            self.episode_costs.append(self.env.total_distance)

            # Callback
            if self.callback and (episode + 1) % 10 == 0:
                avg_cost = np.mean(self.episode_costs[-10:])
                self.callback(episode + 1, avg_cost)

        training_time = time.time() - start_time

        return {
            "episodes": episodes,
            "training_time": training_time,
            "final_epsilon": self.epsilon,
            "avg_reward_last_10": np.mean(self.episode_rewards[-10:]),
            "avg_cost_last_10": np.mean(self.episode_costs[-10:]),
            "best_cost": min(self.episode_costs),
            "avg_actor_loss": np.mean(self.actor_losses[-10:])
            if self.actor_losses
            else 0,
            "avg_critic_loss": np.mean(self.critic_losses[-10:])
            if self.critic_losses
            else 0,
        }

    def solve(self, instance: CVRPInstance) -> Tuple[float, List[int]]:
        """
        Solve a specific instance deterministically (Greedy).
        Used for validation during training. No gradients are calculated.

        Args:
            instance: The CVRP instance to solve.

        Returns:
            Tuple[float, List[int]]: (Total Cost, Sequence of actions)
        """
        self.instance = instance
        self.env = CVRPEnvironment(instance)
        state = self.env.reset()

        self.network.eval()

        hidden = self.network.init_hidden(1, self.device)
        actions_taken = []

        with torch.no_grad():
            while True:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break

                action, _, _, hidden = self.select_action(
                    state, valid_actions, hidden, explore=False
                )

                state, _, done, _ = self.env.step(action)
                actions_taken.append(action)

                if done:
                    break

        total_cost = self.env.total_distance

        self.network.train()

        return total_cost, actions_taken

    def generate_solution(self, explore: bool = False) -> Solution:
        """
        Generate solution using trained policy.

        Args:
            explore (bool): Use exploration for diversity

        Returns:
            Solution: Generated CVRP solution
        """
        state = self.env.reset()
        self.network.eval()
        hidden = self.network.init_hidden(1, self.device)

        with torch.no_grad():
            while True:
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break

                action, probs, log_probs, hidden = self.select_action(
                    state, valid_actions, hidden, explore=explore
                )
                state, reward, done, info = self.env.step(action)

                if done:
                    break

        self.network.train()

        # Convert to Solution
        solution_routes = []
        for vehicle_id, customer_ids in enumerate(self.env.routes):
            from app.core.utils.cvrp_helpers import get_customers_by_ids
            from app.core.utils.distance_calculator import calculate_route_distance

            customers = get_customers_by_ids(self.instance, customer_ids)
            total_demand = sum(c.demand for c in customers)

            sequence_with_depot = [0] + customer_ids + [0]
            total_distance = calculate_route_distance(
                sequence_with_depot, self.env.distance_matrix
            )

            route = Route(
                vehicle_id=vehicle_id,
                customer_sequence=customer_ids,
                total_demand=total_demand,
                total_distance=total_distance,
            )
            solution_routes.append(route)

        return Solution(
            id=f"drl_{random.randint(1000, 9999)}",
            instance_id=self.instance.id,
            algorithm="drl_actor_critic",
            routes=solution_routes,
            total_cost=self.env.total_distance,
            computation_time=0,
            is_valid=True,
        )

    def save(self, path: str) -> None:
        """
        Save model weights.

        Args:
            path (str): File path to save the model
        """
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer_actor": self.optimizer_actor.state_dict(),
                "optimizer_critic": self.optimizer_critic.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model weights.

        Args:
            path (str): File path to load the model from
        """
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor"])
        self.optimizer_critic.load_state_dict(checkpoint["optimizer_critic"])
        self.epsilon = checkpoint["epsilon"]
