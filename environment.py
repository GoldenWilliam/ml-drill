import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from interpolator import interpolate
from load_field_data import GetFields

class SoilEnvironment(gym.Env):
    def __init__(self, data: GetFields, weight_hole_number: float = -2.0, weight_accuracy: float = 10, rmse_threshold: float = 20.0) -> None:
        super(SoilEnvironment, self).__init__()

        # Set constants and data
        self.weight_hole_number = weight_hole_number
        self.weight_accuracy = weight_accuracy
        self.rmse_threshold = rmse_threshold
        self.data = data

        # Grid dimensions
        self.grid_size = (80, 805)
        self.action_space = spaces.Discrete(self.grid_size[1])
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, self.grid_size[0], self.grid_size[1]), dtype=np.uint8)

        # For tracking and efficiency
        self.rmse = 100
        self.step_count = 0
        self.dug_positions = {}
        self.repeat_penalty = -20
        self.total_episode_reward = 0
        self.num_holes = 0

        # Pre-allocate arrays for hole coordinates and values, assuming a max of 10 digs with full depth
        self.max_digs = 10
        self.x_coords = np.zeros(self.max_digs * self.grid_size[0], dtype=np.float32)
        self.y_coords = np.tile(np.arange(self.grid_size[0]), self.max_digs)
        self.ic_values = np.zeros((self.max_digs * self.grid_size[0], 3), dtype=np.float32)
        self.current_hole_index = 0  # Track where to add new data

    def reset(self, seed: int = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.real_field = self.data.get_field().astype(np.uint8)
        self.field_with_holes = np.full_like(self.real_field, 0, dtype=np.uint8)
        self.ip_field = np.full_like(self.real_field, 0, dtype=np.float32)

        # Reset state
        self.num_holes = 0
        self.current_hole_index = 0
        self.dug_positions.clear()
        self.total_episode_reward = 0
        self.step_count = 0
        self.rmse = 100  # Reset RMSE to start value

        state = np.transpose(self.real_field, (2, 0, 1))
        return state, {}

    def get_hole(self, x: int) -> np.ndarray:
        if self._is_done():
            return np.zeros((3, self.grid_size[0], self.grid_size[1]), dtype=np.uint8)

        self.num_holes += 1
        self.dug_positions[x] = self.dug_positions.get(x, 0) + 1

        # Efficiently store coordinates and RGB values without reallocating
        start = self.current_hole_index
        end = start + self.grid_size[0]
        self.x_coords[start:end] = x
        self.ic_values[start:end] = self.real_field[:, x, :]
        self.current_hole_index += self.grid_size[0]

        # Dig the hole visually in the field representation
        self.field_with_holes[:, x, :] = self.real_field[:, x, :]

        return np.transpose(self.field_with_holes, (2, 0, 1))

    def _is_done(self) -> bool:
        """Directly checks both the total digs and RMSE threshold for termination."""
        return self.num_holes >= 10 or self.rmse < self.rmse_threshold

    def _calc_rmse(self) -> float:
        if self.ip_field is not None:
            return np.sqrt(np.mean((self.real_field.astype(np.float32) - self.ip_field) ** 2))
        return 10e2

    # rest of the code remains unchanged...


    def _calc_reward(self) -> float:
        max_rmse = 120
        normalized_rmse = min(self.rmse / max_rmse, 1.0)  # Use stored rmse

        # Hole Penalty: Cumulative and increases with each additional hole
        hole_penalty = self.weight_hole_number * self.num_holes

        # Accuracy Reward: Calculated fresh each step based on normalized RMSE
        accuracy_reward = self.weight_accuracy * (1 - normalized_rmse)

        # Efficiency Reward: Calculated fresh each step, favors fewer holes.
        efficiency_reward = 5 / max(self.num_holes, 1)

        # Total base reward before any exploration bonus
        total_reward = accuracy_reward + hole_penalty + efficiency_reward

        # Consolidated output for reward and RMSE
        print(f"Calculated RMSE: {self.rmse}")
        print(f"\nReward Calculation (Step {self.step_count}):")
        print(f"  - Hole Penalty (Cumulative): {hole_penalty}")
        print(f"  - Accuracy Reward (Based on RMSE): {accuracy_reward}")
        print(f"  - Efficiency Reward (Based on RMSE and num_holes): {efficiency_reward}")
        print(f"  - Total Reward (before Exploration Bonus): {total_reward}")

        return total_reward


    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        x = int(action)  # x represents the chosen coordinate along the width for digging
        self.step_count += 1

        print(f"\nStep {self.step_count}: Digging at x position {x}")

        # Check if the hole at this x position has already been dug
        if x in self.dug_positions:
            # Apply cumulative penalty for re-digging the same hole, increasing with each repeat
            penalty = self.repeat_penalty * self.dug_positions[x]
            reward = penalty
            print(f"Warning: Hole at x position {x} has already been dug. Applying penalty: {penalty}")
            self.num_holes += 1
        else:
            # Mark this x position as having been dug
            self.dug_positions[x] = 1  # Set initial visit count for this x position

            # Dig a hole at the specified x position
            hole = self.get_hole(x)

            # Interpolate field and calculate RMSE once
            self.ip_field = interpolate(self.x_coords, self.y_coords, self.ic_values, self.grid_size, method="linear")
            self.rmse = self._calc_rmse()  # Calculate and store RMSE once

            # Calculate base reward and add exploration bonus
            base_reward = self._calc_reward()
            reward = base_reward + 5  # Exploration bonus
            print(f"  - Exploration Bonus Added: 15")
            print(f"  - Final Reward for new dig: {reward}")

        # Add this step’s reward to the total episode reward
        self.total_episode_reward += reward
        print(f"Updated total episode reward: {self.total_episode_reward}")

        # Check if the episode is terminated based on total digs or RMSE threshold
        terminated = bool(self._is_done() or self.rmse < self.rmse_threshold)
        if terminated:
            if self.rmse < self.rmse_threshold:
                print(f"\nEpisode terminated as RMSE {self.rmse} fell below threshold {self.rmse_threshold}.")
            else:
                print(f"\nEpisode terminated as the total number of digs reached threshold.")
            print(f"Final Total Episode Reward: {self.total_episode_reward}")

        truncated = False
        state = np.transpose(self.real_field, (2, 0, 1))

        return state, reward, terminated, truncated, {"rmse": self.rmse}

    def render(self, mode="human") -> None:
        fig, ax = plt.subplots(3, 1, figsize=(6, 12))

        # Plot the real field
        ax[0].imshow(self.real_field, origin='lower')
        ax[0].set_title('Original Field')

        # Plot the field with holes
        ax[1].imshow(self.field_with_holes, origin='lower')
        ax[1].set_title(f'Field with Holes: {self.num_holes}')

        # Plot the interpolated field
        ax[2].imshow(self.ip_field.astype(np.uint8), origin='lower')
        ax[2].set_title('Interpolated Field')

        plt.tight_layout()
        plt.show()









# # Test the environment
# data = GetFields()
# data.load_data_from_file("generate_simulated_fields/training_data/test_data_10_1.txt")
#
# env = SoilEnvironment(data=data)
#
# # Check the environment for issues
# check_env(env)
#
# # Test environment interaction with basic steps
# done = False
# state = env.reset()
#
# while not done:
#     action = env.action_space.sample()  # Random action for testing
#     state, reward, done, truncated, info = env.step(action)
#     if done or truncated:
#         print("Resetting due to done or truncated.")
#         state = env.reset()  # Reset only when done or truncated is True











