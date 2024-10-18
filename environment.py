import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from interpolator import interpolate
from load_field_data import GetFields


class SoilEnvironment(gym.Env):
    def __init__(self, data: GetFields, f1: float = -0.3, f2: float = -10, starting_position_x: int = 0, rmse_threshold: float = 0.1) -> None:
        super(SoilEnvironment, self).__init__()

        # Set constants
        self.f1 = f1
        self.f2 = f2
        self.start_position_x = starting_position_x
        self.rmse_threshold = rmse_threshold  # Stop condition based on RMSE

        # Set data
        self.data = data

        # Action space: Steps from 1 to 40
        self.steps_from_current_position = list(range(1, 81))  # Steps between 1 and 40
        self.action_space = spaces.Discrete(len(self.steps_from_current_position))

        # Define observation space (same as before)
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 805, 3), dtype=np.uint8)

        # For logging
        self.rmse = 100

    def reset(self, seed: int = None) -> tuple[np.ndarray, dict]:
        """Reset the environment to the initial state"""
        super().reset(seed=seed)

        # Simulate the fields
        self.real_field = self.data.get_field().astype(np.uint8)  # Ensure data is uint8 for RGB
        self.field_with_holes = np.full_like(self.real_field, 0, dtype=np.uint8)
        self.ip_field = np.full_like(self.real_field, 0, dtype=np.float32)  # Keep interpolation in float32
        self.current_position = self.start_position_x
        self.grid_size = self.real_field.shape

        # Save holes and ic_values
        self.num_holes = 0
        self.x_coords = np.array([], dtype=np.float32)
        self.y_coords = np.array([], dtype=np.float32)
        self.ic_values = np.array([], dtype=np.float32)

        # Get the initial hole
        hole = self.get_hole(self.current_position)

        return self.real_field, {}

    def get_hole(self, hole_x: int) -> np.ndarray:
        """Return the full RGB data (80 pixels deep) at the specified x position."""
        if self._is_done():
            return np.zeros((80, 805, 3), dtype=np.uint8)  # Return zeros if out of bounds

        # Save coords from hole
        self.x_coords = np.hstack((self.x_coords, np.full(self.grid_size[0], hole_x)))
        self.y_coords = np.hstack((self.y_coords, np.arange(self.grid_size[0])))

        # Collect RGB values for interpolation
        rgb_values = self.real_field[:, hole_x, :]
        self.ic_values = np.vstack((self.ic_values, rgb_values)) if self.ic_values.size else rgb_values

        # Dig the hole
        self.field_with_holes[:, hole_x, :] = self.real_field[:, hole_x, :]
        self.num_holes += 1

        return self.field_with_holes

    def _is_done(self) -> bool:
        # Check if position is out of grid
        return self.current_position >= self.grid_size[1]

    def _calc_reward(self) -> float:
        """Calculate the reward, balancing fewer holes and better interpolation."""
        # Normalize RMSE to range [0, 1]
        max_rmse = 120  # Example value, adjust based on your data
        normalized_rmse = min(self._calc_rmse() / max_rmse, 1.0)

        # Adjust reward scaling
        hole_penalty = self.f1 * self.num_holes  # Penalize digging more holes
        accuracy_reward = 100*(1 - normalized_rmse)  # Reward for better accuracy

        # Combine penalties and rewards
        reward = accuracy_reward + hole_penalty

        return reward

    def _calc_rmse(self) -> float:
        """Calculate RMSE of interpolated field"""
        if self.ip_field is not None:
            rmse = np.sqrt(np.mean((self.real_field.astype(np.float32) - self.ip_field) ** 2))
            return rmse
        return 10e2

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment based on the discrete action."""
        # Move according to the action, which is an index into the steps_from_current_position list
        step_size = self.steps_from_current_position[action]
        self.current_position += step_size

        # Get a hole and move
        hole = self.get_hole(self.current_position)

        # Check if the episode is done
        truncated = self._is_done()

        # Interpolate field
        self.ip_field = interpolate(self.x_coords, self.y_coords, self.ic_values, self.grid_size, method="linear")

        # Calculate reward
        reward = self._calc_reward()

        # Calculate RMSE and stop if the RMSE is below the threshold
        rmse = self._calc_rmse()
        if rmse < self.rmse_threshold:
            done = True
        else:
            done = truncated

        # Ensure state is in uint8 for observation space
        state = self.real_field.astype(np.uint8)

        return state, reward, done, truncated, {"rmse": rmse}

    def render(self, mode="human") -> None:
        # Create a figure and a set of subplots
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

        # Show the plots
        plt.tight_layout()
        plt.show()


# Test the environment
data = GetFields()
data.load_data_from_file("generate_simulated_fields/training_data/test_data_10_1.txt")

env = SoilEnvironment(data=data)
check_env(env)




