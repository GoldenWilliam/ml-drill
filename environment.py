import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from interpolator import interpolate
from load_field_data import GetFields


class SoilEnvironment(gym.Env):
    def __init__(self, data: GetFields, weight_hole_number: float = -0.3, weight_accuracy: float = -10, starting_position_x: int = 0, rmse_threshold: float = 0.1) -> None:
        super(SoilEnvironment, self).__init__()

        # Set constants
        self.weight_hole_number = weight_hole_number
        self.weight_accuracy = weight_accuracy
        self.start_position_x = starting_position_x
        self.rmse_threshold = rmse_threshold  # Stop condition based on RMSE

        # Set data
        self.data = data

        # Action space: Steps from 1 to 80
        self.steps_from_current_position = list(range(1, 81))  # Steps between 1 and 80
        self.action_space = spaces.Discrete(len(self.steps_from_current_position))

        # Define observation space: (channels, height, width)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 80, 805), dtype=np.uint8)

        # For logging
        self.rmse = 100
        self.step_count = 0  # Initialize step count

    def reset(self, seed: int = None) -> tuple[np.ndarray, dict]:
        """Reset the environment to the initial state"""
        super().reset(seed=seed)

        # Log that reset is being called
        print("Environment reset, resetting step count to 0. Reset triggered due to episode end or explicit call.")

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

        # Reset step count
        self.step_count = 0

        # Convert (height, width, channels) to (channels, height, width)
        state = np.transpose(self.real_field, (2, 0, 1))

        return state, {}

    def get_hole(self, hole_x: int) -> np.ndarray:
        """Return the full RGB data (80 pixels deep) at the specified x position."""
        if self._is_done():
            return np.zeros((3, 80, 805), dtype=np.uint8)  # Return zeros if out of bounds

        # Save coords from hole
        self.x_coords = np.hstack((self.x_coords, np.full(self.grid_size[0], hole_x)))
        self.y_coords = np.hstack((self.y_coords, np.arange(self.grid_size[0])))

        # Collect RGB values for interpolation
        rgb_values = self.real_field[:, hole_x, :]
        self.ic_values = np.vstack((self.ic_values, rgb_values)) if self.ic_values.size else rgb_values

        # Dig the hole
        self.field_with_holes[:, hole_x, :] = self.real_field[:, hole_x, :]
        self.num_holes += 1

        # Return the field with holes in (channels, height, width) format
        return np.transpose(self.field_with_holes, (2, 0, 1))

    def _is_done(self) -> bool:
        # Check if position is out of grid
        return self.current_position >= self.grid_size[1]

    def _calc_reward(self) -> float:
        """Calculate the reward, balancing fewer holes and better interpolation."""
        # Normalize RMSE to range [0, 1]
        max_rmse = 120  # Example value, adjust based on your data
        normalized_rmse = min(self._calc_rmse() / max_rmse, 1.0)

        # Adjust reward scaling
        hole_penalty = self.weight_hole_number * self.num_holes  # Penalize digging more holes
        accuracy_reward = 100 * (1 - normalized_rmse)  # Reward for better accuracy

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

        # Increment step count
        self.step_count += 1

        # Ensure the current position doesn't exceed the width of the field
        new_position = self.current_position + step_size
        if new_position >= self.grid_size[1]:
            # If the new position exceeds the field width, cap it at the last valid position
            self.current_position = self.grid_size[1] - 1
        else:
            self.current_position = new_position

        # Print step number, step size, and current position
        print(
            f"Step number: {self.step_count}, taking step of size: {step_size}, current position: {self.current_position}")

        # Get a hole and move
        hole = self.get_hole(self.current_position)

        # Interpolate field
        self.ip_field = interpolate(self.x_coords, self.y_coords, self.ic_values, self.grid_size, method="linear")

        # Calculate RMSE
        rmse = self._calc_rmse()
        print(f"Current RMSE: {rmse}")

        # Calculate reward
        reward = self._calc_reward()
        print(f"Reward calculated: {reward}, RMSE: {rmse}")

        # Check if the episode is done based on RMSE threshold
        terminated = bool(rmse < self.rmse_threshold)
        if terminated:
            print(f"Stopping episode as RMSE {rmse} is below the threshold {self.rmse_threshold}")

        # Check if the agent has reached the end of the grid
        truncated = bool(self.current_position == self.grid_size[1] - 1)
        if truncated:
            print(f"Stopping episode as the agent has reached the last valid position: {self.current_position}")
            terminated = True  # If truncated, the episode is also terminated

        # Log the specific reset condition
        if terminated:
            print(f"Reset due to RMSE dropping below {self.rmse_threshold}.")
        if truncated:
            print(f"Reset due to reaching the last valid position: {self.current_position}.")

        # Convert (height, width, channels) to (channels, height, width)
        state = np.transpose(self.real_field, (2, 0, 1))

        return state, reward, terminated, truncated, {"rmse": rmse}

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

# Check the environment for issues
check_env(env)

# Test environment interaction with basic steps
done = False
state = env.reset()

while not done:
    action = env.action_space.sample()  # Random action for testing
    state, reward, done, truncated, info = env.step(action)
    if done or truncated:
        print("Resetting due to done or truncated.")
        state = env.reset()  # Reset only when done or truncated is True











