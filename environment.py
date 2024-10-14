import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from interpolator import interpolate
from load_field_data import GetFields


class SoilEnvironment(gym.Env):
    def __init__(self, data: GetFields, f1: float = -1, f2: float = -3, starting_position_x: int = 0,
                 starting_position_y: int = 0) -> None:
        super(SoilEnvironment, self).__init__()

        # Set constants
        self.f1 = f1
        self.f2 = f2
        self.start_position_x = starting_position_x
        self.start_position_y = starting_position_y

        # Set data
        self.data = data

        # Number of steps we can move in from_position (in x and y directions)
        self.steps_from_current_position = [2, 5, 8, 12, 16]

        # Define action and observation space: Move in 4 directions (up, down, left, right)
        self.action_space = spaces.Discrete(4)  # 4 possible moves (up, down, left, right)

        # Observation space to handle 3x3 patches from the field
        self.observation_space = spaces.Box(low=0, high=255, shape=(9,), dtype=np.float64)

        # For logging
        self.rmse = 100

    def reset(self, seed: int = None) -> tuple[np.ndarray, dict]:
        """Reset the environment to the initial state"""
        super().reset(seed=seed)

        # Simulate the fields
        self.real_field = self.data.get_field()  # Assume this returns a 2D or 3D array
        self.field_with_holes = np.full_like(self.real_field, 0)
        self.ip_field = np.full_like(self.real_field, 0)
        self.current_position_x = self.start_position_x
        self.current_position_y = self.start_position_y
        self.grid_size = self.real_field.shape

        # Save holes and ic_values
        self.num_holes = 0
        self.x_coords = np.array([])
        self.y_coords = np.array([])
        self.ic_values = np.array([])

        # Get initial observation (3x3 sub-array at current position)
        observation = self.get_hole(self.current_position_x, self.current_position_y)

        return observation, {}

    def get_hole(self, hole_x: int, hole_y: int) -> np.ndarray:
        """Dig a 'hole' at the specified x, y position. Return a 3x3 patch of the field."""
        # Define a 3x3 region (or patch) around the current position
        patch_x_min = max(0, hole_x - 1)
        patch_x_max = min(self.grid_size[1], hole_x + 2)
        patch_y_min = max(0, hole_y - 1)
        patch_y_max = min(self.grid_size[0], hole_y + 2)

        # Extract the patch (it could be smaller than 3x3 at edges)
        hole = self.real_field[patch_y_min:patch_y_max, patch_x_min:patch_x_max]

        # If the patch is smaller than 3x3 (near the borders), pad it to fit 3x3
        hole = np.pad(hole, ((max(0, 1 - hole_y), max(0, (hole_y + 2) - self.grid_size[0])),
                             (max(0, 1 - hole_x), max(0, (hole_x + 2) - self.grid_size[1]))),
                      mode='constant')

        # Save coordinates and values of the 'hole' (sub-array)
        self.x_coords = np.hstack((self.x_coords, np.full(hole.shape[1], hole_x)))
        self.y_coords = np.hstack((self.y_coords, np.arange(patch_y_min, patch_y_max)))
        self.ic_values = np.hstack((self.ic_values, hole.flatten()))

        # Dig hole in the field (only update the parts of field_with_holes that overlap)
        self.field_with_holes[patch_y_min:patch_y_max, patch_x_min:patch_x_max] = hole

        self.num_holes += 1

        # Flatten the hole for observation and return
        return hole.flatten()

    def _is_done(self) -> bool:
        """Check if the position is out of the grid"""
        if self.current_position_x >= self.grid_size[1] or self.current_position_y >= self.grid_size[0]:
            return True
        return False

    def _calc_reward(self) -> float:
        """Calculate the reward"""
        reward = self.f1 * self.num_holes + self.f2 * self._calc_rmse()
        return reward

    def _calc_rmse(self) -> float:
        """Calculate RMSE of interpolated field"""
        if self.ip_field is not None:
            rmse = np.sqrt(np.mean((self.real_field - self.ip_field) ** 2))
            return rmse
        return 10e2

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment based on the action (0=up, 1=down, 2=left, 3=right)"""

        # Move based on action (up, down, left, right)
        if action == 0:  # Move up
            self.current_position_y = max(0, self.current_position_y - self.steps_from_current_position[1])
        elif action == 1:  # Move down
            self.current_position_y = min(self.grid_size[0] - 1,
                                          self.current_position_y + self.steps_from_current_position[1])
        elif action == 2:  # Move left
            self.current_position_x = max(0, self.current_position_x - self.steps_from_current_position[1])
        elif action == 3:  # Move right
            self.current_position_x = min(self.grid_size[1] - 1,
                                          self.current_position_x + self.steps_from_current_position[1])

        # Get the new "hole" (sub-array/patch) from the current position
        observation = self.get_hole(self.current_position_x, self.current_position_y)

        # Check if the agent has completed the episode
        done = self._is_done()
        truncated = self._is_done()

        # Interpolate the field based on the dug holes
        self.ip_field = interpolate(self.x_coords, self.y_coords, self.ic_values, self.grid_size, method="linear")

        # Calculate reward based on the current state
        reward = self._calc_reward()

        return observation, reward, done, truncated, {"rmse": self._calc_rmse()}

    def render(self) -> None:
        """Render the environment using Matplotlib"""
        fig, ax = plt.subplots(3, 1, figsize=(6, 12))

        # Plot the real field
        ax[0].imshow(self.real_field, cmap='viridis', origin='lower')
        ax[0].set_title('Original Field')

        # Plot the field with holes
        ax[1].imshow(self.field_with_holes, cmap='viridis', origin='lower')
        ax[1].set_title('Field with Holes')

        # Plot the interpolated field
        ax[2].imshow(np.round(self.ip_field), cmap='viridis', origin='lower')
        ax[2].set_title('Interpolated Field')

        # Show the plots
        plt.tight_layout()
        plt.show()


# Example usage
data = GetFields()
data.load_data_from_file("data/train_data_10_1.txt")  # Load data from file (or use load_data_from_arrays for arrays)

env = SoilEnvironment(data=data)
check_env(env)

