
import sys
import os
module_path = os.path.abspath(os.path.join('..',r'C:\Users\Bruker\Documents\IN5490\git_project\ml-drill'))
sys.path.append(module_path)

import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from interpolater import interpolate
from load_field_data import GetFields


class SoilEnvirment(gym.Env):
    def __init__(self, data: GetFields, f1: float = -1, f2: float = 10, starting_position_x: int=0) -> None:
        super(SoilEnvirment, self).__init__()

        # Set konstant
        self.f1 = f1
        self.f2 = f2
        self.start_position_x = starting_position_x
        self.max_num_holes = 20

        # Set data
        self.data = data

        # Set gridsize
        self.grid_size = self.data.get_field().shape

        # Define actions steps
        self.steps = np.arange(0,101,2)
        

        # Define action and ans observation space
        self.action_space = spaces.Discrete(51)
        self.observation_space = spaces.Box(low=0,high=255,shape=(2,36,100),dtype=np.uint8)

        # For logging
        self.rmse = 100

        
    def reset(self, seed: int | None = None):
        """Reset the envirment to initial state"""
        super().reset(seed=seed)


        # Simulate the fields
        self.real_field = self.data.get_field()
        self.field_with_holes = np.full_like(self.real_field, 0)
        self.ip_field = np.full_like(self.real_field, 0)
        self.current_position = self.start_position_x
        self.position_map = np.zeros_like(self.real_field)

        # Save holes and ic_values
        self.num_holes = 0
        self.x_coords = np.array([])
        self.y_coords = np.array([])
        self.ic_values = np.array([])

        # self.get_hole(self.start_position_x)

        state = self.make_observastions()

        return state, {}
    

    def get_hole(self, hole_x: int) -> np.ndarray | None: 
        self.num_holes += 1

        # Check if agent is digging the same hole
        if self.current_position in self.x_coords:
            return None
        
        # Checks if its out of bounds
        if self._is_truncated():
            return None
        
        # Make position map
        self.position_map[:,hole_x] = 1

        # Save coords from hole
        self.x_coords = np.hstack((self.x_coords,np.full(self.grid_size[0], hole_x)))
        self.y_coords = np.hstack((self.y_coords,np.arange(self.grid_size[0])))
        self.ic_values = np.hstack((self.ic_values, self.real_field[:, hole_x]))

        # Digg hole
        self.field_with_holes[:,hole_x] = self.real_field[:,hole_x]
        

        return self.real_field[:,hole_x]

    def _is_done(self) -> bool:      
        if self.current_position >= self.grid_size[1]:
            return True
          
        # Check is pososion out of grid
        if self.num_holes >= self.max_num_holes:
            return True
        
        # Check good rmse
        if self._calc_rmse() <= 0.8:
            return True
        
        return False
    
    def _is_truncated(self) -> bool:        
        # Check is pososion out of grid
        if self.current_position >= self.grid_size[1] or self.current_position < 0:
            return True
        
        return False
    
    def _calc_reward(self) -> float:
        """Calculate reward"""
        rmse = max(self._calc_rmse(), 1e-2)
        same_action_penalty  = -100 if self.current_position in self.x_coords else 0
        reward = self.f1 * self.num_holes + self.f2**(1/(rmse + 0.001)) + same_action_penalty

        return reward
    
    def _calc_rmse(self) -> float:
        """Calculate rmse of interpolated field"""
        if self.ip_field is not None:
            rmse =  np.sqrt(np.mean((self.real_field - self.ip_field) ** 2))
            return rmse
        return 10
    
    def position_map(self) -> np.ndarray:
        position = np.zeros(self.grid_size)

        position[:,self.current_position] = 1
        return position 



    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:

        # Gravet nytt hull
        self.current_position = self.steps[action]
        self.get_hole(self.current_position)

    
        # Interpolate field
        self.ip_field = interpolate(self.x_coords,self.y_coords,self.ic_values,self.grid_size,method="linear")

        # Caluclate reward
        reward = self._calc_reward()

        # Update state
        state = self.make_observastions()


        return state, reward, self._is_done(), self._is_truncated(), {"rmse": self._calc_rmse()}


    def render(self) -> None:
        # Create a figure and a set of subplots
        fig, ax = plt.subplots(3, 1, figsize=(6, 12))

        # Max and min values in the plot
        vmin = 0
        vmax = 6

        # Plot the first zi on the first subplot
        contour1 = ax[0].imshow(self.real_field,cmap='viridis', origin='lower',vmin=vmin,vmax=vmax)
        ax[0].set_title('Original fiels')
        ax[0].set_xlabel('X-axis')
        ax[0].set_ylabel('Z-axis')
        fig.colorbar(contour1, ax=ax[0])

        # Plot the second zi on the second subplot
        contour2 = ax[1].imshow(self.field_with_holes,cmap='viridis', origin='lower',vmin=vmin,vmax=vmax)
        ax[1].set_title('Field with holes')
        ax[1].set_xlabel('X-axis')
        ax[1].set_ylabel('Z-axis')
        fig.colorbar(contour2, ax=ax[1])

        # Plot the second zi on the second subplot
        contour3 = ax[2].imshow(np.round(self.ip_field),cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        #contour3 = ax[2].imshow(np.round(self.ip_field),cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        ax[2].set_title('Interpolated Field')
        ax[2].set_xlabel('X-axis')
        ax[2].set_ylabel('Z-axis')
        fig.colorbar(contour3, ax=ax[2])

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Show the plots
        plt.show()

    def resize_fields(self, fields):
        resized_obs = np.zeros((36,100))
        resized_obs[:self.grid_size[0],:self.grid_size[1]] = fields
        return resized_obs
    
    def make_observastions(self):
        resize_ip_field = self.resize_fields(self.ip_field)
        resize_field_with_holes = self.resize_fields(self.field_with_holes)
        #resized_positions = self.resize_fields(self.position_map)

        obs = np.stack([resize_ip_field, resize_field_with_holes]).astype(np.uint8)
        return obs


    

data = GetFields()
data.load_data("data/train_data_1.txt")

env = SoilEnvirment(data=data)
check_env(env)










