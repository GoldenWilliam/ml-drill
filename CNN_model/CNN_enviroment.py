
import sys
import os

module_path = os.path.abspath(os.path.join('..', os.getcwd()))
sys.path.append(module_path)

import matplotlib.pyplot as plt
import gymnasium as gym
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from interpolater import interpolate
from load_field_data import GetFields
import cv2


class SoilEnvirment(gym.Env):
    def __init__(self, data: GetFields, f1: float = -1, f2: float = 20, rmse_threshold: float = 0.7) -> None:
        super(SoilEnvirment, self).__init__()

        # Set konstant
        self.f1 = f1
        self.f2 = f2
        self.rmse_threshold = rmse_threshold
        self.max_num_holes = 15
        self.max_numbers_actions = 20


        # Set data
        self.data = data

        # Set gridsize
        self.grid_size = self.data.get_field().shape

        # Define where in the grid the agen can digg
        self.positions_in_env = np.arange(0,100)
        

        # Define action and and observation space
        self.action_space = spaces.Discrete(100)
        self.observation_space = spaces.Box(low=0,high=255,shape=(1,36,360),dtype=np.uint8)

        
    def reset(self, seed: int | None = None):
        """Reset the envirment to initial state"""
        super().reset(seed=seed)


        # Simulate the fields
        self.real_field = self.data.get_field()
        self.field_with_holes = np.full_like(self.real_field, 0)
        self.ip_field = np.full_like(self.real_field, 0)
        self.current_position = 0
        self.position_map = np.zeros_like(self.real_field)

        self.last_rmse = 10


        # Save holes and ic_values
        self.num_holes = 0
        self.x_coords = np.array([])
        self.y_coords = np.array([])
        self.ic_values = np.array([])

 

        state = self.make_state()

        self.num_actions = 0

        return state, {}
    

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.num_actions += 1

        # Check if we are doing the same position again
        # if self.positions_in_env[action] in self.x_coords:
        #     reward = -100
        #     state = self.make_state()

        #     return state, reward, False, False, {"rmse" : self._calc_rmse(), "reward" : reward, "holes" : self.num_holes}
        

        # Change current position
        self.current_position = self.positions_in_env[action]

        # Digg hole
        self.get_hole(self.current_position)

        # Interpolate field
        self.ip_field = interpolate(self.x_coords,self.y_coords,self.ic_values,self.grid_size,method="linear")

        # Caluclate reward
        reward = self._calc_reward()

        # Update state
        state = self.make_state()

        return state, reward, self._is_done(), self._is_truncated(), {"rmse": self._calc_rmse(), "reward" : reward, "holes" : self.num_holes}
    
    def get_hole(self, hole_x: int) -> np.ndarray | None: 

        # Check if agent is digging the same hole
        if self.current_position in self.x_coords:
            return None
        
        # # Checks if its out of bounds
        # if self._is_truncated():
        #     return None
        
        # Make position map
        self.position_map[:,hole_x] = 1

        self.num_holes += 1
        # Save coords from hole
        self.x_coords = np.hstack((self.x_coords,np.full(self.grid_size[0], hole_x)))
        self.y_coords = np.hstack((self.y_coords,np.arange(self.grid_size[0])))
        self.ic_values = np.hstack((self.ic_values, self.real_field[:, hole_x]))

        # Digg hole
        self.field_with_holes[:,hole_x] = self.real_field[:,hole_x]
        

        return self.real_field[:,hole_x]

    def _is_done(self) -> bool:
        # Check if the rmse threshold      
        if self._calc_rmse() <= self.rmse_threshold:
            return True
        
        # Check max number of holes
        if self.num_holes >= self.max_num_holes:
            return True 
        
        if self.num_actions >= self.max_numbers_actions:
            return True
        
        
        return False
    
    def _is_truncated(self) -> bool: 

        # Check is pososion out of grid
        if self.current_position >= self.grid_size[1] or self.current_position < 0:
            return True
        
        # Check for max number of holes
        # if self.num_holes >= self.max_num_holes:
        #     return True

        # Check for max number of actions
        if self.num_actions >= self.max_numbers_actions:
            return True
        
        return False
    
    def _calc_reward(self) -> float:
        """Calculate reward"""
        # Cakuclate rmse
        rmse = max(self._calc_rmse(), 0.4)

        # Penalty for the same action multiple times
        same_action_penalty  = -20 if self.current_position in self.x_coords else 0
        loss_rmse_bonus = (self.last_rmse - rmse) * 10
        

        # Reward_0 : 
        reward = self.f1 * self.num_holes + self.f2**(1/rmse) + same_action_penalty

        # reward = self.f1 * self.num_holes + self.f2**(1/rmse) + same_action_penalty + loss_rmse_bonus

        self.last_rmse = rmse

        return min(reward,400)
    
    def _calc_rmse(self) -> float:
        """Calculate rmse of interpolated field"""
        if self.ip_field is not None:
            rmse =  np.sqrt(np.mean((self.real_field - self.ip_field) ** 2))
            return rmse
        return 0

    def render(self, mode: str = 'human', save_path=None) -> None:
    
        # Create a figure and a set of subplots
        fig, ax = plt.subplots(3, 1, figsize=(6, 12))
        fig.suptitle(f"Soil Envirment Rendering   |  Number of holes: {self.num_holes}  | RMSE: {self._calc_rmse():.2f}",fontsize=8)

        # Max and min values in the plot
        vmin = 0
        vmax = 6

        # Define colormap
        cmap = cm.get_cmap("Set1",7)
        norm = BoundaryNorm(np.arange(vmin,vmax+2), cmap.N)

        # Plot the first zi on the first subplot
        contour1 = ax[0].imshow(self.real_field,cmap=cmap,norm=norm)
        ax[0].set_title('Original field')
        ax[0].set_xlabel('X-axis')
        ax[0].set_ylabel('Z-axis')
        fig.colorbar(contour1, ax=ax[0], ticks=np.arange(7))

        # Plot the second zi on the second subplot
        contour2 = ax[1].imshow(self.field_with_holes,cmap=cmap, norm=norm)
        ax[1].set_title(f'Field with holes : {self.num_holes}')
        ax[1].set_xlabel('X-axis')
        ax[1].set_ylabel('Z-axis')
        fig.colorbar(contour2, ax=ax[1], ticks=np.arange(7))

        # Plot the second zi on the second subplot
        contour3 = ax[2].imshow(np.round(self.ip_field),cmap=cmap, norm=norm)
        #contour3 = ax[2].imshow(np.round(self.ip_field),cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        ax[2].set_title(f'Interpolated Field, rmse = {self._calc_rmse()}')
        ax[2].set_xlabel('X-axis')
        ax[2].set_ylabel('Z-axis')
        fig.colorbar(contour3, ax=ax[2], ticks=np.arange(7))

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Show the plots
        if mode == 'human':
            plt.show()
        
        elif mode == 'rgb_array':
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::1] + (3,))
            return image
        

    def make_state(self):
        """Prøver å lage at  gjøre at matrisen har tall sprett mellom 0 og 255 så CNN-policy ikke klager"""
        observation = np.copy(self.ip_field).astype(np.float32)

        image_values = np.arange(100,255,155/7)
        for i in range(7):
            observation[observation == i] = image_values[i]

        # Legger inn 0 der vi har gravet hull
        for x in self.x_coords:
            observation[:,int(x)] = 0
        
        observation = cv2.resize(observation, (360,36),interpolation=cv2.INTER_NEAREST)
        # plt.imshow(observation)
        # plt.show()

        return np.array([observation]).astype(np.uint8)


    

data = GetFields()
data.load_data("data/train_data_1.txt")

env = SoilEnvirment(data=data)
check_env(env)










