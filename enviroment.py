
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from interpolater import interpolate
from load_training_data import GetTrainingFields


class SoilEnvirment(gym.Env):
    def __init__(self, data: GetTrainingFields, f1: float = -1, f2: float = -3, starting_position_x: int=0):
        super(SoilEnvirment, self).__init__()

        # Set konstants
        self.f1 = f1
        self.f2 = f2
        self.start_position_x = starting_position_x

        # Set data
        self.data = data

        # Number of steps we can move in from_position
        self.steps_from_current_position = [2,5,8,12,16]

        # Define action and ans observation space
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0,high=6,shape=(3,),dtype=np.float64)

        # For logging
        self.rmse = 100

        
    def reset(self,seed=None):
        """Reset the envirment to initial state"""
        # Set random seed
        if seed == None:
            seed = random.randint(0,10e5)

        # Simulate the fields
        self.real_field = self.data.get_field()
        self.field_with_holes: np.array = np.full_like(self.real_field, 0)
        self.ip_field: np.array = None
        self.current_position: int = self.start_position_x
        self.grid_size = self.real_field.shape

        # Save holes and ic_values
        self.number_of_holes: int = 0
        self.x_coords = np.array([])
        self.y_coords = np.array([])
        self.ic_values = np.array([])

        hole = self.get_hole(self.current_position)

        mean_ic: float = np.mean(hole)
        std_ic: float = np.std(hole)

        return np.array([self.current_position, mean_ic, std_ic]), {}
    

    def get_hole(self, hole_x: int): 

        # Checks if its out of bounds
        if self._is_done():
            return self.ip_field[:,-1]
        
        # Get coords and ic values from the hole
        new_x_coords = np.ones(self.grid_size[0])*hole_x
        new_y_coords = np.linspace(0,self.grid_size[0],self.grid_size[0])
        new_ic_values = self.real_field[:,hole_x]

        # Save coords and ic value
        self.x_coords = np.hstack((self.x_coords,new_x_coords))
        self.y_coords = np.hstack((self.y_coords,new_y_coords))
        self.ic_values = np.hstack((self.ic_values,new_ic_values))

        # Digg hole
        self.field_with_holes[:,hole_x] = new_ic_values
        self.number_of_holes += 1
        

        return self.real_field[:,hole_x]

    def _is_done(self):        
        # Check is pososion out of grid
        if self.current_position >= self.grid_size[1]:
            return True
        
        return False
    
    def _calc_reward(self):
        rmse = np.sqrt(np.mean((self.real_field - self.ip_field) ** 2))
        reward = self.f1 * self.number_of_holes + self.f2 * self.rmse
        return reward, rmse


    def step(self, action):
        # Gravet nytt hull
        self.current_position += self.steps_from_current_position[action]
        hole = self.get_hole(self.current_position)

        # Check truncated:
        truncated = self._is_done()
    
        # Interpolate field
        self.ip_field = interpolate(self.x_coords,self.y_coords,self.ic_values,self.grid_size,method="linear")

        # Caluclate reward
        reward, rmse = self._calc_reward()

        # Update state
        mean_ic = np.mean(hole)
        std_ic = np.std(hole)
        state = np.array([self.current_position,mean_ic,std_ic])

        
        done = self._is_done()

        if done:
            self.rmse = rmse

        return state, reward, done, truncated, {"rmse": self.rmse}


    def render(self):
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
        contour3 = ax[2].imshow(self.ip_field,cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        ax[2].set_title('Interpolated Field')
        ax[2].set_xlabel('X-axis')
        ax[2].set_ylabel('Z-axis')
        fig.colorbar(contour3, ax=ax[2])

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Show the plots
        plt.show()

# data = GetTrainingFields()
# data.load_data("train_data_200_1.txt")
# env = SoilEnvirment(data=data)
# # check_env(env)

# for _ in range(2):
#     obs, _ = env.reset()
#     done = False
#     while not done:
#         action = random.randint(0,4)
#         obs, reward, done, t, info = env.step(action)
#     env.render()









