# Compare models with eachother

import sys
import os
module_path = os.path.abspath(os.path.join('..',r'C:\Users\Bruker\Documents\IN5490\git_project\ml-drill'))
sys.path.append(module_path)

from CNN_enviroment import SoilEnvirment as cnn_soil_env
from enviroment import SoilEnvirment as soil_env
from CNN_policy import policy_kwargs
from stable_baselines3 import PPO
from load_field_data import GetFields
import numpy as np
import torch as th
import torch.nn as nn


data1 = GetFields(train=False)
data1.load_data("data/train_data_6.txt")

data2 = GetFields(train=False)
data2.load_data("data/train_data_6.txt")

print(data2.training_data[0]==data1.training_data[0])



# env_rand = cnn_soil_env(data1)
env_cnn = cnn_soil_env(data1)
env_norm = soil_env(data2)

env_norm.reset()
env_cnn.reset()

model_cnn = PPO.load("CNN_model/models/PPO/30000", env=env_cnn)
model_norm = PPO.load("models/PPO/50000",env=env_norm)


episodes = 10

rmse_cnn = []
hull_cnn = []
rmse_norm = []
hull_nrom = []

for ep in range(episodes):
    obs , _ = env_cnn.reset()
    done = False
    t = False
    # print("Start")
    while (not done) or (not t):
        action, _states = model_cnn.predict(obs)
        #action = random.randint(0,49)
        obs, reward, done, t, info = env_cnn.step(action)
        # print(f"Action : {action}")
        # print(f"Reward: {reward}") 
    env_cnn.render()
    rmse_cnn.append(env_cnn._calc_rmse())
    hull_cnn.append(env_cnn.num_holes)
    # print("End")

for ep in range(episodes):
    obs , _ = env_norm.reset()
    done = False
    # print("Start")
    while not done:
        action, _states = model_norm.predict(obs)
        #action = random.randint(0,49)
        obs, reward, done, t, info = env_norm.step(action)
        # print(f"Action : {action}")
        # print(f"Reward: {reward}") 
    # env_norm.render()
    rmse_norm.append(env_norm._calc_rmse())
    hull_nrom.append(env_norm.num_holes)
    # print("End")



print(f"CNN : {np.mean(rmse_cnn)}, {np.std(rmse_cnn)},  {np.mean(hull_cnn)}")
print(f"NORM : {np.mean(rmse_norm)}, {np.std(rmse_norm)}, {np.mean(hull_nrom)}")



