


"""
###### Kan kjørest for å sammenlikte rmse for de to modellene (den enkle og CNN) #######
"""

import sys
import os
module_path = os.path.abspath(os.path.join('..',os.getcwd()))
sys.path.append(module_path)

from CNN_enviroment import SoilEnvirment as cnn_soil_env
from enviroment import SoilEnvirment as soil_env
from CNN_policy import policy_kwargs
from stable_baselines3 import PPO, DQN, A2C
from load_field_data import GetFields
import numpy as np
import torch as th
import torch.nn as nn


data1 = GetFields(train=False)
data1.load_data("data/train_data_4.txt")
data1.load_data("data/train_data_5.txt")
data1.load_data("data/scaled_down_matrix.txt")

data2 = GetFields(train=False)
data2.load_data("data/train_data_4.txt")
data2.load_data("data/train_data_5.txt")
data2.load_data("data/scaled_down_matrix.txt")






# env_rand = cnn_soil_env(data1)
env_cnn = cnn_soil_env(data1)
env_simple = soil_env(data2)

env_simple.reset()
env_cnn.reset()



# model_cnn = DQN.load("CNN_model/models/DQN/99000", env=env_cnn)
# model_cnn = DQN.load("CNN_model/models_ex/DQN/99000", env=env_cnn)
model_cnn = DQN.load("CNN_model/models_third_run/DQN/22000", env=env_cnn)
model_simple = PPO.load("models/PPO/50000",env=env_simple)



episodes = 100


rmse_cnn = []
hull_cnn = []
rmse_simple = []
hull_nrom = []

for ep in range(episodes):
    obs , _ = env_cnn.reset()
    done = False
    t = False
    # print("Start")
    while ((not done) and (not t)) and env_cnn.num_holes < 12:
        action, _states = model_cnn.predict(obs)
        #action = random.randint(0,49)
        obs, reward, done, t, info = env_cnn.step(action)
        # print(f"Action : {action}")
        # print(f"Reward: {reward}") 
    # env_cnn.render()
    rmse_cnn.append(env_cnn._calc_rmse())
    hull_cnn.append(env_cnn.num_holes)
    # print("End")

for ep in range(episodes):
    obs , _ = env_simple.reset()
    done = False
    # print("Start")
    while not done:
        action, _states = model_simple.predict(obs)
        #action = random.randint(0,49)
        obs, reward, done, t, info = env_simple.step(action)
        # print(f"Action : {action}")
        # print(f"Reward: {reward}") 
    # env_simple.render()
    rmse_simple.append(env_simple._calc_rmse())
    hull_nrom.append(env_simple.num_holes)
    # print("End")



print(f"CNN : {np.mean(rmse_cnn)}, {np.std(rmse_cnn)},  {np.mean(hull_cnn)}")
print(f"simple : {np.mean(rmse_simple)}, {np.std(rmse_simple)}, {np.mean(hull_nrom)}")


# def test_model(env, done):
#     rmse = []
#     hull = []

#     for ep in range(episodes):
#         done = False
#         truncated = False
#         obs, _ = env.reset()
#         while (not done) or (not truncated):
#             action, _ = env.predict(obs)
#             obs, reward, done, truncated, info = env.step(action)


