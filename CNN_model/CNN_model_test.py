import sys
import os
module_path = os.path.abspath(os.path.join('..',r'C:\Users\Bruker\Documents\IN5490\git_project\ml-drill'))
sys.path.append(module_path)

import multiprocessing as mp
from CNN_enviroment import SoilEnvirment
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from load_field_data import GetFields
import random

current_dir = os.getcwd()


parent_dir = os.path.dirname(current_dir)
print(parent_dir)
model_dir = "CNN_model\models"

path = os.path.join(current_dir,model_dir,"PPO","50000")
print(path)

data = GetFields()
data.load_data("data/train_data_1.txt")
env = SoilEnvirment(data=data)
env.reset()

model = PPO.load(path,env=env)

# obs, _ = env.reset()
# print(env.step(30))
# env.render()



episodes = 20

for ep in range(episodes):
    obs , _ = env.reset()
    done = False
    print("Start")
    while not done:
        action, _states = model.predict(obs)
        #action = random.randint(0,49)
        obs, reward, done, t, info = env.step(action)
        env.render()
        print(f"Action : {action}")
        print(f"Reward: {reward}") 
        env.render()
    print("End")


