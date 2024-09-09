import gymnasium as gym
from enviroment import SoilEnvirment
from stable_baselines3 import DQN, A2C
import os



models_dir = f"models/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = SoilEnvirment()
env.reset()
model = A2C("MlpPolicy",env,verbose=1,tensorboard_log=logdir)


TIMESTEPS = 1000
for i in range(0,30):
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

