from stable_baselines3 import PPO,DQN,A2C
from mlp_enviroment import SoilEnvirment
from load_field_data import GetFields
import os
import numpy as np

current_dir = os.getcwd()


parent_dir = os.path.dirname(current_dir)
print(parent_dir)
model_dir = "models"

path = os.path.join(current_dir,model_dir,"PPO","90000")

data = GetFields()
data.load_data("data/train_data_1.txt")
env = SoilEnvirment(data=data)
env.reset()

model = PPO.load(path,env=env)

# imag3es = []

episodes = 20

for ep in range(episodes):
    obs , _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, t, info = env.step(action)
        # img = model.env.render(mode="rgb_array")
        #images.append(img)
        # print(f"States : {obs}")
        # print(f"Reward = {reward}")
        # print(f"Action : {action}")
    # print("END")
    env.render()


## make a gif
#imageio.mimsave("env.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=1)


def test_models(models, envirmoments, episodes, model_names):

    for model, env in zip(models, envirmoments):
        for ep in episodes:
            obs, _ = env.reset()
            done = False
            while not done:




