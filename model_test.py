from stable_baselines3 import PPO,DQN,A2C
from enviroment import SoilEnvirment
from load_field_data import GetFields
import os

current_dir = os.getcwd()


parent_dir = os.path.dirname(current_dir)
print(parent_dir)
model_dir = "models"

path = os.path.join(current_dir,model_dir,"A2C","290000")

data = GetFields()
data.load_data("data/train_data_1.txt")
env = SoilEnvirment(data=data)
env.reset()

model = A2C.load(path,env=env)


episodes = 10

for ep in range(episodes):
    obs , _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, t, info = env.step(action)
        print(f"States : {obs}")
        print(f"Reward = {reward}")
        print(f"Action : {action}")
    print("END")
    env.render()


