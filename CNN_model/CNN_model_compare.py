import os
import sys
sys.path.append(os.path.abspath("."))

from models_testing_func import models_compare
from stable_baselines3 import PPO, DQN, A2C
from load_field_data import GetFields
from CNN_enviroment import SoilEnvirment
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi


model_names = ['A2C','PPO']
models = []
envs = []
script_dir = os.path.dirname(os.path.abspath(__file__))


for name in model_names:
    data = GetFields(train=False)
    data.load_data("data/train_data_1.txt")

    env = SoilEnvirment(data=data)

    model_path = os.path.join(script_dir,"models",name,"2000")

    if name == "PPO":
        model = PPO.load(model_path,env=env)
    elif name == "A2C":
        model = A2C.load(model_path,env=env)
    else:
        model = DQN.load(model_path, env=env)
    
    models.append(model)
    envs.append(env)


    

df = models_compare(models,model_names,envs,100)

print(df)

fig,ax = plt.subplots(figsize=(7,2))
fig.suptitle("MLP: runs = 100")

ax.axis('off')

tabel = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center',loc='center')
tabel.set_fontsize(10)
tabel.scale(1,1)

# plt.savefig("mlp_train_test_100")

plt.show()

