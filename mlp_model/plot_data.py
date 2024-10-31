
import sys
import os

module_path = os.path.abspath(os.path.join('..', os.getcwd()))
sys.path.append(module_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names = ["A2C","DQN","PPO"]
# /uio/hume/student-u19/skstorst/IN5490/ml_drill/mlp_model/loggig_data/ep_len_mean/run-A2C_0-tag-rollout_ep_len_mean.csv

plt.figure(figsize=(10,6))
for name in names:
    df = pd.read_csv(os.path.join("mlp_model/loggig_data","ep_len_mean",f"run-{name}_0-tag-rollout_ep_len_mean.csv"))

    plt.plot(df["Step"],df["Value"], label=name)
plt.xlabel("Steps")
plt.ylabel("ep_len_mean")
plt.title("MLP : Mean Episode Length")
plt.legend()
plt.savefig("mlp_mean_ep_len")

plt.figure(figsize=(10,6))
for name in names:
    df = pd.read_csv(os.path.join("mlp_model/loggig_data","ep_rew_mean",f"run-{name}_0-tag-rollout_ep_rew_mean.csv"))

    plt.plot(df["Step"],df["Value"], label=name)
plt.xlabel("Steps")
plt.ylabel("ep_rew_mean")
plt.title("MLP : Mean Episode Reward")
plt.legend()
plt.savefig("mlp_mean_ep_rew")








