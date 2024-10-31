import numpy as np
import pandas as pd


def run_episodes(model, env, episodes, render=False):
    rmse = []
    num_holes = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done :
            action, _ = model.predict(obs)
            obs, reward, done, t, info = env.step(action)
        
        rmse.append(env._calc_rmse())
        num_holes.append(env.num_holes)
        
        if render:
            env.render()
    
    return np.mean(rmse), np.std(rmse), np.mean(num_holes)


def models_compare(models,model_names, envs, episodes):
    n = len(models)
    df = {
        'model' : model_names,
        'mean_num_holes' : [],
        'mean_rmse' : [],
        'mean_std' : [],
    }

    for i in range(n):
        model = models[i]
        env = envs[i]

        mean_rmse, mean_std, num_holes = run_episodes(model, env, episodes, False)
        df['mean_num_holes'].append(num_holes)
        df['mean_rmse'].append(mean_rmse)
        df['mean_std'].append(mean_std)

    df = pd.DataFrame(df)
    df['mean_rmse'] = df['mean_rmse'].round(3)
    df['mean_std'] = df['mean_std'].round(3)
    
    return df


        





