import multiprocessing as mp
import os
from mlp_enviroment import SoilEnvirment
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from load_field_data import GetFields
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


class CustumTensorboardCallback(BaseCallback):
    """CallBack to log rmse"""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos",None)
        rmse = infos[0].get("rmse")
        reward = infos[0].get("reward")
        holes = infos[0].get('holes')
        self.logger.record("testing/rmse",rmse)
        self.logger.record("testing/reward",reward)
        self.logger.record("testing/holes",holes)
        self.logger.dump(step=self.num_timesteps)
        return True
    
def train_model(model_id: str):

    models_dir = f"{script_dir}/models/{model_id}"
    logdir = f"{script_dir}/logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Load traning data
    data = GetFields()
    data.load_data("data/train_data_1.txt")
    data.load_data("data/train_data_2.txt")
    data.load_data("data/train_data_3.txt")
    data.load_data("data/train_data_4.txt")
    data.load_data("data/train_data_5.txt")
    data.load_data("data/train_data_6.txt")
    
    # Set up enviroment
    env = SoilEnvirment(data=data)
    env.reset()
    
    if model_id == "A2C":
        model = A2C("MlpPolicy",env,verbose=0,tensorboard_log=logdir)
    if model_id == "PPO":
        model = PPO("MlpPolicy",env,verbose=0,tensorboard_log=logdir, batch_size=25, n_steps=100)
    if model_id == "DQN":
        model = DQN("MlpPolicy",env,verbose=0,tensorboard_log=logdir, exploration_initial_eps=4)

    TIMESTEPS = 10000
    for i in range(0,10):
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False,
                    tb_log_name=model_id,
                    callback=CustumTensorboardCallback())
        
        model.save(f"{models_dir}/{TIMESTEPS*i}")


if __name__ == "__main__":

    p1 = mp.Process(target=train_model, args=("A2C",))
    p2 = mp.Process(target=train_model, args=("PPO",))
    p3 = mp.Process(target=train_model, args=("DQN",))

    p1.start()
    p2.start()
    p3.start()