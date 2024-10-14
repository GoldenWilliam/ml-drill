import multiprocessing as mp
import os
from environment import SoilEnvironment
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from load_field_data import GetFields


class CustomTensorboardCallback(BaseCallback):
    """CallBack to log rmse"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        rmse = infos[0].get("rmse")
        self.logger.record("testing/rmse", rmse)
        self.logger.dump(step=self.num_timesteps)

        return True


def train_model(model_id: str):
    models_dir = f"models/{model_id}"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Load training data
    data = GetFields()
    data.load_data_from_file("data/train_data_10_1.txt")
    # data.load_data_from_file("data/train_data_2.txt")
    # data.load_data_from_file("data/train_data_3.txt")
    # data.load_data_from_file("data/train_data_4.txt")

    # Set up environment
    env = SoilEnvironment(data=data, f1=-2, f2=-5)
    env.reset()

    # Initialize model
    if model_id == "A2C":
        model = A2C("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
    if model_id == "PPO":
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=logdir)
    if model_id == "DQN":
        model = DQN("MlpPolicy", env, verbose=0, tensorboard_log=logdir, exploration_initial_eps=4)

    # Train the model
    TIMESTEPS = 10000
    for i in range(0, 30):
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False,
                    tb_log_name=model_id,
                    callback=CustomTensorboardCallback())

        # Save the model
        model.save(f"{models_dir}/{TIMESTEPS * i}")


if __name__ == "__main__":
    p1 = mp.Process(target=train_model, args=("A2C",))
    # p2 = mp.Process(target=train_model, args=("PPO",))
    p3 = mp.Process(target=train_model, args=("DQN",))

    p1.start()
    # p2.start()
    p3.start()