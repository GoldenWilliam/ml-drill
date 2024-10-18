from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from load_field_data import GetFields
from environment import SoilEnvironment
from custom_cnn import CustomCNNExtractor  # Custom CNN extractor
import os
import multiprocessing as mp

from stable_baselines3.common.callbacks import BaseCallback


class CustomTensorboardCallback(BaseCallback):
    """Callback to log RMSE and reward to TensorBoard"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_reward = 0  # To accumulate reward over each episode
        self.episode_length = 0  # To track episode length

    def _on_step(self) -> bool:
        # Get RMSE from the environment info
        infos = self.locals.get("infos", None)
        rmse = infos[0].get("rmse") if infos else None

        # Get the reward from the current step
        reward = self.locals.get("rewards", [0])[0]  # Rewards can be in a list, take the first value

        # Accumulate episode reward and episode length
        self.episode_reward += reward
        self.episode_length += 1

        # Log RMSE and reward if available
        if rmse is not None:
            self.logger.record_mean("metrics/rmse", rmse)
        self.logger.record("metrics/reward", reward)  # Log reward at each step

        # Check if the episode is done
        done = self.locals.get("dones", [False])[0]
        if done:
            # Log total episode reward and length
            self.logger.record("metrics/episode_reward", self.episode_reward)
            self.logger.record("metrics/episode_length", self.episode_length)

            # Reset the reward and episode length
            self.episode_reward = 0
            self.episode_length = 0

        # Dump the logs to TensorBoard
        self.logger.dump(step=self.num_timesteps)

        return True


def get_incrementing_directory(base_dir: str) -> str:
    """Get the next available directory with an incrementing number."""
    existing_dirs = [int(name) for name in os.listdir(base_dir) if name.isdigit()]
    next_index = max(existing_dirs) + 1 if existing_dirs else 0
    new_dir = os.path.join(base_dir, str(next_index))
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def linear_schedule(initial_value: float):
    """
    Returns a function that computes a linearly decaying learning rate.

    :param initial_value: Initial learning rate value.
    :return: Function that takes the progress (from 1 to 0) and returns the current learning rate.
    """

    def func(progress: float):
        # Learning rate will decrease linearly from initial_value to 0
        return initial_value * progress

    return func


def train_model(model_id: str, data_files: list, rmse_threshold: float = 20):
    # Get incrementing directories for models and logs
    models_base_dir = f"models/{model_id}"
    logs_base_dir = f"logs/{model_id}"

    models_dir = get_incrementing_directory(models_base_dir)
    logdir = get_incrementing_directory(logs_base_dir)

    print(f"Models will be saved in: {models_dir}")
    print(f"Logs will be saved in: {logdir}")

    # Custom CNN Policy configuration
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # Initialize the model once
    data = GetFields()
    data.load_data_from_file(data_files[0])  # Load the first file for environment setup

    # Set up environment
    env = SoilEnvironment(data=data, weight_hole_number=-2, weight_accuracy=-5)

    if model_id == "A2C":
        model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log=logdir)
    elif model_id == "PPO":
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log=logdir)
    elif model_id == "DQN":
        model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log=logdir, learning_rate=linear_schedule(1e-3),
                    exploration_initial_eps=4, exploration_fraction=0.2, exploration_final_eps=0.1, buffer_size=15000)

    # Loop through the list of data files and train on all models
    for i, data_file in enumerate(data_files):
        print(f"Processing file {i + 1}/{len(data_files)}: {data_file}")

        # Load data from each file
        data.load_data_from_file(data_file)

        # Train the model on each of the 10 models in the current file
        for model_index in range(10):
            print(f"Training on model {model_index + 1} from file: {data_file}")

            # Reset the environment to load the next model
            env.reset()

            # Train the model until RMSE goes below the threshold or until a maximum number of steps
            MAX_TIMESTEPS = 100  # Maximum number of timesteps before giving up
            total_steps = 0
            rmse = float("inf")  # Start with a high RMSE

            while rmse > rmse_threshold and total_steps < MAX_TIMESTEPS:
                model.learn(total_timesteps=1,  # Do 1 step at a time to check RMSE
                            reset_num_timesteps=False,
                            tb_log_name=model_id,
                            callback=CustomTensorboardCallback())

                total_steps += 1
                rmse = env._calc_rmse()  # Check the current RMSE
                print(f"Current RMSE after step {total_steps}: {rmse}")

            print(f"Model {model_index + 1} from {data_file} finished in {total_steps} steps with RMSE: {rmse}")

        # Save the model after each file is processed
        model.save(f"{models_dir}/model_after_file_{i + 1}")

    # Save the final trained model
    model.save(f"{models_dir}/final_model")


if __name__ == "__main__":
    # Define the list of data files
    data_dir = 'generate_simulated_fields/training_data/'
    data_files = [os.path.join(data_dir, f"test_data_10_{i}.txt") for i in range(1, 81)]  # 80 files

    # Start process for training one model across all files
    p1 = mp.Process(target=train_model, args=("DQN", data_files))

    p1.start()
    p1.join()




















# from stable_baselines3 import DQN, A2C, PPO
# from stable_baselines3.common.callbacks import BaseCallback
# from load_field_data import GetFields
# from environment import SoilEnvironment
# from custom_cnn import CustomCNNExtractor  # Custom CNN extractor
# import os
# import multiprocessing as mp
#
#
# class CustomTensorboardCallback(BaseCallback):
#     """Callback to log RMSE"""
#
#     def __init__(self, verbose: int = 0):
#         super().__init__(verbose)
#
#     def _on_step(self) -> bool:
#         infos = self.locals.get("infos", None)
#         rmse = infos[0].get("rmse")
#         self.logger.record("testing/rmse", rmse)
#         self.logger.dump(step=self.num_timesteps)
#         return True
#
#
# def get_incrementing_directory(base_dir: str) -> str:
#     """Get the next available directory with an incrementing number."""
#     existing_dirs = [int(name) for name in os.listdir(base_dir) if name.isdigit()]
#     next_index = max(existing_dirs) + 1 if existing_dirs else 0
#     new_dir = os.path.join(base_dir, str(next_index))
#     os.makedirs(new_dir, exist_ok=True)
#     return new_dir
#
#
# def train_model(model_id: str, data_files: list):
#     # Get incrementing directories for models and logs
#     models_base_dir = f"models/{model_id}"
#     logs_base_dir = f"logs/{model_id}"
#
#     models_dir = get_incrementing_directory(models_base_dir)
#     logdir = get_incrementing_directory(logs_base_dir)
#
#     print(f"Models will be saved in: {models_dir}")
#     print(f"Logs will be saved in: {logdir}")
#
#     # Loop through the list of data files
#     for i, data_file in enumerate(data_files):
#         print(f"Processing file {i + 1}/{len(data_files)}: {data_file}")
#
#         # Load data once from each file
#         data = GetFields()
#         data.load_data_from_file(data_file)
#
#         # Set up environment with data loaded
#         env = SoilEnvironment(data=data, f1=-2, f2=-5)
#
#         # Custom CNN Policy configuration
#         policy_kwargs = dict(
#             features_extractor_class=CustomCNNExtractor,
#             features_extractor_kwargs=dict(features_dim=256),
#         )
#
#         # Initialize model with CNN policy
#         if model_id == "A2C":
#             model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log=logdir)
#         elif model_id == "PPO":
#             model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log=logdir)
#         elif model_id == "DQN":
#             model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log=logdir,
#                         exploration_initial_eps=4, buffer_size=10000)
#
#         # Train the model on each of the 10 models in the current file
#         for model_index in range(10):  # Loop through the 10 models
#             print(f"Training on model {model_index + 1} from file: {data_file}")
#
#             # Reset the environment to load the next model
#             env.reset()
#
#             # Train the model for each model in the file
#             TIMESTEPS = 10  # Number of timesteps for training
#             for step in range(1):  # Reduced for demonstration (you can increase)
#                 print(f"Training on {data_file}, model {model_index + 1}, step {step + 1}")
#                 model.learn(total_timesteps=TIMESTEPS,
#                             reset_num_timesteps=False,
#                             tb_log_name=model_id,
#                             callback=CustomTensorboardCallback())
#
#                 # Save the model after training each model
#                 model.save(f"{models_dir}/{data_file.split('/')[-1].replace('.txt', '')}_model_{model_index}_step_{TIMESTEPS * step}")
#
#
# if __name__ == "__main__":
#     # Define the list of data files
#     data_dir = 'generate_simulated_fields/training_data/'
#     data_files = [os.path.join(data_dir, f"test_data_10_{i}.txt") for i in range(1, 81)]  # 80 files
#
#     # Start process for training models on multiple files
#     #p1 = mp.Process(target=train_model, args=("A2C", data_files))
#
#
#     # p2 = mp.Process(target=train_model, args=("PPO",))
#     p3 = mp.Process(target=train_model, args=("DQN", data_files))
#
#     #p1.start()
#     # p2.start()
#     p3.start()
#
#     # Wait for the processes to finish
#     #p1.join()
#     # p2.join()
#     p3.join()
