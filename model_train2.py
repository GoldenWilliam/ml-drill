from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from load_field_data import GetFields
from environment import SoilEnvironment
from custom_cnn import CustomCNNExtractor  # Custom CNN extractor
import os
import multiprocessing as mp


class CustomTensorboardCallback(BaseCallback):
    """CallBack to log rmse"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        rmse = infos[0].get("rmse")
        self.logger.record_mean("testing/rmse", rmse)
        self.logger.dump(step=self.num_timesteps)

        return True



def get_incrementing_directory(base_dir: str) -> str:
    """Create a new incremented directory for models/logs."""
    existing_dirs = [int(name) for name in os.listdir(base_dir) if name.isdigit()]
    next_index = max(existing_dirs) + 1 if existing_dirs else 0
    new_dir = os.path.join(base_dir, str(next_index))
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


def linear_schedule(initial_value: float):
    """Returns a linearly decaying learning rate."""
    return lambda progress: initial_value * progress


def train_model(model_id: str, data_files: list, rmse_threshold: float = 15.0):
    models_dir = get_incrementing_directory(f"models/{model_id}")
    logdir = get_incrementing_directory(f"logs/{model_id}")

    print(f"Models will be saved in: {models_dir}")
    print(f"Logs will be saved in: {logdir}")

    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # Set up environment
    data = GetFields()
    data.load_data_from_file(data_files[0])  # Load the first file for environment setup
    env = SoilEnvironment(data=data, weight_hole_number=-2, weight_accuracy=-5, rmse_threshold=rmse_threshold)
    env.reset()

    # Initialize model with specified algorithm
    if model_id == "A2C":
        model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)
    elif model_id == "PPO":
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)
    elif model_id == "DQN":
        model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir,
                    learning_rate=linear_schedule(1e-3), exploration_initial_eps=1, exploration_fraction=0.2,
                    exploration_final_eps=0.1, buffer_size=15000)

    # Training loop over all data files
    for i, data_file in enumerate(data_files):
        print(f"\nProcessing file {i + 1}/{len(data_files)}: {data_file}")
        data.load_data_from_file(data_file)

        for model_index in range(10):  # Train on 10 different models in the current file
            print(f"Training model {model_index + 1} from {data_file}")
            model.learn(total_timesteps=10,  # Training for a fixed number of steps
                        reset_num_timesteps=False,
                        tb_log_name=model_id,
                        callback=CustomTensorboardCallback())

        model.save(f"{models_dir}/model_after_file_{i + 1}")
        print(f"Model saved after processing file {i + 1}")

    model.save(f"{models_dir}/final_model")
    print("Final model saved.")


if __name__ == "__main__":
    data_dir = 'generate_simulated_fields/training_data/'
    data_files = [os.path.join(data_dir, f"test_data_10_{i}.txt") for i in range(1, 81)]  # 80 files

    # Start training in a separate process
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
