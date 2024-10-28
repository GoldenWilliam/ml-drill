import os
import multiprocessing as mp
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from load_field_data import GetFields
from environment import SoilEnvironment
from custom_cnn import CustomCNNExtractor  # Custom CNN extractor


class CustomTensorboardCallback(BaseCallback):
    """Callback to log RMSE, per-step rewards, cumulative episode rewards, and final episode RMSE to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_reward = 0  # Track total reward for each episode
        self.episode_rmse = 0  # Track final RMSE for each episode
        self.episode_num = 0  # Track the episode number

    def _on_step(self) -> bool:
        # Log RMSE if available
        infos = self.locals.get("infos", None)
        if infos is not None:
            rmse = infos[0].get("rmse", None)
            if rmse is not None:
                self.logger.record("testing/rmse", rmse)  # Log RMSE at each step
                self.episode_rmse = rmse  # Update episode RMSE with the latest RMSE value

        # Get current reward and add to episode total
        reward = self.locals.get("rewards", [0])[0]
        self.episode_reward += reward

        # Log the reward for the current step
        self.logger.record("testing/step_reward", reward)
        self.logger.dump(step=self.num_timesteps)  # Immediately log the step reward for each step

        # Check if the episode is done to log episode rewards and RMSE
        done = self.locals.get("dones", [False])[0]
        if done:
            # Log cumulative episode reward
            self.logger.record("testing/episode_reward", self.episode_reward)

            # Log final episode RMSE
            self.logger.record("testing/episode_rmse", self.episode_rmse)
            self.logger.dump(step=self.episode_num)  # Log the episode metrics at the end of each episode

            # Reset counters for the next episode
            self.episode_reward = 0
            self.episode_rmse = 0
            self.episode_num += 1

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


def train_model(model_id: str, data_files: list, rmse_threshold: float = 20.0):
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
    data.load_data_from_file(data_files[0])  # Load the first file to initialize environment
    env = SoilEnvironment(data=data, weight_hole_number=-0.2, weight_accuracy=100, rmse_threshold=rmse_threshold)
    env.reset()

    # Initialize model based on the specified algorithm
    if model_id == "A2C":
        model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)
    elif model_id == "PPO":
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)
    elif model_id == "DQN":
        model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir,
                    learning_rate=linear_schedule(1e-3), exploration_initial_eps=1, exploration_fraction=0.15,
                    exploration_final_eps=0.15, buffer_size=50000)


    # Create a single instance of CustomTensorboardCallback outside the loop
    callback = CustomTensorboardCallback()

    # Training loop over all data files
    for i, data_file in enumerate(data_files):
        print(f"\nProcessing file {i + 1}/{len(data_files)}: {data_file}")
        data.load_data_from_file(data_file)

        for model_index in range(10):  # Train on 10 different models within the current file
            print(f"Training model {model_index + 1} from {data_file}")
            model.learn(
                total_timesteps=10,  # Larger value to avoid frequent resets
                reset_num_timesteps=False,
                tb_log_name=model_id,
                callback=callback  # Use the single callback instance
            )

        # Save intermediate model
        model.save(f"{models_dir}/model_after_file_{i + 1}")
        print(f"Model saved after processing file {i + 1}")

    # Save the final model
    model.save(f"{models_dir}/final_model")
    print("Final model saved.")


if __name__ == "__main__":
    data_dir = 'generate_simulated_fields/training_data/'
    data_files = [os.path.join(data_dir, f"test_data_10_{i}.txt") for i in range(1, 81)]  # Assuming 80 files

    # Start training in separate processes for each algorithm
    processes = []
    for model_id in ["DQN", "A2C", "PPO"]:
        process = mp.Process(target=train_model, args=(model_id, data_files))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
