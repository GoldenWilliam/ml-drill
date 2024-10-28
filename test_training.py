import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from load_field_data import GetFields
from environment import SoilEnvironment
from custom_cnn import CustomCNNExtractor

class TrainingTestCallback(BaseCallback):
    """
    Callback for testing purposes to log RMSE and rewards to the console
    and TensorBoard for each episode.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.rmse_values = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        rmse = infos[0].get("rmse") if infos else None
        reward = self.locals.get("rewards", [0])[0]

        # Log RMSE if available
        if rmse is not None:
            self.rmse_values.append(rmse)
            self.logger.record("test/rmse", rmse)
            if self.verbose:
                print(f"Step {self.num_timesteps} - RMSE: {rmse}")

        # Accumulate rewards
        self.episode_rewards[-1] += reward

        done = self.locals.get("dones", [False])[0]
        if done:
            # Log episode reward and length
            self.logger.record("test/episode_reward", self.episode_rewards[-1])
            self.logger.record("test/episode_length", self.episode_lengths[-1])
            if self.verbose:
                print(f"Episode finished - Total Reward: {self.episode_rewards[-1]}, Steps: {self.episode_lengths[-1]}")

            # Reset for next episode
            self.episode_rewards.append(0)
            self.episode_lengths.append(0)
        else:
            # Increment step count for the current episode
            self.episode_lengths[-1] += 1

        self.logger.dump(step=self.num_timesteps)
        return True

    def _on_rollout_start(self) -> None:
        self.episode_rewards.append(0)
        self.episode_lengths.append(0)


def test_training(model_id="DQN"):
    """Runs a test training session on the environment to ensure functionality."""
    data_dir = 'generate_simulated_fields/training_data/'
    data_files = [os.path.join(data_dir, f"test_data_10_{i}.txt") for i in range(1, 3)]  # Use 2 files for testing

    # Environment setup
    data = GetFields()
    data.load_data_from_file(data_files[0])
    env = SoilEnvironment(data=data, weight_hole_number=-2, weight_accuracy=-5, rmse_threshold=0.1)

    # Model setup
    logdir = "./logs/test_run"
    os.makedirs(logdir, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)

    print("\nStarting test training...")
    for data_file in data_files:
        print(f"\nProcessing data file: {data_file}")
        data.load_data_from_file(data_file)

        # Test training on 3 episodes to check logging and reward accumulation
        callback = TrainingTestCallback(verbose=1)
        model.learn(total_timesteps=300, tb_log_name="test_dqn", callback=callback)

    print("\nTest training completed.")
    print("Final RMSE values from test:", callback.rmse_values)


if __name__ == "__main__":
    print("\nRunning training test...")
    test_training()


