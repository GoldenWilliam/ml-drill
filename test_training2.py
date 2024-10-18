import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from load_field_data import GetFields
from environment import SoilEnvironment
from custom_cnn import CustomCNNExtractor
import os

# Test Tensorboard callback for logging
class CustomTensorboardCallback(BaseCallback):
    """Callback to log RMSE for testing model performance."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rmse_values = []  # Track RMSE values during training for testing

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        rmse = infos[0].get("rmse")
        self.rmse_values.append(rmse)  # Log RMSE for test inspection
        print(f"Step {self.num_timesteps}: RMSE = {rmse}")
        return True

def test_model_training_with_fixes():
    """Test the training process after applying fixes to reduce cyclical behavior."""

    data_file = "generate_simulated_fields/training_data/test_data_10_1.txt"

    # Load data
    data = GetFields()
    data.load_data_from_file(data_file)

    # Set up environment
    env = SoilEnvironment(data=data, f1=-2, f2=-5)
    env.reset()

    # Custom CNN Policy configuration
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # Initialize model (using A2C as an example, you can replace with PPO or DQN)
    model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-4)

    # Callback to track RMSE during training
    callback = CustomTensorboardCallback(verbose=1)

    # Run training
    model.learn(total_timesteps=20000, callback=callback)  # Increase timesteps for better training

    # After training, check RMSE values
    rmse_values = callback.rmse_values
    print("RMSE Values During Training:", rmse_values)

    # Verify if the model reduces RMSE more steadily
    assert len(rmse_values) > 0, "No RMSE values were logged during training"
    assert rmse_values[-1] < rmse_values[0], "RMSE did not decrease during training"

# Run the test
if __name__ == "__main__":
    test_model_training_with_fixes()
