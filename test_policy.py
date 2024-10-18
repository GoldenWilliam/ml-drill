import torch as th
import numpy as np
from stable_baselines3 import DQN
from environment import SoilEnvironment
from load_field_data import GetFields
from custom_cnn import CustomCNNExtractor  # Make sure this matches the filename for the custom CNN


def test_custom_cnn_policy():
    """Test to check if the custom CNN policy works with the environment."""

    # Load the training data
    data = GetFields()
    data.load_data_from_file("generate_simulated_fields/training_data/test_data_10_1.txt")

    # Create an instance of the environment
    env = SoilEnvironment(data=data)

    # Define custom policy kwargs with the custom CNN extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=512)  # The final feature vector size after CNN
    )

    # Initialize the DQN model with the custom CNN
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # Run one learning step to see if it works
    try:
        print("Starting to test the custom CNN policy...")
        model.learn(total_timesteps=10)  # Perform 10 timesteps of learning for testing
        print("Custom CNN policy test passed successfully!")
    except Exception as e:
        print(f"Custom CNN policy test failed: {e}")


# Run the test
if __name__ == "__main__":
    test_custom_cnn_policy()
