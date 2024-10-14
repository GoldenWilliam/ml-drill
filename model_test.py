from stable_baselines3 import PPO, DQN, A2C
from environment import SoilEnvirment
from load_field_data import GetFields
import os

# Get current and parent directories
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
print(f"Parent directory: {parent_dir}")

# Define model directory and path to load the model
model_dir = "models"
model_path = os.path.join(current_dir, model_dir, "A2C", "290000")

# Load training data (Use the correct method 'load_data_from_file' instead of 'load_data')
data = GetFields()
data.load_data_from_file("data/train_data_10_1.txt")

# Set up the environment
env = SoilEnvirment(data=data)
env.reset()

# Load the pre-trained model
model = A2C.load(model_path, env=env)
print(f"Loaded model from {model_path}")

# Define number of test episodes
episodes = 10

# Loop over each episode
for ep in range(episodes):
    # Reset the environment and retrieve the initial observation
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    print(f"===== Episode {ep + 1} =====")

    # Continue stepping through the environment until done
    while not done:
        # Use the trained model to predict the next action
        action, _states = model.predict(obs)

        # Take the step in the environment
        obs, reward, done, truncated, info = env.step(action)

        # Accumulate rewards and print current state
        episode_reward += reward
        print(f"Observation: {obs}")
        print(f"Action: {action}, Reward: {reward}")

    # End of episode
    print(f"Total reward for episode {ep + 1}: {episode_reward}")
    print("END OF EPISODE")

    # Optionally render the environment at the end of each episode
    env.render()