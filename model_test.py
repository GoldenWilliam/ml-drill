import os
from stable_baselines3 import PPO, DQN, A2C
from environment import SoilEnvironment
from load_field_data import GetFields
from stable_baselines3.common.monitor import Monitor

# Define model type and path (change "A2C" to the model you want to test)
model_type = "A2C"
model_dir = f"models/"+f"{model_type}"
model_path = os.path.join(os.getcwd(), model_dir, "4/model_after_file_1.zip")

# Load testing data
data = GetFields()
data.load_data_from_file('generate_simulated_fields/test_data/test_data_10_1.txt')

# Set up the environment with parameters matching the training setup
env = SoilEnvironment(data=data, weight_hole_number=-0.2, weight_accuracy=100)
env = Monitor(env)  # Monitor helps track rewards and episode info

# Load the pre-trained model
if model_type == "A2C":
    model = A2C.load(model_path, env=env)
elif model_type == "PPO":
    model = PPO.load(model_path, env=env)
elif model_type == "DQN":
    model = DQN.load(model_path, env=env)
else:
    raise ValueError(f"Unsupported model type: {model_type}")
print(f"Loaded {model_type} model from {model_path}")

# Define the number of test episodes
episodes = 3

# Loop over each episode
for ep in range(episodes):
    # Reset the environment and retrieve the initial observation
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    final_rmse = None  # Placeholder for final RMSE at the end of the episode

    print(f"===== Episode {ep + 1} =====")

    # Continue stepping through the environment until done
    while not done:
        # Use the trained model to predict the next action
        action, _states = model.predict(obs, deterministic=True)

        # Take the step in the environment
        obs, reward, done, truncated, info = env.step(action)

        # Accumulate rewards
        episode_reward += reward

        # Update final RMSE (we assume it’s available in `info`)
        if 'rmse' in info:
            final_rmse = info['rmse']

    # End of episode
    print(f"Total reward for episode {ep + 1}: {episode_reward}")
    if final_rmse is not None:
        print(f"Final RMSE for episode {ep + 1}: {final_rmse}")
    else:
        print("Final RMSE not available.")

    print("END OF EPISODE")

    # Optionally render the environment at the end of each episode
    env.render()




