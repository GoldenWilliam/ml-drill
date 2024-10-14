import gymnasium as gym
from stable_baselines3 import DQN

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")  # For human-readable render

# Initialize and train the model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_cartpole")

del model  # Remove the model to demonstrate saving and loading

# Reload the saved model
model = DQN.load("dqn_cartpole")

# Reset the environment
obs, info = env.reset()

# Run the trained model
while True:
    # Render the environment at each step
    env.render()  # This will display the environment window or update the visualization

    # Get the model's action
    action, _states = model.predict(obs, deterministic=True)

    # Take the action in the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # If the episode ends (terminated or truncated), reset the environment
    if terminated or truncated:
        obs, info = env.reset()

# Close the environment when done
env.close()