import numpy as np
from environment import SoilEnvironment  # Ensure the path is correct
from load_field_data import GetFields


def test_environment():
    # Initialize GetFields and load data from a sample file
    field_data = GetFields()
    field_data.load_data_from_file('generate_simulated_fields/test_data/test_data_10_1.txt')  # Update with the actual path to your data file

    # Test that data has loaded correctly
    try:
        field_sample = field_data.get_field()
        assert field_sample.shape == (80, 805, 3), "Loaded field data shape should be (80, 805, 3)"
        print("Data loading test passed.")
    except ValueError as e:
        print(f"Data loading test failed: {e}")
        return  # Exit if data loading fails

    # Instantiate the environment with loaded field data
    env = SoilEnvironment(data=field_data)

    # Test reset functionality
    state, info = env.reset()
    assert state.shape == (3, env.grid_size[0], env.grid_size[1]), "Initial state shape is incorrect"
    assert env.num_holes == 0, "Initial number of holes should be 0 after reset"
    assert env.total_episode_reward == 0, "Total episode reward should be 0 after reset"
    print("Environment reset test passed.")

    # Test stepping through random actions
    num_steps = 5
    for step in range(num_steps):
        action = env.action_space.sample()  # Randomly sample an action
        state, reward, terminated, truncated, info = env.step(action)

        # Basic checks on state and reward
        assert state.shape == (3, env.grid_size[0], env.grid_size[1]), "State shape mismatch after step"
        assert isinstance(reward, float), "Reward should be a float"

        # Check RMSE in info
        assert "rmse" in info, "RMSE should be present in info dict"
        assert info["rmse"] >= 0, "RMSE should be non-negative"

        # Print some information for each step
        print(
            f"Step {step + 1}: Action = {action}, Reward = {reward}, RMSE = {info['rmse']}, Terminated = {terminated}")

        # Check termination if threshold conditions are met
        if terminated:
            print("Environment terminated correctly based on conditions.")
            break

    # Check if episode terminated correctly by number of digs or RMSE threshold
    assert terminated or env.num_holes < 10, "Episode should terminate when num_holes reaches 10 or RMSE < threshold"

    print("Step test passed.")


if __name__ == "__main__":
    test_environment()


