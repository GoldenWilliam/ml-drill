import numpy as np
from environment import SoilEnvironment
from load_field_data import GetFields

def test_environment_initialization():
    """Test the initialization of the environment."""
    data = GetFields()
    data.load_data_from_file("generate_simulated_fields/training_data/test_data_10_1.txt")
    env = SoilEnvironment(data=data)

    assert env.f1 == -0.3
    assert env.f2 == -10
    assert env.start_position_x == 0
    print("test_environment_initialization passed!")

def test_reset_method():
    """Test the reset method and initial state."""
    data = GetFields()
    data.load_data_from_file("generate_simulated_fields/training_data/test_data_10_1.txt")
    env = SoilEnvironment(data=data)

    observation, _ = env.reset()
    assert observation is not None
    assert observation.shape == (240,)  # This should be 240 for 80x3 RGB fields
    assert env.current_position == env.start_position_x
    print("test_reset_method passed!")

def test_get_hole_method():
    """Test the get_hole method, which retrieves the patch of the field."""
    data = GetFields()
    data.load_data_from_file("generate_simulated_fields/training_data/test_data_10_1.txt")
    env = SoilEnvironment(data=data)

    env.reset()
    hole = env.get_hole(env.current_position)
    assert hole is not None
    assert hole.shape == (80, 3)  # Hole should be 80 pixels deep with 3 RGB channels
    assert (hole >= 0).all() and (hole <= 255).all()  # Check that RGB values are in range
    print("test_get_hole_method passed!")

def test_step_method():
    """Test the step function to ensure agent actions are performed correctly."""
    data = GetFields()
    data.load_data_from_file("generate_simulated_fields/training_data/test_data_10_1.txt")
    env = SoilEnvironment(data=data)

    env.reset()
    action = 2  # Test with a valid action (e.g., move by 8 units)
    observation, reward, done, truncated, info = env.step(action)

    assert observation is not None
    assert observation.shape == (240,)  # Observation should be flattened RGB values
    assert reward is not None
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert 'rmse' in info
    print("test_step_method passed!")

def test_render_method():
    """Test the render function to verify visualization works."""
    data = GetFields()
    data.load_data_from_file("generate_simulated_fields/training_data/test_data_10_1.txt")
    env = SoilEnvironment(data=data)

    env.reset()
    try:
        env.render()
        print("Render method passed!")
    except Exception as e:
        print(f"Render method failed: {e}")

# Run all tests
if __name__ == "__main__":
    test_environment_initialization()
    test_reset_method()
    test_get_hole_method()
    test_step_method()
    test_render_method()

