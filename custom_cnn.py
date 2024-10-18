import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNNExtractor(nn.Module):
    def __init__(self, observation_space, features_dim: int = 256):
        super(CustomCNNExtractor, self).__init__()

        # We are assuming the observation_space is a Box with shape (80, 805, 3)
        n_input_channels = observation_space.shape[2]  # This should be 3 (RGB channels)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass to determine the number of features after the CNN layers
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float().permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample_input).shape[1]

        # Final fully connected layer to get to the desired feature dimension
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        # Set features_dim attribute
        self.features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Permute to fit [batch_size, channels, height, width] required by Conv2D
        cnn_output = self.cnn(observations.permute(0, 3, 1, 2))
        return self.linear(cnn_output)

