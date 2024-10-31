from gymnasium import Space
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from CNN_enviroment import SoilEnvirment


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: Space, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)

        # Input shape (channels, height, weight)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Downsample with stride 2
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Further downsample
                    nn.ReLU(),
                    nn.Flatten()
                )

        # Compute number of fetures afte CNN layers
        with th.no_grad():
            sample_input = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]

        # Output fully connected layer to match the fature dimentions
        self.linear = nn.Sequential(
            nn.Linear(n_flatten,features_dim),
            nn.ReLU()
        )


    def forward(self, obersevations):
        return self.linear(self.cnn(obersevations))
    
 
policy_kwargs = dict(
     features_extractor_class = CustomCNN,
     features_extractor_kwargs=dict(features_dim=126)
 )

def init_weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)




# class CustomCnnPolicy(ActorCriticCnnPolicy):
#     def __init___(self, *args, **kwargs):
#         super(CustomCnnPolicy).__init__(
#             *args,
#             features_extractor_class=CustomCNN,
#             **kwargs
#         )
