import numpy as np 
from torch import nn
from torch.distribitions import Normal
from rl_template import BaseAgent

class Agent(BaseAgent):
    def __init__(self, n_state, n_action):
        super(Agent).__init__()
        self.actor = nn.Sequential(
                nn.Linear(n_state, 128),
                nn.ReLU(),
                nn.Linear(128, n_action)
                )

        self.critic = nn.Sequential(
                nn.Linear(n_state, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
                )
        


    def _init_weight(self):
        for layer in self.actor_layer:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

            actor_out = self.actor_layer[-1]
            nn.init.orthogonal_(actor_out.weight, gain=0.01)
            nn.init.constant_(actor_out.bias, 0.0)   
    
