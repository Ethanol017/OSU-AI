import torch
import torch.nn as nn
from torch.distributions import Normal,Bernoulli

class Actor(nn.Module):

    def __init__(self,log_std_min=-5, log_std_max=0):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # input: 1 * 72 * 128
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 9 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.continuous_head_mean = nn.Linear(128, 2)
        self.continuous_head_std = nn.Linear(128, 2)
        self.discrete_head = nn.Linear(128, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        continuous_mean = self.continuous_head_mean(x)
        continuous_log_std = self.continuous_head_std(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max  - self.log_std_min) * (log_std + 1)
        logits = self.discrete_head(x)
        discrete_prob = torch.softmax(logits, dim=-1)
        return continuous_mean, continuous_log_std, discrete_prob
    
    def get_action(self,state):
        mean, log_std, discrete_prob = self.forward(state)
        # continuous
        normal = Normal(mean, log_std.exp())
        continuous_action = normal.rsample()
        continuous_action = torch.tanh(continuous_action)
        u = torch.atanh(continuous_action)
        cont_log_pi_prob = ( normal.log_prob(u) - torch.log(1 - continuous_action.pow(2) + 1e-6) ).sum(dim=-1, keepdim=True) # shape: (batch_size, 1)
        # discrete
        discrete_action = Bernoulli(probs=discrete_prob)
        discrete_action = discrete_action.sample()
        disc_log_pi_prob = discrete_action.log_prob(discrete_action).sum(dim=-1, keepdim=True) # shape: (batch_size, 1)
        return continuous_action, discrete_action, cont_log_pi_prob, disc_log_pi_prob
    
class Critic(nn.Module):
    def __init__(self):
        # input: 1 * 72 * 128
        super(Critic, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 9 * 16, 512),
            nn.ReLU(),
            nn.Linear(512+3, 256), # 3 for action dim
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    def forward(self, x, action):
        x = self.cnn(x)
        x = self.flatten(x)
        x = torch.cat((x, action), dim=1) # concatenate action
        x = self.fc(x)
        return x