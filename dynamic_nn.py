import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicNeuralNetwork(nn.Module):
    def __init__(self, in_dim=9, out_dim=4, hidden_dim=4, mutation_power=0.1):
        super(DynamicNeuralNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.mutation_power = mutation_power
        
        self.baseline_score = -100000
        self.fitness_score = -100000
        self.avg_reward = -100000
        self.running_rewards = [-100000]*5
        self.best_reward = -100000

        self.age = 0

        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        h = F.relu(self.fc_in(x))
        h2 = F.relu(self.fc_hidden(h))
        out = F.softmax(self.fc_out(h2), dim=0)
        return out
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def init_random_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def load_from_agent(self, d):
        d2 = self.state_dict()
        for k in d:
            if k.endswith('bias'):
                for i, val in enumerate(d[k]):
                    d2[k][i] = val
            else:
                for i,val in enumerate(d[k]):
                    for j,val2 in enumerate(d[k][i]):
                        d2[k][i][j] = val2

    def show_param(self):
        num_params = sum(p.numel() for p in self.parameters())
        details = {
            'age': self.age,
            'fitness_score': self.fitness_score,
            'avg_reward': self.avg_reward,
            'running_rewards': self.running_rewards,
            'baseline_score': self.baseline_score,
            'mutation_power': self.mutation_power,
            'hidden_dim': self.hidden_dim,
            'num_params': num_params,
            'best_reward': self.best_reward
        }
        return details
