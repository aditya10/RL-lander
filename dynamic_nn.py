import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicNeuralNetwork(nn.Module):
    def __init__(self, in_dim=8, out_dim=4, hidden_dim=4, hidden_layers=0, mutation_power=0.1):
        super(DynamicNeuralNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.mutation_power = mutation_power

        self.fitness_score = -100000
        self.avg_reward = -100000
        self.running_rewards = [-100000]*5

        self.life = 0

        self.fc_in = nn.Linear(in_dim, hidden_dim)
        #self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        #self.fc.extend([nn.Linear(hidden_dim, hidden_dim) for i in range(self.hidden_layers)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        h = F.relu(self.fc_in(x))
        #h = F.relu(self.fc_hidden(h))
        out = F.softmax(self.fc_out(h), dim=0)
        return out
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
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
