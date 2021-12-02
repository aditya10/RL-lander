import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim=8, out_dim=4, hidden_dim=128):
        super(NeuralNetwork, self).__init__()

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