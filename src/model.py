# imports
import torch
import torch.nn as nn

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters (to be changed)
input_size = 0 # Unknown for now, change later
hidden_size = 100
num_classes = 14 # accounts for all 10 digits and the four basic operations
batch_size = 100
learning_rate = 0.01

# data (to be added)

# model (to be changed)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size, device=device)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes, device=device)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)