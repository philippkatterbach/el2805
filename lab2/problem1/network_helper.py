import torch
import torch.nn as nn
class MyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size,32)
        self.input_layer_activation = nn.ReLU()

        self.layer1 = nn.Linear(32,32)
        self.layer1_activation = nn.ReLU()
        
        self.output_layer = nn.Linear(32,output_size)
    def forward(self, x):
        l1 = self.input_layer(x)
        l1_act = self.input_layer_activation(l1)
        l2 = self.layer1(l1_act)
        l2_act = self.layer1_activation(l2)
        output = self.output_layer(l2_act)
        return output
model = torch.load('neural-network-1.pth')
torch.save(model.to('cpu'),'neural-network-1.pth')