from torch import nn

class FeatureMapper(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int) -> None:

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        skip = self.fc1(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        #out = self.layer_norm(x)
        out = self.layer_norm(x+skip)
        return out



        

