import torch
import gc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Define the CNN layers
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        # Perform the forward pass of the CNN
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)

        return x
    
model = CNN().to(device)

# Tile the input tensor across the spatial dimensions
input_tensor = torch.randn(100, 3, 224, 224).to(device, dtype=torch.float32)
# tiled_input_tensor = torch.tile(input_tensor, (1, 1, 2, 2)).to(device)

output_tensor = model(input_tensor).to(device)

# Print the output tensor
print(output_tensor)

