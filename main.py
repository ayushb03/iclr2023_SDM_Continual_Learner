import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Step 1: Define the Top-K Activation
class TopKActivation(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        # ReLU to ensure non-negative activations (Dale's principle)
        x = F.relu(x)
        # Subtract the (k+1)-th largest value from all top-k activations
        threshold = torch.topk(x, self.k + 1, dim=1).values[:, -1].unsqueeze(1)
        return F.relu(x - threshold)

# Step 2: Define the SDMLP model
class SDMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)  # No bias
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=False)  # No bias
        self.top_k = TopKActivation(k)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)  # L2 normalization of input
        x = self.top_k(self.linear1(x))  # Top-K activation in hidden layer
        x = F.normalize(x, p=2, dim=1)  # L2 normalization after Top-K
        return self.linear2(x)  # Output layer

# Step 3: Load CIFAR-10 dataset and create data splits
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

cifar10 = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

# Split CIFAR-10 into five tasks, each with 2 classes
splits = [
    torch.utils.data.Subset(cifar10, [i for i, (_, y) in enumerate(cifar10) if y in [2*n, 2*n+1]])
    for n in range(5)
]

# DataLoader for each split
loaders = [torch.utils.data.DataLoader(split, batch_size=64, shuffle=True) for split in splits]

# Step 4: Instantiate the SDMLP model
input_dim = 32 * 32 * 3  # CIFAR-10 image size flattened
hidden_dim = 1000  # Hidden layer neurons
output_dim = 10  # CIFAR-10 has 10 classes
initial_k = 999 
target_k = 10  # End with only 10 neurons active

model = SDMLP(input_dim, hidden_dim, output_dim, initial_k)  # Model instantiation

# Step 5: Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # No momentum to avoid stale updates

# Step 6: Training loop with GABA switch for k-annealing
def anneal_k(epoch, max_k, target_k, total_epochs):
    """Linearly anneal k from max_k to target_k over the training epochs."""
    return max(target_k, int(max_k - (epoch / total_epochs) * (max_k - target_k)))

def train(model, loaders, epochs, max_k, target_k):
    model.train()
    for epoch in range(epochs):
        current_k = anneal_k(epoch, max_k, target_k, epochs)  
        model.top_k.k = current_k  # Set current k value

        for loader in loaders:
            for inputs, labels in loader:
                inputs = inputs.view(inputs.size(0), -1) 
                outputs = model(inputs)  
                loss = F.cross_entropy(outputs, labels)  

                optimizer.zero_grad()  
                loss.backward()  
                optimizer.step()  

        print(f"Epoch {epoch + 1}/{epochs}, k={current_k}, Loss: {loss.item():.4f}")

# Step 7: Train the model on Split CIFAR-10
train(model, loaders, epochs=2, max_k=initial_k, target_k=target_k)

# Step 8: Evaluation function to check for catastrophic forgetting
def evaluate(model, loaders):
    model.eval()
    total_correct = total_samples = 0

    with torch.no_grad():
        for loader in loaders:
            for inputs, labels in loader:
                inputs = inputs.view(inputs.size(0), -1)  
                outputs = model(inputs) 
                predictions = outputs.argmax(dim=1)  
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    return accuracy

evaluate(model, loaders)
