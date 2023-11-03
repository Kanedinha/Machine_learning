# import libraries 
import torch
import torch.nn as nn
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')
print("Using {0}".format(device))

from torch.utils.data import Dataset

class XorDataset(Dataset):
    def __init__(self):
        self.Xs = torch.Tensor([[0., 0.],
                                [0., 1.],
                                [1., 0.],
                                [1., 1.]])
        self.y = torch.Tensor([0., 1., 1., 0.])
        
    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        image = self.Xs[idx]
        label = self.y[idx]
        return image, label
    
from torch.utils.data import DataLoader

training_data = XorDataset()
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
feature = train_features[0]
label = train_labels[0]
print(f"Features: {feature}; Label: {label}")

class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.linear_xor_stack = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_xor_stack(x)
        return logits
    
xor_network = XOR()
model = xor_network.to(device)
print(model)

from torch.utils.data import DataLoader
all_losses=[]

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    all_losses.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")   

learning_rate = 0.3
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 2000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(train_dataloader, model, loss_fn)
print("Done!")

import matplotlib.pyplot as plt

plt.plot(all_losses)
plt.ylabel('Loss')
plt.show()

# show weights and bias
for name, param in xor_network.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# test input
input = torch.Tensor([[0., 0.],
               [0., 1.],
               [1., 0.],
               [1., 1.]])
out = xor_network.forward(input)
print(out.round())

