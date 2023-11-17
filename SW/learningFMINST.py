import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader # loads data in batches
from torchvision import datasets # load Fasion-MNIST
import torchvision.transforms as T # transformers for computer vision 

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # progress bar

mytransform = T.ToTensor() # image (3D array) to Tensor

train_data = datasets.FashionMNIST(root = './', download=True, train = True, transform = mytransform)
test_data = datasets.FashionMNIST(root = './', download=True, train = False, transform = mytransform)

img, label = train_data[0]
img.shape # returns a Tensor of Size 1,28,28

# We could simply plot the tensor
plt.imshow(img.reshape(28,28), cmap = 'gist_yarg'); # gist_yarg plots inverse of W&B
plt.axis('off');
plt.show()

torch.manual_seed(101)

train_loader = DataLoader(train_data, batch_size = 100, shuffle=True)
# the test loader can be bigger and doesn't need to be shuffled
test_loader =  DataLoader(test_data,  batch_size = 500, shuffle=False)

# Plot 10 images
for img, label in train_loader:
    break # we run only one iteration , after that we break
img.shape # bz, ch, W H

myimages = img[:50].numpy() # we now obtain NumPy arrays
myimages.shape

myimages[0].shape # channel, height, width

myimages[0].transpose(1,2,0).shape # height, width, channel

fig, ax = plt.subplots(nrows = 5, ncols = 10, figsize=(8,4), subplot_kw={'xticks': [], 'yticks': []})
for row in range(0,5):
    for col in range(0,10):
        myid = (10*row) + col # (ncols*rows) + cols

        ax[row,col].imshow( myimages[myid].transpose(1,2,0), cmap = 'gist_yarg' ) # W,H,C
        ax[row,col].axis('off')
plt.show()

class MultilayerPerceptron(nn.Module):
    def __init__(self):
        super(MultilayerPerceptron, self).__init__()
        self.linear_ann_stack = nn.Sequential(
            nn.Linear(784, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        logits = self.linear_ann_stack(x)
        return logits
    
torch.manual_seed(101)

mymodel = MultilayerPerceptron() # default params are in_features = 784, out_features=10
mymodel # check topology

learning_rate = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mymodel.parameters(), learning_rate)

params = [p.numel() for p in mymodel.parameters() if p.requires_grad]
print("parameters to evaluate: ", np.sum(params))

myiter = iter(train_loader)
img, label = myiter.__next__() # only one iteration
img.shape # batch_size, channel, Height, Width

print("Batches of: ", img.view(100,-1).shape)

y_pred = mymodel( img.view(100,-1) )
y_pred.shape # 100 x 10, meaning for every batch (100) we obtain  (10 probabilities) predictions

val, idx = torch.max(y_pred, dim=1) # dim 1 is for the output
print(idx) # indices == predictions

# tracking variables

class Loss:
    """ Class to monitor train and test lost"""
    train: list = []
    test: list = []


class Accuracy:
    """ Class to monitor train and test accuracy"""
    train: list = []
    test: list = []

from sklearn.metrics import accuracy_score
# Train for 10 epocs
myiter = iter(train_loader)
img_train, label = myiter.__next__()

epochs=10
losses = Loss()
acc = Accuracy()
for i in range(epochs):
    i= i+1
    y_pred=mymodel.forward(img_train.view(100,-1))
    loss=criterion(y_pred,label)
    losses.train.append(loss)
    accuracy = accuracy_score(y_pred, label)
    acc.train.append(accuracy)
    print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

myiter = iter(test_loader)
img_test, label = myiter.__next__()

y_test = mymodel(img_test.view(10000,-1))
loss = criterion(y_test,label)
losses.test.append(loss)
accuracy = accuracy_score(y_test, label)
acc.test.append(accuracy)
    

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

LossTrainTensor = torch.tensor(Loss.train,requires_grad=True)
LossTestTensor = torch.tensor(Loss.test,requires_grad=True)

ax[0].plot(LossTrainTensor.detach().numpy(), label = 'Training')
ax[0].plot(LossTestTensor.detach().numpy(), label='test/validation')
ax[0].set_ylabel('Loss', fontsize=16)

ax[1].plot(Accuracy.train, label = 'Training')
ax[1].plot(Accuracy.test, label='test/validation')
ax[1].set_yticks(range(85,110,5))
ax[1].axvline(x=2, color='gray', linestyle=':')
ax[1].axhline(y=100, color='gray', linestyle=':')
ax[1].set_ylabel('Accuracy (%)', fontsize=16)
plt.show()

for myax in ax:
    myax.set_xlabel('Epoch', fontsize=16)
    myax.set_xticks(range(epochs))
    myax.legend(frameon=False)

test_loader =  DataLoader(test_data,  batch_size = 10_000, shuffle=False) # the whole test is 10,000 images
myiter = iter(test_loader)
img, label = myiter.__next__()
img.shape

with torch.no_grad():
    correct = 0

    for X, y_label in test_loader:
            y_val = mymodel( X.view(X.shape[0],-1) ) # flatten
            _, predicted = torch.max( y_val, dim = 1)
            correct += (predicted == y_label).sum()

print(f'Test accuracy: = {correct.item()*100/(len(test_data)):2.4f} %')

# Show the confusion matrix

import seaborn as sns
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(predicted, label)
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão')
plt.show()