#importing Libraries
import seaborn as sns
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
print(df.head())

print(df.isnull().sum())
df['Outcome'].hist()

# sns.displot(df,x='Glucose',kind='kde')
# sns.displot(df,x='BMI',kind='kde')
# sns.displot(df,x='BloodPressure',kind='kde')
# sns.pairplot(data=df,x_vars=['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'], y_vars=['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'], hue='Outcome')

y = df['Outcome']
x = df.drop('Outcome',axis=1)
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2)

# Creating Tensors
X_train=torch.FloatTensor(x_train.values)
X_test=torch.FloatTensor(x_test.values)
Y_train=torch.LongTensor(y_train.values)
Y_test=torch.LongTensor(y_test.values)

class ANN_model(nn.Module):
    def __init__(self):
        super(ANN_model, self).__init__()
        self.linear_ann_stack = nn.Sequential(
            nn.Linear(8, 10),
            nn.Sigmoid(),
            nn.Linear(10, 20),
            nn.Sigmoid(),
            nn.Linear(20, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_ann_stack(x)
        return logits

torch.manual_seed(20)
model = ANN_model()
print(model.parameters)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

epochs=600
train_losses=[]
for i in range(epochs):
    i= i+1
    y_pred=model.forward(X_train)
    loss=loss_function(y_pred,Y_train)
    train_losses.append(loss)
    if i % 10 == 1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# tensor1 = torch.tensor(train_losses,requires_grad=True)
plt.plot(train_losses)
plt.ylabel('Loss')
plt.show()

input = torch.Tensor(X_test)
out = model.forward(input)

plt.plot(out.round())
plt.ylabel('predict')
plt.show()



