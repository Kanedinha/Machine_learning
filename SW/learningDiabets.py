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

df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')
df.head()

df.isnull().sum()
df['Outcome'].hist()
sns.displot(df,x='Glucose',kind='kde')
sns.displot(df,x='BMI',kind='kde')
sns.displot(df,x='BloodPressure',kind='kde')
plt.show()

# sns.pairplot(data=df,x_vars=['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'], y_vars=['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'], hue='Outcome')
# plt.show()

from sklearn.datasets import load_diabetes
X,y = load_diabetes(return_X_y=True, as_frame=True)

y = df['Outcome']
x = df[['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2)

X_train=torch.FloatTensor(x_train.values)
X_test=torch.FloatTensor(x_test.values)
Y_train=torch.LongTensor(y_train.values)
Y_test=torch.LongTensor(y_test.values)

# Creating the Model
# -- Complete the code
class ANN_model(nn.Module):
    def __init__(self):
        super(ANN_model, self).__init__()
        self.linear_ann_stack = nn.Sequential(
            nn.Linear(8, 8),
            nn.Sigmoid(),
            nn.Linear(8, 8),
            nn.Sigmoid(),
            nn.Linear(8, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_ann_stack(x)
        return logits

torch.manual_seed(20)
model = ANN_model()

model.parameters

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

epochs=10000
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


tensor1 = torch.tensor(train_losses,requires_grad=True)
plt.plot(tensor1.detach().numpy())
plt.ylabel('Loss')
plt.show()

predictions = model.forward(X_test)
y_test_np = Y_test.numpy()
y_pred_np = predictions.max(1)[1].detach().numpy()  # Obtém o índice da classe com a maior probabilidade

# Criar a matriz de confusão
matrix = confusion_matrix(y_test_np, y_pred_np)

# Plotar a matriz de confusão
from sklearn.metrics import accuracy_score

# Calcular a acurácia
accuracy = accuracy_score(y_test_np, y_pred_np)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Valores Previstos')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão')
plt.show()






