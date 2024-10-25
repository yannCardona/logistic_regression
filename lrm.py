import pandas as pd
import numpy as np
import matplotlib as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

df = pd.read_csv("data_banknote_authentication.txt", header=None)
print(df.head())

X_features = df[[0, 1, 2, 3]].to_numpy()
y_labels = df[4].to_numpy()

# Defining Dataset and Dataloader
class MyDataset(Dataset):
	def __init__(self, X, y):

		self.features = torch.tensor(X, dtype=torch.float32)
		self.labels = torch.tensor(y, dtype=torch.float32)

	def __getitem__(self, index):
		x = self.features[index]
		y = self.labels[index]        
		return x, y

	def __len__(self):
		return self.labels.shape[0]

# 80% of the data is used for training and 20% for validation 
train_size = int(X_features.shape[0]*0.80)
val_size = X_features.shape[0] - train_size

dataset = MyDataset(X_features, y_labels)

train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
	dataset=train_set,
	batch_size=10,
	shuffle=True,
)

val_loader = DataLoader(
	dataset=val_set,
	batch_size=10,
	shuffle=False,
)

# Standardization
train_mean = torch.zeros(X_features.shape[1])

for x, y in train_loader:
    train_mean += x.sum(dim=0)
    
train_mean /= len(train_set)

train_std = torch.zeros(X_features.shape[1])
for x, y in train_loader:
    train_std += ((x - train_mean)**2).sum(dim=0)

train_std = torch.sqrt(train_std / (len(train_set)-1))

def standardize(df, train_mean, train_std):
    return (df - train_mean) / train_std

class LogisticRegression(torch.nn.Module):
	
	def __init__(self, num_features):
		super().__init__()
		self.linear = torch.nn.Linear(num_features, 1)
	
	def forward(self, x):
		logits = self.linear(x)
		probas = torch.sigmoid(logits)
		return probas

model = LogisticRegression(num_features=4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5) ## possible SOLUTION

num_epochs = 2 ## possible SOLUTION

for epoch in range(num_epochs):
	
	model = model.train()
	for batch_idx, (features, class_labels) in enumerate(train_loader):

		features = standardize(features, train_mean, train_std)
		probas = model(features)
		
		loss = F.binary_cross_entropy(probas, class_labels.view(probas.shape))
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if not batch_idx % 20: # log every 20th batch
			print(f'Epoch: {epoch+1:03d}/{num_epochs:03d}'
				f' | Batch {batch_idx:03d}/{len(train_loader):03d}'
				f' | Loss: {loss:.2f}')

def compute_accuracy(model, dataloader):

	model = model.eval()
	
	correct = 0.0
	total_examples = 0
	
	for idx, (features, class_labels) in enumerate(dataloader):
		
		features = standardize(features, train_mean, train_std)
		with torch.no_grad():
			probas = model(features)
		
		pred = torch.where(probas > 0.5, 1, 0)
		lab = class_labels.view(pred.shape).to(pred.dtype)

		compare = lab == pred
		correct += torch.sum(compare)
		total_examples += len(compare)

	return correct / total_examples

train_acc = compute_accuracy(model, train_loader)
print(f"Accuracy: {train_acc*100:.2f}%")

val_acc = compute_accuracy(model, val_loader)
print(f"Accuracy: {val_acc*100:.2f}%")

