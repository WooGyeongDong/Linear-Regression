#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import copy
#%%

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Data 추출
train_target = train_data['SalePrice']
train_data = train_data.drop(columns=['Id','SalePrice'])
test_data = test_data.drop(columns=['Id'])
train_target = np.log(train_target)

# train data와 test data가 같은 더미변수를 가지기 위해 합침
all_data = pd.concat((train_data, test_data))

# 결측값은 0으로 채우고 더미변수로 변환
numerical_features = all_data.dtypes[all_data.dtypes!='object'].index
all_data[numerical_features] = all_data[numerical_features].fillna(0)
all_data = pd.get_dummies(all_data, dummy_na = True, drop_first = True)

# train data와 test data 분리
train_data = all_data.iloc[:1460,:]
test_data = all_data.iloc[1460:,:]


# vaildation dataset split
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=0.25, random_state= 8)

#%%
# Dataset class 선언
class CustomDataset(Dataset):
	def __init__(self, data, target):

		self.inp = data.values
		self.outp = target.values.reshape(-1,1)

	def __len__(self):
		return len(self.inp)

	def __getitem__(self,idx):
		inp = torch.FloatTensor(self.inp[idx])
		outp = torch.FloatTensor(self.outp[idx])
		return inp, outp # 해당하는 idx(인덱스)의 input과 output 데이터를 반환한다.

# Train data와 Validation data를 DataLoader에 넣음
dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_valid, y_valid)

#%%
def get_data(train_ds, valid_ds, bs):
  return (
    DataLoader(train_ds, batch_size=bs, shuffle=True),
    DataLoader(valid_ds, batch_size=bs*2),      
  )

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'    

def preprocess(x, y):
    return x.to(device), y.to(device)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

dataloader, val_dataloader = get_data(dataset, val_dataset, 256)
dataloader = WrappedDataLoader(dataloader, preprocess)
val_dataloader = WrappedDataLoader(val_dataloader, preprocess)

#%%
# 학습 모델
class Net(torch.nn.Module):
    def __init__(self, layer_value):
        super(Net,self).__init__()
        self.layer = nn.ModuleList()
        for i, _ in enumerate(layer_value[:-1]) :
            self.layer.append(nn.Linear(layer_value[i],layer_value[i+1]))
            nn.init.xavier_uniform_(self.layer[i].weight)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self,x):
      out = x
      for layer in self.layer[:-1] :
          out = F.gelu(layer(out))
          out = self.dropout(out)

      output = torch.relu(self.layer[-1](out))

      return output
#%%
# model과 optimizer 선언
def get_model(layer_value):
  model = Net(layer_value)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
  return model, optimizer, scheduler

#%%
# log에 0이 들어가지 않게 조정 후 loss 반환
def MSLE_loss(pred, target):
    log_pred = torch.log(pred + 0.1)
    loss = nn.MSELoss()(log_pred, target)
    return loss
  
#%%
def loss_batch(model, loss_func, xb, yb, opt=None):
  loss = loss_func(model(xb), yb)
  
  if opt is not None:
    opt.zero_grad()
    loss.backward()
    opt.step()
    
  return loss.item(), len(xb)

#%%
def fit(model, loss_func, opt, scheduler, train_dl, valid_dl, nb_epochs):
  best_loss = 10 ** 9 # 매우 큰 값으로 초기값 가정
  patience_limit = 3 # 몇 번의 epoch까지 지켜볼지를 결정
  patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
  val = []
  for epoch in range(nb_epochs):
    model.train()
    for x_train, y_train in train_dl:
      loss_batch(model, loss_func, x_train, y_train, opt)
      
    scheduler.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
      for x_val, y_val in valid_dl:
        losses, nums = loss_batch(model, loss_func, x_val, y_val)
        val_loss += losses/nums
    val.append(val_loss)

    print(epoch, val_loss)
    if abs(val_loss - best_loss) < 1e-3: # loss가 개선되지 않은 경우
      patience_check += 1

      if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
          print("Learning End. Best_Loss:{:6f}".format(best_loss/len(val_dataloader)))
          break

    else: # loss가 개선된 경우
      best_loss = val_loss
      best_model = copy.deepcopy(model)
      patience_check = 0
      
  return val, copy.deepcopy(best_model)
#%%   

model, optimizer, scheduler = get_model([288,1024,1024,1])
vali_loss, best_model = fit(model, MSLE_loss, optimizer, scheduler, dataloader, val_dataloader, 500)
 
import matplotlib.pyplot as plt
plt.plot(vali_loss)

