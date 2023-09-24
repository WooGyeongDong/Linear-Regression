 House Prices Prediction with Deepleaning
 ======================================

### Data 불러오기
```
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
```

### Data 가공
```
# Train, Target, Test Data 추출
train_target = train_data['SalePrice']
train_data = train_data.drop(columns=['Id','SalePrice'])
test_data = test_data.drop(columns=['Id'])

# Target Data Log변환
train_target = np.log(train_target)

# train data와 test data가 같은 더미변수를 가지기 위해 합침
all_data = pd.concat((train_data, test_data))

# 결측값은 0으로 채우고 더미변수로 변환
numerical_features = all_data.dtypes[all_data.dtypes!='object'].index
all_data[numerical_features] = all_data[numerical_features].fillna(0)
all_data = pd.get_dummies(all_data, dummy_na = True, drop_first = True)
```
- 더미변수 변환
$$x\in\left\{ C_1,C_2,\cdots ,C_k\right\}$$
$$x\to\begin{pmatrix}
d_{C_2} & d_{C_3} & \cdots & d_{C_k} \\
\end{pmatrix}$$
$$\ d_{C_i} =
\begin{cases}
1,\;if\; d_{C_i}\ is\ {C_i}\\
0,\;otherwise
\end{cases}$$

```
# train data와 test data 분리
train_data = all_data.iloc[:1460,:]
test_data = all_data.iloc[1460:,:]

# train data와 vaildation data 분리
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=0.25, random_state= 8)
```

```
#Data shape 확인
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
#결과
(1095, 288) (365, 288) (1095,) (365,)
```
- Data 차원
$$X_{train}\in\mathbb{R}^{1095\times288},\;\;\;y_{train}\in\mathbb{R}^{1095}$$
$$X_{valid}\in\mathbb{R}^{365\times288},\;\;\;y_{valid}\in\mathbb{R}^{365}$$

### Data를 DataLoader로 변환
```
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
		return inp, outp
```

```
# Train data와 Validation data를 DataLoader에 넣음
dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_valid, y_valid)

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
- Batch로 분할
$$Z=\begin{pmatrix}
 X_{train}& y_{train} \\
\end{pmatrix},\;\;\;\;Z\prime=shuffle(Z)$$
$$Z\prime=\begin{pmatrix}
X_1 & y_1 \\
X_2 & y_2 \\
\vdots  & \vdots  \\
X_n & y_n \\
\end{pmatrix},\;\;\;\;n=\#\;of\;batch$$
$$X_i\in\mathbb{R}^{batchsize\times288},\;\;\;y_i\in\mathbb{R}^{batchsize}$$
마찬가지로 vaildation data도 분할한다.

### 모형 구축
```
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.hidden_layer1 = nn.Linear(288,1024)
        self.hidden_layer2 = nn.Linear(1024,1024)
        self.output_layer = nn.Linear(1024,1)
        self.dropout = nn.Dropout(p=0.2)
        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.xavier_uniform_(self.hidden_layer2.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
    def forward(self,x):
        inputs = x
        layer1_out = F.gelu(self.hidden_layer1(inputs))
        layer1_out = self.dropout(layer1_out)
        layer2_out = F.gelu(self.hidden_layer2(layer1_out))
        layer2_out = self.dropout(layer2_out)
        output = torch.relu(self.output_layer(layer2_out))
        return output
```
- 가중치 초기화
가중치 초기화로 Xavier Initialization을 수행한다.
$$W\sim U(-\sqrt{\frac{6}{n_{in}+n_{out}}},\sqrt{\frac{6}{n_{in}+n_{out}}})$$

- inputlayer에서 hiddenlayer1
X가 모형에 입력되면 다음과 같은 아핀변환과 비선형변환인 GELU를 수행하여 hiddenlayer로 값을 반환한다. 
$$h_1=\mathbb{R}^{288} \mapsto \mathbb{R}^{1024}$$
$$H^{(1)}=GELU(XW^{(1)T} + 1b^{(1)T})$$
$$W^{(1)}\in\mathbb{R}^{1024\times288},\;\;\;b^{(1)}\in\mathbb{R}^{1024},\;\;\;\;1=(1,\cdots,1)^T\in\mathbb{R}^{batchsize}$$
$$GELU(x)=xP(X\leq x)=x \Phi(x) \;\;\;\;X\sim N(0,1)$$

- Dropout
Dropout은 다음과 같이 표현할 수 있다
$$D=\begin{pmatrix}
d_{1,1} & \cdots & d_{1,1024} \\
\vdots & \ddots & \vdots \\
d_{1024,1} &  \cdots & d_{1024,1024} \\
\end{pmatrix}, \;\;\;\;d_{ij}\sim Ber(1-p)$$
$$H\prime=\frac{1}{1-p}H \circ D$$

- hiddenlayer1에서 hiddenlayer2
$$h_2=\mathbb{R}^{1024} \mapsto \mathbb{R}^{1024}$$
$$H^{(2)}=GELU(H^{(1)}W^{(2)T} + 1b^{(2)T})$$
$$W^{(2)}\in\mathbb{R}^{1024\times1024},\;\;\;b^{(1)}\in\mathbb{R}^{1024},\;\;\;\;1=(1,\cdots,1)^T\in\mathbb{R}^{batchsize}$$

- hiddenlayer2에서 outputlayer
$$o=\mathbb{R}^{1024} \mapsto \mathbb{R}^{}$$
$$O=RELU(H^{(2)}W^{(3)T} + 1b^{(3)})$$
$$W^{(3)}\in\mathbb{R}^{1\times1024},\;\;\;b^{(3)}\in\mathbb{R},\;\;\;\;1=(1,\cdots,1)^T\in\mathbb{R}^{batchsize}$$
$$O\in\mathbb{R}^{batchsize}$$
$$RELU(x)=max(0,x)$$


### optimizer, scheduler, loss 정의
```
# model과 optimizer 선언
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# log에 0이 들어가지 않게 조정 후 loss 반환
def MSLE_loss(pred, target):
    log_pred = torch.log(pred + 0.1)
    loss = nn.MSELoss()(log_pred, target)
    return loss
```
- loss
$$Loss(\hat y,y)=MSE(\hat y,y)=\frac{1}{n}(\hat y-y)^T(\hat y-y)$$
$$\hat y=\log O,\;\;\;\;\;n=batchsize$$

- optimizer
optimizer는 Gradient Decent 알고리즘에 기반한 Adam을 사용한다.
Gradient는 다음과 같은 미분의 연쇄법칙을 통해 계산한다.
$$\frac{\partial Loss()}{\partial W^{(3)}}=\frac{\partial Loss()}{\partial O}\times\frac{\partial O}{\partial W^{(3)}}$$
$$\frac{\partial Loss()}{\partial W^{(2)}}=\frac{\partial Loss()}{\partial O}\times\frac{\partial O}{\partial H^{(2)}}\times\frac{\partial H^{(2)}}{\partial W^{(2)}}$$
$$\frac{\partial Loss()}{\partial W^{(1)}}=\frac{\partial Loss()}{\partial O}\times\frac{\partial O}{\partial H^{(2)}}\times\frac{\partial H^{(2)}}{\partial H^{(1)}}\times\frac{\partial H^{(1)}}{\partial W^{(1)}}$$
그 후 Gradient로 1차 모멘트와 2차모멘트를 구한다.
$$g_t=\frac{\partial Loss()}{\partial W}$$
$$m_t=\beta_{1}m_{t-1}+(1-\beta_1)g_t$$
$$v_t=\beta_{2}v_{t-1}+(1-\beta_2)g^2_t$$
$$\hat m_t=\frac{m_t}{1-\beta^t_1}$$
$$\hat v_t=\frac{v_t}{1-\beta^t_2}$$
learnigrate를 곱한 만큼 W를 조정한다. 
$$W^+=W-\alpha \frac{\hat m_t}{\sqrt {\hat v_t}+\epsilon }$$
$$\alpha=lr_{epoch}$$

- scheduler
StepLR
$$\ lr_{epoch} =
\begin{cases}
gamma\times lr_{epoch-1},\;if\; epoch \;\% \;stepsize = 0 \\
lr_{epoch-1},\;otherwise
\end{cases}$$

### 모형 학습
```
nb_epochs = 500
best_loss = 10 ** 9 # 매우 큰 값으로 초기값 가정
patience_limit = 3 # 몇 번의 epoch까지 지켜볼지를 결정
patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록

for epoch in range(nb_epochs + 1):
  sum_loss = 0
  model.train()
  for batch_idx, samples in enumerate(dataloader):

    x_train, y_train = samples
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    # prediction 계산
    prediction = model(x_train)

    # loss 계산
    loss = MSLE_loss(prediction, y_train)

    # parameter 조정
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    sum_loss += loss.item()
  
  scheduler.step()

  # 10번 epoch마다 평균 loss 출력
  if epoch % 10 == 0:
    print('Epoch {:4d}/{} Loss: {:.6f}'.format(
        epoch, nb_epochs, sum_loss/len(dataloader)))

  ### Validation loss Check
  model.eval()
  val_loss = 0
  for x_val, y_val in val_dataloader:

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    val_pred = model(x_val)
    loss = MSLE_loss(val_pred, y_val)

    val_loss += loss.item()   
      
  ### early stopping 여부를 체크하는 부분 ###
  if val_loss > best_loss: # loss가 개선되지 않은 경우
      patience_check += 1

      if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
          print("Learning End. Best_Loss:{:6f}".format(best_loss/len(val_dataloader)))
          break

  else: # loss가 개선된 경우
      best_loss = val_loss
      best_model = copy.deepcopy(model)
      patience_check = 0
```
```
#결과
Epoch    0/500 Loss: 25.147983
Epoch   10/500 Loss: 0.117929
Epoch   20/500 Loss: 0.099740
Epoch   30/500 Loss: 0.094323
Epoch   40/500 Loss: 0.090017
Epoch   50/500 Loss: 0.087942
Learning End. Best_Loss:0.064814
```

### 모형 검증
```
# Model Test
test_data = torch.tensor(test_data.values, dtype=torch.float32).to(device)
prediction = best_model(test_data)
print(prediction)

#결과
tensor([[159281.7812],
        [201935.8125],
        [187041.9375],
        ...,
        [214291.2656],
        [131444.7344],
        [184093.3594]], device='cuda:0', grad_fn=<ReluBackward0>)
```
