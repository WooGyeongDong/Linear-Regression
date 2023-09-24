 House Prices Prediction with Deepleaning
 ======================================
### Data 개요
- 분석할 자료는 주택의 특성과 가격을 정리한 자료이다. 주택의 특성으로 주택의 가격을 예측하는 것을 목표로 한다.
자료 수 : 1460 / 1459 (train / test)
총 변수 수 : 79
변수에 대한 상세 설명 : [kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

- 수치형 변수 : 
주택 면적(LotArea), 주택 건축 연도(YearBuilt), 차고 크기(GarageArea), 화장실 수(BsmtFullBath), 벽난로 수(Fireplaces), 주택의 상태점수(OverallCond) 등 총 36개의 수치형 변수가 존재한다.
예) 주택 면적의 분포
![area.png](https://www.dropbox.com/scl/fi/0gcx691ryxyjthkuxukfl/area.png?rlkey=gfxbtf7g12ij5qopmya28ldam&dl=0&raw=1)
- 범주형 변수 : 43
건물 위치(Street), 지붕 양식(RoofStyle), 지붕 재료(RoofMatl), 난방 방식(Heating), 벽돌 베리어 양식(MasVnrType) 등 총 43개의 범주형 변수가 존재한다.
예) 지붕 양식의 분포
![roof.png](https://www.dropbox.com/scl/fi/mlc9ulqkshx9x7jefem3k/roof.png?rlkey=0guvum88eas0wm1kel2o9ht6e&dl=0&raw=1)
- 결측치
총 6965개의 결측치가 관찰되었다. 주로 담장상태(Fence), 수영장 상태(PoolQC) 골목 포장상태(Alley)에서 결측치가 발생하였다.

- 주택 가격(y) 분포
![y.png](https://www.dropbox.com/scl/fi/soe7gjg5d952ygnjihoak/y.png?rlkey=3e0a63wl69ybwpivrtzzexw5l&dl=0&raw=1)
log 변환한 주택가격(y) 분포
![y2.png](https://www.dropbox.com/scl/fi/vlxqq7wjhyh26sknl8wjn/y2.png?rlkey=i5qpe5daja1t0gxs8uf3x7mlc&dl=0&raw=1)


### Data 불러오기
```
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

# train data와 test data 분리
train_data = all_data.iloc[:1460,:]
test_data = all_data.iloc[1460:,:]
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

### K Fold Cross Validation
```
# Cross Validation
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True)
```

### Data를 DataSet으로 변환
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
		
dataset = CustomDataset(X_train, y_train)
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
    def __init__(self, layer_value):
        super(Net,self).__init__()
        self.layer = nn.ModuleList()
        for i, value in enumerate(layer_value[:-1]) :
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


### loss 정의
```
# model layer 설정
layer_value = [288,512,1024,1024,512,1]

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
\gamma\times lr_{epoch-1},\;if\; epoch \;\% \;stepsize = 0 \\
lr_{epoch-1},\;otherwise
\end{cases}$$

### 모형 학습
```
validation_loss=[]
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
  # Dataloader에 Data 배분
  train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx) # index 생성
  val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx) # index 생성

  dataloader = DataLoader(dataset, batch_size=128, sampler=train_idx)
  val_dataloader = DataLoader(dataset, batch_size=128, sampler=val_idx)
  
  # 모형 초기화
  model = Net(layer_value).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.6)
  
  nb_epochs = 500
  best_loss = 10 ** 9 # 매우 큰 값으로 초기값 가정
  patience_limit = 3 # 몇 번의 epoch까지 지켜볼지를 결정
  patience_check = 0 # 현재 몇 epoch 연속으로 loss 개선이 안되는지를 기록
  val = []
  
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

    ### Validation loss Check
    model.eval()
    val_loss = 0
    for x_val, y_val in val_dataloader:

      x_val = x_val.to(device)
      y_val = y_val.to(device)

      val_pred = model(x_val)
      loss = MSLE_loss(val_pred, y_val)

      val_loss += loss.item()
    val.append(val_loss)
    ### early stopping 여부를 체크하는 부분 ###
    if abs(val_loss - best_loss) < 1e-3: # loss가 개선되지 않은 경우
        patience_check += 1

        if patience_check >= patience_limit: # early stopping 조건 만족 시 조기 종료
            print("K-fold {} Train_Loss {:6f} Valid_Loss:{:6f}".format(fold, sum_loss/len(dataloader),best_loss/len(val_dataloader)))
            validation_loss.append(best_loss/len(val_dataloader))
            break

    else: # loss가 개선된 경우
        best_loss = val_loss
        best_model = copy.deepcopy(model)
        patience_check = 0
```
### Hyper Parameter 조정
```
#1
Model layer = [288,1024,1024,1], step_size=10, gamma=0.5
Validation Score: 0.0658, ± 0.0069
#2
Model layer = [288,512,1024,1024,512,1], step_size=10, gamma=0.5
Validation Score: 0.0468, ± 0.0090
#3
Model layer = [288,512,1024,1024,512,1], step_size=15, gamma=0.5
Validation Score: 0.0424, ± 0.0029
#4
Model layer = [288,512,1024,1024,512,1], step_size=15, gamma=0.6
Validation Score: 0.0411, ± 0.0032

#최종 모형의 Hyper Parameter
Model layer = [288,512,1024,1024,512,1], step_size=15, gamma=0.6
```
- validation loss 시각화
![out.png](https://www.dropbox.com/scl/fi/lwcwvk9jidtn4805d74ub/out.png?rlkey=5fgv0rmvue85ol7el6svy2mps&dl=0&raw=1)
epoch = 10부터 validation loss
![output3.png](https://www.dropbox.com/scl/fi/8wzalg7z7my26kcqdoxoe/output3.png?rlkey=gmfst5gxa11niwr1e612r7t07&dl=0&raw=1)
### 모형 검증
```
# Model Test
test_data = torch.tensor(test_data.values, dtype=torch.float32).to(device)
prediction = best_model(test_data)
print(prediction)

#결과
tensor([[137011.6094],
        [182386.9531],
        [178240.7969],
        ...,
        [175137.2344],
        [ 88928.8750],
        [229283.5000]], grad_fn=<ReluBackward0>)
```

- 평가지표
RMSE = 0.21168
출처 : kaggle

### $$R^2$$를 평가지표로 사용할 수 없는 이유

$$R^2=\frac{SS_{reg}}{SS_{total}}$$
$$SS_{total}=\sum ( y_i -\bar y)^2$$로 데이터가 주어지면 불변
$$\therefore \underset{arg}{max}R^2\Leftrightarrow \underset{arg}{max}SS_{reg}=\sum (\hat y_i -\bar y)^2$$
$$\therefore \hat y_i$$이 평균에서 멀어지게 학습된다.

$$SS_{total}=\sum ( y_i -\bar y)^2=\sum ( y_i-\hat y_i +\hat y_i -\bar y)^2=\sum (\hat y_i -\bar y)^2+\sum (y_i -\hat y_i)^2-2\sum (y_i -\hat y_i)(\hat y_i -\bar y)$$
최소제곱추정치에서는 $$\sum (y_i -\hat y_i)=\sum (y_i -\hat y_i)\hat y_i=0$$이지만 딥러닝 학습에서는 이를 만족하지 않기 때문에 $$R^2$$와 오차가 모두 커지게 학습될 수 있다.

### 고전적 회귀분석과 비교
- 고전적 회귀분석의 목적함수
$$\hat y=X\hat\beta$$
$$Loss(\hat\beta)=(y-\hat y)^T(y-\hat y)=\hat\beta^TX^TX\hat\beta-2\hat\beta^TX^Ty+y^Ty$$

- 목적함수의 미분
$$\frac{\partial Loss(\hat\beta)}{\partial \hat\beta}=2X^TX\hat\beta-2X^Ty=0$$
$$\hat\beta=(X^TX)^{-1}X^Ty$$
목적함수의 미분값이 $$0$$이 되는 $$\hat\beta$$를 구할 수 있다.

- 딥러닝 회귀분석의 목적함수
$$\hat y=f(f(XW^{(1)} + 1b^{(1)})W^{(2)}+1b^{(2)}),\;\;\;\;f(x):elemental-wise\;nonlinear\;activation$$
$$Loss(\hat y,y)=MSE(\hat y,y)=\frac{1}{n}(\hat y-y)^T(\hat y-y)$$

- 목적함수의 미분
$$\frac{\partial Loss()}{\partial w^{(1)}_{11}}=(\hat y-y)f\prime(f(x_{\cdot 1}w_{11}+1b^{(1)}_1)W^{(2)}_{1\cdot}+1b^{(2)})f(x_{1 \cdot}w_{11}+1b^{(1)}_1)W^{(2)}_{1\cdot}f\prime(x_{1 \cdot}w_{11}+1b^{(1)}_1)x_{1 \cdot}=0$$
$$\vdots$$
$$\frac{\partial Loss()}{\partial w^{(2)}_{11}}=(\hat y-y)f\prime(f(XW^{(1)}_{\cdot1} + 1b^{(1)})w^{(2)}_{11}+1b^{(2)T})f(XW^{(1)T}_{\cdot1} + 1b^{(1)T})w^{(2)}_{11}=0$$
$$\vdots$$
$$\frac{\partial Loss()}{\partial b^{(1)}_{1}}=(\hat y-y)f\prime(f(XW^{(1)}_{\cdot 1} + 1b^{(1)}_1)W^{(2)}_{1\cdot}+1b^{(2)})f(XW^{(1)}_{\cdot1} + 1b^{(1)}_1)W^{(2)}_{1\cdot}f\prime(XW^{(1)}_{\cdot1} + 1b^{(1)}_1)=0$$
$$\vdots$$
$$\frac{\partial Loss()}{\partial b^{(2)}_{1}}=(\hat y-y)f\prime(f(XW^{(1)} + 1b^{(1)})W^{(2)}+1b^{(2)})=0$$
위를 모두 만족하는 연립방정식의 해, $$W^{(1)},W^{(2)},b^{(1)},b^{(2)}$$ 를 구하는 것은 상당히 힘든 일이므로 Gradient Decent 알고리즘을 사용하여 추정한다.