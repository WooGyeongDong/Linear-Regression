 House Prices Prediction with Deepleaning
 ======================================
### Data Intro
Goal : Predict house prices    
Data size : 1460 / 1459 (train / test)    
Total variable # : 79     
Detail of variable : [kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)      

- Numeric variable : 
36 numeric variable. LotArea, YearBuilt, GarageArea, BsmtFullBath, Fireplaces, OverallCond etc.
- ex) Distribution of LotArea    
![area.png](https://www.dropbox.com/scl/fi/0gcx691ryxyjthkuxukfl/area.png?rlkey=gfxbtf7g12ij5qopmya28ldam&dl=0&raw=1)
- Categorycal variable : 43
43 categorycal variable. Street, RoofStyle, RoofMatl, Heating, MasVnrType etc.
- ex) Distribution of RoofStyle     
![roof.png](https://www.dropbox.com/scl/fi/mlc9ulqkshx9x7jefem3k/roof.png?rlkey=0guvum88eas0wm1kel2o9ht6e&dl=0&raw=1)
- Missing Value : 6965

- Distribution of House Prices(y)
  
![y.png](https://www.dropbox.com/scl/fi/soe7gjg5d952ygnjihoak/y.png?rlkey=3e0a63wl69ybwpivrtzzexw5l&dl=0&raw=1)

- Distribution of Logarithm House Prices(log y) 

![y2.png](https://www.dropbox.com/scl/fi/vlxqq7wjhyh26sknl8wjn/y2.png?rlkey=i5qpe5daja1t0gxs8uf3x7mlc&dl=0&raw=1)

### Model Design
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
- Model : Multi Layer Neural Net
- Non-Linear Activation : GeLU
- Use Xavier Initialization
- Dropout to prevent overfitting


### Hyper Parameter Tuning
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

#Final Hyper Parameter
Model layer = [288,512,1024,1024,512,1], step_size=15, gamma=0.6
```

### Result

Validation Loss

![out.png](https://www.dropbox.com/scl/fi/lwcwvk9jidtn4805d74ub/out.png?rlkey=5fgv0rmvue85ol7el6svy2mps&dl=0&raw=1)

Validation loss after epoch = 10

![output3.png](https://www.dropbox.com/scl/fi/8wzalg7z7my26kcqdoxoe/output3.png?rlkey=gmfst5gxa11niwr1e612r7t07&dl=0&raw=1)

- Evaluation
RMSE = 0.21168
Reference : kaggle
