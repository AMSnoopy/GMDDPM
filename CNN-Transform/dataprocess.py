from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
def preprocess_data():
    # 加载和预处理数据
 
  
    # 读取CSV文件
    train = pd.read_csv('..//data//train_data.csv')

    test = pd.read_csv('..//data//test_data.csv')
    train = train.sample(frac=1).reset_index(drop=True)
    X_train=train.iloc[:,:-1]
    y_train=train.iloc[:,-1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return train_loader, test_loader