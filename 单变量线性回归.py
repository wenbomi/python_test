import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 文件路径
path = "lab42data.txt"

#读取文件
data = pd.read_csv(path,names=['Population','Profit'])
print(data)

#构造数据集
data.insert(0,'ones',1)
print(data)

#标签
X = data.iloc[:,0:-1] #前两列为标签
X = X.values
# print(X)

#真实值
y = data.iloc[:,-1]
y = y.values.reshape(97,1)

#定义损失函数
def costfunction(X,y,theta):
    inner = np.power(X@theta - y ,2)
    return np.sum(inner/(2 * len(X)))

#theta初始化
theta = np.zeros((2,1))

