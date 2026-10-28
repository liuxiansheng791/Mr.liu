import torch
import matplotlib.pyplot as plt
def synthetic_data(w,b,num_examples):
    """生成y=Xw+b+噪声"""
    #生成特征：均值0，方差1的正态分布
    X=torch.normal(0,1,(num_examples,len(w)))
    #线性计算
    y=torch.matmul(X,w)+b
    #增加噪声，标准差0.01
    y+=torch.normal(0,0.01,y.shape)
    #reshape成列向量
    return X,y.reshape((-1,1))
    #真实参数
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)
#打印第一条样本
print('features:',features[0],'\nlabel:',labels[0])
#绘制散点图：第二个特征和label的关系
plt.figure(figsize=(6,4))
#features[:,1]是第二列特征
plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),s=10)
plt.xlabel('Feature1')
plt.ylabel('Label')
plt.show()