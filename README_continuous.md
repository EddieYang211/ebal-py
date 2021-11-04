### Entropy Balancing for Continuous Treatment

#### 简介
截面数据中离线因果推断的方法。针对多取值（multi-valued）策略。通过交叉熵和加权的方法消除策略和特征的相关性，同时最大化保证权重均匀分布。

参考文献：https://web.stanford.edu/~jhain/Paper/PA2012.pdf
R包：https://github.com/cran/ebal

#### 输入
* T: vector, 策略变量(e.g. 每个用户红点下发数)，必须提供
* X: matrix, 用户特征(每个特征用列表示)，必须提供
<!-- optional -->
* base_weight: 理想权重，默认每个用户为1，optional

#### 初始化超参（可以修改）
* coefs：拉格朗日乘子初始值（默认全部为零），不熟悉代码不建议修改
* max_iterations：算法最大迭代次数，默认500
* constraint_tolerance: 判断优化已经收敛的阀值。代表可接受的控制组和实验组变量的最大差异，默认0.0001
* print_level：是否打印运行细节。可设为-1减少打印细节。默认0
* lr: 步长，默认为1. 取值越大收敛越快，但容易出现exploding gradient
* PCA：是否对特征矩阵做PCA分解。主要解决特征共线性问题。默认True


#### 使用例子

```
import pandas as pd

df = pd.read_csv('~/data.csv') # load data

Y = df.Y.values
X = df.iloc[:,0:10].copy()                        #特征
T = df.T                              #策略， e.g.红点下发个数

e = ebal_con()
out = e.ebalance(T, X)

# 检验策略与特征相关性
e.check_balance(X, T, out['w'])
```

#### 输出
* converged: boolean, 算法有没有收敛
* maxdiff: 约束最大偏差
* w: vector, 每一个用户的权重。在下游拟合曲线的时候把这个加进去

#### 注意
* continuous版本输出的权重：sum(权重)=1；视情况自行调整
* 当PCA=False的时候，特征矩阵X需要full rank，不能出现共线的特征，会报错

#### 常见错误
* loss/权重出现nan：一般是因为步长过大导致gradient爆炸了。适当调小lr或者修改初始coefs
* "collinearity in covariate matrix for controls (remove collinear covariates)"：特征矩阵rank deficient，需要去掉共线的特征或者设置PCA为True
* 跑完500 iteration没有收敛：适当修改初始coefs
* PCA过后仍然报错collinearity：适当修改代码中PCA过后选取特征的条件