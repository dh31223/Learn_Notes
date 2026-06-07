# 数学笔记

跟着3周数学补充计划，每天学完后的核心理解。

## 第一周：线性代数 + 矩阵求导

### 第1天 | 特征值与特征向量

- 特征向量就是坐标轴中经过矩阵A变换之后，方向不会改变的向量。
- 特征值就是矩阵A在对应的特征向量中，拉伸的程度。

### 第二天 | 特征分解与正定矩阵

- 正定矩阵A对所有的非零向量x，都有 x^T A x > 0（二次型大于0，不是Ax>0，Ax是向量没法比大小）。
- 所有特征值 > 0 就是正定矩阵。
- Hessian正定就代表该点是极小值。

### 第三天 | 奇异值分解与SVD

- SVD公式：A = U∑V^t。
- ∑是个对角矩阵，对角线上的元素就是奇异值，它反映了矩阵A在各个维度的拉伸程度。
- 几何直觉：任意矩阵 = 旋转 + 拉伸 + 旋转

### 第四天 | SVD与伪逆矩阵

- SVD能做矩阵压缩，也就是特征降维。（取前K个最大的奇异值）
- 当矩阵不可逆的时候，我们可以对奇异值求倒数，来构建伪逆矩阵
- 伪逆矩阵可以近似的求矩阵A的逆过程。（矩阵对应的线性方程有解时）
- 伪逆矩阵还可以求L2范数最小的矩阵A的逆过程。（矩阵对应的线性方程无解）
- SVD可用于PCA降维

### 第五天 | 矩阵范数

- L1范数就是所有元素绝对值之和
- L2范数就是所有元素平方和的平方根
- 谱范数就是最大的奇异值
- L1的图像因为是尖锐的，所以有很大概率出现不重要的维度的权重被降为0，所以导致稀疏性

### 第六天 | 矩阵求导入门

- 梯度本身就是一个向量，因为在机器学习和深度学习中，权重是向量，标量对向量求导得到的梯度也是向量
- 雅可比矩阵：向量对向量求导得到的是矩阵

|求导结果|x是标量|x是向量|
|-|------|------|
|**y是标量**|标量|向量|
|**y是向量**|向量|矩阵|


$$
\frac{\partial Y}{\partial X} = \begin{bmatrix}
\frac{\partial y_1}{\partial X} \\ 
\frac{\partial y_2}{\partial X} \\ 
\vdots \\ 
\frac{\partial y_m}{\partial X}
\end{bmatrix} = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_n} \\ 
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_2}{\partial x_n} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{bmatrix}
$$

$$
\frac{\partial (X^TAX)}{\partial X} = (A + A^T)X \quad \text{（若A对称则}=2AX\text{）}
\\
其中a是向量，A是矩阵
\\
\frac{\partial (a^tX)}{\partial X} = a
\\
\frac{\partial (AX)}{\partial X} = A
$$

<img src="img/矩阵求导1.png" alt="图片" style="zoom:30%;" />

<img src="img/矩阵求导2.png" alt="图片" style="zoom:30%;" />



### 第七天 | 休息

## 第二周：矩阵求导续 + 概率论升级

### 第八天 | 黑塞矩阵 + 链式法则

#### 8.1 对于一个多元标量函数 f(x1,x2,…,xn)，它的 **黑塞矩阵** 是一个 n×n 的方阵，第i行第j列的元素是：
$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

例如：
$$
f(x, y) = x^2 + xy + y^2
$$

第一步：求一阶偏导（梯度）
$$
\frac{\partial f}{\partial x} = 2x + y, \quad \frac{\partial f}{\partial y} = x + 2y
$$

第二步：求二阶偏导
$$
\frac{\partial^2 f}{\partial x^2} = 2, \quad \frac{\partial^2 f}{\partial y^2} = 2, \quad \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}(x + 2y) = 1, \quad \frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}(2x + y) = 1
$$

第三步：写成矩阵
$$
H = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

这是一个常数矩阵，它的值不依赖x和y，它的特征值都为正，是**正定矩阵**。

#### 8.2 几何意义：

- 对于一元函数$f(x)$，二阶导数$f''(x)$就表示曲线在该点的弯曲方向和弯曲程度，如果是二次函数，则处处弯曲方向和弯曲程度都相等，因为$f''(x)$是常数。
- 对于二元函数$z = f(x, y)$，黑塞矩阵描述了曲面在某个点附近的**局部弯曲形状**：
	+ 它决定了沿**切平面内的某个方向**的曲率
	+ 比如，上面的例子$f(x, y) = x^2 + xy + y^2$是个抛物面，在所有方向上都向上弯曲，也就对应了黑塞矩阵式正定矩阵，黑塞矩阵正是反应弯曲的矩阵。
>**直观类比：**梯度告诉你“该往哪个山坡走最陡”，黑塞告诉你“这个山坡是缓坡还是陡坡，以及坡度的变化趋势”。




### 第9天 常见矩阵求导公式 + 泰勒展开

>补充： ML里对称矩阵到处都是——Hessian 矩阵（二阶偏导的对称性）、协方差矩阵、Gram
矩阵，它们天然对称。所以你遇到的 xᵀAx 求导，十有八九 A 就是对称的，公式直接简化为
2Ax。这就是为什么很多教材直接写 2Ax 而不写 (A+Aᵀ)x。

>补充第二个原因：
原因1（你说了）：计算量——求 n×n 矩阵的逆是 O(n³)
原因2（你没说）：存储——100万参数的模型，Hessian 是 100万×100万 = 1万亿个元素，单精度浮点也要 4TB
显存，根本塞不进 GPU
原因3（bonus）：深度学习里 Hessian
还是病态的（ill-conditioned），特征值分布极不均匀，数值上求逆也不稳定
更精确的表述： Newton 法每步解 x_{new} = x -
 H⁻¹∇f（直接用二阶泰勒展开求极小值的解析解），梯度下降每步 x_{new} = x -
 η∇f（只沿负梯度方向走固定步长）。前者一步到位但贵，后者便宜但需要多走几步。


### 第10天 极大似然估计 MLE（上）



1. 似然和概率的区别
- 概率：已知参数（比如硬币正面概率 $ p = 0.5$），问“出现某个结果（7正3反）的可能性多大？”
- 似然：已知结果（7正3反），问“参数$p$等于多少时，这个结果出现的可能性最大？”
>通俗的讲：**“什么样的参数值，最有可能产生我手头这些数据？”**
>我们就是找一组参数 θ，θ使得当前观测到的数据出现的概率最大。
>也就是机器学习中参数拟合的过程

例子：
问题：抛硬币10次，7次正面，3次反面。假设每次正面概率为$p$，用MLE估计$p$。

**写出第一步：写出似然函数**
$$
P(7\text{正}3\text{反} \mid p) = p^7 \cdot (1-p)^3
$$

**写出第二步：取对数**
$$
\ln L(p) = 7\ln p + 3\ln(1-p)
$$

**写出第三步：对$p$求导并令导数为0**
$$
\frac{d}{dp}\ln L = \frac{7}{p} - \frac{3}{1-p} = 0
$$

解方程：
$$
\frac{7}{p} = \frac{3}{1-p} \quad \Rightarrow \quad 7(1-p) = 3p \quad \Rightarrow \quad 7-7p = 3p \quad \Rightarrow \quad 7 = 10p \quad \Rightarrow \quad p = 0.7
$$


2. LME三步法
- 写出似然函数：$L(\theta) = \prod_{i} P(x_i \mid \theta)$
- 取对数：$\ell(\theta) = \ln L(\theta)$
- 对$\theta$求导，令导数为0，解出$\theta$

3. 为什么逻辑回归用交叉熵损失，背后也是MLE。

>你让损失变小，就是让“模型产生真实标签的概率”变大。
这就是极大似然估计的核心思想：找到参数，让已经发生的事（真实标签）看起来尽可能合理。
>而让损失变小，本质也就是找一组参数，在该特征下预测正确的概率最大，也就是找一组参数让这个特征对应的标签合理。

### 第11天 MLE（下）+ 最大后验 MAP

数学知识补充：

数学上：

- `min` 表示“最小值”。比如 `min(2, 5, 1) = 1`，它返回**最小**的数值。
- `arg min` 表示“使后面式子取最小值时，对应的自变量值”。`arg` 是 argument（自变量）的缩写。

> 简单记忆：
> `min f(x)` 返回一个**数**（最小值）
> `arg min f(x)` 返回一个**x值**（在哪个x处取到最小值）

**例子**：
函数 $f(x) = (x - 3)^2$

- `min f(x) = 0`
- `arg min f(x) = 3`，因为当 $x = 3$ 时 f(x) 最小。




**用 MLE 推导线性回归的均方误差损失**

**问题**：数据集 $\{(x_i, y_i)\}$，假设 $y_i = w^T x_i + \epsilon_i$，其中 $\epsilon_i \sim N(0, \sigma^2)$ 独立同分布。

**思路**：MLE 要找到参数 $w$，使得观察到这些 $y_i$ 的概率最大。

1. 第1步：写出似然函数

每个 $y_i$ 服从正态分布：均值 $\mu_i = w^T x_i$，方差 $\sigma^2$。

$$
P(y_i \mid x_i, w) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - w^T x_i)^2}{2\sigma^2}\right)
$$

所有样本独立：

$$
L(w) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - w^T x_i)^2}{2\sigma^2}\right)
$$

2. 第2步：取对数

$$
\ln L(w) = -\frac{n}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - w^T x_i)^2
$$

3. 第3步：最大化 $\ln L$ 等价于最小化 $\sum (y_i - w^T x_i)^2$

因为前面两项与 $w$ 无关（$\sigma$ 固定时）。所以：

$$
\hat{w}_{\text{MLE}} = \arg \min_{w} \sum_{i=1}^{n} (y_i - w^T x_i)^2
$$

> **结论**：在误差服从高斯分布的假设下，**MLE 等价于最小化均方误差（MSE）**。这就是线性回归为什么用 MSE 损失的理论根源。
