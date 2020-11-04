#### 环境

```python
python == 3.7.9 (anaconda)
numpy == 1.19.1
matplotlib == 3.3.1
opencv-python == 4.4.0
```

#### 问题

##### Q1

`cv2.imread()`按照$BGR$的顺序排列的

##### Q2 灰度化(Grayscale)

灰度表示为：$Y=0.2126R+0.7152G+0.0722B$

##### Q3 二值化(Thresholding)

将灰度的阈值设置为128进行二值化
$$
y= \left\{
\begin{aligned} 
&0   &(\text{if}\quad y < 128) \\ 
&255 &(\text{else}) 
\end{aligned} 
\right.
$$

##### Q4 大津二值化算法(Otsu's Method)

最大类间方差法，自动确定二值化中阈值的算法

##### Q5 HSV变换

HSV即色相(Hue)，饱和度(Saturation)，明度(Value)来表示色彩的一种方式。

色相：将颜色使用$0^{\circ}$到$360^{\circ}$表示

饱和度：色彩的纯度，饱和度越低颜色越黯淡($0\leq S<1$)

明度：色彩的明暗程度。数值越高越接近白色，数值越低越接近黑色($0\leq V<1$)

从$\text{RGB}$色彩表示转换到$\text{HSV}$色彩表示通过以下方式计算：

$\text{RGB}$的取值范围为$[0, 1]$，令：
$$
\text{Max}=\max(R,G,B)\
$$

$$
\text{Min}=\min(R,G,B)
$$

色相： 
$$
H=\left\{\begin{aligned} &0&(\text{if}\ \text{Min}=\text{Max})\\&60\times \frac{G-R}{\text{Max}-\text{Min}}+60&(\text{if}\ \text{Min}=B)\\ &60\times \frac{B-G}{\text{Max}-\text{Min}}+180&(\text{if}\ \text{Min}=R)\\ &60\times \frac{R-B}{\text{Max}-\text{Min}}+300&(\text{if}\ \text{Min}=G) \end{aligned}
\right.
$$
 饱和度：
$$
S=\text{Max}-\text{Min}
$$
明度： 
$$
V=\text{Max}
$$
从$\text{HSV}$色彩表示转换到$\text{RGB}$色彩表示通过以下方式计算:
$$
C = S
$$

$$
\ H' = \frac{H}{60}
$$

$$
X = C\ (1 - |H' \mod 2 - 1|)
$$

$$
\ (R,G,B)=(V-C)\times (1,1,1)+\left\{
\begin{aligned}&(0, 0, 0) &(\text{if H is undefined})\\ &(C, X, 0) &(\text{if}\quad 0 \leq H' < 1)\\ &(X, C, 0) &(\text{if}\quad 1 \leq H' < 2)\\ &(0, C, X) &(\text{if}\quad 2 \leq H' < 3)\\ &(0, X, C) &(\text{if}\quad 3 \leq H' < 4)\\ &(X, 0, C) &(\text{if}\quad 4 \leq H' < 5)\\ &(C, 0, X) &(\text{if}\quad 5 \leq H' < 6) \end{aligned}\right.
$$

##### Q6 减色处理

将图像的值从$256^{3}$压缩到$4^{3}$，将RGB的值只取$\{32,96,160,224\}$,这被称作色彩量化。

##### Q7 平均池化(Average Pooling)

将图片按照固定大小网格分割，网格内的像素值取网格内所有像素的平均值。

我们将这种把图片使用均等大小网格分割，并求网格内代表值的操作称为**池化（Pooling）**。

池化操作是**卷积神经网络（Convolutional Neural Network）**中重要的图像处理方式。平均池化按照下式定义：
$$
v=\frac{1}{|R|}\ \sum\limits_{i=1}^R\ v_i
$$

##### Q8 最大池化(Max Pooling)

网格内的值不取平均值，而是取最大值进行池化操作

##### Q9 高斯滤波(Gaussian Filter)

高斯滤波器是一种可以使图像**平滑**的滤波器，用于去除**噪声**。

高斯滤波器将中心像素周围的像素按照高斯分布加权平均进行平滑化。这样的（二维）权值通常被称为**卷积核（kernel）**或者**滤波器（filter）**。

由于图像的长宽可能不是滤波器大小的整数倍，因此我们需要在图像的边缘补$0$。这种方法称作**Zero Padding**。并且权值$g$（卷积核）要进行[归一化操作](https://blog.csdn.net/lz0499/article/details/54015150)（$\sum\ g = 1$）。

按下面的高斯分布公式计算权值： $$ g(x,y,\sigma)=\frac{1}{2\ \pi\ \sigma^2}\ e^{-\frac{x^2+y^2}{2\ \sigma^2}} $$

标准差$\sigma=1.3$的 $8-$近邻高斯滤波器如下： $$ K=\frac{1}{16}\cdot \left[ \begin{matrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{matrix} \right] $$

##### Q10 中值滤波(Median Filter)

直接取 $8-$临近的中值输出，可以用来降噪。

##### Q11 均值滤波

直接取 $8-$临近的均值输出。

##### Q12 Motion Filter

使用$3\times3$的Motion Filter进行滤波。

Motion Filter取对角线方向的像素的平均值，像如下矩阵：
$$
K=\left[ \begin{matrix} \frac{1}{3}&0&0\\ 0&\frac{1}{3}&0\\ 0 & 0& \frac{1}{3} \end{matrix} \right]
$$

##### Q13 Max-Min 滤波器

使用网格内的最大值和最小值的差值对网格内像素重新赋值，用来**边缘检测**。

边缘检测用于检测图像中的线，这种操作称为**特征提取**。

边缘检测通常在**灰度图像**上进行。

##### Q14 差分滤波器(Differential Filter)

纵向： 
$$
K=\left[ \begin{matrix} 0&-1&0\\ 0&1&0\\ 0&0&0 \end{matrix} \right]
$$
横向： 
$$
K=\left[ \begin{matrix} 0&0&0\\ -1&1&0\\ 0&0&0 \end{matrix} \right]
$$

##### Q15 Sobel滤波器

Sobel滤波器可以提取特定方向（纵向或横向）的边缘，滤波器按下式定义：

纵向： 
$$
K=\left[ \begin{matrix} 1&2&1\\ 0&0&0\\ -1&-2&-1 \end{matrix} \right]
$$


横向： 
$$
K=\left[ \begin{matrix} 1&0&-1\\ 2&0&-2\\ 1&0&-1 \end{matrix} \right]
$$


##### Q16 Prewitt滤波器

Prewitt滤波器是用于边缘检测的一种滤波器，使用下式定义：

纵向： 
$$
K=\left[ \begin{matrix} -1&-1&-1\\ 0&0&0\\ 1&1&1 \end{matrix} \right]
$$
横向： 
$$
K=\left[ \begin{matrix} -1&0&1\\ -1&0&1\\ -1&0&1 \end{matrix} \right]
$$

##### Q17 Laplacian滤波器

Laplacian滤波器是对图像亮度进行二次微分从而**检测边缘**的滤波器。

卷积核是下面这样的： 
$$
K= \left[ \begin{matrix} 0&1&0\\ 1&-4&1\\ 0&1&0 \end{matrix} \right]
$$


##### Q18 Emboss滤波器

Emboss滤波器可以使物体轮廓更加清晰，按照以下式子定义： 
$$
K= \left[ \begin{matrix} -2&-1&0\\ -1&1&1\\ 0&1&2 \end{matrix} \right]
$$


##### Q19 LoG滤波器

LoG即高斯-拉普拉斯（Laplacian of Gaussian）的缩写，使用高斯滤波器使图像平滑化之后再使用拉普拉斯滤波器使图像的轮廓更加清晰。用于**边缘检测**。

LoG 滤波器使用以下式子定义： 
$$
\text{LoG}(x,y)=\frac{x^2 + y^2 - s^2}{2 \ \pi \ s^6} \ e^{-\frac{x^2+y^2}{2\ s^2}}
$$

##### Q41-43 Canny边缘检测

Canny边缘检测法的理论介绍。

1. 使用高斯滤波；
2. 在 $x$ 方向和 $y$ 方向上使用Sobel滤波器，在此之上求出边缘的强度和边缘的梯度；
3. 对梯度幅值进行非极大值抑制（Non-maximum suppression）来使边缘变得更细；
4. 使用滞后阈值来对阈值进行处理。



按照以下步骤进行处理：

1. 将图像进行灰度化处理；
2. 将图像进行高斯滤波（$5\times5$，$s=1.4$）；
3. 在$x$方向和$y$方向上使用Sobel滤波器，在此之上求出边缘梯度$f_x$和$f_y$。边缘梯度可以按照下式求得： $$ \text{edge}=\sqrt{{f_x}^2+{f_x}^2}\ \text{tan}=\arctan(\frac{f_y}{f_x}) $$
4. 使用下面的公式将梯度方向量化： $$ \text{angle}=\left\{\begin{aligned} &0  (\text{if}\quad -0.4142 < \tan \leq 0.4142)\\ &45  (\text{if}\quad 0.4142 < \tan < 2.4142)\\ &90 (\text{if}\quad |\tan| \geq 2.4142)\\ &135 (\text{if}\quad -2.4142 < \tan \leq -0.4142) \end{aligned}\right. $$
5. 我们比较我们我们所关注的地方梯度的法线方向邻接的三个像素点的梯度幅值，如果该点的梯度值不比其它两个像素大，那么这个地方的值设置为0。
6. 在这里我们将通过设置高阈值（HT：high threshold）和低阈值（LT：low threshold）来将梯度幅值二值化。



效果：

<img src="D:\tf2\github\LearningCv100\assert\Q43.jpg" style="zoom:25%;" />

<img src="D:\tf2\github\LearningCv100\assert\Q43result.jpg" style="zoom:25%;" />