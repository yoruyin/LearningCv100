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

$$
y= \begin{cases} 0& (\text{if}\quad y < 128) \ 255& (\text{else}) \end{cases}
$$



