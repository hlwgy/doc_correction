opencv基础-想为女秘书减压：文档矫正

# 一、缘起
> 公司的美女秘书知道我是做图像识别的程序员，她专门找到我。她说，各行业我都接触过，唯独没有接触过你们IT界的男人，想问一下，你们是直接做吗？我有点懵，不知所措……她接着说，就是做之前是否需要有准备工作？我说有的。她很开心，太好了，那就说是，识别图像之前，肯定会对图像做一些预处理喽。正好我手里有一些文件扫描件，但是不规范，你能帮忙做一做吗？

女秘书给我这样一张图，说这个机密文件的空白区域太大了，她想只要文字区域，要我用程序标示出来。

![图片.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3607648c89e54485a3b0d7d99d6d0727~tplv-k3u1fbpfcp-watermark.image?)

# 二、boundingRect 边界矩形

这个需求太简单了。

我首先想到的就是`vc2`里面的`boundingRect`方法，就是框矩形区域的。

通过灰度、反色、二值化，处理一下，最后交给boundingRect识别，我很快就做出来了，效果如下：

![img1.gif](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/3b5e991a92a0408c9228d718f5ee9c09~tplv-k3u1fbpfcp-watermark.image?)

代码如下：

```python
def boundingRect(image_path):

    # 读入图片3通道 [[[255,255,255],[255,255,255]],[[255,255,255],[255,255,255]]]
    image = cv2.imread(image_path)
    # 转为灰度单通道 [[255 255],[255 255]]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 黑白颠倒
    gray = cv2.bitwise_not(gray)
    # 二值化
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # 获取最小包裹正矩形 x-x轴位置, y-y轴位置, w-宽度, h-高度
    x, y, w, h = cv2.boundingRect(thresh)
    left, top, right,  bottom = x, y, x+w, y+h

    # 把框画在图上
    cv2.rectangle(image,(left, top), (right, bottom), (0, 0, 255), 2)

    # 将处理好的文件保存到当前目录
    #cv2.imwrite('img2_1_rotate.jpg', image)
    # 将处理好的文件弹窗展示
    # cv2.imshow("output", image)
    # cv2.waitKey(0)

boundingRect('img1_0_origin.jpg')
```

***我想鼓起勇气去找女秘书，打算告诉她我的实现思路。由于害怕磕磕巴巴讲不清楚，我就先私下打好草稿。***

首先调用`cv2.imread`读入图片，这时候读取的是`3`通道的原图，读完了之后，如果调用`cv2.imshow("output", image)`展示一下，就是彩色原图。图形的数据是这样的`[[[255,255,255],[255,255,255]],[[255,255,255],[255,255,255]]]`，每个像素点有`RGB`三个值。

对于识别文本边界的话，这个数据还是有点复杂。因为不用关心它是红的还是绿的，只需要关心有没有图像就行。因此，需要调用`cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`对图形进行灰度处理，处理后的图形数据变简单了，变为单通道的像素集合`[[255 255],[255 255]]`。

![img1_1_gray.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d164f5d5ed9d44a3b26006b2c9b41e7d~tplv-k3u1fbpfcp-watermark.image?)

此时，白色区域的数值接近`255`，黑色区域接近`0`。我们更关注于黑色区域的字，它的值居然是被忽视的`0`，这不可以。计算机一般对于0是忽略的，对于`255`是要关注的（这就是为什么很多文字识别的训练集都是黑底白字）。所以，需要来一个黑白反转。

![img1_2_bitwise.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/282700a8726d41e2bc5486c3b0fe34e2~tplv-k3u1fbpfcp-watermark.image?)

黑白反转完了，其实它的算法很简单，用`255-`就可以。`255-255=0,255-0=255`。这样黑的就变为白的，白的变为黑的。但是，灰度图是`0~255`之间的数字，会存在`127、128`这种不黑不白的像素。也会存在一些`5、6、7`这类毛边或者阴影，说它是字吧，还看不清，说不是吧，隐隐约约还有。人生就要果断的断舍离，程序更要如此。要么是字，要么是空白，需要二分法。

![img1_3_thresh.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5631b89114304dadafbc11791e501943~tplv-k3u1fbpfcp-watermark.image?)

现在，图片只存在`0`或者`255`了。把它交给`boundingRect`它可以返回我们需要的数值。

![img1_4_rect.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2f9db83f3f4f497684ab30da7cefdfc3~tplv-k3u1fbpfcp-watermark.image?)


我把程序交付给了女秘书，我最终还是啥也没说，她很忙，说了声谢谢，她说试试看。


# 二、minAreaRect 最小面积矩形

女秘书叫我去一趟。我特意去了一趟厕所。

我想象着该如何回应她的感谢，我要面带微笑，我说客气啥，都是应该的。不行，我得表现的傲娇一些，这都是小事，分分钟搞定，以后有这种事记得找我。这样是不是不够友好……

我还是见到了她，她说程序好像有点问题，识别出的小卡片不是她想要的，我看了看。

![img1-img2.gif](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b8bbcd8d5f7b4df9a201585ac56dcb03~tplv-k3u1fbpfcp-watermark.image?)

原来是这个机密文件扫描斜了，所以框出来也是斜的。这属于异常情况。

因为她不是产品经理，所以我忍住没有发火。我说我回去再看看。

我找到了另一种方法，就是`minAreaRect`，它可以框选出一个区域的最小面积。即便是图片倾斜了，为了达到最小面积，它的框也得倾斜。有了倾斜角度，再旋转回来，就是正常的了，最终测试成功。

![img2.gif](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d8914e74ae2143f38f10e020f66c7ff8~tplv-k3u1fbpfcp-watermark.image?)

我又开始对着镜子训练了，我这次一定要鼓起勇气告诉她实现思路，同时，把上次失败的原因也一并说出来。

代码如下：
```python
def minAreaRect(image_path):

    # 读入图片3通道 [[[255,255,255],[255,255,255]],[[255,255,255],[255,255,255]]]
    image = cv2.imread(image_path)
    # 转为灰度单通道 [[255 255],[255 255]]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 黑白颠倒
    gray = cv2.bitwise_not(gray)
    # 二值化
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # %% 把大于0的点的行列找出来
    ys, xs = np.where(thresh > 0)
    # 组成坐标[[306  37][306  38][307  38]],里面都是非零的像素
    coords = np.column_stack([xs,ys])
    # 获取最小矩形的信息 返回值(中心点，长宽，角度) 
    rect = cv2.minAreaRect(coords)
    angle = rect[-1] # 最后一个参数是角度
    print(rect,angle) # ((26.8, 23.0), (320.2, 393.9), 63.4)

    # %%  通过换算，获取四个顶点的坐标
    box = cv2.boxPoints(rect)
    box = np.int0(cv2.boxPoints(rect))
    print(box) # [[15 181][367  5][510 292][158 468]]
    
    # 画框，弹窗展示
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    cv2.imshow("output", image)
    cv2.waitKey(0)

    return angle
```

前面读入图片、灰度、黑白颠倒、二值化，与`boundingRect`处理一样。

有区别的地方就是为了能够找到最小面积，`minAreaRect`需要你给它提供所有非空白的坐标。它通过自己的算法，计算这些坐标，可以画出一个非平行XY轴，刚刚包裹这个区域的矩形区域。

`minAreaRect`的返回值需要解读一下`((248.26095581054688, 237.67669677734375), (278.31488037109375, 342.6839904785156), 53.530765533447266)`，它分为三个部分(中心点坐标x、y，长宽h、w，角度a)。

我们先看角度a，如果角度a是正数。

![img2_2_rotate_mark.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dbbc09162ac44397aad311058dced75d~tplv-k3u1fbpfcp-watermark.image?)

不想记这些也没有关系，想搞到它框选的位置有一个方法叫`cv2.boxPoints(rect)`，它可以从`minAreaRect`的返回值，直接转换出来四个顶点的坐标`[[15 181][367  5][510 292][158 468]]`。

至于如何旋转图片……你先记下就好，能用就行，因为这不是重点，没有必要为所有事情耗费精力，人生总要留点遗憾才能去奋力弥补。

```python
# 传入图片数据数组和需要旋转的角度
def rotate_bound(image, angle):
    #获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # 提取旋转矩阵 sin cos 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    nH = h
    # 调整旋转矩阵
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
```
求角度，再旋转图片

```phton
# 调用求角度
angle = minAreaRect('img2_0_rotate.jpg')  
# 旋转图片,查看效果
image = rotate_bound(cv2.imread('img2_0_rotate.jpg'), 90-angle)
cv2.imshow("output", image)
cv2.waitKey(0) 
```

我带着程序去找女秘书，推开门，看到她刚把外套挂在衣架上，在她转身的那一刻，我看到她穿着低胸装，很低的那种，很火。我是正人君子，非礼勿视。我说了一句，程序改好了，你再试试吧。说完我就走了。

不一会，她又叫我过去，说程序好像有点问题。我再去时，发现她居然把外套穿上了，天气不是很冷，为什么要穿外套。

她跟我描述了一下现象，总之，处理出来的图依然不是她想要的，她就是想要摆正了的文字图片。

![img2-img3.gif](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/19756f2051f74665b57cb820a1f95a16~tplv-k3u1fbpfcp-watermark.image?)

我一看，这个异常情况太异常了，首先一个斜的矩形文本区域，这个区域内的文本又是斜的。这玩意，你怎么画框它也转不正啊。

美女秘书娇羞羞地问：大工程师，这个有难度吗？

“难度，哈哈哈，不存在的！我先回去了，抽个空给你搞定！”，我直挺挺地走出房间，关上门瞬间泄了气，这玩意怎么整啊？

# 三、HoughLinesP 霍夫线变换

这时候不能再想什么框，什么区域的问题了，一切框都是不起作用的，更不能想天气热不热的问题，这些都只会扰乱思绪。

我最终还是找到一种方法，把倾斜的问题给旋转正了。

![img3.gif](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b8da932063004788a441c32c5475fd1e~tplv-k3u1fbpfcp-watermark.image?)

我采用的是一种叫`霍夫变换`的方法。

`霍夫变换`不仅能识别`直线`，也能够识别任何形状，常见的有圆形、椭圆形。

我不贪心，我这里只用它识别直线。

```python
# 读入图片3通道 [[[255,255,255],[255,255,255]],[[255,255,255],[255,255,255]]]
image = cv2.imread(image_path)
# 转为灰度单通道 [[255 255],[255 255]]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 # 处理边缘
edges = cv2.Canny(gray, 500, 200)
# 求得所有直线的集合
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, maxLineGap=200)
print(lines) # [[[185 153 369 337]] [[128 172 355 362]]]
```

图像的灰度处理也和之前一样，主要目的是要让数据即简洁又有效。

这里多了一个`cv2.Canny(gray, 500, 200)`的边缘处理，处理的效果如下。

![img3_1_cut_ed.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f32f76c405674410b812a925d1d05f57~tplv-k3u1fbpfcp-watermark.image?)

这样做的目的依然是进一步让数据即简洁又有效。

说一下`Canny(gray, 500, 200)`的`3`个参数。第`1`个参数是灰度图像的数据，因为它只能处理`灰度图像`。第`2`个参数是`大阈值`，用于设置`刻画边缘的力度`，数值大边缘越粗犷，大到一定程度，边缘就断断续续的连不成块。第`3`个参数是`小阈值`，用于`修补断开的边缘`，数值决定修补的精细程度。

通过边缘处理，我们就得到图像的轮廓，这时候是不影响原图结构的。虽然`2`个点确定一条直线，但是一条直线也可以有3个点、4个点、5个点，如果条件合适，是可以从点定出直线的。

![图片.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/04eb510c53e74f5290bf7383578998e3~tplv-k3u1fbpfcp-watermark.image?)

`cv2.HoughLinesP`就在轮廓的基础上根据条件，在图片上辗转反侧地去画线，看看能画出多少直线。肯定很多条，但并不是都符合条件，根据参数它能找到符合条件的所有线段如下所示。

![img3_3_cut.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/77707a71dcf94f91825b9b156c3ce4fc~tplv-k3u1fbpfcp-watermark.image?)

那么，它是根据什么找的。我们解读它的参数 `cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, maxLineGap=200)`。
1. 第`1`个参数`edges`是边缘的像素数据。
2. 第`2`个参数是搜索直线时，每次移动几个像素。
3. 第`3`个参数是搜索直线时，每次旋转多少角度。
4. 第`4`个参数`threshold`最小是多少个点相交可以算一条直线。
5. 第`5`个参数`maxLineGap`两点之间，最大多少像素距离后，就不能再算作一条直线了。

经过这些筛选，直线们就出来了。

有了直线，我们就可以计算直线的角度了。

```python
# 计算一条直线的角度
def calculateAngle(x1,y1,x2,y2):

    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)

    if x2 - x1 == 0:
        result=90 # 直线是竖直的
    elif y2 - y1 == 0:
        result=0 # 直线是水平的
    else:
        # 计算斜率
        k = -(y2 - y1) / (x2 - x1)
        # 求反正切，再将得到的弧度转换为度
        result = np.arctan(k) * 57.29577
    return result
```

我们计算所有直线的角度。然后选出哪个角度出现的最多，频次最高的角度，基本代表整体的倾斜角度。
```python
# 存储所有线段的倾斜角度
angles = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0,0,255))
    angle = calculateAngle(x1, y1, x2, y2)
    angles.append(round(angle))
# 找到最多出现的一个
mostAngle = Counter(angles).most_common(1)[0][0]
print("mostAngle:",mostAngle)
```

最后,我们调用成果，展示出来旋转后的效果。

```python
mostAngle = houghImg("img3_0_cut.jpg")
# 旋转图片,查看效果
image = rotate_bound(cv2.imread('img3_0_cut.jpg'), mostAngle)
cv2.imshow("output", image)
cv2.waitKey(0)
```

这样，这种文档我们也可矫正了。

![img3_5_cut.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4976aec5900741569c8626fcf620a41b~tplv-k3u1fbpfcp-watermark.image?)

我连忙去找美女秘书，要把这个好消息告诉她。

她确实很高兴，并且说自己最近也在研究程序，还问我要源码。

于是，我就把github地址 https://github.com/hlwgy/doc_correction 给了她。



