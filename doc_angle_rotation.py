 
#%% 导入必要的包
import numpy as np
import cv2
from collections import Counter

# %% 最小包裹正矩形 ================================================================
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

    # 写入文件
    #cv2.imwrite('img2_1_rotate.jpg', image)
    # 弹出展示图片
    cv2.imshow("output", image)
    cv2.waitKey(0)


# %% 最小面积矩形 ========================================================================
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

    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    cv2.imshow("output", image)
    cv2.waitKey(0)

    return angle
# %% 霍夫线 ========================================================================

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

# 霍夫线获得角度
def houghImg(image_path):
    # 读入图片3通道 [[[255,255,255],[255,255,255]],[[255,255,255],[255,255,255]]]
    image = cv2.imread(image_path)
    # 转为灰度单通道 [[255 255],[255 255]]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 处理边缘
    edges = cv2.Canny(gray, 500, 200, 3)
    # 求得
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, maxLineGap=200)
    print(lines)

    # 得到所有线段的端点
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0,0,255))
        angle = calculateAngle(x1, y1, x2, y2)
        angles.append(round(angle))

    mostAngle = Counter(angles).most_common(1)[0][0]
    print("mostAngle:",mostAngle)

    cv2.imshow("output", image)
    cv2.waitKey(0)

    return mostAngle


# %% 图片旋转 ============================================================================

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
# %%
if __name__ == '__main__':

    print("请选择需要的代码解除注释，查看效果")

    # 【A】 最小边框矩形
    boundingRect('img1_0_origin.jpg')   

    # 【B】 最小面积矩形，获得角度
    # angle = minAreaRect('img2_0_rotate.jpg')  
    # 旋转图片,查看效果
    # image = rotate_bound(cv2.imread('img2_0_rotate.jpg'), 90-angle)
    # cv2.imshow("output", image)
    # cv2.waitKey(0)

    # 【C】霍夫线，获得角度
    # mostAngle = houghImg("img3_0_cut.jpg")
    # 旋转图片,查看效果
    # image = rotate_bound(cv2.imread('img3_0_cut.jpg'), mostAngle)
    # cv2.imshow("output", image)
    # cv2.waitKey(0)
