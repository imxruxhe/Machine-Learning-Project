import numpy as np
import matplotlib.pyplot as plt

x=np.arange(0,6,0.1)
y1=np.sin(x)
y2=np.cos(x)

plt.plot(x,y1,label='sin')
plt.plot(x,y2,label='cos')
plt.xlabel("x-axis")
plt.ylabel('y-axis')
plt.legend(loc="lower left")
plt.show()

from matplotlib.image import imread
img = imread("./sunflower.jpeg")
plt.imshow(img)
plt.show()

from matplotlib.image import imread
img = imread('../digital/openCV/images/bees-flowers-header.jpeg')  # 다시 해보기
# 시작할 때는 .이 붙어야함 /현재위치에서 들어가기 ./현재 위치에서 나가기 다음에는 
plt.imshow(img)
plt.show()

test = 0


