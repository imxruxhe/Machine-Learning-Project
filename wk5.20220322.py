'''
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


# 데이터 준비
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# 그래프 그리기
plt.plot(x,y1,label='sin plot')
plt.plot(x,y2,label="cos plot")
plt.title("sin & cos plot")
plt.legend(loc="upper right")
plt.show()

img = plt.imread("./images/star-wars-the-rise-of-skywalker-theatrical-poster-1000_ebc74357.jpeg")
plt.imshow(img)
plt.show()
'''
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

def AND(x1,x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    sum= x1*w1 + x2*w2
    
    if sum <= theta:
        return 0
    else:
        return 1
    

def NAND(x1,x2):
    theta= -0.7
    w=np.array([-0.5,-0.5])
    x=np.array([x1,x2])
    
    if np.sum(w*x)<=theta:  # sum=w*x -> 배열이 나와버림
        return 0
    else:
        return 1    

def OR(x1,x2):
    theta= 0.5
    w=np.array([1,1])   # w = [w1,w2]
    x=np.array([x1,x2])    # x = [x1,x2]
    
    if np.sum(w*x)<=theta:  # sum=w*x -> 배열이 나와버림
        return 0
    else:
        return 1

def XOR(x1,x2):
    
    s1=NAND(x1,x2);
    s2=OR(x1,x2);
    y = AND(s1,s2)
    
    return y 

def sigmoid(x):
    y = np.exp(-x)
    return 1 / (1+y)

def step(x):
    if x>=0:         # array 불가능? /  출력값 자체를 이걸로 써야함 / 
        y=1
    else:
        y=0
    return y
        
def step1(x):
    y = (x >= 0)
    return y.astype('uint8')   # 각각을 전부 다 step  boolean 을 integer로 바꾸는 이야기 return y.astype(np.int)
   #return y.astype(np.int)도 위와 동일
   
def ReLU1(x):   # 하나하나는 맞지만 array로 들어오면 True,False 이런식으로 나옴 그래서 또 마줘줘야함 
    if x>=0:
        y=x
    else:
        y=0
    return y

def ReLU(x):
    return np.maximum(0,x)
    
      
        
    


if __name__=='__main__':
        
    print("AND===================")
    for x1 in [0,1]:
        for x2 in [0,1]:
            
            y=AND(x1,x2)
            print("x1:{}, x2:{}, y_AND:{}".format(x1,x2,y));
         
    print("NAND===================")   
    for x1 in [0,1]:
        for x2 in [0,1]:
            
            y=NAND(x1,x2)
            print("x1:{}, x2:{}, y_NAND:{}".format(x1,x2,y));          
            
    print("OR===================")   
    for x1 in [0,1]:
        for x2 in [0,1]:
            
            y=OR(x1,x2)
            print("x1:{}, x2:{}, y_OR:{}".format(x1,x2,y)); 

    print("XOR===================")   
    for x1 in [0,1]:
        for x2 in [0,1]:
            
            y=XOR(x1,x2)
            print("x1:{}, x2:{}, y_XOR:{}".format(x1,x2,y)); 


    x = np.arange(-10,10,0.01)
    y1 = sigmoid(x)

    plt.plot(x,y1)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('sigmoid function')
    plt.grid(True)
    plt.show()

    y2 = step1(x)
    #plt.plot(x,y2)
    #plt.show()

    y3 = ReLU(x)

    plt.plot(x,y1,'b-',label='sigmoid')
    plt.plot(x,y2,'r:',label='step')
    plt.plot(x,y3,'m-.',label='ReLU')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    #plt.xlim(-5,5)
    #plt.ylim(0.8,2)
    plt.title('sigmoid & step & ReLu function')
    plt.grid(True)
    plt.legend(loc='lower right')   # loc 안 쓰면 plot 피해서 자동으로 위치함
    plt.show()

    tmp=0