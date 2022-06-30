from tkinter import N
import cv2
import numpy as np
import matplotlib.pyplot as plt

def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1
    
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1
    
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1
    
def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y
    
    
def sigmoid(x):
    y=1+np.exp(-x)
    return 1/(1+y)

def step(x):
    y=x>0
    return y.astype(np.uint8)

def ReLU(x):
    return np.maximum(0,x)

def softmax(x):
    c_Prime=np.exp(x-c_Prime)
    num=np.exp(x-c_Prime)
    den=np.sum(np.exp(x-c_Prime))
    return num/den

if __name__=='__main__':
    
    print("AND==================")
    for x1 in [0,1]:
        for x2 in [0,1]:
            y=AND(x1,x2)
            print("x1:{}, x2:{}, y_AND:{}".format(x1,x2,y))
    
    print("NAND=================")
    for x1 in [0,1]:
        for x2 in [0,1]:
            y=NAND(x1,x2)
            print("x1:{}, x2:{}, y_NAND:{}".format(x1,x2,y))    
    
    print("OR==================")
    for x1 in [0,1]:
        for x2 in [0,1]:
            y=OR(x1,x2)
            print("x1:{}, x2:{}, y_OR:{}".format(x1,x2,y))
            
    print("XOR================")
    for x1 in [0,1]:
        for x2 in [0,1]:
            y=XOR(x1,x2)
            print("x1:{}, x2:{}, y_XOR:{}".format(x1,x2,y))
    
    x=np.arange(-5,5,0.01)
    y1=sigmoid(x)
    y2=step(x)
    y3=ReLU(x)
    
    plt.plot(x,y1,label='sigmoid')
    plt.plot(x,y2,label='step')
    plt.plot(x,y3,label='ReLU')
    plt.title('sigmoid & step * ReLU function')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    x1=np.array([990,1100,1000])
    y4=softmax(x1)
    print(y4)