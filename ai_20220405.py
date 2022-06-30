import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def softmax(x):
    '''
    #yi = exp(x) / sum(exp(xi))
    ''' 
    c_Prime= np.sum(x)/len(x)   # 990,1000,1010 의 평균값으로 해준다는 의미
    num=np.exp(x-c_Prime)
    den=np.sum(np.exp(x-c_Prime))
    #y=np.exp(x) / np.sum(np.exp(x))
    return num/den


if __name__=='__main__':
    
    x=np.array([990,1000,1010])
    y=softmax(x)   # 여기서 실제로 동작함
    print(y)
    
    tmp=0
    
    c_Prime=np.sum(x)/len(x)
    num=np.exp(x-c_Prime)
    den=np.sum(np.exp(x-c_Prime))