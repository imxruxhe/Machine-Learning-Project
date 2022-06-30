import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    C_Prime=np.sum(x)/len(x)
    num=np.exp(x-C_Prime)
    den=np.sum(np.exp(x-C_Prime))
    return num/den


if __name__=='__main__':
    x=np.array([0.3,2.9,4.0])
    y=softmax(x)
    print(y)