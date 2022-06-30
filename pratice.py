import numpy as np
import matplotlib.pyplot as plt

def add(x1,x2):
    return x1+x2

def sub(x1,x2):
    return x1-x2

def mult(x1,x2):
    return x1*x2

def div(x1,x2):
    min_value=0.0000000001
    x2=max(min_value,x2)
    return x1/x2

if __name__=='__main__':
    x1=60
    x2=30
    
    add_out=add(x1,50)
    sub_out=sub(x1,x2)
    mult_out=mult(x1,x2)
    div_out=div(40,x2)

    print("add_out:{}".format(add_out))
    print("sub_out:{}".format(sub_out))
    print("mult_out:{}".format(mult_out))
    print("div_out:{}".format(div_out))