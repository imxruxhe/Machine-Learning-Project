from tkinter import E
import numpy as np
# 파이썬 함수 먼저 하고 수업 진행하셨슴.

#def function_name(input_args):
#    return function_of_musisi(input_args)

# 사칙연산기
def add(in1,in2):  # python은 default 값이 먼저 나와야해서 in1=3 설정 먼저 불가능하다. in2가 디폴트가 되기 때문에
    out = in1+in2
    return out

def substract(in1,in2):
    return in1-in2

def mult(in1,in2):
    return in1*in2

def div(in1,in2):
    min_value=0.0000000001
    in2=max(in2,min_value)
    return in1/in2


if __name__=='__main__': 
    
    in1=1
    in2=2
    
    add_out=add(in1,in2)
    substract_out=substract(1,2)
    mult_out=mult(1,2)
    div_out=div(4,2)

    #print('add_out:{}'.format(add_out))
    #print('substract_out:{}\nmult_out:{}'.format(substract_out,mult_out))
    print('add_out:{}'.format(add_out))
    print('substract_out:{}'.format(substract_out))
    print('mult_out:{}'.format(mult_out))
    print('div_out:{}'.format(div_out))

# 이 구조는 무조건 유지해주어야 한다.