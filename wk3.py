'''


hungry = False # hungry test 이 위에서부터만 동작합니다.
if hungry:
    print("I am hungry") # True 이니까 아래로 내려가지 않음
else:
    print("I'am sleepy")    #들여쓰기 통일시킨다.
    
for i in [1,2,3]: #for test  # 종단점 커서가 이 줄에 있으면 조사식 알 수 없음 실행 X
    print(i)

aa =[i**2 for i in [2,3,4]]
test=0 #더미코드
'''
import numpy as np

def adder(a,b):
    return a + b

def sub(a,b):
    return a - b

def mult(a,b):
    return a * b   # return a * b 를 해줘야 a,b의 색이 달라진다 함수는 return 값이 필요해서

def div(a,b):
    min_value = 0.00000000001
    b = max(b,min_value) # min_value 를 정해준다.  # b > 0 => boolean 이 되어버린다. 그래서 이렇게 표현 해주어야한다.
    return a / b

# 다음에 이 함수 쓰고 싶으면 import wk.3 이렇게 불러줄 수 있다. 다른 곳에서 쓰기 위해서 아래 34번 줄의 과정을 거침

if __name__ == '__main__':  # 위에는 함수 정의 원래 코드는 이렇게 짜야함 entrycode 이걸 안하면 wk.3에서만 쓰이게 설정하는 것
    
    x = 3.5
    y = 4.8

    z1 = adder(x,y)

    x1 = 3.2
    y2 = 0.000000000000000000000000000000001
    z2 = div(x,y)

    test = 0


