'''
import tensorflow as tf
import numpy as np
import utils # utils를 가져와서 쓰겠다는 의미임.

#from utils import add,substract,mult 사용시 c=add(a,b) 다른파일에 같은 함수가 있다면 쫑날 확률이 높음


if __name__=='__main__': # __name__=='__main__' 이 어떻게 쓰이는데
    
    a=1
    b=2
    c=utils.add(a,b) # utils의 __name__=='__main__' 이전까지만 동작한다.(중요함) -> 함수 이름이 길어져도 안전하긴함.
'''
import tensorflow as tf
mnist = tf.keras.datasets.mnist # tf.keras.datasets -> 나는 정교화된 교육용 dataset을 불러올 준비가 되어있다.

# load dataset
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

#  preprocessing : (전처리)
x_train, x_test = x_train / 255.0, x_test / 255.0

# build model    -> Sequential이 포함하는 패키지 일렬로 세워야지 면이 되면 안된다.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'), # 128개의 노드
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax') # 출력 확률적 softmax 사용 
])

# compile model
model.compile(optimizer='adam', # 모델훈련 방법 adam
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) # metrics 정밀도 측정


# train model
model.fit(x_train, y_train, epochs=5) #  모델을 훈련 시킨다.

# test model 평가, 테스트 모드
model.evaluate(x_test, y_test)

# 훈련시 loss 가 줄어든다 오류값을 계속 줄도록 5번 정도 돌려준다. {훈련}
# 이것을 import def __main__ 함수 형태로 바꿀 줄 알아야 한다 다음 시간에 수업 할듯?