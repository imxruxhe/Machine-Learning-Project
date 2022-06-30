# tensorflow 이 부분 중간고사 범위는 아님
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os

# load data
#(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
def load_data():
    (_x_train,y_train),(_x_test,y_test)=tf.keras.datasets.mnist.load_data()
    
    x_train = _x_train.reshape(-1,28,28,1) #reshape 후 형상을 바꿔준다. 후에 아래 4줄로 확인을 해준다.
    x_test = _x_test.reshape(-1,28,28,1) #똑같이 바꿔준다. (N,H,W,C)
    
    
    # data form을 스탠다드 형태로 만ㅡ어줘야한다.
    #num = 12345
    #plt.imshow(x_train[num,:,:]) #위의 모양처럼 4개라 [num,:,:,:] 해줘야하는데 컴퓨터가 자동으로 해주는 것 같다.
    #plt.title('Label:'+str(y_train[num])) # title 안에는 문자열만 들어갈 수 있어서 str()로 형 변환을 해주어야 한다.
    #plt.show()
    # 내가 만든 사용할 데이터가 그런 테스트 데이터 이며 맞은 label인지 보기 위해서 일단 실행함 확인후 코드 막고 실행하기
    
    return (x_train,y_train),(x_test,y_test)


# Preprocessing 전처리 과정
#x_train, x_test = x_train / 255.0, x_test / 255.0

def preprocesing(x_train,x_test):
    x_train=x_train/255.0
    x_test=x_test/255.0
    return x_train, x_test

# build  model
def build_model():
    '''
    model = tf.keras.models.Sequential([ # Sequential 모델을 묶어준다.
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # 쭉 찢어
    tf.keras.layers.Dense(128, activation='relu'), # 128개
    tf.keras.layers.Dropout(0.2), # 1대1 통과 
    tf.keras.layers.Dense(10, activation='softmax') # 노드 10개 짜리
    ])
    '''
    input = tf.keras.Input(shape=(28, 28, 1), name="img")
    x = tf.keras.layers.Flatten()(input)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(input,output,name='funtional_model')
    
    return model

    # build_model 이 http://localhost:6006/ 에 있는 것이다.

# compile model
def compile_model(model,lr):  # Input 값이 있어야함
    model.compile(#optimizer='adam',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) # metrics=['accuracy'] 정밀도를 측정할거야
    return model

# train model 훈련에 필요한 정보
def train_model(model,epochs):
    
    # tensorboard --logdir=logs
    log_dir = result_dir+"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") #현재시간을 string으로 만들어라?
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)        
    model.fit(x=x_train, y=y_train,
              epochs=10, # train 데이터를 몇 번 훈련 시킬 것인가 epochs 
              callbacks=[tensorboard_callback],
              validation_data = (x_test, y_test), # 테스트 성능도 같이나온다
              use_multiprocessing=True) # epochs는 전 dataset을 한번씩 돌리는것 epochs=5 5번 돌려본다.
    return model
    # 이 줄이 끝나면 훈련이 끝난다.

# eval model 평가 모델
def eval_model(model):
    model.evaluate(x_test, y_test)
    return model
# module화 -> 내가 만든 코드를 다른 부분에서 쓸 수 있게 하기 위해서
# 1. import 2. def 함수 정의 3. if__name__=='__main':

if __name__=='__main__':
    
    result_dir='results/' # global 변수로 설정해준다.

    
    import os
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    epoch = 2**0 #가로 세로 3차원으로 계산한다. 계산될 숫자들이 2의 power승
    lr = 1e-3
    
    (x_train,y_train),(x_test,y_test) = load_data()
    x_train,x_test=preprocesing(x_train,x_test)
    model=build_model()
    model=compile_model(model,lr)
    model.summary()
    
    with open('modelsummary.txt', 'w') as f: #modelsummary로 저장해준다. 문서에 이용할수 있게 오
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    model=train_model(model,lr)
    model=eval_model(model)   # 변수 & 함수 이름은 길게 써줘도 괜찮음
    
    #파일을 보관하는 것ㅡㅗ 만들어줘야하는데 
