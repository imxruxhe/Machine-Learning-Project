from cProfile import label
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
import pdb


# load data
#(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
def load_data():
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    '''
    x_train.shape == (50000, 32, 32, 3)
    x_test.shape == (10000, 32, 32, 3)
    y_train.shape == (50000, 1)
    y_test.shape == (10000, 1)
    '''

    #num = 12345
    #plt.imshow(x_train[num, : , :, :])
    #plt.title('Label:' + str(y_train[num]))
    #plt.show()

    return (x_train, y_train),(x_test, y_test)

# preprocessing
def preprocessing(x_train_, x_test_):
    x_train = x_train_ / 255.0
    x_test = x_test_ / 255.0
    return x_train, x_test


# build model
def build_model():
    '''
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    '''
   
    #x = tf.keras.layers.Flatten()(input)
    #x = tf.keras.layers.Dense(128, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    input = tf.keras.Input(shape=(32, 32, 3), name='img') # 입력은 28/28/1이다.
    
    
    # output shape(H,W,C) 출력은 (28,28,4) padding이 same이어서 28,28 그대로 나머지 4는 filter임
    # Conv, ReLu를 1줄로 만들어준거임.
    
    # 채널 수는 filter Number
    # Conv2D - ReLu - MaxPool2D 가 이렇게 3개가 한 쌍이다. activation='relu'에 들어가 있다.
    x = tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3),strides=(1, 1),  padding='Same',activation='relu', use_bias=False)(input) # 28/28/4
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2,2), padding='Same')(x) #(28,28,4) stride(14,14,4) 채널수는 같고 가로세로 하나씩 줄어든다.
    
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3),strides=(1, 1),  padding='Same',activation='relu', use_bias=False)(x) # (H,w,C) (14,14,8)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2,2), padding='Same')(x) #(7,7,8) 
    
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3),strides=(1, 1),  padding='Same',activation='relu', use_bias=False)(x) # (H,w,C) (7,7,16)
    x = tf.keras.layers.Dropout(0.5)(x)
    #x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2,2), padding='Same')(x) # (,,16) # 공간성이 맞지 않는다. 4인지 3인지 그래서 padding 써주지 않는다
       
    #x=tf.keras.activations.relu()(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(512, activation='softmax')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(100, activation='softmax')(x) #node개수 만큼 10 은 class 개수 뒤에는 무조건 softmax 10개를 통해서 나오다 softmax를 통해서 하나만 나온다.

    model = tf.keras.Model(inputs=input, outputs=output, name = 'functional_model') #input,output 순서 바뀌면 오류

    return model


# compile model
def compile_model(model_, lr):
    model_.compile(
                  optimizer = tf.keras.optimizers.Adam(learning_rate = lr), # 성능을 결정짓는 하이퍼 파라메타
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model_

# train model
def train_model(model, epoch):

    # tensorboard --logdir=logs
    log_dir = result_dir+"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
   
    # 훈련 데이터 
    history = model.fit(x=x_train, y=y_train,
              epochs = epoch,
              callbacks=[tensorboard_callback],
              validation_data = (x_test, y_test),
              use_multiprocessing=True)
    
    # 훈련중에 history가 생기고 이 것을 기반으로 성능 평가를 하겠다 
    
    return model, history
'''
def plot_performance(history):
    train_loss=history.history['loss']
    train_accuracy = history.history['accuracy']
    val_loss=history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    
    plt.plot(train_loss,label='train_loss')
    plt.plot(val_loss,label='val_loss')
    plt.title("loss_performance")
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plt.savefig(result_dir+'train_performance.png'), plt.close()
    
    plt.plot(train_accuracy,label='train_accuracy')
    plt.plot(val_accuracy,label='val_accuracy')
    plt.title("accuracy_performance")
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plt.savefig(result_dir+'validation_performance.png'), plt.close()'''
    
def plot_performance(history):
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_accuracy = history.history['val_accuracy']

    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='validation loss')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plt.savefig(result_dir + 'loss_performance.png'), plt.close()

    plt.plot(train_accuracy, label='train accuracy')
    plt.plot(val_accuracy, label='validation accuracy')
    plt.title('accuracy performance')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plt.savefig(result_dir + 'accuracy_performance.png'), plt.close()

    


# eval model
def eval_model(model):
    model.evaluate(x_test, y_test)
    return model


if __name__=='__main__':
    
    result_dir='results/' # global 변수로 설정해준다.

    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # hyperparameter
    epoch = 2**4  # 전 트레이닝 데이터
    lr = 1e-3     # running rate


    (x_train, y_train),(x_test, y_test) = load_data()
    x_train, x_test = preprocessing(x_train, x_test)
    model = build_model()
    model = compile_model(model, lr)
    model.summary()
    
    with open(result_dir+'modelsummary.txt', 'w') as f: #modelsummary로 저장해준다. 문서에 이용할수 있게 오
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    model, history = train_model(model, epoch) # 같이 반환해주어야 한다.
    #  _, history = train_model(model, epoch)
    #model, _ = train_model(model, epoch) 메모리 잡아먹는 것 방지 But 순서는 지켜줘야함
    
    plot_performance(history)
    
    #pdb.set_trace()
    
    model = eval_model(model)