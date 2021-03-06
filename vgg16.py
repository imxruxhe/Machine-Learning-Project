from cProfile import label
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
import pdb


# load data
#(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
def load_data():
    (_x_train, y_train),(_x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = _x_train.reshape(-1,28,28,1)
    x_test = _x_test.reshape(-1,28,28,1)

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
    input = tf.keras.Input(shape=(224,224,3), name='img')
    
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(input) #(H,W,C)=(224,224,64)
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)     #(224,224,64)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='Same')(x)                                                #(112,112,64)
    
    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(112,112,128)
    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(112,112,128)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='Same')(x)                                                #(56,56,128)
    
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(56,56,256)
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(56,56,256)
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(56,56,256)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='Same')(x)                                                #(28,28,256)
    
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(28,28,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(28,28,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(14,14,512)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='Same')(x)
    
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(14,14,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(14,14,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='Same',activation='relu',use_bias=False)(x)    #(14,14,512)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='Same')(x)                                                #(7,7,512)
    
    
    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(4096, activation='softmax')(x)
    x = tf.keras.layers.Dense(4096, activation='softmax')(x) #node?????? ?????? 10 ??? class Number ????????? ????????? softmax 10?????? ????????? ????????? softmax??? ????????? ????????? ?????????.
    output = tf.keras.layers.Dense(1000, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=input, outputs=output, name = 'functional_model') #input,output ?????? ????????? ??????

    return model
    
    x = tf.keras.layer
    '''
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    '''
'''   
    #x = tf.keras.layers.Flatten()(input)
    #x = tf.keras.layers.Dense(128, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.2)(x)
    input = tf.keras.Input(shape=(28, 28, 1), name='img') # ????????? 28/28/1??????.
    
    
    # output shape(H,W,C) ????????? (28,28,4) padding??? same????????? 28,28 ????????? ????????? 4??? filter???
    # Conv, ReLu??? 1?????? ??????????????????.
    
    # ?????? ?????? filter Number
    # Conv2D - ReLu - MaxPool2D ??? ????????? 3?????? ??? ?????????. activation='relu'??? ????????? ??????.
    x = tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3),strides=(1, 1),  padding='Same',activation='relu', use_bias=False)(input) # 28/28/4
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2,2), padding='Same')(x) #(28,28,4) stride(14,14,4) ???????????? ?????? ???????????? ????????? ????????????.
    
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3),strides=(1, 1),  padding='Same',activation='relu', use_bias=False)(x) # (H,w,C) (14,14,8)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2,2), padding='Same')(x) #(7,7,8) 
    
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3),strides=(1, 1),  padding='Same',activation='relu', use_bias=False)(x) # (H,w,C) (7,7,16)
    #x = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2,2), padding='Same')(x) # (,,16) # ???????????? ?????? ?????????. 4?????? 3?????? ????????? padding ????????? ?????????
       
    #x=tf.keras.activations.relu()(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    output = tf.keras.layers.Dense(10, activation='softmax')(x) #node?????? ?????? 10 ??? class ?????? ????????? ????????? softmax 10?????? ????????? ????????? softmax??? ????????? ????????? ?????????.

    model = tf.keras.Model(inputs=input, outputs=output, name = 'functional_model') #input,output ?????? ????????? ??????

    return model
    '''



# compile model
def compile_model(model_, lr):
    model_.compile(
                  optimizer = tf.keras.optimizers.Adam(learning_rate = lr), # ????????? ???????????? ????????? ????????????
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model_

# train model
def train_model(model, epoch):

    # tensorboard --logdir=logs
    log_dir = result_dir+"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
   
    # ?????? ????????? 
    history = model.fit(x=x_train, y=y_train,
              epochs = epoch,
              callbacks=[tensorboard_callback],
              validation_data = (x_test, y_test),
              use_multiprocessing=True)
    
    # ???????????? history??? ????????? ??? ?????? ???????????? ?????? ????????? ????????? 
    
    return model, history

def plot_performance(history):
    train_loss=history.history['loss']
    train_accuracy = history.history['accuracy']
    val_loss=history.history['val_loss']
    val_accuracy = history.history['val_accuracy']
    
    plt.plot(train_loss,label='train_loss')
    plt.plot(train_accuracy,label='train_accuracy')
    plt.title("train_performance")
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    plt.savefig(result_dir+'train_performance.png'), plt.close()
    
    plt.plot(val_loss,label='val_loss')
    plt.plot(val_accuracy,label='val_accuracy')
    plt.title("validation_performance")
    plt.legend()
    plt.show()
    plt.savefig(result_dir+'validation_performance.png'), plt.close()

    


# eval model
def eval_model(model):
    model.evaluate(x_test, y_test)
    return model


if __name__=='__main__':
    
    result_dir='results/' # global ????????? ???????????????.

    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # hyperparameter
    epoch = 2**2  # ??? ???????????? ?????????
    lr = 1e-3


    (x_train, y_train),(x_test, y_test) = load_data()
    x_train, x_test = preprocessing(x_train, x_test)
    model = build_model()
    model = compile_model(model, lr)
    model.summary()
    
    with open(result_dir+'modelsummary.txt', 'w') as f: #modelsummary??? ???????????????. ????????? ???????????? ?????? ???
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    model, history = train_model(model, epoch) # ?????? ?????????????????? ??????.
    #  _, history = train_model(model, epoch)
    #model, _ = train_model(model, epoch) ????????? ???????????? ??? ?????? But ????????? ???????????????
    
    plot_performance(history)
    
    #pdb.set_trace()
    
    model = eval_model(model)