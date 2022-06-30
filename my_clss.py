from multiprocessing import pool
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os
import pdb

from torch import softmax
#

def download_custom_dataset(key_word, saved_dir, number):
    # ref:https://icrawler.readthedocs.io/en/latest/ 
    #     https://garudabyte.com/how-to-crawl-images-from-bing/

    from icrawler.builtin import BingImageCrawler

    bing_crawler = BingImageCrawler(downloader_threads=4,
                                    storage={'root_dir': saved_dir})
    bing_crawler.crawl(keyword=key_word, filters=None, offset=0, max_num=number)


def make_custom_dataset():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
            './datasets/train/',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
            './datasets/validation/',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

    return train_generator, validation_generator















# load data
def load_data():
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    #x_train = _x_train.reshape(-1,28,28,1)
    #x_test = _x_test.reshape(-1,28,28,1)

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

'''
def bulid_VGG16_model():
    
    input = tf.keras.Input(shape=(224,224,3))
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(input) # (H,W,C) = (224,224,64)
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)     # (H,W,C) = (224,224,64)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x)                                   # (H,W,C) = (112,112,64)
    
    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (112,112,128)
    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (112,112,128)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x)                                   # (H,W,C) = (56,56,128)
    
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (56,56,256)
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (56,56,256)
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (56,56,256)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x)                                   # (H,W,C) = (28,28,256)
    
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (28,28,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (28,28,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (28,28,512)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x)                                   # (H,W,C) = (14,14,512)
    
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (14,14,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (14,15,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (14,14,512)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096,activation='softmax')(x)
    x = tf.keras.layers.Dense(4096,activation='softmax')(x)
    output = tf.keras.layers.Dense(1000,activation='softmax')(x)
    
    model = tf.keras.Model(inputs=input,outputs=output,name='funtional_model')
    
    return model
'''
# build_model()
def VGG16_build_model():
    
    input = tf.keras.Input(shape=(224,224,3))
    
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(input) # (H,W,C) = (224,224,64)
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)     # (H,W,C) = (224,224,64)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x)                                   # (H,W,C) = (122,122,64)

    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (112,112,128)
    x = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (122,112,128)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x)                                   # (H,W,C) = (56,56,128)
    
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (56,56,256)
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (56,56,256)
    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (56,56,256)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x)                                   # (H,W,C) = (28,28,128)
    
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (28,28,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (28,28,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (28,28,512)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x)                                   # (H,W,C) = (14,14,512)
    
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (14,14,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (14,14,512)
    x = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x)    # (H,W,C) = (14,14,512)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096,activation='softmax')(x)
    x = tf.keras.layers.Dense(4096,activation='softmax')(x)
    output = tf.keras.layers.Dense(1000,activation='softmax')(x)
    
    model = tf.keras.Model(inputs=input,outputs=output,name='functional_model')
    
    return model
    
    
# build_model
def bulid_model():
    
    input = tf.keras.layers.Input(shape=(28,28,3),name='img')
    
    # Conv2D - Relu - MaxPool2D
    x = tf.keras.layers.Conv2D(filters=4,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(input)  # (H,W,C) = (28,28,4) 
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x) # (H,W,C) = (14,14,4)
    
    x = tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x) # (H,W,C) = (14,14,8)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),stride=(2,2),padding='same')(x) # (H,W,C) = (7,7,8)
    
    x = tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='relu',use_bias=False)(x) # (H,W,C) = (7,7,16)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(512,activation='softmax')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(2,activation='softmax')(x)
    
    model = tf.keras.Model(inputs=input,outputs=output,name='funtional_model')
    
    return model

    

# build model
def build_model():
    
    input = tf.keras.Input(shape=(28, 28, 3), name='img')

    # Conv2D - ReLU - MaxPool2D
    x = tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3), padding='SAME', activation='relu', use_bias=False)(input)  # (H,W,C) (28,28,4)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x) # (14, 14, 4)

    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding='SAME', activation='relu', use_bias=False)(x)  # (H,W,C) (14, 14, 8)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x) # (7, 7, 8)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='SAME', activation='relu', use_bias=False)(x)  # (H,W,C) (7, 7, 16)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(512, activation='softmax')(x)  # shape 
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(2, activation='softmax')(x)  # shape 

    model = tf.keras.Model(inputs=input, outputs=output, name='functional_model')

    return model

    


# compile model
def compile_model(model_, lr):
    model_.compile(
                  optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy']
                  )
    return model_

# train model
def train_model(model, epoch):

    # tensorboard --logdir=logs
    log_dir = result_dir + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = result_dir + 'logs')
    
    history = model.fit(train_generator, 
              epochs = epoch, 
              callbacks = tensorboard_callback,  # if you have several callback, use list like [tensorboard_callback, cp_callback]
              validation_data = validation_generator,
              use_multiprocessing=True)

    return model, history

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
    #plt.show()
    plt.savefig(result_dir + 'loss_performance.png'), plt.close()

    plt.plot(train_accuracy, label='train accuracy')
    plt.plot(val_accuracy, label='validation accuracy')
    plt.title('accuracy performance')
    plt.xlabel('epoch')
    plt.legend()
    #plt.show()
    plt.savefig(result_dir + 'accuracy_performance.png'), plt.close()



# eval model
def eval_model(model):
    model.evaluate(x_test, y_test)
    return model


if __name__=='__main__':

    '''
    number = 100
    classes = ['cat', 'dog']
    root_dir = './datasets'

    for i in range(len(classes)):
        download_custom_dataset(classes[i], root_dir + '/' + classes[i], number)
    '''

    result_dir = 'results/'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # hyperparameter
    epoch = 2**4
    lr = 1e-4


    #(x_train, y_train),(x_test, y_test) = load_data()
    #x_train, x_test = preprocessing(x_train, x_test)

    train_generator, validation_generator = make_custom_dataset()


    model = build_model()
    model = compile_model(model, lr)
    model.summary()
    with open(result_dir + 'model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model, history = train_model(model, epoch)
    plot_performance(history)
    model = eval_model(model)
    