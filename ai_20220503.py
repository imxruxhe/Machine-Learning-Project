import tensorflow as tf
import matplotlib.pyplot as plt
import datetime


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
    '''
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
    '''
   
    input = tf.keras.Input(shape=(28, 28, 1), name='img')
    x = tf.keras.layers.Flatten()(input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(input, output, name = 'functional_model')

    return model


# compile model
def compile_model(model_, lr):
    model_.compile(
                  #optimizer='adam',
                  optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model_

# train model
def train_model(model, epoch):

    # tensorboard --logdir=logs
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
   
    model.fit(x=x_train, y=y_train,
              epochs = epoch,
              callbacks=[tensorboard_callback],
              validation_data = (x_test, y_test),
              use_multiprocessing=True)
    return model

# eval model
def eval_model(model):
    model.evaluate(x_test, y_test)
    return model


if __name__=='__main__':

    # hyperparameter
    epoch = 2**4
    lr = 1e-3


    (x_train, y_train),(x_test, y_test) = load_data()
    x_train, x_test = preprocessing(x_train, x_test)
    model = build_model()
    model = compile_model(model, lr)
    model = train_model(model, epoch)
    model = eval_model(model)