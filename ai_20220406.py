import tensorflow as tf
from tensorflow import Flatten,Dense,Dropout

#mnist = tf.keras.datasets.mnist
#https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data 
# (60000,28,28) 28*28 1channel이 60000개 붙어있음

(x_train, y_train),(x_test, y_test) = tf.keras.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # normalization

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # 28*28=784개 노드
  tf.keras.layers.Dense(128, activation='relu'), # Dense 밀집  128개 거의 ReLU 사용
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax') # Dense 10개 mnist 클래스 number랑 같아야함 softmax 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)