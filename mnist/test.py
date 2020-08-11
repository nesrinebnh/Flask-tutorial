import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.models import model_from_json

#mnist = tf.keras.datasets.mnist
import cv2
"""
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)
score = model.evaluate(x_test, y_test)


"""
"""import keras
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()
print(y_train[0])

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

input_shape = (28,28,1)

x_train = x_train.astype("float32")
x_train /= 255


x_test = x_test.astype("float32")
x_test /= 255

print(y_train[0])

y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
print(y_train[0])

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(keras.layers.Conv2D(64,(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dropout((0.5)))
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train,
 batch_size=120, epochs = 20,
 validation_data=(x_test, y_test))

score = model.evaluate(x_test,y_test)

print('Test loss: ', score[0])
print('Test accuracy ',score[1])


img = cv2.imread('output.png',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(255-img,(28,28))
test = img.flatten()/255.0
test = test.reshape((1,28,28,1))
print('The answer is ',np.argmax(model.predict(test), axis=1))"""

json_file = open('model/model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("model/model.h5")
print("Loaded Model from disk")
#compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
loaded_model.run_eagerly = True
print("eager mode:: "+str(loaded_model.run_eagerly))

img = cv2.imread('output.png',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(28,28))
test = img.reshape((1,28,28,1))
print('The answer is ',np.argmax(loaded_model.predict(test), axis=-1))