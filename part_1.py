from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras import optimizers
import matplotlib.pyplot as plt

class Model():
  def __init__(self):
    self.model = keras.Sequential([Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    Conv2D(filters=16,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
    Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=128,activation='relu'),
    Dense(units=128,activation='relu'),
    Dense(units=10,activation='softmax')])

    self.datagen = ImageDataGenerator(width_shift_range=0.15, height_shift_range=0.15, horizontal_flip=True)

  def compileModel(self, lr = 0.001, momentum = 0.9):
    self.sgd = optimizers.SGD(lr=lr, momentum=0.9)
    self.model.compile(loss='categorical_crossentropy', optimizer=self.sgd, metrics = ['accuracy'])

  def fitModel(self, x_train, y_train, x_val, y_val, is_datagen = True):
    if is_datagen:
      self.datagen.fit(x_train)
    history = self.model.fit(self.datagen.flow(x_train, y_train, batch_size = 100), epochs=50, steps_per_epoch = len(x_train) // 100, validation_data = (x_val, y_val))
    # print(history.history)
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)
#convert to float32, so that devision does not make the fearues all zero
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#devide features by 255 so that pixels have value in the range of [0,1]
X_train /= 255
X_test /= 255
# convert class vectors to binary class matrices (one-hot encoding)
n_output=10
Y_train = np_utils.to_categorical(y_train, n_output)
Y_test = np_utils.to_categorical(y_test, n_output)


a = Model()
a.compileModel(lr = 0.001)
a.fitModel(X_train, Y_train, X_test, Y_test)

a = Model()
a.compileModel(lr = 0.1)
a.fitModel(X_train, Y_train, X_test, Y_test)


a = Model()
a.compileModel(lr = 1)
a.fitModel(X_train, Y_train, X_test, Y_test)