import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

from keras.datasets import cifar10
from keras.utils import np_utils

from utils import *

import pickle

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train=X_train.astype(np.float32)
X_test=X_test.astype(np.float32)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
X_train /= 255
X_test /= 255
X_train=2*X_train-1
X_test=2*X_test-1


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

batch_size=100
lr=0.001
Training=True
Compressing=False

def get_model():
	batch_norm_alpha=0.9
	batch_norm_eps=1e-4

	model=Sequential()

	model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid',input_shape=[32,32,3]))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

	model.add(Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

	model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	#model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

	model.add(Flatten())

	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	return model

def convert_to_masked_model(model):
	prev_layer = None
	#implement a function that takes a model and returns another model with masked_conv and masked_dense layers
	ret_model = Sequential()
	for i,layer in enumerate(model.layers):

		if i == 0:
			curr_shape = model.input_shape
			prev_layer = layer
		else:
			curr_shape = prev_layer.compute_output_shape(curr_shape)
	 
		if isinstance(layer, Conv2D):
			pruned_layer = pruned_Conv2D(filters=64, kernel_size=3)#, strides=(1, 1), padding='valid',input_shape=[32,32,3]))
			pruned_layer.set_config(layer.get_config())
			pruned_layer.build(input_shape = curr_shape)
			pruned_layer.set_weights(layer.get_weights()+[pruned_layer.get_mask()])
		
		elif isinstance(layer, Dense):
			pruned_layer = pruned_Dense(n_neurons_out = 10)
			pruned_layer.set_config(layer.get_config())
			pruned_layer.build(input_shape = curr_shape)
			pruned_layer.set_weights(layer.get_weights()+[pruned_layer.get_mask()])
	 
		else:
		 	pruned_layer = layer
			 
		ret_model.add(pruned_layer)

		prev_layer = layer
		# pruned_layer.set_weights(layer.get_weights())
	return ret_model

model=get_model()

weights_path='pretrained_cifar10.h5'
model.load_weights(weights_path)
opt = keras.optimizers.Adam(lr=0.001,decay=1e-6)

#complie the model with sparse categorical crossentropy loss function as you did in part 1
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])

#make sure weights are loaded correctly by evaluating the model here and printing the output
print('Evaluating the original model:')
model.evaluate(X_test, y = Y_test)

#convert the layers to maskable_layers:
prunable_model=convert_to_masked_model(model)


opt = keras.optimizers.Adam(lr=0.001,decay=1e-6)
#now complie the prunable model with sparse categorical crossentropy loss function as you did in part 1
prunable_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])

#make sure weights are loaded correctly by evaluating the prunable model here and printing the output
print('Evaluating the masked model:')
print(prunable_model.evaluate(X_test, y = Y_test))


#do the rest of the project:
def prune_model(model, frac, layer_id):
  ctr = 1
  layer_to_prune = model.layers[0]
  for i, layer in enumerate(model.layers):
    if isinstance(layer, pruned_Conv2D) or isinstance(layer, pruned_Dense):
      if ctr == layer_id:
        layer_to_prune = layer
        break
      ctr += 1

  weight_list = layer_to_prune.get_weights()
  weights = weight_list[0]
  mask = get_mask(weights, frac)
  layer_to_prune.set_mask(mask)
  return model

def undo_pruning(model, layer_id):
  ctr = 1
  layer_to_reset = model.layers[0]
  for i, layer in enumerate(model.layers):
    if isinstance(layer, pruned_Conv2D) or isinstance(layer, pruned_Dense):
      if ctr == layer_id:
        layer_to_reset = layer
        break
      ctr += 1

  weight_list = layer_to_reset.get_weights()
  weights = weight_list[0]
  mask = np.ones(weights.shape)
  layer_to_reset.set_mask(mask)
  return model

def get_mask(weights, frac):
    weights_flat = np.abs(np.ravel(weights)) #np.abs needs to be done here
    print(weights_flat.shape)
    n_zeros = int(frac * weights_flat.shape[0])
    weights_flat_args = np.argpartition(weights_flat, n_zeros)
    mask = np.ones(weights_flat_args.shape[0])
    mask[weights_flat_args[:n_zeros]] = 0
    mask = np.reshape(mask, weights.shape)
    print(np.sum(mask))
    return mask


FRAC_LIST = []
i = 0.1
while i < 1:
  FRAC_LIST.append(i)
  i += 0.05
print(FRAC_LIST)

for layer in range(1, 10):
  accuracy = np.zeros((len(FRAC_LIST),)) #accuracy per layer
  loss = np.zeros((len(FRAC_LIST),))
  for i,frac in enumerate(FRAC_LIST):
    pruned_model = prune_model(prunable_model, frac, layer)
    pruned_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics = ['accuracy'])
    print('Evaluating model for LAYER : {} and PRUNED_FRACTION : {}'.format(layer, frac))
    result = pruned_model.evaluate(X_test, y = Y_test)
    accuracy[i] = result[1]
    loss[i] = result[0]
    print(result)
    prunable_model = undo_pruning(pruned_model, layer)
    # print(prunable_model.evaluate(X_test, y = Y_test))

  with open('accuracy_{}.pkl'.format(layer), 'wb+') as f:
    pickle.dump(accuracy, f)

  with open('loss_{}.pkl'.format(layer), 'wb+') as f:
    pickle.dump(loss, f)

  print("LAYER : {} DONE!".format(layer))
