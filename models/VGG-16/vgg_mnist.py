# -*- coding: utf-8 -*-

import keras
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
import keras.backend as K
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.callbacks import EarlyStopping
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,GlobalAvgPool2D,BatchNormalization,Concatenate,Input,Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def plot_graphs(history,model_str):
  acc1_fig = plt.figure()
  acc_fig = acc1_fig.add_subplot(1,1,1)
  acc_fig.plot(history.history['accuracy'])
  acc_fig.plot(history.history['val_accuracy'])
  acc_fig.set_title('model accuracy')
  acc_fig.set_ylabel('accuracy')
  acc_fig.set_xlabel('epoch')
  acc_fig.legend(['train', 'validation'], loc='upper left')
  #plt.show()
  acc1_fig.savefig(model_str+'_accuracy')


  # summarize history for loss
  loss_fig1 = plt.figure()
  loss_fig = loss_fig1.add_subplot(1,1,1)
  loss_fig.plot(history.history['loss'])
  loss_fig.plot(history.history['val_loss'])
  loss_fig.set_title('model loss')
  loss_fig.set_ylabel('loss')
  loss_fig.set_xlabel('epoch')
  loss_fig.legend(['train', 'validation'], loc='upper left')
  #plt.show()
  loss_fig1.savefig(model_str +'_loss')

"""**VGG MODEL IMPLEMENTATION**"""

def VGGNet(data_input_shape, num_of_classes):
  input_layer = Input(data_input_shape)
  x = Conv2D(64,3,padding='same',activation='relu')(input_layer)
  x = Conv2D(64,3,padding='same',activation='relu')(x)
  x = MaxPool2D(2,strides=2,padding='same')(x)

  x = Conv2D(128,3,padding='same',activation='relu')(x)
  x = Conv2D(128,3,padding='same',activation='relu')(x)
  x = MaxPool2D(2,strides=2,padding='same')(x)

  x = Conv2D(256,3,padding='same',activation='relu')(x)
  x = Conv2D(256,3,padding='same',activation='relu')(x)
  x = Conv2D(256,3,padding='same',activation='relu')(x)
  x = MaxPool2D(2,strides=2,padding='same')(x)

  x = Conv2D(512,3,padding='same',activation='relu')(x)
  x = Conv2D(512,3,padding='same',activation='relu')(x)
  x = Conv2D(512,3,padding='same',activation='relu')(x)
  x = MaxPool2D(2,strides=2,padding='same')(x)

  x = Conv2D(512,3,padding='same',activation='relu')(x)
  x = Conv2D(512,3,padding='same',activation='relu')(x)
  x = Conv2D(512,3,padding='same',activation='relu')(x)
  x = MaxPool2D(2,strides=2,padding='same')(x)

  x =Flatten()(x)
  x = Dense(4096,activation='relu')(x)
  x = Dense(4096,activation='relu')(x)
  output_layer = Dense(num_of_classes,activation='softmax')(x)
  model = Model(input_layer,output_layer)
  return model

"""two models will be trained on the same dataset.
One with the original data and the second model with data augmentation.
"""
(x_train, y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test  = x_test.reshape(x_test.shape[0],28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#normalizing the data
x_train /=255
x_test /=255
#x_train = (x_train-0.5)/0.5
#x_test = (x_test-0.5)/0.5

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

training_data = x_train[:48000]
training_labels = y_train[:48000]
val_data = x_train[48000:]
val_labels = y_train[48000:]

input_shape = (28,28,1) #mnist shape
num_of_classes = 10 #output classes
es = EarlyStopping(monitor='val_accuracy',mode='max',patience=5)
opt = SGD(learning_rate=0.01,momentum=0.9)
BATCH_SIZE = 100
EPOCHS = 15
vgg_model = VGGNet(input_shape,num_of_classes)
vgg_model.compile(opt,loss='categorical_crossentropy',metrics=['accuracy'])

print("\n TRAINING ON ORIGINAL MNIST DATA \n")
mnist_history = vgg_model.fit(training_data,training_labels,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=1,callbacks=[es],validation_data=(val_data,val_labels))
score = vgg_model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)

print("mnist TEST DATA: loss = %f accuracy = %f"%(score[0],score[1]))
plot_graphs(mnist_history,'VGG_mnist_model')

batchSize = 32
vgg_model2 = VGGNet(input_shape,num_of_classes)
vgg_model2.compile(opt,loss='categorical_crossentropy',metrics=['accuracy'])
datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,width_shift_range=0.2,height_shift_range=0.2)
datagen.fit(training_data)

# fits the model on batches with real-time data augmentation:
print("\n TRAINING ON AUGMENTED MNIST DATA \n")

mnist_history2 = vgg_model2.fit_generator(datagen.flow(training_data, training_labels, batch_size=batchSize),
                    steps_per_epoch=len(training_data) / batchSize, epochs=EPOCHS,callbacks=[es],validation_data=(val_data,val_labels))
score = vgg_model2.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print("mnist TEST DATA: loss = %f accuracy = %f"%(score[0],score[1]))
plot_graphs(mnist_history2,'VGG_mnist_with_augmentation')
