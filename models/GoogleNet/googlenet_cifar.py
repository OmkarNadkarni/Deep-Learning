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
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,GlobalAvgPool2D,BatchNormalization,Concatenate,Input,Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def inception_block(input_tensor,num_of_filters):
  l1 = Conv2D(num_of_filters[0],kernel_size=1,activation='relu')(input_tensor)

  l2 = Conv2D(num_of_filters[1],kernel_size=1,activation='relu')(input_tensor)
  l2 = Conv2D(num_of_filters[2],kernel_size=3,padding='same',activation='relu')(l2)

  l3 = Conv2D(num_of_filters[3],kernel_size=1,activation='relu')(input_tensor)
  l3 = Conv2D(num_of_filters[4],kernel_size=5,padding='same',activation='relu')(l3)

  l4 = MaxPool2D(3,strides=1,padding='same')(input_tensor)
  l4 = Conv2D(num_of_filters[5],kernel_size=1,activation='relu')(l4)

  output_layer = Concatenate(axis=-1)([l1,l2,l3,l4])
  return output_layer

def GoogleNetModel(data_input_shape,num_of_classes):
  input_tensor = Input(data_input_shape)
  x = Conv2D(64,kernel_size=7,strides=2,padding='same',activation='relu')(input_tensor)
  x = MaxPool2D(3,strides=2,padding='same')(x)
  x = BatchNormalization()(x)

  x = Conv2D(64,1,activation='relu')(x)
  x = Conv2D(192,kernel_size=3,strides=1,padding='same',activation='relu')(x)
  x = BatchNormalization()(x)
  x = MaxPool2D(3,strides=2,padding='same')(x)

  x = inception_block(x,[64,96,128,16,32,32])
  x = inception_block(x,[128,128,192,32,96,64])
  x = MaxPool2D(3,strides=2,padding='same')(x)

  x = inception_block(x,[192,96,208,16,48,64])
  x = inception_block(x,[160,112,224,24,64,64])
  x = inception_block(x,[128,128,256,24,64,64])
  x = inception_block(x,[112,144,288,32,64,64])
  x = inception_block(x,[256,160,320,32,128,128])
  x = MaxPool2D(3,strides=2,padding='same')(x)

  x = inception_block(x,[256,160,320,32,128,128])
  x = inception_block(x,[384,192,384,48,128,128])
  x = GlobalAvgPool2D()(x)
  x = Dropout(0.4)(x)
  output_layer = Dense(num_of_classes,activation='softmax')(x)
  model = Model(input_tensor,output_layer)
  return model

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

(x_train, y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#normalizing the data
x_train /=255
x_test /=255

x_train = (x_train-0.5)/0.5
x_test = (x_test-0.5)/0.5

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

training_data = x_train[:40000]
training_labels = y_train[:40000]
val_data = x_train[40000:]
val_labels = y_train[40000:]


input_shape = (32,32,3) #mnist shape
num_of_classes = 10 #output classes
es = EarlyStopping(monitor='val_accuracy',mode='max',patience=5)
opt = SGD(learning_rate=0.01,momentum=0.9)
EPOCHS = 15
BATCH_SIZE = 100

GNmodel = GoogleNetModel(input_shape,num_of_classes)
GNmodel.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
print("\nTRAINING MODEL ON ORIGINAL DATA\n")
mnist_history = GNmodel.fit(training_data,training_labels,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=1,callbacks=[es],validation_data=(val_data,val_labels))
score = GNmodel.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print("mnist TEST DATA: loss = %f accuracy = %f"%(score[0],score[1]))
plot_graphs(mnist_history,'GoogleNet_cifar_model')

GNmodel2 = GoogleNetModel(input_shape,num_of_classes)
GNmodel2.compile(opt,loss='categorical_crossentropy',metrics=['accuracy'])
datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
datagen.fit(training_data)
# fits the model on batches with real-time data augmentation:
print("\nTRAINING MODEL ON AUGMENTED DATA\n")
batchSize = 32
mnist_history2 = GNmodel2.fit_generator(datagen.flow(training_data, training_labels, batch_size=batchSize),
                    steps_per_epoch=len(training_data) / batchSize, epochs=EPOCHS,callbacks=[es],validation_data=(val_data,val_labels))
score = GNmodel2.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print("mnist TEST DATA: loss = %f accuracy = %f"%(score[0],score[1]))
plot_graphs(mnist_history2,'GoogleNet_cifar_with_augmentation')
