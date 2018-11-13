from keras.callbacks import ModelCheckpoint, TensorBoard

import LoadBatches
from Models import FCN8,FCN32
from keras import optimizers
import math

#############################################################################
train_images_path = "data/dataset1/images_prepped_train/"
train_segs_path = "data/dataset1/annotations_prepped_train/"
train_batch_size = 16
n_classes =11

epochs = 200

input_height=320
input_width=320



val_images_path = "data/dataset1/images_prepped_test/"
val_segs_path = "data/dataset1/annotations_prepped_test/"
val_batch_size = 16

key="fcn32"


##################################

method={"fcn32":FCN32.FCN32,"fcn8":FCN8.FCN8}

m = method[key](n_classes,input_height=input_height,input_width=input_width)
m.compile(loss='categorical_crossentropy',optimizer="adadelta",metrics=['acc'])

G = LoadBatches.imageSegmentationGenerator(train_images_path,
                                   train_segs_path, train_batch_size, n_classes=n_classes, input_height=input_height,input_width=input_width)

G_test=LoadBatches.imageSegmentationGenerator(val_images_path,
                                   val_segs_path, val_batch_size, n_classes=n_classes, input_height=input_height,input_width=input_width)

checkpoint = ModelCheckpoint(filepath="output/%s_model.h5" % key, monitor='acc', mode='auto', save_best_only='True')
tensorboard = TensorBoard(log_dir='output/log_%s_model' % key,write_graph=False,histogram_freq=1,write_grads=True,write_images=True)

m.fit_generator(generator=G,
                      steps_per_epoch=math.ceil(367./train_batch_size),
                      epochs=epochs,callbacks=[],
                      verbose=1,
                      validation_data=G_test,
                      validation_steps=8,
                      shuffle=True)
