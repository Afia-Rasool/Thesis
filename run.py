import keras
import numpy as np
import tensorflow as tf
from Segmentation_Models import load_DATA, pre_process
from Segmentation_Models import FPN, FPN_ASPP, SegNet, UNet, FCN
from Segmentation_Models import path
from Segmentation_Models import performance_measures, visualize_predictions, save_history, Plot_history
from keras.utils import plot_model
from segmentation_models.losses import bce_jaccard_loss
from math import floor
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard

dir_data, dir_seg, dir_img, dirr_data, dirr_seg, dirr_img = load_DATA()
X, Y, X_test, y_test = pre_process(dir_data, dir_seg, dir_img, dirr_data, dirr_seg, dirr_img)






config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.99
sess = tf.Session(config=config)



Model = 'FPN_ASPP'
model = FPN_ASPP('resnext50', input_shape=X[0].shape, classes=2, activation='sigmoid', encoder_weights='imagenet')
model.compile(keras.optimizers.Adam(lr=0.001), loss= bce_jaccard_loss , metrics=['accuracy'])
model.summary()
plot_model(model, to_file=path(Model) + '/model.png')



def step_decay(epoch):

    initial_lrate = 0.001
    drop = 0.1
    epochs_drop =5
    lrate = initial_lrate * pow(drop, floor((1+epoch)/epochs_drop))
    print('Learning rate at this epoch is: {}'.format(lrate))
    return lrate

filePATH=path(Model) + "/weights_best.hdf5"
checkpoint = ModelCheckpoint(filePATH, monitor='acc', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
tb=TensorBoard(log_dir=path(Model) +'/log', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False,
               write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
               embeddings_data=None, update_freq='epoch')
lrate = LearningRateScheduler(step_decay)
callbacks_list = [checkpoint, lrate, tb]
results= model.fit(
    x=X,
    y=Y,
    batch_size = 2,
    epochs=40,
    validation_split=0.1,
    verbose = 2,
    callbacks=callbacks_list
)



y_pred = model.predict(X_test, batch_size=2)
y_predi = np.argmax(y_pred, axis=3)
y_testi = np.argmax(y_test, axis=3)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)




save_history(results, Model)
Plot_history(Model)
performance_measures(y_testi, y_predi, Model)
visualize_predictions(X_test,y_predi ,y_testi)

####tensorboard --logdir=/home/afia/PycharmProjects/Thesis/Segmentation_Models/FPN/results/log

