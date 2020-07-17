# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
import tensorflow as tf


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

model1 = tf.keras.models.load_model('model.h5')


print("\nPREDICTIONS FOR ORIGINAL")
for img in range(0,4):
    predict_class = model1.predict_classes(X_test[[img],:])
    print("\nPrediction for index", img, "is -> " + str(predict_class), "-> Real result is ->", y_test[img])
    print("Probability matrix for index", img, " ->\n", model1.predict_proba(X_test[[img],:]))

model2 = tf.keras.models.load_model('model.h6')

print("\nPREDICTIONS FOR UPDATED")
for img in range(0,4):
    predict_class = model2.predict_classes(X_test[[img],:])
    print("\nPrediction for index", img, "is -> " + str(predict_class), "-> Real result is ->", y_test[img])
    print("Probability matrix for index", img, " ->\n", model2.predict_proba(X_test[[img], :]))

