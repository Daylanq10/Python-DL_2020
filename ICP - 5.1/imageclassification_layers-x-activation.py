from keras import Sequential
from keras.datasets import mnist
import numpy as np
from keras.layers import Dense
from keras.utils import to_categorical
import os

# this with import os to fix error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape[1:])
# process the data
# 1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
print(dimData)
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
# scale data
train_data /= 255.0
test_data /= 255.0
# change the labels frominteger to one-hot encoding. to_categorical is doing the same thing as LabelEncoder()
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# creating network
model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=(dimData,)))
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(512, activation='sigmoid')) # extra layer added and all changed to sigmoid
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=10, verbose=1,
                    validation_data=(test_data, test_labels_one_hot))

# look at loss and accuracy
[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation results on Test Data: Loss = {}, accuracy = {}".format(test_loss, test_acc))

# ran this with 'sigmoid', 'tanh', and 'relu' to see what happened.  Both sigmoid and tanh
# caused the data loss to be higher and accuracy to be lower.
# In one test Loss = 0.202 and 0.936 for sigmoid
# Similar results for tanh
# Accuracy for relu was a little higher in accuracy but also higher in data loss
# Also took a lot longer to run
