from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
from keras.datasets import mnist, fashion_mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

history = autoencoder.fit(x_train, x_train,
                          epochs=5,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test))

# get prediction of selected image data
img = x_test[0]  # get image at location
test_img = img.reshape((1, 784))  # reshape to fit
prediction = autoencoder.predict(test_img)

# before reconstruction
plt.imshow(x_train[0].reshape(28, 28))
plt.show()

# after reconstruction
plt.imshow(prediction.reshape(28, 28))
plt.show()

# summarize history for accuracy/loss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy/loss')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

encoder = Model(input_img, encoded)
img = x_test[0]  # get image at location
test_img = img.reshape((1, 784))  # reshape to fit
hidden_layer = encoder.predict(test_img)
plt.imshow(hidden_layer)
plt.show()

##x_test_encoded = autoencoder.predict(x_test, batch_size=269)
##plt.figure(figsize=(6, 6))
##plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
##plt.colorbar()
##plt.show()

##decoded_imgs = autoencoder.predict(x_test)

##plt.show()
