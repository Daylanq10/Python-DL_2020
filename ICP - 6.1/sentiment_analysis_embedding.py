import numpy
from keras.layers import Embedding, Flatten
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('imdb_master.csv', encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values

# Gives possible y values for final layer
print("Classes: ")
print(numpy.unique(y))

# tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
# getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)  # This creates the matrix corresponding to words

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

X_train = pad_sequences(X_train, maxlen=2000)
X_test = pad_sequences(X_test, maxlen=2000)

input_dim = 2000  # since num_words = 2000

model = Sequential()
model.add(Embedding(2000, 50, input_length=2000))
model.add(Flatten())
model.add(layers.Dense(300, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
# 3 for output layer because 3 options for result
# softmax because we are not dealing with binary for sigmoid
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test), batch_size=256)

[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation results on Test Data: Loss = {}, accuracy = {}".format(test_loss, test_acc))

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predict = model.predict_classes(X_test[[2000],:])
print(predict[0])
print(y_test[2000])