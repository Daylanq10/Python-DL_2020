from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# get dataset
dataset = pd.read_csv("breastcancer.csv")

# binary for diagnosis instead of M or B
label_enc = preprocessing.LabelEncoder()
dataset['diagnosis'] = label_enc.fit_transform(dataset['diagnosis'])

# Train data
X_train, X_test, Y_train, Y_test = train_test_split(dataset.iloc[:, 2:32], dataset['diagnosis'],
                                                    test_size=0.25, random_state=87)
# start modeling
np.random.seed(155)
my_first_nn = Sequential()  # create model
my_first_nn.add(Dense(60, input_dim=30, activation='relu'))  # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid'))  # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test))

# START OF SCALED VERSION
sc = StandardScaler()
to_scale = dataset.iloc[:, 2:32]
sc.fit(to_scale)
scaled = sc.transform(to_scale)

# data has been scaled
df = pd.DataFrame(data=scaled)

# training data
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(df, dataset['diagnosis'],
                                                            test_size=0.25, random_state=87)
# start modeling
np.random.seed(155)
my_second_nn = Sequential()  # create model
my_second_nn.add(Dense(60, input_dim=30, activation='relu'))  # hidden layer
my_second_nn.add(Dense(1, activation='sigmoid'))  # output layer
my_second_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_second_nn_fitted = my_second_nn.fit(X_train_2, Y_train_2, epochs=100,
                                       initial_epoch=0)
print(my_second_nn.summary())
print(my_second_nn.evaluate(X_test_2, Y_test_2))
