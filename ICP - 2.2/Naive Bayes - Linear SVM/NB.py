import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# DATA SETUP
data = pd.read_csv("glass.csv")

# CHECKED FOR NULLS (NONE)
print(data.isnull().sum())

# SPLIT TO TRAIN AND TEST
X_test, X_train = train_test_split(data, test_size=0.2, random_state=0)

X = data.drop("Type", axis=1)
Y = data["Type"]

# SPLIT TO TRAIN AND TEST AGAIN
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

# GNB SETUP
gnb = GaussianNB()

# TRAIN
gnb.fit(X_train, Y_train)

# PREDICTION
Y_pred = gnb.predict(X_test)

# OUTPUT
acc_gnb = round(gnb.score(X_train, Y_train) * 100, 2)
print(acc_gnb)

print(classification_report(Y_test, Y_pred))
