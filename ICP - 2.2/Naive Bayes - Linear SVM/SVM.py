import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# DATA SETUP
data = pd.read_csv("glass.csv")

# SPLIT TO TRAIN AND TEST
X_test, X_train = train_test_split(data, test_size=0.2, random_state=0)

X = data.drop("Type", axis=1)
Y = data["Type"]

# SPLIT TO TRAIN AND TEST AGAIN
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

# SVC SETUP
svc = SVC()

# TRAIN
svc.fit(X_train, Y_train)

# PREDICTION
Y_pred = svc.predict(X_test)

# OUTPUT
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_svc)

print(classification_report(Y_test, Y_pred))
