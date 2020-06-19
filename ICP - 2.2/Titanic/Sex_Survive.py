import pandas as pd
import seaborn as sns

# PLACE CSV CONTENT
X_train = pd.read_csv("train_preprocessed.csv")

print(X_train.columns.values)

# CHECK FOR CORRELATION
corr = X_train['Survived'].corr((X_train['Sex']))

print(corr)
