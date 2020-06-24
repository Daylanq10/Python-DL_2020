import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# CSV CONTENTS
train = pd.read_csv('train.csv')
train.SalePrice.describe()

# CHECK FOR NULL VALUES
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum() != 0))

# SCATTER PLOT WITH OUTLIERS
plt.xlabel("Garage Area")
plt.ylabel("Sale Price")
plt.scatter(x=train["GarageArea"], y=train["SalePrice"])
plt.show()

# RECONSTRUCT DATA WITHOUT OUTLIERS USING Z-SCORE

outlier = train[np.abs(stats.zscore(train["GarageArea"]) > 3)]
updated = train[np.abs(stats.zscore(train['GarageArea']) < 2.9)]
updated = updated[np.abs(stats.zscore(updated['GarageArea']) > -2)]

# SCATTER PLOT WITHOUT OUTLIERS
plt.xlabel("Garage Area")
plt.ylabel("Sale Price")
plt.scatter(x=updated.GarageArea, y=updated.SalePrice)
plt.show()


# SHOW HOW OUTLIERS COULD SKEW FURTHER DATA
box = [train.GarageArea, updated.GarageArea]
plt.boxplot(box)
plt.show()

