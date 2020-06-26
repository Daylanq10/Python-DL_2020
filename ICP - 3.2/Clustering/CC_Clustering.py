import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics

# TAKE IN DATA FROM THE CSV
dataset = pd.read_csv('CC.csv')

# CHECK FOR NULLS
nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# REPLACE NULL VALUES WITH MEAN OF COLUMNS
dataset = dataset.fillna(dataset.mean())
print(sum(dataset.isnull().sum() != 0), '\n')

# USED ONLY 'PURCHASES' AND 'ONEOFF_PURCHASES'
x = dataset[["PURCHASES", "ONEOFF_PURCHASES"]].iloc[:, :]

# LOOK AT DATA
sns.FacetGrid(x, height=6).map(plt.scatter, "PURCHASES", "ONEOFF_PURCHASES").add_legend()
plt.title("LOOK AT DATA")
plt.show()

# ELBOW GRAPH
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# GIVE NUMBER OF CLUSTERS TO BE FOUND
nclusters = 3
seed = 0


# SCORE FOR X
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(x)
y_cluster_kmeans = km.predict(x)
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("The score for X is ", score)

# SCORE FOR SCALED X
scaler = preprocessing.StandardScaler()
scaler.fit(x)
x_scaled_array = scaler.transform(x)
x_scaled = pd.DataFrame(x_scaled_array, columns=x.columns)
km.fit(x_scaled)
y_cluster_kmeans = km.predict(x_scaled)
score = metrics.silhouette_score(x_scaled, y_cluster_kmeans)
print("The score for X scaled is ", score)

#PCA
pca = PCA(2)
x_pca = pca.fit_transform(x_scaled)
df2 = pd.DataFrame(data=x_pca, columns=["PURCHASES", "ONEOFF_PURCHASES"])
df2['results'] = km.predict(df2)
score = metrics.silhouette_score(x_pca, df2['results'])
print("the score for X pca is ", score)

# GRAPH FOR X_PCA
sns.FacetGrid(df2, hue="results", height=4).map(plt.scatter, "PURCHASES", "ONEOFF_PURCHASES").add_legend()
plt.title("X_PCA CLUSTERED")
plt.show()

# GRAPH FOR X_SCALED
df2 = pd.DataFrame(data=x_scaled_array, columns=["PURCHASES", "ONEOFF_PURCHASES"])
df2['results'] = km.predict(df2)
sns.FacetGrid(df2, hue="results", height=4).map(plt.scatter, "PURCHASES", "ONEOFF_PURCHASES").add_legend()
plt.title("X_SCALED CLUSTERED")
plt.show()
