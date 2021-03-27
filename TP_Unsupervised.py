#librairies
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage , fcluster
import numpy as np
import pandas
from sklearn import cluster
from sklearn import metrics

#chargement des données
data = pd.read_csv("dataset.csv",header=0)
data.drop('class',1,inplace=True)
print(data.shape)

#statistiques descriptives
print(data.describe())

#Croisement 2 à 2 des variables
scatter_matrix(data,figsize=(9,9))
plt.show()

#CAH
Z = linkage(data,method='ward',metric='euclidean') 
dendrogram(Z,truncate_mode='lastp',p=10,leaf_rotation=45,leaf_font_size=10,show_contracted=True)
plt.title('CAH')
plt.xlabel('Cluster Size')
plt.ylabel('distance')
plt.axhline(y=85) #le trait dans le dendrograme
plt.show()

#création et affichage des groupes :
clusters = fcluster(Z,t='700',criterion='distance')
print(clusters)

idg = np.argsort(clusters)
print(pd.DataFrame(data.index[idg],clusters[idg]))


#k-means
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(data)
idk = np.argsort(kmeans.labels_)

#affichage des observations et leurs groupes
print(pd.DataFrame(data.index[idk],kmeans.labels_[idk]))

#distances aux centres de classes des observations
print(kmeans.transform(data))

#correspondance K-means et  CAH
pd.crosstab(clusters,kmeans.labels_)

#utilisation de la métrique "silhouette"
res = np.arange(9,dtype="double")
for k in np.arange(9):
   km = cluster.KMeans(n_clusters=k+2)
   km.fit(data)
   res[k] = metrics.silhouette_score(data,km.labels_)
print(res)
plt.title("Silhouette")
plt.xlabel("number of clusters")
plt.plot(np.arange(2,11,1),res)
plt.show()