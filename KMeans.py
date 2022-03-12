''' KMeans :
Select random centers and create categories.
Then take the average points of each category and select a new center until 'K' can kneel at its loss
'''

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

X , Y = make_blobs(1000 , centers=[[1,2],[3,2],[5,5],[6,4],[2,1]]) # make clusters

k_clusters = KMeans(n_clusters=5,      # number of output clusters
                    n_init=16,         # number of times selection centers
                    init="k-means++"   # algoritym of selection centers
                    )

k_clusters.fit(X)
y_hat = k_clusters.predict(X)

plt.scatter(x=X[:,0],y=X[:,1],c=y_hat)
plt.show()
