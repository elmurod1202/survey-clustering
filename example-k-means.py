# An example of k-means clustering in python.
# This code was inspired by (and mostly taken from) a tutorial: https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
# Author: Elmurod Kuriyozov (elmurod1202@gmail.com)
# Date: March 25, 2022


# How it works
# 1. Select K (i.e. 2) random points as cluster centers called centroids
# 2. Assign each data point to the closest cluster by calculating its distance with respect to each centroid
# 3. Determine the new cluster center by computing the average of the assigned points
# 4. Repeat steps 2 and 3 until none of the cluster assignments change

# Importing nececcary libraries:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans



# Let's generate our own example data using the make_blobs function from the sklearn.datasets module.
# The centers parameter specifies the number of clusters.
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Uncomment these lines if you want to see the created plot:
# plt.scatter(X[:,0], X[:,1])
# plt.show()



# One problem that may arise is that you may need the optimal number of clusters, this can also be solved:
# WCSS is defined as the sum of the squared distance between each member of the cluster and its centroid.


# Now, let's try to find out the optimal number of clusters for the plot we have created using the elbow method.
# To get the values used in the graph, we train multiple models using a different number of clusters
#  and storing the value of the intertia_ property (WCSS) every time.

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Uncomment these lines if you want to see the created plot:
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()


# We graph the relationship between the number of clusters and Within Cluster Sum of Squares (WCSS),
#  then we select the number of clusters where the change in WCSS begins to level off (elbow method).
# From our example we can see that the optimal number of clusters is 4.


# Next, we’ll categorize the data using the optimum number of clusters (4) we determined in the last step. 
# k-means++ ensures that you get don’t fall into the random initialization trap.

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)

# Uncomment these lines if you want to see the created plot:
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

