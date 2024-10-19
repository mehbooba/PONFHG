from sklearn.cluster import KMeans
import numpy as np
def spectral_clust(adj_mat):
	D = np.diag(adj_mat.sum(axis=1))
	L = D-adj_mat
	vals, vecs = np.linalg.eig(L)
	vecs = vecs[:,np.argsort(vals)]
	vals = vals[np.argsort(vals)]
	kmeans = KMeans(n_clusters=4)
	kmeans.fit(vecs[:,1:4])
	colors = kmeans.labels_

	print("Clusters:", colors)

