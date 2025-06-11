import math
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class KMeans:
    def __init__(self, X,n_centroid, n_train):
        self.n_centroid = n_centroid
        self.n_train = n_train
        self.X = X


    def distance(self,x1:np.array,x2:np.array):
        return math.sqrt(np.sum((x1-x2)**2))
    
    def initialize_centroids(self):
        X = self.X
        idx = np.random.choice(np.shape(X)[0],self.n_centroid, replace = False)
        self.centroids = X[idx]
        return self.centroids
    def find_center(self,Cluster):
        center = np.mean(Cluster, axis=0)
        return center
    def build_cluster(self):
        centers = self.centroids
        x,y = self.X.shape
        z = self.n_centroid
        self.Clusters =  [[] for _ in range(z)]

        for i in range(x):
            distances = [self.distance(self.X[i], centers[j]) for j in range(len(centers))]
            center_idx = np.argmin(distances)
            self.Clusters[center_idx].append(self.X[i])
        return self.Clusters
    def update_center(self ):
        self.centroids = np.array([self.find_center(c) for c in self.Clusters])
        return self.centroids
    def train(self):
        n = self.n_train
        self.initialize_centroids()
        for _ in tqdm(range(n)):
            self.build_cluster()
            self.update_center()
            
    def plot_clusters(self):
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        plt.figure(figsize=(8, 6))

        # Tracer les points par cluster
        for i, cluster in enumerate(self.Clusters):
            cluster = np.array(cluster)
            if len(cluster) == 0:
                continue
            plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6)

        # Tracer les centres
        centers = np.array(self.centroids)
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centres')

        plt.title('K-means Clustering')
        plt.legend()
        plt.grid(True)
        plt.show()

# Exemple
X = np.random.rand(100, 2) 
model = KMeans(X, n_centroid=5, n_train=100)
model.train()

# Résultat :
print("Centres finaux :\n", model.centroids)

print([tuple(model.distance(X[i],model.centroids[j]) for j in range(model.n_centroid)) for i in range(X.shape[0])])
model.plot_clusters()