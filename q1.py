# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:40:25 2019

@author: yashd
"""
import numpy as np
import matplotlib.pyplot as plt

class mykmeans():
    def __init__(self, limit=0.001, iter_limit=10000,):
        self.limit = limit
        self.iter_limit = iter_limit
        
    # Fitting Function for K-means 
    def fit(self,data,k,cent):
        self.centroids = {}
        
        for i in range(k):
            self.centroids[i] = cent[i]

        for i in range(self.iter_limit):
            self.classifications = {}
            
            for i in range(k):
                self.classifications[i] = []
            
            for feature in data:
                distances = [np.linalg.norm(feature-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature)
            
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            
            optimized = True
            
            for c in self.centroids:
                current_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-current_centroid)/current_centroid*100.0) > self.limit:
                    print(np.sum((current_centroid-current_centroid)/current_centroid*100.0))
                    optimized = False
                
            if optimized:
                break
    
    # Plotting Function for K-means
    def plot_graph(self,centroids,classifications):
        colors = 10*["g","r","c","b","k"]
        for centroid in cluster.centroids:
            plt.scatter(cluster.centroids[centroid][0], cluster.centroids[centroid][1], s = 130, marker = "D")
        
        for classification in cluster.classifications:
            color = colors[classification]
            for feature in cluster.classifications[classification]:
                plt.scatter(feature[0], feature[1], color = color,s = 30, marker = ".")
        #print("Final Centroid co-ordinates are:")
        #print(centroids)
        plt.show()
    
if __name__ == '__main__':
    # Initializing 2D Gaussian Random Data    
    mu1 = [1,0]
    mu2 = [0,1.5]
    cov1=[[0.9,0.4],[0.4,0.9]]
    cov2=[[0.9,0.4],[0.4,0.9]]
    
    # Generating 2D Gaussian Random Data
    x,y= np.random.multivariate_normal(mu1,cov1,500).T
    G1 = np.array(list(zip(x,y)))
    a,b = np.random.multivariate_normal(mu2,cov2,500).T
    G2 = np.array(list(zip(a,b)))
    X =np.concatenate((G1,G2),axis=0)
   
    #Initializing Centroids and No.of Clusters
    cent=[[10,10],[-10,-10],[-10,10],[10,-10]]
    k=4
    
    # Calling the K-Means Clustering Algorithm and fitting it to the Gaussian Data
    cluster = mykmeans()
    cluster.fit(X,k,cent)
    cluster.plot_graph(cluster.centroids,cluster.classifications)



