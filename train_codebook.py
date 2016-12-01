# -*- coding: utf-8 -*-
from get_local_features import get_local_features
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
def train_codebook(nclusters,normalized_descriptors):
    km = MiniBatchKMeans(nclusters)
    km.fit(normalized_descriptors)
    i = km.get_params() #carreguem els params del vector k
    nc = i['n_clusters'] #obtenim el numero de clusters des de la clase minibatchkmeans
    return km, nc
    
    #return kmeans(normalized_descriptors,nclusters) #obtenim els centroides de les imatges

if __name__ == "__main__":
    desc = get_local_features("../imagen_primerscript/people.jpg") #obtenim els descriptors de la imatge
    centroides,nc = train_codebook(2,desc) #ccalculem els centroides de la imatge
    #plt.scatter(desc[:,0],desc[:,1]),plt.scatter(centroides[:,0],centroides[:,1],color = 'r'),plt.show() #mostrem els centroides i els descriptors