# -*- coding: utf-8 -*-
import os
import pickle #carreguem la llibreria pickle per poder treballar amb els diccionaris
import numpy as np
from sklearn.metrics import pairwise_distances

def rank(features_path,save_path,features_train,val_or_test,annotation):
    out = []
    ordenada = []
    featuresfile = open(features_path+'/bow_'+val_or_test+'.p','r') #obrim el diccionari de vectors de característiques de validació o de test
    train_featuresfile = open(features_train,'r') #obrim el diccionari de vectors de característiques d'entrenament
    annot = open(annotation+'/'+val_or_test+'/annotation.txt','r') #obrim el fixer anotacions del conjunt de validacio o de test
    desconegut = [] #Creem el vector desconegut en el qual guardarem totes les ids del conjunt de validació o test que tinguin categoria desconegut.
    for line in annot:
        tab = line.index("\t")
        fin = len(line)
        if str(line[tab+1:fin-1]) == "desconegut":
            desconegut.append(line[0:tab]) #Entrem totes les ids les quals la seva categoria sigui desconegut.
    rankfiles = pickle.load(featuresfile) #Carreguem el diccionari validació o test
    train = pickle.load(train_featuresfile) #carreguem el diccionari entrenament
    entrenament = train.keys() #afegim totes les ids segons les tenim ordenades en el diccionari en el vector entrenament
    for k in rankfiles.keys(): #per cada clau del diccionari dels vectors de caracteristiques de validació
                                #o test ens crearà un fitxer .txt guardat a la seva carpeta corresponent,
                                #el qual tindrà el rànking aleatori de les claus del diccionari d'entrenament
        if desconegut.count(k) == 0: #Si la id (key del diccionari de validacio o test) no apareix en el vector desconeguts, que ens calculi el rànking.
            outfile = open(save_path+'/ranking_'+val_or_test+'/'+k+'.txt','w')
            bow = rankfiles[k] #treiem el vector de caracteristiques de cada id de validació o de test 
            for k2 in train.keys():
                bow_train = train[k2] #treiem el vector de caracteristiques de cada id d'entrenament
                dist = pairwise_distances(bow,bow_train,metric='euclidean',n_jobs=1) #calculem les distancies euclidees entre el Bow de validació/test i els BOW d'entrenament
                out.append(dist) #Guardem les distancies en out (cada distància esta guardada en aquest vector que la seva corresponent id en el vector d'entrenament)
            ordenada = sorted(out) #Ordenem la llista out de menys a més distància a la imatge que estem estudiant. Ho guardem a l'array ordenada.
            for item in ordenada:
                position = out.index(item) #busquem la posicio en el vector out de les distancies ordenades
                outfile.write(entrenament[position]+"\n") #el fitxer de sortida marcarà les ids que més s'aproximin (en distàncies) en ordre.
            out = []
            ordenada = []
            outfile.close()
            # Finalment, totes les ids (key del diccionari de validacio o test) que pertenyen a la classe desconegut seran ignorades a l'hora de crear els rànkings
    featuresfile.close()
    train_featuresfile.close()
    annot.close()

if __name__ == "__main__":
    ruta = os.path.dirname(os.path.abspath(__file__)) #ruta absoluta del projecte
    rank(ruta+'/files',ruta+'/files',ruta+'/files/bow_train.p',"val",ruta+'/TerrassaBuildings900') #crida a la funció rank pel diccionari de validació