#-*- coding: utf-8 -*-    
import os
from scipy.cluster.vq import vq
from get_local_features import get_local_features
from train_codebook import train_codebook
from sklearn.cluster import MiniBatchKMeans

ruta = os.path.dirname(os.path.abspath(__file__)) # Definim la instrucció principal que busca la ruta absoluta del fitxer

def compute_assignments(codebook,desc):
    assignments = codebook.predict(desc) #calculem els assignaments
    return assignments
    
if __name__ == "__main__":
    dsc = []
    desc = get_local_features("../imagen_primerscript/people.jpg")
    desc2 = get_local_features("../imagen_primerscript/tiger.jpg")
    for feat in desc:
        dsc.append(feat)
    for feat in desc2:
        dsc.append(feat)
    codebook,_ = train_codebook(5,dsc) #entrenem el codebook
    clase = compute_assignments(codebook,dsc) #observem els assignaments