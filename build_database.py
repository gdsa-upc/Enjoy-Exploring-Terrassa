# -*- coding: utf-8 -*-
import os # llibreria os per poder accedir al directori de treball
def build_database(dir_entrada,val_or_train,dir_sortida):
    images = os.listdir(dir_entrada) #Llegim el nom dels fitxers que hi ha en el directori que volem (el d'entrada)
    outfile = open(dir_sortida+'/outfile_'+val_or_train+'.txt','w') # Creem un fitxer per guardar les id's  
    for file in images:
        outfile.write(file[0:-4]+"\n") #cada linea del nou fitxer serà el nom de les imatge que hi ha en el directori menys l'extensio .jpg

ruta = os.path.dirname(os.path.abspath(__file__)) #ruta absoluta del projecte   
build_database(ruta+'/TerrassaBuildings900/train/images',"train",ruta+'/files'); #creacio train database
build_database(ruta+'/TerrassaBuildings900/val/images',"val",ruta+'/files'); #creacio val database