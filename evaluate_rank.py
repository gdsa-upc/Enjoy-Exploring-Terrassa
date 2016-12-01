# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image,ImageOps

def border(name_image): #Funcio creada per fer un top5 del rànking creat que escollim
    mostra = open("../files/ranking_val/"+name_image+".txt",'r')
    print "Imatge mostra:"
    imagemostra = Image.open("../TerrassaBuildings900/val/images/168-2743-15592.jpg")
    y = ImageOps.expand(imagemostra,border=50,fill='blue')
    plt.figure(1),plt.imshow(y),plt.show()
    print "\n"
    print "Rànking mostra:"
    mostra_val = open("../TerrassaBuildings900/val/annotation.txt",'r')
    for line in mostra_val:
        id_im = line[0:line.index("\t")]
        if str(id_im) == name_image:
            clase_mostra = line[line.index("\t")+1:line.index("\n")]

    top = 0
    for line in mostra:
        if top<5:
            id_image = line[0:line.index("\n")]
            image_rank = Image.open("../TerrassaBuildings900/train/images/"+id_image+".jpg")
            mostra_train = open("../TerrassaBuildings900/train/annotation.txt",'r')
            for lin in mostra_train:
                id_train = lin[0:lin.index("\t")]
                if id_train == id_image:
                    clase = lin[lin.index("\t")+1:lin.index("\n")]
            if clase == clase_mostra:
                y2 = ImageOps.expand(image_rank,border=50,fill='green')
                plt.figure(top+2),plt.imshow(y2),plt.show()
            else:
                y3 = ImageOps.expand(image_rank,border=50,fill='red')
                plt.figure(top+2),plt.imshow(y3),plt.show()
            clase.strip()
        mostra_train.close()
        top += 1
        
        
def evaluate_rank(dir_rank):
    nfiles = os.listdir(dir_rank)
    ground_truth_val = open("../TerrassaBuildings900/val/annotation.txt", "r")
    ground_truth_train = open("../TerrassaBuildings900/train/annotation.txt","r")
    truth = {} #inicialitzem una taula on l'index es la id de la imatge i conté la seva categoria
    AP = {}
    next(ground_truth_val)#eliminem la primera linia de l'arxiu ja que no ens interessa
    next(ground_truth_train)#eliminem la primera linia de l'arxiu ja que no ens interessa
    for line in ground_truth_val:
        id_foto = line.index("\t")
        final = line.index("\n")
        truth[line[0:id_foto]] = line[id_foto+1:final] #guardem la categoria de cada imatge a un vector
    for line in ground_truth_train:
        id_foto = line.index("\t")
        final = line.index("\n")
        truth[line[0:id_foto]] = line[id_foto+1:final] #guardem la categoria de cada imatge a un vector
    MAP = 0
    APC = 0
    for file in nfiles:
        ranking = open(dir_rank+"/"+file,"r")#obrim l'arxiu rank d'una imatge de cerca
        filename = file[0:file.index(".")]
        categoria = truth[filename] #assignem la categoria que té la imatge de cerca
        relevants = 0
        precision = 0
        AP[filename] = 0
        irrelevants = 0
        k = 0
        for line in ranking:
            final = line.index("\n")
            k += 1
            if truth[line[0:final]] == categoria: #si la id de la imatge coincideix amb la categoria sumem + 1 a relevants
                relevants += 1 
                precision = precision + (float(relevants)/float(k)) #calculem la precisio per cada k
            else:
                irrelevants += 1
        AP[filename] = float(precision)/float(relevants) #calculem la AP de cada imatge de cerca
        APC += AP[filename]  #calculem la AP acumulada de cada imatge de cerca
        ranking.close()
    MAP = APC/len(nfiles) #calcul del MAN
    return AP, MAP #retornem els valors de AP de cada imatge i de MAN

AP,MAP = evaluate_rank("../files/ranking_val")