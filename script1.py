# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os #carreguem la llibreria os per tal d'obtenir la ruta absoluta de la carpeta del projecte

ruta = os.path.dirname(os.path.abspath(__file__)) #obtenim la ruta absoluta de la carpeta del projecte
img = cv2.imread(ruta+'/desconegut_257.jpg',1) #obrim la imatge que hi es a la carpeta images
surf = cv2.SURF(4000) #posem el llindar hessian a 4000
kp, des = surf.detectAndCompute(img, None) 
img2 = cv2.drawKeypoints(img, kp, None, (255,0,0), 4) #dibuixem els punts d'interes i ho guardem a la variable img2
plt.imshow(img2),plt.show() #mostrem la imatge amb els punts d'interes dibuixats
print len(kp) #mostra el nom de punts d'interés que hi han a la imatge
