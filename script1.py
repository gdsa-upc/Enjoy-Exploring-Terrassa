# -*- coding: utf-8 -*-
# Proyecto 101.2

import cv2
import numpy as np

img = cv2.imread('C:\Users\Albert\Documents\UNI\Q-5\GDSA\Projecte\testimage.jpg',0)

# Pasamos a nivel de gris
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Iniciamos el detector SIFT
sift = cv2.SIFT()

# Detectamos puntos de interés, los computamos y los ubicamos
kp, des = sift.detectAndCompute(gray,None)

# Dibuja los puntos en una imagen de salida llamada 'sift_keypoints.jpg'
img = cv2.drawKeypoints(gray,kp)

# Creamos la imagen resultante
cv2.imwrite('sift_keypoints.jpg',img)