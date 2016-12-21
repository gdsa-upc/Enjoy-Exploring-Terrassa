# -*- coding: utf-8 -*-
import os
from params import get_params
import sys
import os, time
import numpy as np
import pickle
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import os, sys
import pandas as pd
import kaggle_scripts as kaggle_scripts
from build_bow import *
from build_database import *
from eval_rankings import *
from get_assignments import *
from get_features import *
from get_local_features import *
from kaggle_scripts import *
from params import *
from rank import *
from train_codebook import *
tt = time.time()

# BUILD DATABASE---------------
params = get_params()
sys.path.insert(0,params['src'])
    
# Build image list for validation set
build_database(params)
    
# Switch to training set
params['split'] = 'train'
    
# Build image list for training set
build_database(params)

# GET FEATURES-----------------
params = get_params()

# Change to training set
params['split'] = 'train'
    
print "Apilant descriptors..."
# Save features for training set
t = time.time()
X, pca, scaler = stack_features(params)
print "Fet! Temps utilitzat:", time.time() - t
#print "Nombre de descriptors d'entrenament", np.shape(X)

print "Entrenant codebook..."
t = time.time()
train_codebook(params,X)
print "Fet! Temps utilitzat:", time.time() - t
    
print "Emmagatzemant baul de descriptors per al set d'entrenament..."
t = time.time()
get_features(params, pca,scaler)
print "Fet! Temps utilitzat:", time.time() - t

params['split'] = 'val'
    
print "Emmagatzemant baul de descriptors per al set de validacio..."
t = time.time()
get_features(params)
print "Fet! Temps utilitzat", time.time() - t

# RANK-------------------------
params = get_params()
rank(params)

# EVAL RANKINGS----------------
params = get_params()
    
ap_list, dict_ = eval_rankings(params)
    
print np.mean(ap_list)

for id in dict_.keys():
    if not id == 'desconegut':
        # We divide by 10 because it's the number of images per class in the validation set.
        print id, dict_[id]/10
'''
# ID de la imatge de validacio sense el '.jpg'
query_id='168-2743-15592'
single_eval(params,query_id)
'''

print "Temps total: ", time.time() - tt