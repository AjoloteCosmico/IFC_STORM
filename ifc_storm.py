from __future__ import division   # impone aritmética no entera en la división
from PIL import Image             # funciones para cargar y manipular imágenes
import numpy as np                # funciones numéricas (arrays, matrices, etc.)
import matplotlib.pyplot as plt 
import cv2
from sklearn.cluster import DBSCAN
import sys
np.set_printoptions(threshold=sys.maxsize)

def findCentroids(I,epsilon,tresh,minimo):
   # I1 = I.convert('L')
   # M=np.array(I1) 
    M=I
    i=0 
    K=np.zeros((M.size,2))
    row, col = M.shape              #Guardamos las coordenadas de todos los pixeles con una intensidad mayor a 20
    for y in range(0, row):
        for x in range (1,col):
            if(M[y,x]>tresh):  
                K[i,0]=x
                K[i,1]=y
                i+=1
            
    db = DBSCAN(eps=epsilon, min_samples=minimo).fit(K[0:i,])

    #Hallar los centroides de cada cluster
    n=np.amax(db.labels_)
    Centros=np.zeros((n+1,3))  

    
    for t in range(0, i):
        if(db.labels_[t]!=-1):
            Centros[db.labels_[t],0]+=K[t,0]*M[int(K[t,1]),int(K[t,0])] #numerador del componente en x
            Centros[db.labels_[t],1]+=K[t,1]*M[int(K[t,1]),int(K[t,0])] #numerador del componente en y
            Centros[db.labels_[t],2]+=M[int(K[t,1]),int(K[t,0])]        #denominador de ambos componentes
            
    return Centros
